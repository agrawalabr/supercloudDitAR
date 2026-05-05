"""GPU CSV ETL — supercloud_power/configs/gpuETL.yaml

Pipeline (per job):
  1. Discover node CSVs, group by job_id.
  2. Per node: outlier-filter sampling rate → per-GPU rollup → bin to Δt grid → ffill.
  3. Cross-node aggregation per bin (sum/mean per metric spec).
  4. Dense regular grid 0..k_max; fill gaps; emit relative-time parquet.
  5. Embed trace metadata (nodes_used, length, duration_sec, delta_t) in the
     parquet footer so gpu_traces.csv can be built/extended at any time without
     reprocessing, and without any shared-state or locking.

Output layout: (timestamp, gpu_used_pct, memory_used_pct, memory_used_MiB, power_draw_W)
"""
import os, re, sys, time, yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "gpuETL.yaml"
BATCH_SIZE = 500

# ── Module-level helpers (must be picklable for ProcessPoolExecutor) ──────────x
def _parquet_ok(args):
    """Return True if the parquet at path exists, is large enough, and is readable."""
    path, min_bytes = args
    try:
        return path.is_file() and path.stat().st_size >= min_bytes and bool(pq.read_metadata(path))
    except Exception:
        return False


def _estimate_dt(path, ts_col, n_probe):
    """Mean Δt from first n_probe rows, or None if the file is unreadable."""
    try:
        t = pd.to_numeric(
            pd.read_csv(path, usecols=[ts_col], nrows=n_probe, engine="c")[ts_col],
            errors="coerce",
        ).dropna().to_numpy()
        t = np.sort(np.unique(t))
        return float((t[-1] - t[0]) / (len(t) - 1)) if len(t) >= 2 else None
    except Exception:
        return None


def _process_node(path, p, t0_ref):
    """Outlier-filter → per-GPU rollup → bin to grid → ffill.  Returns DataFrame or None."""
    ts_col, ms, dt = p["timestamp_col"], p["metric_sources"], p["dt_seconds"]

    est = _estimate_dt(path, ts_col, p["outlier_probe_rows"])
    if est is None or not (p["delta_t_min"] <= est <= p["delta_t_max"]):
        return None

    try:
        d = pd.read_csv(path, usecols=[ts_col] + ms, engine="c")
    except Exception:
        return None

    if not set([ts_col] + ms).issubset(d.columns):
        return None

    d = d[[ts_col] + ms].apply(pd.to_numeric, errors="coerce").dropna(subset=[ts_col])
    if len(d) < 2:
        return None

    # Per-GPU rollup (multiple GPU rows share the same timestamp)
    d = d.groupby(ts_col, as_index=False).agg({m: p["agg_map"][m] for m in ms})

    # Bin to global k-grid anchored at t0_ref
    k = np.floor((d[ts_col].to_numpy(np.float64) - t0_ref) / dt + 1e-12).astype(np.int64)
    binned = (
        d.assign(k=k)
        .drop(columns=[ts_col])
        .groupby("k", as_index=False)
        .agg({m: p["within_bin_agg"] for m in ms})
    )
    if len(binned) == 0:
        return None

    k_min, k_max = int(binned["k"].min()), int(binned["k"].max())
    return (
        pd.DataFrame({"k": np.arange(k_min, k_max + 1, dtype=np.int64)})
        .merge(binned, on="k", how="left")
        .ffill()
    )


def _run_job(p):
    """Process one job → write parquet with embedded trace metadata.
    Returns (job_id, error_str_or_None).
    No shared state touched; fully safe for ProcessPoolExecutor.
    """
    ms, agg, dt = p["metric_sources"], p["agg_map"], p["dt_seconds"]

    # Anchor the bin grid to the earliest timestamp across all node files
    t0s = []
    for path in p["paths"]:
        try:
            row = pd.read_csv(path, usecols=[p["timestamp_col"]], nrows=1, engine="c")
            t = pd.to_numeric(row[p["timestamp_col"]], errors="coerce").dropna()
            if len(t):
                t0s.append(float(t.iloc[0]))
        except Exception:
            pass
    if not t0s:
        return p["job_id"], "no_readable_t0"

    t0_ref = min(t0s)
    per_node = [
        node for path in p["paths"]
        if (node := _process_node(Path(path), p, t0_ref)) is not None
    ]
    n_used, n_total = len(per_node), len(p["paths"])

    if n_total > 0 and n_used / n_total < p["min_node_keep_ratio"]:
        return p["job_id"], f"low_node_ratio:{n_used}/{n_total}"
    if not per_node:
        return p["job_id"], "no_usable_nodes"

    # Cross-node aggregation per bin
    cross = (
        pd.concat(per_node, ignore_index=True)
        .groupby("k", as_index=False)
        .agg({m: agg[m] for m in ms})
    )
    k_max = int(cross["k"].max())
    if k_max < 0:
        return p["job_id"], "neg_kmax"

    # Dense grid 0..k_max — fill all gaps, then rename and cast
    out = (
        pd.DataFrame({"k": np.arange(0, k_max + 1, dtype=np.int64)})
        .merge(cross, on="k", how="left")
    )
    out[ms] = out[ms].ffill().bfill().fillna(0.0)
    out["timestamp"] = out["k"].to_numpy(np.float64) * dt
    out = (
        out.drop(columns=["k"])
        .rename(columns=p["rename_after_agg"])
        [p["output_columns"]]
    )
    out["timestamp"] = out["timestamp"].astype(np.float64)
    for col in p["output_columns"]:
        if col != "timestamp":
            out[col] = out[col].astype(np.float32)

    # Embed trace stats in the parquet footer — no CSV writes from workers
    trace_meta = {
        b"job_id":       str(p["job_id"]).encode(),
        b"nodes_used":   str(n_used).encode(),
        b"length":       str(len(out)).encode(),
        b"duration_sec": str(round(k_max * dt, 3)).encode(),
        b"delta_t":      str(dt).encode(),
    }
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **trace_meta})

    # Atomic write (POSIX rename — readers never see a partial file)
    op  = Path(p["out_path"])
    tmp = op.with_suffix(".tmp")
    try:
        op.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, tmp, compression=p["compression"], row_group_size=int(p["row_group_size"]))
        os.replace(tmp, op)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return p["job_id"], f"parquet_err:{e}"

    return p["job_id"], None


# ── Main ETL class ────────────────────────────────────────────────────────────
class GpuETL:
    def __init__(self, cfg_path=None):
        self.cfg_path = cfg_path or CONFIG
        with open(self.cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.paths      = cfg["paths"]
        self.source     = cfg["source"]
        self.columns    = cfg["columns"]
        self.resample   = cfg["resample"]
        self.processing = cfg["processing"]
        self.parquet    = cfg["parquet"]
        self.base       = Path.cwd()
        print(f"Loaded config: {self.cfg_path}")

    def _rp(self, v):
        p = Path(v)
        return p if p.is_absolute() else (self.base / p).resolve()

    # ── Discover node CSVs grouped by job_id ─────────────────────────────────
    def _discover(self):
        print(f"Discovering jobs from {self.paths['source']}...")
        pat  = re.compile(self.processing["filename_pattern"])
        jobs = {}
        for path in self._rp(self.paths["source"]).glob(self.source["glob_pattern"]):
            if path.is_file() and (m := pat.match(path.stem)):
                jobs.setdefault(m.group("job_id"), []).append(path)
        print(f"Discovered {len(jobs)} jobs")
        return jobs

    # ── Build picklable payload dicts for workers ─────────────────────────────
    def _payloads(self, jobs):
        ts  = self.columns["timestamp"]
        met = self.columns["metrics"]
        ms  = [m["source"] for m in met]
        od  = self._rp(self.paths["intermediate"])
        dt  = float(self.processing["delta_t_seconds"])
        pr  = self.processing
        return [
            {
                "job_id":             k,
                "paths":              [str(x) for x in v],
                "dt_seconds":         dt,
                "timestamp_col":      ts,
                "metric_sources":     ms,
                "agg_map":            {m["source"]: m["cross_node_agg"] for m in met},
                "rename_after_agg":   {m["source"]: m["output"]         for m in met},
                "output_columns":     [ts] + [m["output"] for m in met],
                "within_bin_agg":     self.resample["within_bin_agg"],
                "min_node_keep_ratio":float(pr.get("min_node_keep_ratio", 0.75)),
                "delta_t_min":        float(pr.get("delta_t_min", 0.001)),
                "delta_t_max":        float(pr.get("delta_t_max", 0.500)),
                "outlier_probe_rows": int(pr.get("outlier_probe_rows", 50)),
                "out_path":           str(od / self.parquet["filename_template"].format(job_id=k)),
                "compression":        self.parquet["compression"],
                "row_group_size":     int(self.parquet["row_group_size"]),
            }
            for k, v in jobs.items()
        ]

    # ── Build gpu_traces.csv from parquet footer metadata (single-threaded) ──
    def _build_traces(self, od: Path = None, traces_path: Path = None):
        """Scan every parquet, read embedded trace metadata, and overwrite traces file.
        Since we are scanning every file, always overwrite output traces file.
        Duration bounds (lower_bound_sec / upper_bound_sec) are applied here.
        """
        od = self._rp(self.paths["intermediate"]) if not od else od
        traces_path = self._rp(self.paths["gpu_traces"]) if not traces_path else traces_path

        lower = float(self.processing.get("lower_bound_sec", float("-inf")))
        upper = float(self.processing.get("upper_bound_sec", float("inf")))

        rows = []
        for f in tqdm(sorted(od.glob("*.parquet")), desc="building traces", leave=False):
            try:
                raw = pq.read_metadata(f).metadata or {}
            except Exception:
                continue
            meta = {k.decode(): v.decode() for k, v in raw.items() if not k.startswith(b"pandas")}
            if "job_id" not in meta:
                continue
            dur = float(meta.get("duration_sec", 0))
            if not (lower <= dur <= upper):
                continue
            rows.append({
                "job_id":       meta["job_id"],
                "nodes_used":   int(meta.get("nodes_used", 0)),
                "file_path":    str(f.resolve()),
                "length":       int(meta.get("length", 0)),
                "duration_sec": dur,
                "delta_t":      float(meta.get("delta_t", 0)),
            })

        if rows:
            pd.DataFrame(rows).to_csv(traces_path, mode="w", header=True, index=False)
            print(f"Traces: wrote {len(rows)} rows → {traces_path}")
        else:
            # Always overwrite. If no rows meet criteria, still clear/create file with header.
            pd.DataFrame(columns=["job_id","nodes_used","file_path","length","duration_sec","delta_t"]).to_csv(traces_path, mode="w", header=True, index=False)
            print("Traces: no rows to write, file cleared")
        return len(rows)
 
    # ── Main orchestration ────────────────────────────────────────────────────
    def run(self):
        root        = self._rp(self.paths["source"])
        od          = self._rp(self.paths["intermediate"])
        traces_path = self._rp(self.paths["gpu_traces"])

        od.mkdir(parents=True, exist_ok=True)
        traces_path.parent.mkdir(parents=True, exist_ok=True)

        if not root.is_dir():
            print(f"Source missing: {root}")
            return 1

        jobs = self._discover()
        pl   = self._payloads(jobs)
        print(f"Discovered {len(pl)} jobs")

        # ── Skip already-completed parquets (resume support) ──────────────────
        min_bytes = int(self.parquet.get("skip_if_min_bytes", 256))
        args = [(Path(x["out_path"]), min_bytes) for x in pl]
        with ProcessPoolExecutor() as ex:
            done_flags = list(tqdm(ex.map(_parquet_ok, args), total=len(pl), desc="skip-check", leave=False))

        pl = [x for x, done in zip(pl, done_flags) if not done]
        print(f"  {sum(done_flags)} already done, {len(pl)} to process")

        # ── Parallel processing in batches ────────────────────────────────────
        nw  = max(1, (os.cpu_count() or 4) - int(self.processing.get("reserved_cpus", 2)))
        cap = self.processing.get("max_workers")
        if cap:
            nw = min(nw, int(cap))

        batches = [pl[i : i + BATCH_SIZE] for i in range(0, len(pl), BATCH_SIZE)]
        ok, bad = [], []
        t0 = time.time()
        print(f"Processing {len(pl)} jobs across {len(batches)} batch(es) with {nw} workers ...")

        for b_idx, batch in enumerate(batches, 1):
            t_batch = time.time()
            with ProcessPoolExecutor(max_workers=nw) as ex:
                futs = {ex.submit(_run_job, x): x for x in batch}
                with tqdm(total=len(batch), desc=f"Batch {b_idx}/{len(batches)}") as pbar:
                    for fu in as_completed(futs):
                        x = futs[fu]
                        try:
                            j, err = fu.result()
                        except Exception as e:
                            j, err = x["job_id"], f"crash:{e}"
                        (bad if err else ok).append(
                            (j, err, len(x["paths"])) if err else j
                        )
                        if err:
                            tqdm.write(f"  FAIL {j}: {err}")
                        pbar.update(1)
            print(f"  Batch {b_idx} done in {time.time() - t_batch:.1f}s")

        elapsed = time.time() - t0

        # ── Build gpu_traces.csv from parquet metadata (no locks, no races) ───
        self._build_traces(od, traces_path)
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(GpuETL()._build_traces())
