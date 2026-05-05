"""SeqETL — supercloud_power/configs/seqETL.yaml

Phases:
  1. Load the already merged SLURM + GPU feature table.
  2. Stratified 9:1 train/test split at job level.
  3. Compute global normalization stats from TRAIN parquets only
     (log1p+zscore for memory_used_MiB and power_draw_W; deterministic linear
      scaling for the two percentage channels — no fit needed).
  4. Per job: read GPU parquet, normalize 4 channels, clip outliers, atomic
     write (4, T) float32 NPY. Resume by skipping NPYs that already exist.
  5. Build chunk-index parquets — one row per (job_id, chunk_idx). Job-level
     SLURM features stored separately to avoid 24-column duplication across
     ~35M chunk rows.

Outputs:
  npy_dir/{job_id}.npy            (4, T) float32, normalized
  train_jobs.parquet              (job_id, npy_path, length, [24 SLURM cols])
  test_jobs.parquet               (job_id, npy_path, length, [24 SLURM cols])
  train_chunks.parquet            (job_id, chunk_idx)
  test_chunks.parquet             (job_id, chunk_idx)
  norm_stats.npz                  per-channel mean/std

The PowerTraceDataset class at the bottom of this file is imported directly
by training/inference code.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
import os, sys, time, yaml, math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "seqETL.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (must be picklable for ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────
def _stats_worker(args):
    """Compute (sum, sumsq, count) of log1p(x) for each log1p-zscored channel.
    Returns None if the file is unreadable or empty.
    """
    parquet_path, log_channels = args
    try:
        d = pq.read_table(parquet_path, columns=log_channels).to_pandas()
    except Exception:
        return None
    if len(d) == 0:
        return None
    out = {}
    for ch in log_channels:
        v = np.log1p(d[ch].to_numpy(np.float64))
        out[ch] = (float(v.sum()), float((v**2).sum()), len(v))
    return out


class _NormTask(NamedTuple):
    job_id:         str
    parquet_path:   str
    npy_path:       str
    channel_names:  tuple
    channel_norms:  tuple
    log_means:      dict
    log_stds:       dict
    clip_lo:        float
    clip_hi:        float
    skip_min_bytes: int


def _normalize_worker(t: _NormTask):
    """Normalize one job's parquet → write (4, T) float32 NPY (atomic).
    Returns (job_id, length_or_None, error_str_or_None).
    """
    npy_path = Path(t.npy_path)
    if npy_path.is_file() and npy_path.stat().st_size >= t.skip_min_bytes:
        try:
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[0] == len(t.channel_names):
                return t.job_id, int(arr.shape[1]), None
        except Exception:
            pass  # fall through and reprocess

    try:
        d = pq.read_table(t.parquet_path, columns=list(t.channel_names)).to_pandas()
    except Exception as e:
        return t.job_id, None, f"read_err:{e.__class__.__name__}"
    if len(d) < 1:
        return t.job_id, None, "empty_parquet"

    T = len(d)
    out = np.empty((len(t.channel_names), T), dtype=np.float32)
    for i, (ch, norm) in enumerate(zip(t.channel_names, t.channel_norms)):
        x = d[ch].to_numpy(np.float64)
        if norm == "linear_pct":
            z = (x / 100.0) * 2.0 - 1.0
        elif norm == "log1p_zscore":
            z = (np.log1p(x) - t.log_means[ch]) / t.log_stds[ch]
        else:
            return t.job_id, None, f"unknown_norm:{norm}"
        out[i] = np.clip(z, t.clip_lo, t.clip_hi).astype(np.float32)

    # Atomic write via a .tmp.npy so the final path is only visible once complete.
    tmp = npy_path.with_name(npy_path.stem + ".tmp.npy")
    try:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp, out, allow_pickle=False)
        os.replace(tmp, npy_path)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return t.job_id, None, f"write_err:{e.__class__.__name__}"
    return t.job_id, T, None


# ─────────────────────────────────────────────────────────────────────────────
# Resource detection helper (used across train.py / inference.py too)
# ─────────────────────────────────────────────────────────────────────────────
def detect_resources(reserved_cpus: int = 2) -> dict:
    """Detect CPU and GPU resources available to this process."""
    info = {
        "cpu_count":      os.cpu_count() or 4,
        "workers":        max(1, (os.cpu_count() or 4) - reserved_cpus),
        "gpu_count":      0,
        "gpu_name":       None,
        "vram_gb":        0.0,
        "bf16_supported": False,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"]       = props.name
            info["vram_gb"]        = props.total_memory / 1e9
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        pass
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Main ETL class
# ─────────────────────────────────────────────────────────────────────────────
class SeqETL:
    def __init__(self, cfg_path: str | None = None):
        self.cfg_path = Path(cfg_path).resolve() if cfg_path else CONFIG.resolve()
        with open(self.cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.paths      = cfg["paths"]
        self.channels   = cfg["channels"]
        self.sequence   = cfg["sequence"]
        self.norm       = cfg["normalization"]
        self.split_cfg  = cfg["split"]
        self.slurm_cols = list(cfg["slurm_feature_cols"])
        self.storage    = cfg["storage"]
        self.workers    = cfg["workers"]
        self.base       = Path.cwd()
        print(f"Loaded config: {self.cfg_path}")

        self.channel_names = [c["name"] for c in self.channels]
        self.channel_norms = [c["norm"] for c in self.channels]
        self.log_channels  = [c["name"] for c in self.channels if c["norm"] == "log1p_zscore"]

    def _rp(self, v) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (self.base / p).resolve()

    def _n_workers(self) -> int:
        nw  = max(1, (os.cpu_count() or 4) - int(self.workers.get("reserved_cpus", 2)))
        cap = self.workers.get("max_workers")
        return min(nw, int(cap)) if cap else nw

    # ── Phase 1: Load merged table ───────────────────────────────────────────
    def _load_and_merge(self) -> pd.DataFrame:
        merged_path = self._rp(self.paths["merged"])
        if not merged_path.is_file():
            raise FileNotFoundError(f"Missing merged features parquet: {merged_path}")

        merged = pd.read_parquet(merged_path)
        required = ["job_id", "file_path", "length", "duration_sec"] + self.slurm_cols
        missing  = [c for c in required if c not in merged.columns]
        if missing:
            raise KeyError(f"Merged parquet missing columns: {missing}")

        merged["job_id"]       = merged["job_id"].astype(str)
        merged["length"]       = merged["length"].astype(np.int64)
        merged["duration_sec"] = merged["duration_sec"].astype(np.float64)

        check_cols  = self.slurm_cols + ["length", "duration_sec"]
        finite_mask = np.isfinite(merged[check_cols].to_numpy()).all(axis=1)
        n_inf = (~finite_mask).sum()
        if n_inf:
            print(f"  Dropped {n_inf} jobs with non-finite values")
        merged = merged[finite_mask].reset_index(drop=True)

        n_zero = (merged["length"] <= 0).sum()
        if n_zero:
            print(f"  Dropped {n_zero} jobs with length <= 0")
        merged = merged[merged["length"] > 0].reset_index(drop=True)

        print(f"  Loaded merged features for {len(merged):,} jobs")
        return merged

    # ── Phase 2: Stratified 9:1 split ────────────────────────────────────────
    def _stratified_split(self, df: pd.DataFrame):
        rng   = np.random.default_rng(int(self.split_cfg["seed"]))
        strat = self.split_cfg["stratify_cols"]

        # If no stratification columns, just do a random split.
        if not strat:
            test_frac = float(self.split_cfg["test_frac"])
            idx = np.arange(len(df))
            rng.shuffle(idx)
            n_test = int(round(len(df) * test_frac))
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            print(f"  Train: {len(train_df):,}  Test: {len(test_df):,} (no stratification)")
            return train_df, test_df

        missing = [c for c in strat if c not in df.columns]
        if missing:
            raise KeyError(f"stratify columns not found: {missing}")

        key       = df[strat].astype(str).agg("_".join, axis=1)
        test_frac = float(self.split_cfg["test_frac"])
        test_idx  = []
        for _, sub in df.assign(_k=key).groupby("_k"):
            n_test = max(1, int(round(len(sub) * test_frac)))
            test_idx.extend(rng.choice(sub.index.values, size=min(n_test, len(sub)), replace=False))

        test_mask = df.index.isin(test_idx)
        train_df  = df[~test_mask].reset_index(drop=True)
        test_df   = df[test_mask].reset_index(drop=True)

        train_v = set(train_df[strat].astype(str).agg("_".join, axis=1).unique())
        test_v  = set(test_df[strat].astype(str).agg("_".join, axis=1).unique())
        print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")
        print(f"  Variants — total {len(key.unique())}, in train {len(train_v)}, in test {len(test_v)}")
        return train_df, test_df
 

    # ── Phase 3: Norm stats from TRAIN parquets only ─────────────────────────
    def _fit_norm_stats(self, train_df: pd.DataFrame) -> dict:
        sample_frac = self.norm.get("stats_sample_frac")
        if sample_frac and 0 < float(sample_frac) < 1.0:
            sub = train_df.sample(n=max(1, int(len(train_df) * float(sample_frac))),
                                  random_state=int(self.split_cfg["seed"]))
            print(f"  Fitting stats on {len(sub):,} of {len(train_df):,} train jobs (sample)")
        else:
            sub = train_df
            print(f"  Fitting stats on all {len(train_df):,} train jobs")

        args = [(p, self.log_channels) for p in sub["file_path"].tolist()]
        agg  = {ch: [0.0, 0.0, 0] for ch in self.log_channels}
        with ProcessPoolExecutor(max_workers=self._n_workers()) as ex:
            futures = [ex.submit(_stats_worker, a) for a in args]
            for fu in tqdm(as_completed(futures), total=len(futures), desc="Stats", leave=False):
                res = fu.result()
                if res is None:
                    continue
                for ch, (s, sq, n) in res.items():
                    agg[ch][0] += s
                    agg[ch][1] += sq
                    agg[ch][2] += n

        min_std = float(self.norm.get("min_std", 1e-6))
        means_dict, stds_dict = {}, {}
        for ch in self.log_channels:
            s, sq, n = agg[ch]
            if n == 0:
                raise RuntimeError(f"No data for channel {ch} during stats fit")
            mean = s / n
            std  = math.sqrt(max(sq / n - mean**2, min_std**2))
            means_dict[ch] = mean
            stds_dict[ch]  = std
            print(f"  {ch}: log1p_mean={mean:.6f}, log1p_std={std:.6f}, n={n:,}")

        return {
            "channels":  np.array(self.channel_names, dtype=object),
            "norms":     np.array(self.channel_norms,  dtype=object),
            "log_means": np.array([means_dict.get(ch, 0.0) for ch in self.channel_names], dtype=np.float64),
            "log_stds":  np.array([stds_dict.get(ch,  1.0) for ch in self.channel_names], dtype=np.float64),
        }

    # ── Phase 4: Normalize all jobs to NPY ───────────────────────────────────
    def _process_npy(self, df: pd.DataFrame, stats: dict):
        npy_dir = self._rp(self.paths["npy_dir"])
        npy_dir.mkdir(parents=True, exist_ok=True)
        clip_lo, clip_hi = map(float, self.norm["clip_range"])
        skip_min_bytes   = int(self.storage.get("npy_skip_min_bytes", 64))

        log_means = {ch: float(m) for ch, m in zip(self.channel_names, stats["log_means"])}
        log_stds  = {ch: float(s) for ch, s in zip(self.channel_names, stats["log_stds"])}

        task_args = [
            _NormTask(
                str(row["job_id"]), row["file_path"],
                str(npy_dir / f"{row['job_id']}.npy"),
                tuple(self.channel_names), tuple(self.channel_norms),
                log_means, log_stds, clip_lo, clip_hi, skip_min_bytes,
            )
            for _, row in df.iterrows()
        ]

        nw         = self._n_workers()
        batch_size = int(self.workers.get("batch_size", 500))
        results, bad = {}, []
        t0 = time.time()
        total_jobs = len(task_args)
        total_batches = (total_jobs + batch_size - 1) // batch_size
        print(f"Total jobs: {total_jobs}, Total batches: {total_batches}")

        for i, b_start in enumerate(range(0, total_jobs, batch_size), start=1):
            batch = task_args[b_start : b_start + batch_size]
            with ProcessPoolExecutor(max_workers=nw) as ex:
                futs = {ex.submit(_normalize_worker, a): a for a in batch}
                with tqdm(total=len(batch), desc=f"NPY {i}", leave=False) as pbar:
                    for fu in as_completed(futs):
                        try:
                            jid, length, err = fu.result()
                        except Exception as e:
                            jid, length, err = futs[fu].job_id, None, f"crash:{e}"
                        if err:
                            bad.append((jid, err))
                            tqdm.write(f"  FAIL {jid}: {err}")
                        else:
                            results[jid] = length
                        pbar.update(1)
            print(f"Batch {i} completed")
                   

        print(f"  NPY: {len(results):,} ok, {len(bad):,} failed in {time.time()-t0:.1f}s")
        return results, bad

    # ── Phase 5: Build chunk index ───────────────────────────────────────────
    def _build_chunk_index(self, jobs_df: pd.DataFrame, lengths: dict, kind: str):
        """Build per-job chunk index. kind in {"train", "test"} for path lookup."""
        stride  = int(self.sequence["stride"])
        npy_dir = self._rp(self.paths["npy_dir"])

        jobs_df = jobs_df[jobs_df["job_id"].isin(lengths)].copy().reset_index(drop=True)
        jobs_df["length"]   = jobs_df["job_id"].map(lengths).astype(np.int64)
        jobs_df["npy_path"] = jobs_df["job_id"].apply(lambda jid: str(npy_dir / f"{jid}.npy"))

        jobs_out = jobs_df[["job_id", "npy_path", "length"] + self.slurm_cols].copy()

        chunk_rows = [
            (jid, ci)
            for jid, L in zip(jobs_out["job_id"].values, jobs_out["length"].values)
            for ci in range(max(1, math.ceil(L / stride)))
        ]
        chunks_out = pd.DataFrame(chunk_rows, columns=["job_id", "chunk_idx"])
        print(f"  {kind}: {len(jobs_out):,} jobs → {len(chunks_out):,} chunks")
        return jobs_out, chunks_out

    # ── Orchestration ────────────────────────────────────────────────────────
    def run(self):
        report_path = self._rp(self.paths["report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        log = []
        def _say(s):
            print(s)
            log.append(s)

        _say("\n── Phase 1: Load merged table ──")
        merged = self._load_and_merge()
        _say(f"  Loaded jobs: {len(merged):,}")

        _say("\n── Phase 2: Stratified split ──")
        train_df, test_df = self._stratified_split(merged)

        _say("\n── Phase 3: Norm stats fit (train only) ──")
        stats     = self._fit_norm_stats(train_df)
        norm_path = self._rp(self.paths["norm_stats"])
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(norm_path, **stats)
        _say(f"  Wrote {norm_path}")

        _say("\n── Phase 4: Normalize parquets → NPY ──")
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        lengths, bad = self._process_npy(all_df, stats)
        _say(f"  Successful: {len(lengths):,}, Failed: {len(bad):,}")

        _say("\n── Phase 5: Build chunk index ──")
        train_jobs, train_chunks = self._build_chunk_index(train_df, lengths, "train")
        test_jobs,  test_chunks  = self._build_chunk_index(test_df,  lengths, "test")

        cmp = self.storage.get("parquet_compression", "zstd")
        for chunks, jobs, kind in [(train_chunks, train_jobs, "train"),
                                   (test_chunks,  test_jobs,  "test")]:
            chunks_path = self._rp(self.paths[f"{kind}_chunks"])
            jobs_path   = chunks_path.with_name(
                chunks_path.stem.replace("chunks", "jobs") + ".parquet"
            )
            chunks.to_parquet(chunks_path, index=False, compression=cmp)
            jobs.to_parquet(jobs_path,     index=False, compression=cmp)
            _say(f"  Wrote {chunks_path} ({len(chunks):,} chunks)")
            _say(f"  Wrote {jobs_path} ({len(jobs):,} jobs)")

        _say(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
        with open(report_path, "w") as f:
            f.write("\n".join(log))
        print(f"Report: {report_path}")
        return 1 if bad else 0
        

if __name__ == "__main__":
    sys.exit(SeqETL().run())
