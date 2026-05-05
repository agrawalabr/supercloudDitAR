"""SeqETL — supercloud_power/configs/seqETL.yaml

Phases:
  1. Load & merge slurm_log.parquet with gpu_traces.csv on job_id.
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
import os, sys, time, yaml, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "seqETL.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (must be picklable for ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────
def _stats_worker(args):
    """Compute (sum, sumsq, count) of log1p(x) for the two log1p-zscored
    channels in one parquet file. Returns a dict keyed by channel name.
    Returns None if file is unreadable or empty.
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
        out[ch] = (float(v.sum()), float((v ** 2).sum()), int(len(v)))
    return out


def _normalize_worker(args):
    """Normalize one job's parquet → write (4, T) float32 NPY (atomic).
    Returns (job_id, length_or_None, error_str_or_None).
    """
    (job_id, parquet_path, npy_path,
     channel_names, channel_norms,
     log_means, log_stds, clip_lo, clip_hi, skip_min_bytes) = args

    npy_path = Path(npy_path)
    # Resume: skip if NPY already exists & non-trivial
    if npy_path.is_file() and npy_path.stat().st_size >= skip_min_bytes:
        try:
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[0] == len(channel_names):
                return job_id, int(arr.shape[1]), None
        except Exception:
            pass  # fall through and reprocess

    try:
        d = pq.read_table(parquet_path, columns=list(channel_names)).to_pandas()
    except Exception as e:
        return job_id, None, f"read_err:{e.__class__.__name__}"
    if len(d) < 1:
        return job_id, None, "empty_parquet"

    T = len(d)
    out = np.empty((len(channel_names), T), dtype=np.float32)
    for i, (ch, norm) in enumerate(zip(channel_names, channel_norms)):
        x = d[ch].to_numpy(np.float64)
        if norm == "linear_pct":
            z = (x / 100.0) * 2.0 - 1.0
        elif norm == "log1p_zscore":
            z = (np.log1p(x) - log_means[ch]) / log_stds[ch]
        else:
            return job_id, None, f"unknown_norm:{norm}"
        z = np.clip(z, clip_lo, clip_hi)
        out[i] = z.astype(np.float32)

    # Atomic write. np.save appends ".npy" if not present, so use an explicit
    # ".tmp.npy" so the final on-disk file matches what we'll os.replace with.
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
        return job_id, None, f"write_err:{e.__class__.__name__}"
    return job_id, int(T), None


# ─────────────────────────────────────────────────────────────────────────────
# Resource detection helper (used across train.py / inference.py too)
# ─────────────────────────────────────────────────────────────────────────────
def detect_resources(reserved_cpus: int = 2) -> dict:
    """Detect CPU and GPU resources available to this process."""
    info = {
        "cpu_count":     os.cpu_count() or 4,
        "workers":       max(1, (os.cpu_count() or 4) - reserved_cpus),
        "gpu_count":     0,
        "gpu_name":      None,
        "vram_gb":       0.0,
        "bf16_supported": False,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["vram_gb"] = props.total_memory / 1e9
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        pass
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Main ETL class
# ─────────────────────────────────────────────────────────────────────────────
class SeqETL:
    def __init__(self, cfg_path: Optional[str] = None):
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

        # Derived
        self.channel_names = [c["name"] for c in self.channels]
        self.channel_norms = [c["norm"] for c in self.channels]
        # Identify log1p-zscored channels for stats fitting
        self.log_channels = [
            c["name"] for c in self.channels if c["norm"] == "log1p_zscore"
        ]

    def _rp(self, v) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (self.base / p).resolve()

    def _n_workers(self) -> int:
        nw = max(1, (os.cpu_count() or 4)
                 - int(self.workers.get("reserved_cpus", 2)))
        cap = self.workers.get("max_workers")
        if cap:
            nw = min(nw, int(cap))
        return nw

    # ── Phase 1: Load & merge ────────────────────────────────────────────────
    def _load_and_merge(self) -> pd.DataFrame:
        slurm_path = self._rp(self.paths["slurm_features"])
        traces_path = self._rp(self.paths["gpu_traces"])
        if not slurm_path.is_file():
            raise FileNotFoundError(f"Missing SLURM features: {slurm_path}")
        if not traces_path.is_file():
            raise FileNotFoundError(f"Missing GPU traces: {traces_path}")

        slurm = pd.read_parquet(slurm_path)
        slurm["job_id"] = slurm["job_id"].astype(str)
        traces = pd.read_csv(traces_path, dtype={"job_id": str})

        # Confirm all required columns exist
        missing = [c for c in self.slurm_cols if c not in slurm.columns]
        if missing:
            raise KeyError(f"slurm_log.parquet missing columns: {missing}")
        for col in ["job_id", "file_path", "length", "duration_sec"]:
            if col not in traces.columns:
                raise KeyError(f"gpu_traces.csv missing column: {col}")

        # Inner join — only keep jobs present in both
        keep_cols = ["job_id", "file_path", "length", "duration_sec"]
        merged = traces[keep_cols].merge(
            slurm[["job_id"] + self.slurm_cols], on="job_id", how="inner"
        )
        merged["length"] = merged["length"].astype(np.int64)
        merged["duration_sec"] = merged["duration_sec"].astype(np.float64)

        # Sanity: drop jobs with NaN/Inf in any required column
        check_cols = self.slurm_cols + ["length", "duration_sec"]
        ok = np.isfinite(merged[check_cols].to_numpy()).all(axis=1)
        n_dropped = (~ok).sum()
        if n_dropped:
            print(f"  Dropped {n_dropped} jobs with non-finite values")
        merged = merged.loc[ok].reset_index(drop=True)
        # Drop jobs with length <= 0
        ok2 = merged["length"] > 0
        if (~ok2).sum():
            print(f"  Dropped {(~ok2).sum()} jobs with length <= 0")
        merged = merged.loc[ok2].reset_index(drop=True)

        print(f"  Loaded & merged {len(merged):,} jobs")
        return merged

    # ── Phase 2: Stratified 9:1 split ────────────────────────────────────────
    def _stratified_split(self, df: pd.DataFrame):
        rng = np.random.default_rng(int(self.split_cfg["seed"]))
        strat = self.split_cfg["stratify_cols"]
        for col in strat:
            if col not in df.columns:
                raise KeyError(f"stratify column not found: {col}")

        # Build cross-product key
        key = df[strat].astype(str).agg("_".join, axis=1)
        test_frac = float(self.split_cfg["test_frac"])

        # Per-cell split: ensure each variant appears in both train and test
        test_idx = []
        for cell, sub in df.assign(_k=key).groupby("_k"):
            n = len(sub)
            n_test = max(1, int(round(n * test_frac)))
            picked = rng.choice(sub.index.values, size=min(n_test, n), replace=False)
            test_idx.extend(picked.tolist())
        test_set = set(test_idx)
        train_df = df.loc[[i for i in df.index if i not in test_set]].reset_index(drop=True)
        test_df  = df.loc[sorted(test_set)].reset_index(drop=True)

        # Verify variant coverage
        train_v = set(train_df[strat].astype(str).agg("_".join, axis=1).unique())
        test_v  = set(test_df[strat].astype(str).agg("_".join, axis=1).unique())
        all_v = set(key.unique())
        print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")
        print(f"  Variants — total {len(all_v)}, in train {len(train_v)}, in test {len(test_v)}")
        return train_df, test_df

    # ── Phase 3: Norm stats from TRAIN parquets only ─────────────────────────
    def _fit_norm_stats(self, train_df: pd.DataFrame) -> dict:
        # Optional subsample for faster fitting
        sample_frac = self.norm.get("stats_sample_frac")
        if sample_frac is not None and 0 < sample_frac < 1.0:
            n = max(1, int(len(train_df) * float(sample_frac)))
            sub = train_df.sample(n=n, random_state=int(self.split_cfg["seed"]))
            print(f"  Fitting stats on {len(sub):,} of {len(train_df):,} train jobs (sample)")
            paths = sub["file_path"].tolist()
        else:
            print(f"  Fitting stats on all {len(train_df):,} train jobs")
            paths = train_df["file_path"].tolist()

        nw = self._n_workers()
        args = [(p, self.log_channels) for p in paths]

        # Streaming reduction: per-channel (sum, sumsq, count)
        agg = {ch: [0.0, 0.0, 0] for ch in self.log_channels}
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = [ex.submit(_stats_worker, a) for a in args]
            for fu in tqdm(as_completed(futures), total=len(futures),
                           desc="Stats", leave=False):
                res = fu.result()
                if res is None:
                    continue
                for ch, (s, sq, n) in res.items():
                    agg[ch][0] += s
                    agg[ch][1] += sq
                    agg[ch][2] += n

        min_std = float(self.norm.get("min_std", 1e-6))
        out = {"channels": np.array(self.channel_names, dtype=object)}
        means_dict, stds_dict = {}, {}
        for ch in self.log_channels:
            s, sq, n = agg[ch]
            if n == 0:
                raise RuntimeError(f"No data for channel {ch} during stats fit")
            mean = s / n
            var = max(sq / n - mean ** 2, min_std ** 2)
            std = math.sqrt(var)
            means_dict[ch] = mean
            stds_dict[ch] = std
            print(f"  {ch}: log1p_mean={mean:.6f}, log1p_std={std:.6f}, n={n:,}")
        # Save in arrays aligned to channel ordering for cheap restore
        out["log_means"] = np.array(
            [means_dict.get(ch, 0.0) for ch in self.channel_names], dtype=np.float64
        )
        out["log_stds"] = np.array(
            [stds_dict.get(ch, 1.0) for ch in self.channel_names], dtype=np.float64
        )
        out["norms"] = np.array(self.channel_norms, dtype=object)
        return out

    # ── Phase 4: Normalize all jobs to NPY ───────────────────────────────────
    def _process_npy(self, df: pd.DataFrame, stats: dict):
        npy_dir = self._rp(self.paths["npy_dir"])
        npy_dir.mkdir(parents=True, exist_ok=True)
        clip_lo, clip_hi = map(float, self.norm["clip_range"])
        skip_min_bytes = int(self.storage.get("npy_skip_min_bytes", 64))

        # log_means / log_stds aligned to channel ordering — workers want a dict
        # keyed by channel name (only the log1p_zscored ones matter; others ignored)
        log_means = {ch: float(m) for ch, m in zip(self.channel_names, stats["log_means"])}
        log_stds  = {ch: float(s) for ch, s in zip(self.channel_names, stats["log_stds"])}

        args = [
            (
                str(row["job_id"]), row["file_path"],
                str(npy_dir / f"{row['job_id']}.npy"),
                tuple(self.channel_names), tuple(self.channel_norms),
                log_means, log_stds, clip_lo, clip_hi, skip_min_bytes,
            )
            for _, row in df.iterrows()
        ]

        nw = self._n_workers()
        batch_size = int(self.workers.get("batch_size", 500))
        results = {}  # job_id -> length
        bad = []
        t0 = time.time()
        for b_start in range(0, len(args), batch_size):
            batch = args[b_start : b_start + batch_size]
            with ProcessPoolExecutor(max_workers=nw) as ex:
                futs = {ex.submit(_normalize_worker, a): a for a in batch}
                with tqdm(total=len(batch), desc=f"NPY {b_start//batch_size + 1}",
                          leave=False) as pbar:
                    for fu in as_completed(futs):
                        try:
                            jid, length, err = fu.result()
                        except Exception as e:
                            jid, length, err = futs[fu][0], None, f"crash:{e}"
                        if err:
                            bad.append((jid, err))
                            tqdm.write(f"  FAIL {jid}: {err}")
                        else:
                            results[jid] = length
                        pbar.update(1)
        elapsed = time.time() - t0
        print(f"  NPY: {len(results):,} ok, {len(bad):,} failed in {elapsed:.1f}s")
        return results, bad

    # ── Phase 5: Build chunk index ───────────────────────────────────────────
    def _build_chunk_index(self, jobs_df: pd.DataFrame, lengths: dict, kind: str):
        """Build per-job chunk index. kind in {"train", "test"} for path lookup.
        Returns (jobs_table, chunks_table) parquets to be written."""
        stride = int(self.sequence["stride"])
        npy_dir = self._rp(self.paths["npy_dir"])

        # Drop jobs that didn't successfully produce an NPY
        jobs_df = jobs_df[jobs_df["job_id"].isin(lengths.keys())].copy().reset_index(drop=True)
        # Patch the length column from the actually-written NPY (gpuETL lengths
        # should match, but we trust the NPY since that's what training reads)
        jobs_df["length"] = jobs_df["job_id"].map(lengths).astype(np.int64)
        # Add npy_path column
        jobs_df["npy_path"] = jobs_df["job_id"].apply(
            lambda jid: str(npy_dir / f"{jid}.npy")
        )

        # Job-level table (light): job_id, npy_path, length, [24 SLURM cols]
        jobs_out = jobs_df[["job_id", "npy_path", "length"] + self.slurm_cols].copy()

        # Chunk-level table (heavy): job_id, chunk_idx
        # n_chunks = ceil(length / stride), at least 1.
        chunk_rows = []
        for jid, L in zip(jobs_out["job_id"].values, jobs_out["length"].values):
            n_chunks = max(1, math.ceil(L / stride))
            for ci in range(n_chunks):
                chunk_rows.append((jid, ci))
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

        # 1. Load & merge
        _say("\n── Phase 1: Load & merge ──")
        merged = self._load_and_merge()
        _say(f"  Merged jobs: {len(merged):,}")

        # 2. Stratified split
        _say("\n── Phase 2: Stratified split ──")
        train_df, test_df = self._stratified_split(merged)

        # 3. Norm stats from TRAIN ONLY
        _say("\n── Phase 3: Norm stats fit (train only) ──")
        stats = self._fit_norm_stats(train_df)
        norm_path = self._rp(self.paths["norm_stats"])
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(norm_path,
                 channels=stats["channels"],
                 norms=stats["norms"],
                 log_means=stats["log_means"],
                 log_stds=stats["log_stds"])
        _say(f"  Wrote {norm_path}")

        # 4. Normalize all jobs (train + test) to NPY
        _say("\n── Phase 4: Normalize parquets → NPY ──")
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        lengths, bad = self._process_npy(all_df, stats)
        _say(f"  Successful: {len(lengths):,}, Failed: {len(bad):,}")

        # 5. Build chunk index for each split
        _say("\n── Phase 5: Build chunk index ──")
        train_jobs, train_chunks = self._build_chunk_index(train_df, lengths, "train")
        test_jobs,  test_chunks  = self._build_chunk_index(test_df,  lengths, "test")

        cmp = self.storage.get("parquet_compression", "zstd")
        train_chunks_path = self._rp(self.paths["train_chunks"])
        test_chunks_path  = self._rp(self.paths["test_chunks"])
        # Save jobs table alongside chunks: same dir, *_jobs.parquet
        train_jobs_path = train_chunks_path.with_name(
            train_chunks_path.stem.replace("chunks", "jobs") + ".parquet"
        )
        test_jobs_path = test_chunks_path.with_name(
            test_chunks_path.stem.replace("chunks", "jobs") + ".parquet"
        )

        train_chunks.to_parquet(train_chunks_path, index=False, compression=cmp)
        test_chunks .to_parquet(test_chunks_path,  index=False, compression=cmp)
        train_jobs  .to_parquet(train_jobs_path,   index=False, compression=cmp)
        test_jobs   .to_parquet(test_jobs_path,    index=False, compression=cmp)
        _say(f"  Wrote {train_chunks_path} ({len(train_chunks):,} chunks)")
        _say(f"  Wrote {test_chunks_path}  ({len(test_chunks):,} chunks)")
        _say(f"  Wrote {train_jobs_path}   ({len(train_jobs):,} jobs)")
        _say(f"  Wrote {test_jobs_path}    ({len(test_jobs):,} jobs)")

        elapsed = time.time() - t_start
        _say(f"\nTotal elapsed: {elapsed:.1f}s")
        with open(report_path, "w") as f:
            f.write("\n".join(log))
        print(f"Report: {report_path}")
        return 1 if bad else 0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class — imported by training/inference code
# ─────────────────────────────────────────────────────────────────────────────
class PowerTraceDataset:
    """Sliding-window chunks for v5 DiT-AR training/inference.

    For chunk_idx i of a job with length L:
      ctx       = bins [i*stride - W_ctx, i*stride)
                  Out-of-range positions are filled with 0 (first chunks have
                  partial or no context — model sees a "fresh start" signal).
      future_aux= bins [i*stride, i*stride + W_pred), aux channels (0..2),
                  zero-padded past job end.
      target    = bins [i*stride, i*stride + W_pred), power channel (3),
                  zero-padded past job end.
      pred_mask = 1 where target is real, 0 where padded past job end.
      cond      = 24-dim SLURM vector for the job.
    """

    def __init__(self, chunks_parquet, jobs_parquet,
                 W_ctx, W_pred, stride, slurm_cols, n_channels=4):
        import pandas as _pd
        self.W_ctx    = int(W_ctx)
        self.W_pred   = int(W_pred)
        self.stride   = int(stride)
        self.n_ch     = int(n_channels)
        self.slurm_cols = list(slurm_cols)

        self.chunks = _pd.read_parquet(chunks_parquet)
        jobs = _pd.read_parquet(jobs_parquet)

        # Pre-build dicts for O(1) per-chunk lookups
        self._npy_path = dict(zip(jobs["job_id"].values, jobs["npy_path"].values))
        self._length   = dict(zip(jobs["job_id"].values, jobs["length"].values))
        # SLURM features as a dict job_id -> (n_slurm,) float32
        slurm_arr = jobs[self.slurm_cols].to_numpy(np.float32)
        self._slurm = dict(zip(jobs["job_id"].values, slurm_arr))

        # mmap cache (per-process). Lazy-loaded on first access.
        self._mmap = {}

    def __len__(self):
        return len(self.chunks)

    def _get_mmap(self, npy_path):
        m = self._mmap.get(npy_path)
        if m is None:
            m = np.load(npy_path, mmap_mode="r")
            self._mmap[npy_path] = m
        return m

    def __getitem__(self, idx):
        # row.iloc is faster than dict lookup at chunk granularity
        row = self.chunks.iloc[idx]
        job_id, ci = str(row["job_id"]), int(row["chunk_idx"])

        npy_path = self._npy_path[job_id]
        L = int(self._length[job_id])
        cond = self._slurm[job_id]

        arr = self._get_mmap(npy_path)        # shape (n_ch, L) float32
        chunk_start = ci * self.stride

        # ── Past context: bins [chunk_start - W_ctx, chunk_start) ──
        ctx = np.zeros((self.n_ch, self.W_ctx), dtype=np.float32)
        ctx_real_start = max(0, chunk_start - self.W_ctx)
        ctx_real_end   = chunk_start
        if ctx_real_end > ctx_real_start:
            offset = self.W_ctx - (ctx_real_end - ctx_real_start)
            ctx[:, offset:] = arr[:, ctx_real_start:ctx_real_end]

        # ── Future window: bins [chunk_start, chunk_start + W_pred) ──
        future = np.zeros((self.n_ch, self.W_pred), dtype=np.float32)
        future_real_end = min(L, chunk_start + self.W_pred)
        n_real = max(0, future_real_end - chunk_start)
        if n_real > 0:
            future[:, :n_real] = arr[:, chunk_start:future_real_end]

        future_aux = future[:3]               # channels 0..2 (gpu_pct, mem_pct, mem_MiB)
        target     = future[3:4]              # channel 3   (power_draw_W) shape (1, W_pred)

        pred_mask = np.zeros(self.W_pred, dtype=np.float32)
        pred_mask[:n_real] = 1.0

        # Return numpy; downstream collation in DataLoader's collate converts
        return {
            "ctx":        ctx,                 # (4, W_ctx)   float32
            "future_aux": future_aux,          # (3, W_pred)  float32
            "target":     target,              # (1, W_pred)  float32
            "pred_mask":  pred_mask,           # (W_pred,)    float32
            "cond":       cond,                # (24,)        float32
            "job_id":     job_id,              # str (for inference; ignored by training)
            "chunk_idx":  ci,                  # int (for inference; ignored by training)
        }


def _collate_v5(batch):
    """Convert numpy outputs of PowerTraceDataset into stacked torch tensors."""
    import torch
    return {
        "ctx":        torch.from_numpy(np.stack([b["ctx"]        for b in batch])),
        "future_aux": torch.from_numpy(np.stack([b["future_aux"] for b in batch])),
        "target":     torch.from_numpy(np.stack([b["target"]     for b in batch])),
        "pred_mask":  torch.from_numpy(np.stack([b["pred_mask"]  for b in batch])),
        "cond":       torch.from_numpy(np.stack([b["cond"]       for b in batch])),
        "job_id":     [b["job_id"]    for b in batch],
        "chunk_idx":  [b["chunk_idx"] for b in batch],
    }


if __name__ == "__main__":
    sys.exit(SeqETL().run())