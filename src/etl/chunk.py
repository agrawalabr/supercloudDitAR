"""chunkETL.py — PyTorch Dataset & collate fn for v5 sliding-window training.

Imported by train.py and inference.py. Reads (job_id, chunk_idx) from the
chunk-index parquet, mmap-loads the corresponding NPY, and slices out
ctx/future_aux/target/pred_mask/cond per __getitem__.

Notes on the implementation:
  - Inherits torch.utils.data.Dataset for proper DataLoader integration.
  - chunk-index columns are pre-extracted to numpy arrays at __init__ for
    fast __getitem__ (pandas .iloc is slow at this scale, ~36M rows).
  - mmap is opened lazily per file with a bounded LRU so we never exceed
    the OS file-descriptor limit (typical default 1024).
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Bounded LRU keeps memory + fd footprint small. mmap reopen is cheap (~µs),
# so we don't lose much on cache miss. Set MMAP_LRU_MAX = 256 — comfortably
# under the typical 1024 fd limit.
MMAP_LRU_MAX = 256


class PowerTraceDataset(Dataset):
    """Sliding-window chunks for v5 DiT-AR training/inference.

    For chunk_idx i of a job with length L:
      ctx       = bins [i*stride - W_ctx, i*stride)  (4, W_ctx)
                  Zero-padded on the left for chunks where ctx pre-dates
                  bin 0. The model treats all-zero ctx as the "fresh start"
                  signal — chunk_idx=0 always has zero ctx.
      future_aux= bins [i*stride, i*stride + W_pred), channels 0..2
                  Zero-padded on the right past job end.
      target    = bins [i*stride, i*stride + W_pred), channel 3 (power)
                  Zero-padded on the right past job end.
      pred_mask = 1 where target is real, 0 past job end.
      cond      = 24-dim SLURM vector for the job (constant across chunks).
    """

    def __init__(
        self,
        chunks_parquet: str,
        jobs_parquet: str,
        W_ctx: int,
        W_pred: int,
        stride: int,
        slurm_cols: list,
        n_channels: int = 4,
    ):
        self.W_ctx      = int(W_ctx)
        self.W_pred     = int(W_pred)
        self.stride     = int(stride)
        self.n_ch       = int(n_channels)
        self.slurm_cols = list(slurm_cols)

        # Load chunk index. Pre-extract to numpy for fast __getitem__:
        # pandas .iloc[i] does index validation per call which is slow at
        # ~36M rows × multiple epochs.
        chunks = pd.read_parquet(chunks_parquet)
        self._chunk_job_id    = chunks["job_id"].astype(str).to_numpy()
        self._chunk_chunk_idx = chunks["chunk_idx"].astype(np.int64).to_numpy()

        # Job-level metadata: per-job dicts indexed by job_id (str).
        jobs = pd.read_parquet(jobs_parquet)
        jobs_jid = jobs["job_id"].astype(str).to_numpy()
        self._npy_path = dict(zip(jobs_jid, jobs["npy_path"].astype(str).to_numpy()))
        self._length   = dict(zip(jobs_jid, jobs["length"].astype(np.int64).to_numpy()))
        # SLURM as one stacked array; each row referenced via job_id index.
        slurm_arr = jobs[self.slurm_cols].to_numpy(np.float32)
        self._slurm = dict(zip(jobs_jid, slurm_arr))

        # Bounded LRU mmap cache. Per-process state — DataLoader workers each
        # maintain their own copy after fork.
        self._mmap_lru: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._chunk_job_id)

    def _get_mmap(self, npy_path: str) -> np.ndarray:
        m = self._mmap_lru.get(npy_path)
        if m is not None:
            self._mmap_lru.move_to_end(npy_path)   # mark as recently used
            return m
        # Cache miss: open and insert
        m = np.load(npy_path, mmap_mode="r")
        self._mmap_lru[npy_path] = m
        # Evict oldest if over capacity
        while len(self._mmap_lru) > MMAP_LRU_MAX:
            self._mmap_lru.popitem(last=False)
        return m

    def __getitem__(self, idx: int) -> dict:
        job_id = str(self._chunk_job_id[idx])
        ci     = int(self._chunk_chunk_idx[idx])

        arr = self._get_mmap(self._npy_path[job_id])   # shape (n_ch, L) float32
        L   = int(self._length[job_id])
        chunk_start = ci * self.stride

        # Past context: bins [chunk_start - W_ctx, chunk_start) — left-zero-padded
        ctx = np.zeros((self.n_ch, self.W_ctx), dtype=np.float32)
        ctx_start = max(0, chunk_start - self.W_ctx)
        if chunk_start > ctx_start:
            offset = self.W_ctx - (chunk_start - ctx_start)
            ctx[:, offset:] = arr[:, ctx_start:chunk_start]

        # Future window: bins [chunk_start, chunk_start + W_pred) — right-zero-padded
        future = np.zeros((self.n_ch, self.W_pred), dtype=np.float32)
        future_real_end = min(L, chunk_start + self.W_pred)
        n_real = max(0, future_real_end - chunk_start)
        if n_real > 0:
            future[:, :n_real] = arr[:, chunk_start:future_real_end]

        pred_mask = np.zeros(self.W_pred, dtype=np.float32)
        pred_mask[:n_real] = 1.0

        return {
            "ctx":        ctx,                  # (n_ch, W_ctx) float32
            "future_aux": future[:3],           # (3, W_pred)  float32
            "target":     future[3:4],          # (1, W_pred)  float32
            "pred_mask":  pred_mask,            # (W_pred,)    float32
            "cond":       self._slurm[job_id],  # (24,)        float32
            "job_id":     job_id,
            "chunk_idx":  ci,
        }


def _collate_v5(batch: list) -> dict:
    """Stack numpy outputs from PowerTraceDataset into torch tensors."""
    tensor_keys = ("ctx", "future_aux", "target", "pred_mask", "cond")
    out = {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in tensor_keys}
    out["job_id"]    = [b["job_id"]    for b in batch]
    out["chunk_idx"] = [b["chunk_idx"] for b in batch]
    return out
