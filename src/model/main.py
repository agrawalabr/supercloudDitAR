"""main.py — Top-level orchestrator for v5 pipeline.

Usage:
  python main.py --config configs/v5.yaml [--skip_seqetl] [--skip_train] [--skip_inference]

Reads a single top-level config that points to all stage-specific configs and
output paths. Runs SeqETL → train → inference in order, with each stage
skippable via flag.

Top-level config schema (v5.yaml):
  paths:
    seq_etl_config: configs/seqETL.yaml
    train_chunks:   data/p/train_chunks.parquet
    train_jobs:     data/p/train_jobs.parquet
    test_chunks:    data/p/test_chunks.parquet
    test_jobs:      data/p/test_jobs.parquet
    norm_stats:     data/p/norm_stats.npz
    ckpt_dir:       data/ckpt/v5
    inference_dir:  data/inf/v5
  sequence:
    W_ctx: 2048
    W_pred: 1024
    stride: 1024
    patch_size: 16
  model:
    d_model: 384
    n_heads: 6
    n_layers: 12
    mlp_ratio: 4
    dropout: 0.0
    patch_size: 16
    W_ctx: 2048
    W_pred: 1024
    cond_dim: 24
    n_channels: 4
    n_aux_channels: 3
    diffusion_T: 1000
  train:
    batch_size: 64
    n_epochs: 50
    lr: 1.0e-4
    weight_decay: 0.01
    warmup_steps: 2000
    grad_clip: 1.0
    cfg_dropout_p: 0.05
    loss_skip_threshold: 5.0
    log_every: 50
    ckpt_every_steps: 2000
    ema_decay: 0.999
    scheduled_sampling_p_max: 0.3
    scheduled_sampling_t_min: 50
    scheduled_sampling_t_max: 200
    num_workers: null     # null = auto
  inference:
    subsample_n: 1000
    subsample_stratify_cols: [type_batch, type_interactive, type_map, type_other]
    max_windows_per_job: 25     # 25 * 1024 bins ≈ 6 hours per job
    ddim_steps: 50
    cfg_scale: 1.5
    save_per_job_traces: false
    seed: 42
  slurm_feature_cols: [...]    # 24 names
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqETL
import train as train_mod
import inference as inf_mod


def run_seqetl(cfg: dict) -> int:
    print("\n" + "=" * 70)
    print(" Stage 1: SeqETL")
    print("=" * 70)
    etl_cfg = cfg["paths"]["seq_etl_config"]
    return seqETL.SeqETL(cfg_path=etl_cfg).run()


def run_train(cfg: dict) -> int:
    print("\n" + "=" * 70)
    print(" Stage 2: Train")
    print("=" * 70)
    return train_mod.run(cfg)


def run_inference(cfg: dict) -> int:
    print("\n" + "=" * 70)
    print(" Stage 3: Inference")
    print("=" * 70)
    return inf_mod.run(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Top-level v5 config YAML")
    ap.add_argument("--skip_seqetl", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_inference", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    rc = 0
    if not args.skip_seqetl:
        rc = run_seqetl(cfg)
        if rc:
            print(f"SeqETL returned {rc} — continuing")
    if not args.skip_train:
        rc = run_train(cfg)
        if rc:
            print(f"Train returned {rc}")
            return rc
    if not args.skip_inference:
        rc = run_inference(cfg)
        if rc:
            print(f"Inference returned {rc}")
            return rc

    print(f"\n{'=' * 70}\n Pipeline complete in {(time.time() - t0) / 60:.1f} min\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
