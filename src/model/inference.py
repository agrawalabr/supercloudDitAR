"""inference.py — Sliding-window AR power-trace generation for v5.

For each evaluation job:
  1. Load ground-truth NPY (4 channels: gpu_used_pct, mem_used_pct, mem_MiB, power)
  2. Build SLURM cond from test_jobs.parquet
  3. Iterate AR windows (each generates W_pred=1024 bins of power):
       chunk 0: ctx = zeros, future_aux = real aux for window
       chunk i: ctx = last W_ctx bins of (generated power + real aux), future_aux = real aux for window
       Run DDIM 50 steps with CFG to denoise → predicted x_0 in z-score space
  4. Concatenate windows → full synthetic trace; denormalize
  5. Compute metrics vs. ground-truth: Pearson r, RMSE, watt MAE

Configurable:
  - max_windows_per_job: cap inference at first N windows
  - subsample: stratified sampling of test jobs (e.g., 1000)
  - cfg_scale: classifier-free guidance scale (default 1.5)
  - ddim_steps: number of DDIM steps (default 50)
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC/batch jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ditArV5 as M


# ─────────────────────────────────────────────────────────────────────────────
# DDIM sampler with CFG
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_sample(net: torch.nn.Module,
sched: M.DiffusionSchedule,
    ctx: torch.Tensor,             # (B, n_ch, W_ctx)
    future_aux: torch.Tensor,      # (B, 3, W_pred)
    cond: torch.Tensor,            # (B, cond_dim)
    n_steps: int = 50,
    cfg_scale: float = 1.5,
    eta: float = 0.0,
) -> torch.Tensor:
    """Returns predicted x_0 of shape (B, 1, W_pred)."""
    device = ctx.device
    B = ctx.size(0)
    W_pred = future_aux.size(-1)

    # DDIM step indices: T-1 down to 0, in n_steps increments
    timesteps = torch.linspace(sched.T - 1, 0, n_steps + 1, dtype=torch.long, device=device)

    # Start from pure noise
    x = torch.randn(B, 1, W_pred, device=device)

    for i in range(n_steps):
        t = timesteps[i].repeat(B)
        t_next = timesteps[i + 1].repeat(B)

        # CFG: run cond + uncond forward passes
        if cfg_scale != 1.0:
            x_in    = torch.cat([x, x], dim=0)
            ctx_in  = torch.cat([ctx, ctx], dim=0)
            aux_in  = torch.cat([future_aux, future_aux], dim=0)
            cond_in = torch.cat([cond, cond], dim=0)
            t_in    = torch.cat([t, t], dim=0)
            drop = torch.zeros(2 * B, dtype=torch.bool, device=device)
            drop[B:] = True
            v = net(ctx_in, aux_in, x_in, t_in, cond_in, cond_drop_mask=drop)
            v_cond, v_uncond = v.chunk(2, dim=0)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = net(ctx, future_aux, x, t, cond)

        # v → x_0 prediction
        x0_pred = sched.predict_x0_from_v(x, t, v_pred)
        # v → eps
        eps_pred = sched.predict_eps_from_v(x, t, v_pred)

        # DDIM update to t_next
        ac_t      = sched.alphas_cumprod[t].view(-1, 1, 1)
        if t_next[0].item() >= 0:
            ac_t_next = sched.alphas_cumprod[t_next].view(-1, 1, 1)
        else:
            ac_t_next = torch.ones_like(ac_t)

        # Standard DDIM (eta=0 deterministic)
        sigma = eta * torch.sqrt((1 - ac_t_next) / (1 - ac_t) * (1 - ac_t / ac_t_next))
        dir_xt = torch.sqrt((1 - ac_t_next) - sigma ** 2) * eps_pred
        x = torch.sqrt(ac_t_next) * x0_pred + dir_xt
        if eta > 0:
            x = x + sigma * torch.randn_like(x)

    return x  # final x ≈ x_0


# ─────────────────────────────────────────────────────────────────────────────
# Per-job AR generation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_job(net: torch.nn.Module, sched: M.DiffusionSchedule, npy_arr: np.ndarray, cond: np.ndarray, W_ctx: int, W_pred: int, stride: int, max_windows: int, n_steps: int, cfg_scale: float, device: torch.device) -> np.ndarray:
    """Returns generated power channel of shape (T_capped,) in z-score space."""
    L = npy_arr.shape[1]
    T_capped = min(L, max_windows * W_pred)
    n_windows = math.ceil(T_capped / W_pred)

    cond_t = torch.from_numpy(cond.copy()).float().unsqueeze(0).to(device)
    aux_full = torch.from_numpy(npy_arr[:3].copy()).float().unsqueeze(0).to(device)  # (1, 3, T)
    pwr_full = torch.from_numpy(npy_arr[3:4].copy()).float().unsqueeze(0).to(device) # (1, 1, T)

    # Output buffer (z-score space, channel 3)
    gen_pwr = torch.zeros(1, 1, T_capped, device=device)

    for w in range(n_windows):
        start = w * W_pred
        end_real = min(T_capped, start + W_pred)

        # ── Build ctx: last W_ctx bins of [aux + generated power] ──
        ctx = torch.zeros(1, 4, W_ctx, device=device)
        ctx_real_start = max(0, start - W_ctx)
        n_have = start - ctx_real_start
        if n_have > 0:
            offset = W_ctx - n_have
            # Aux channels (0..2): use real aux from NPY
            ctx[:, :3, offset:] = aux_full[:, :, ctx_real_start:start]
            # Power channel (3): use generated power from previous windows
            ctx[:, 3:4, offset:] = gen_pwr[:, :, ctx_real_start:start]

        # ── Build future_aux ──
        future_aux = torch.zeros(1, 3, W_pred, device=device)
        n_real = end_real - start
        if n_real > 0:
            future_aux[:, :, :n_real] = aux_full[:, :, start:end_real]

        # ── Generate ──
        x0 = ddim_sample(
            net, sched, ctx, future_aux, cond_t,
            n_steps=n_steps, cfg_scale=cfg_scale,
        )
        # Store predicted power into the buffer (only the real region)
        gen_pwr[:, :, start:end_real] = x0[:, :, :n_real]

    return gen_pwr.squeeze(0).squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified subsample
# ─────────────────────────────────────────────────────────────────────────────
def stratified_subsample(df: pd.DataFrame, n: int, stratify_cols: list, seed: int = 42) -> pd.DataFrame:
    """Pick n jobs across the cross-product of stratify_cols, proportional."""
    if n >= len(df) or not stratify_cols:
        return df.sample(n=min(n, len(df)), random_state=seed)
    rng = np.random.default_rng(seed)
    key = df[stratify_cols].astype(str).agg("_".join, axis=1)
    cells = df.assign(_k=key).groupby("_k")
    take_per_cell = max(1, n // cells.ngroups)
    out_idx = []
    for _, sub in cells:
        take = min(take_per_cell, len(sub))
        out_idx.extend(rng.choice(sub.index.values, size=take, replace=False).tolist())
    if len(out_idx) > n:
        out_idx = rng.choice(out_idx, size=n, replace=False).tolist()
    return df.loc[out_idx].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Denormalization
# ─────────────────────────────────────────────────────────────────────────────
def denormalize_power(z: np.ndarray, log_mean: float, log_std: float) -> np.ndarray:
    """Invert log1p+zscore: x = expm1(z * std + mean)."""
    return np.expm1(z * log_std + log_mean).astype(np.float32)


def metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    """Pearson r, RMSE, watt MAE."""
    real = real.astype(np.float64)
    fake = fake.astype(np.float64)
    if len(real) == 0:
        return {"pearson_r": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    r_std = real.std()
    f_std = fake.std()
    if r_std < 1e-9 or f_std < 1e-9:
        pr = float("nan")
    else:
        pr = float(np.corrcoef(real, fake)[0, 1])
    rmse = float(np.sqrt(np.mean((real - fake) ** 2)))
    mae  = float(np.mean(np.abs(real - fake)))
    return {"pearson_r": pr, "rmse": rmse, "mae": mae}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_samples_from_disk(synth_dir: Path, test_jobs: pd.DataFrame, log_mean_pwr: float, log_std_pwr: float, n_samples: int = 6, save_path: Optional[Path] = None) -> None:
    """Plot real vs generated power traces using saved .npy files from synth_dir.

    Picks n_samples jobs spread across the duration range so the figure covers
    short, medium, and long jobs rather than clustering at one end.
    """
    synth_dir = Path(synth_dir)
    if save_path is None:
        save_path = synth_dir.parent / "samples.png"

    # Keep only rows for which a generation exists on disk
    job_ids = test_jobs["job_id"].astype(str)
    mask = job_ids.apply(lambda jid: (synth_dir / f"{jid}.npy").exists())
    available = test_jobs[mask].copy()
    if available.empty:
        print(f"No generations found in {synth_dir} — nothing to plot.")
        return

    # Sort by trace duration so picks span the full range
    if "duration_sec" in available.columns:
        available = available.sort_values("duration_sec").reset_index(drop=True)
    pick_rows = available.iloc[
        np.linspace(0, len(available) - 1, min(n_samples, len(available)), dtype=int)
    ]

    fig, axes = plt.subplots(len(pick_rows), 1, figsize=(15, 2.5 * len(pick_rows)))
    if len(pick_rows) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, pick_rows.iterrows()):
        jid = str(row["job_id"])
        nodes_req = int(row["nodes_req"]) if "nodes_req" in row.index else 1

        try:
            arr   = np.load(row["npy_path"], mmap_mode="r")   # (4, L)
            gen_z = np.load(synth_dir / f"{jid}.npy").astype(np.float32)
        except Exception as exc:
            print(f"  Skip {jid}: {exc}")
            ax.set_visible(False)
            continue

        real_z = np.asarray(arr[3, : len(gen_z)], dtype=np.float32)
        real_W = denormalize_power(real_z, log_mean_pwr, log_std_pwr)
        gen_W  = denormalize_power(gen_z,  log_mean_pwr, log_std_pwr)

        if real_z.std() > 1e-8 and gen_z.std() > 1e-8:
            pr = float(np.corrcoef(real_z, gen_z)[0, 1])
        else:
            pr = 0.0

        dur_min = len(gen_z) * 0.103 / 60.0
        ax.plot(real_W, label="Real",      lw=0.6, alpha=0.85)
        ax.plot(gen_W,  label="Generated", lw=0.6, alpha=0.85)
        ax.set_title(
            f"job={jid}  nodes={nodes_req}  dur={dur_min:.1f} min  "
            f"pearson_r={pr:+.3f}  "
            f"real(μ/max)={real_W.mean():.0f}/{real_W.max():.0f} W  "
            f"gen(μ/max)={gen_W.mean():.0f}/{gen_W.max():.0f} W",
            fontsize=9,
        )
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("step (0..L-1)")
    fig.suptitle(
        "Real vs Generated GPU Power Traces — DiT v5 Test Set",
        fontsize=11, y=1.0,
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main inference run
# ─────────────────────────────────────────────────────────────────────────────
def run(cfg: dict, ckpt_dir: str = None) -> int:
    inf_cfg = cfg["inference"]
    paths   = cfg["paths"]
    seq_cfg = cfg["sequence"]
    out_dir = Path(paths["inference_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_dir = out_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    # ── Device + checkpoint ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = ckpt_dir if ckpt_dir else paths["ckpt_dir"]

    # Load EMA checkpoint
    ckpt_path = Path(ckpt_dir) / "ema.pt"
    if not ckpt_path.is_file():
        ckpt_path = Path(ckpt_dir) / "last.pt"
        print(f"Warning: ema.pt not found, using {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)

    net = M.build_model(cfg["model"]).to(device)
    if "ema" in sd and sd["ema"] is not None:
        net.load_state_dict(sd["ema"])
        print(f"Loaded EMA weights from {ckpt_path}")
    else:
        net.load_state_dict(sd["model"])
        print(f"Loaded model weights from {ckpt_path}")
    net.eval()

    sched = M.DiffusionSchedule(T=cfg["model"].get("diffusion_T", 1000)).to(device)

    # ── Load norm stats (for denormalization) ──
    stats = np.load(paths["norm_stats"], allow_pickle=True)
    channels = stats["channels"].tolist()
    pwr_idx = channels.index("power_draw_W")
    log_mean_pwr = float(stats["log_means"][pwr_idx])
    log_std_pwr  = float(stats["log_stds"][pwr_idx])
    print(f"Power denorm: log1p mean={log_mean_pwr:.4f}, std={log_std_pwr:.4f}")

    # ── Load test jobs + optional subsample ──
    test_jobs = pd.read_parquet(paths["test_jobs"])
    print(f"Loaded {len(test_jobs):,} test jobs")
    n_subsample = inf_cfg.get("subsample_n")
    if n_subsample and n_subsample < len(test_jobs):
        test_jobs = stratified_subsample(
            test_jobs, n_subsample,
            stratify_cols=inf_cfg.get("subsample_stratify_cols", []),
            seed=inf_cfg.get("seed", 42),
        )
        print(f"Subsampled to {len(test_jobs):,} jobs")

    # ── Per-job loop ──
    max_windows = int(inf_cfg.get("max_windows_per_job", 25))
    n_steps = int(inf_cfg.get("ddim_steps", 50))
    cfg_scale = float(inf_cfg.get("cfg_scale", 1.5))
    save_traces = bool(inf_cfg.get("save_per_job_traces", False))

    summary = []
    t_start = time.time()
    for _, row in tqdm(test_jobs.iterrows(), total=len(test_jobs), desc="Generating"):
        job_id = str(row["job_id"])
        try:
            arr = np.load(row["npy_path"], mmap_mode="r")  # (4, L)
        except Exception as e:
            print(f"FAIL load {job_id}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[0] != 4:
            print(f"FAIL shape {job_id}: {arr.shape}")
            continue

        cond = row[cfg["slurm_feature_cols"]].to_numpy(dtype=np.float32)

        gen_z = generate_job(
            net, sched,
            np.asarray(arr),  # materialize from mmap
            cond,
            W_ctx=seq_cfg["W_ctx"], W_pred=seq_cfg["W_pred"], stride=seq_cfg["stride"],
            max_windows=max_windows, n_steps=n_steps, cfg_scale=cfg_scale,
            device=device,
        )

        # Real power (z-score) → real watts
        real_z = arr[3, :len(gen_z)]
        real_W = denormalize_power(real_z, log_mean_pwr, log_std_pwr)
        gen_W  = denormalize_power(gen_z,  log_mean_pwr, log_std_pwr)

        np.save(synth_dir / f"{job_id}.npy", gen_z)

        m = metrics(real_W, gen_W)
        m["job_id"] = job_id
        m["length"] = len(gen_z)
        m["duration_min"] = len(gen_z) * 0.103 / 60.0
        summary.append(m)

        if save_traces:
            trace_df = pd.DataFrame({
                "timestamp_s": np.arange(len(gen_z)) * 0.103,
                "real_power_W": real_W,
                "synthetic_power_W": gen_W,
            })
            trace_df.to_csv(out_dir / f"trace_{job_id}.csv", index=False)

    elapsed = time.time() - t_start
    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Print headline metrics ──
    print(f"\n── Inference summary ({len(summary_df):,} jobs in {elapsed:.0f}s) ──")
    if len(summary_df):
        finite = summary_df.dropna(subset=["pearson_r"])
        print(f"  Pearson r — mean: {finite['pearson_r'].mean():.3f}, "
              f"median: {finite['pearson_r'].median():.3f}, "
              f"frac>0.30: {(finite['pearson_r'] > 0.30).mean():.3f}")
        print(f"  RMSE (watts) — mean: {summary_df['rmse'].mean():.1f}, "
              f"median: {summary_df['rmse'].median():.1f}")
        print(f"  MAE  (watts) — mean: {summary_df['mae'].mean():.1f}, "
              f"median: {summary_df['mae'].median():.1f}")
    print(f"Wrote {summary_path}")

    # ── Plot real vs generated samples ──
    n_plot = int(inf_cfg.get("plot_n_samples", 6))
    
    plot_samples_from_disk(synth_dir=synth_dir,test_jobs=test_jobs, log_mean_pwr=log_mean_pwr, log_std_pwr=log_std_pwr, n_samples=n_plot, save_path=out_dir / "samples.png")

    return 0


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sys.exit(run(cfg, args.ckpt_dir))
