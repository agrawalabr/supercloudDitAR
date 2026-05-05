"""train.py — Training driver for DiT-AR v5.

Auto-scales to available hardware:
  - Detects GPU count → uses DDP across all visible CUDA devices
  - Detects bf16 support → uses bf16 mixed precision when available, fp16 otherwise
  - Detects CPU count → sets DataLoader num_workers
  - Reads batch_size from config (override) or defaults from VRAM detection

Design choices:
  - v-prediction diffusion target with cosine β schedule
  - Loss is masked MSE: positions past job end (pred_mask=0) contribute zero
  - Per-batch outlier guard: if any sample's loss > 5.0, the batch is skipped
  - EMA of model weights (decay 0.999) — inference uses EMA exclusively
  - "Scheduled noise" exposure-bias mitigation in last 20% of training:
    with probability p (ramped 0→0.3), the past-power channel of ctx is
    replaced by a noisy version (q_sample with t ∈ [50, 200])
  - CFG dropout at 5%
  - Resume from latest checkpoint by default
"""
from __future__ import annotations
import argparse, math, os, sys, time, copy, signal
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M
import chunkETL as cE
import seqETL


# ─────────────────────────────────────────────────────────────────────────────
# DDP setup helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ddp_setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


# ─────────────────────────────────────────────────────────────────────────────
# EMA wrapper — keeps a shadow copy of model weights
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for ema_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(d).add_(m_p.detach(), alpha=1 - d)
        for ema_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(m_b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.shadow.load_state_dict(sd)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────
def _lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ─────────────────────────────────────────────────────────────────────────────
# Single training run on one rank
# ─────────────────────────────────────────────────────────────────────────────
def train_one_rank(rank: int, world_size: int, cfg: dict) -> None:
    is_ddp = world_size > 1
    if is_ddp:
        _ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    main = _is_main(rank)

    # ── Setup precision (bf16 preferred on H100/H200; fp16 fallback) ─────────
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    elif device.type == "cuda":
        amp_dtype = torch.float16
        use_grad_scaler = True
    else:
        amp_dtype = torch.float32
        use_grad_scaler = False
    if main:
        print(f"[rank {rank}] device={device}, amp_dtype={amp_dtype}, world_size={world_size}")

    # ── Build dataset + dataloader ───────────────────────────────────────────
    seq_cfg = cfg["sequence"]
    train_cfg = cfg["train"]
    ds = cE.PowerTraceDataset(
        chunks_parquet=cfg["paths"]["train_chunks"],
        jobs_parquet=cfg["paths"]["train_jobs"],
        W_ctx=seq_cfg["W_ctx"], W_pred=seq_cfg["W_pred"], stride=seq_cfg["stride"],
        slurm_cols=cfg["slurm_feature_cols"],
        n_channels=cfg["model"]["n_channels"],
    )
    if main:
        print(f"[rank {rank}] dataset: {len(ds):,} chunks")

    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if is_ddp else None
    )
    n_workers = train_cfg.get("num_workers")
    if n_workers is None:
        n_workers = max(2, (os.cpu_count() or 4) // max(1, world_size) - 2)
    loader = DataLoader(
        ds, batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(n_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=cE._collate_v5,
        persistent_workers=int(n_workers) > 0,
    )

    # ── Model + optimizer + EMA ──────────────────────────────────────────────
    net = M.build_model(cfg["model"]).to(device)
    sched = M.DiffusionSchedule(T=cfg["model"].get("diffusion_T", 1000)).to(device)
    if main:
        print(f"[rank {rank}] model: {M.count_params(net) / 1e6:.1f}M params")

    if is_ddp:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    optim = torch.optim.AdamW(
        (net.module if is_ddp else net).parameters(),
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler() if use_grad_scaler else None
    ema = EMA(net.module if is_ddp else net, decay=train_cfg["ema_decay"]) if main else None

    # ── Resume from checkpoint if present ─────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"

    start_epoch, global_step = 0, 0
    if last_ckpt.is_file():
        sd = torch.load(last_ckpt, map_location=device)
        (net.module if is_ddp else net).load_state_dict(sd["model"])
        optim.load_state_dict(sd["optim"])
        if ema is not None and "ema" in sd:
            ema.load_state_dict(sd["ema"])
        if scaler is not None and "scaler" in sd and sd["scaler"] is not None:
            scaler.load_state_dict(sd["scaler"])
        start_epoch = sd.get("epoch", 0)
        global_step = sd.get("step", 0)
        if main:
            print(f"[rank {rank}] resumed from {last_ckpt} at step {global_step}, epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────────────
    n_epochs = train_cfg["n_epochs"]
    steps_per_epoch = len(loader)
    total_steps = n_epochs * steps_per_epoch
    warmup = train_cfg.get("warmup_steps", 2000)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    cfg_dropout_p = train_cfg.get("cfg_dropout_p", 0.05)
    loss_skip_thr = train_cfg.get("loss_skip_threshold", 5.0)
    log_every = train_cfg.get("log_every", 50)
    ckpt_every = train_cfg.get("ckpt_every_steps", 2000)

    # Scheduled-noise exposure bias mitigation (final 20% of epochs)
    ss_start_epoch = int(0.8 * n_epochs)
    ss_p_max = float(train_cfg.get("scheduled_sampling_p_max", 0.3))
    ss_t_min = int(train_cfg.get("scheduled_sampling_t_min", 50))
    ss_t_max = int(train_cfg.get("scheduled_sampling_t_max", 200))

    log_path = ckpt_dir / "train_log.csv"
    if main and not log_path.exists():
        log_path.write_text("step,epoch,lr,loss,n_skipped\n")

    def _save(tag: str = "last", skipped_count: int = 0):
        if not main:
            return
        sd = {
            "model": (net.module if is_ddp else net).state_dict(),
            "optim": optim.state_dict(),
            "ema":   ema.state_dict() if ema else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": current_epoch,
            "step":  global_step,
            "cfg":   cfg,
        }
        torch.save(sd, ckpt_dir / f"{tag}.pt")

    n_skipped_total = 0
    t_start = time.time()
    current_epoch = start_epoch
    try:
        for current_epoch in range(start_epoch, n_epochs):
            if is_ddp and sampler is not None:
                sampler.set_epoch(current_epoch)
            net.train()

            # Scheduled-noise probability ramps 0 → ss_p_max in last 20%
            if current_epoch >= ss_start_epoch:
                ss_p = ss_p_max * (current_epoch - ss_start_epoch + 1) / max(1, n_epochs - ss_start_epoch)
            else:
                ss_p = 0.0

            for batch in loader:
                ctx        = batch["ctx"].to(device, non_blocking=True)         # (B, n_ch, W_ctx)
                future_aux = batch["future_aux"].to(device, non_blocking=True)  # (B, 3, W_pred)
                target     = batch["target"].to(device, non_blocking=True)      # (B, 1, W_pred)
                pred_mask  = batch["pred_mask"].to(device, non_blocking=True)   # (B, W_pred)
                cond       = batch["cond"].to(device, non_blocking=True)        # (B, 24)
                B = ctx.size(0)

                # Sample diffusion timestep + noise
                t = torch.randint(0, sched.T, (B,), device=device)
                noise = torch.randn_like(target)
                x_t = sched.q_sample(target, t, noise)
                v_target = sched.get_v(target, t, noise)

                # CFG dropout: drop conditioning with probability cfg_dropout_p
                cond_drop = torch.rand(B, device=device) < cfg_dropout_p

                # Scheduled-noise exposure bias: replace power channel of ctx
                # with q_sampled noisy version at moderate t
                if ss_p > 0:
                    do_ss = torch.rand(B, device=device) < ss_p
                    if do_ss.any():
                        ss_t = torch.randint(ss_t_min, ss_t_max, (B,), device=device)
                        # ctx[:, 3:4, :] is the power channel
                        ctx_power = ctx[:, 3:4, :]
                        ctx_noise = torch.randn_like(ctx_power)
                        ctx_power_noisy = sched.q_sample(ctx_power, ss_t, ctx_noise)
                        ctx_new_power = torch.where(
                            do_ss[:, None, None], ctx_power_noisy, ctx_power
                        )
                        ctx = ctx.clone()
                        ctx[:, 3:4, :] = ctx_new_power

                # ── Forward ──
                # LR schedule
                for pg in optim.param_groups:
                    pg["lr"] = train_cfg["lr"] * _lr_factor(global_step, warmup, total_steps)

                if device.type == "cuda":
                    cm = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                else:
                    cm = torch.amp.autocast(device_type="cpu", enabled=False)

                with cm:
                    v_pred = net(ctx, future_aux, x_t, t, cond, cond_drop_mask=cond_drop)
                    # Masked MSE: zero loss past job end (pred_mask=0)
                    sq = (v_pred - v_target) ** 2                                # (B, 1, W_pred)
                    sq = sq.squeeze(1) * pred_mask                                # (B, W_pred)
                    valid = pred_mask.sum(dim=1).clamp_min(1.0)                   # (B,)
                    per_sample_loss = sq.sum(dim=1) / valid                        # (B,)
                    loss = per_sample_loss.mean()

                # ── Outlier guard: skip the whole batch if any sample is bad ──
                if torch.isfinite(loss) and per_sample_loss.max().item() > loss_skip_thr:
                    n_skipped_total += 1
                    optim.zero_grad(set_to_none=True)
                    global_step += 1
                    continue
                if not torch.isfinite(loss):
                    n_skipped_total += 1
                    optim.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                # ── Backward + step ──
                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(
                        (net.module if is_ddp else net).parameters(), grad_clip
                    )
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (net.module if is_ddp else net).parameters(), grad_clip
                    )
                    optim.step()

                if ema is not None:
                    ema.update(net.module if is_ddp else net)

                global_step += 1

                # Logging (rank 0 only)
                if main and global_step % log_every == 0:
                    cur_lr = optim.param_groups[0]["lr"]
                    elapsed = time.time() - t_start
                    rate = global_step / max(1.0, elapsed)
                    eta_h = (total_steps - global_step) / max(1.0, rate) / 3600.0
                    print(
                        f"step {global_step}/{total_steps} ep {current_epoch} "
                        f"lr {cur_lr:.2e} loss {loss.item():.4f} "
                        f"skipped {n_skipped_total} rate {rate:.1f}/s eta {eta_h:.1f}h "
                        f"ss_p {ss_p:.2f}",
                        flush=True,
                    )
                    with open(log_path, "a") as f:
                        f.write(f"{global_step},{current_epoch},{cur_lr},{loss.item()},{n_skipped_total}\n")

                # Checkpoint
                if main and global_step % ckpt_every == 0:
                    _save("last")

            # Epoch end
            if main:
                _save("last")
                print(f"[rank 0] epoch {current_epoch} done at step {global_step}", flush=True)
    except KeyboardInterrupt:
        if main:
            print("[rank 0] KeyboardInterrupt — saving checkpoint and exiting", flush=True)
            _save("last")
    finally:
        if main:
            _save("final")
            if ema is not None:
                # Save EMA-only checkpoint for inference convenience
                torch.save({"ema": ema.state_dict(), "cfg": cfg},
                           ckpt_dir / "ema.pt")
        _ddp_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(cfg: dict) -> int:
    res = seqETL.detect_resources()
    print(f"Detected resources: {res}")
    world = max(1, res["gpu_count"])
    if world > 1:
        torch.multiprocessing.spawn(
            train_one_rank, args=(world, cfg), nprocs=world, join=True
        )
    else:
        train_one_rank(0, 1, cfg)
    return 0


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to top-level config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sys.exit(run(cfg))
