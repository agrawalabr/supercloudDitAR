"""train.py — Training driver for DiT-AR v5.

Auto-scales to available hardware:
  - Detects GPU count → uses DDP across all visible CUDA devices
  - Detects bf16 support → uses bf16 mixed precision when available, fp16 otherwise
  - Detects CPU count → sets DataLoader num_workers
  - Reads batch_size from config (override) or defaults from VRAM detection

Design choices:
  - v-prediction diffusion target with cosine β schedule
  - Loss is masked MSE: positions past job end (pred_mask=0) contribute zero
  - Per-sample outlier guard: samples with loss > 5.0 are zeroed out so the
    rest of the batch still trains. If every sample in a batch is bad (NaN
    after masking), both DDP ranks coordinate via all_reduce and skip backward
    together — a one-sided skip would desync the gradient all_reduce and hang.
  - EMA of model weights (decay 0.999) — inference uses EMA exclusively
  - "Scheduled noise" exposure-bias mitigation in last 20% of training:
    with probability p (ramped 0→0.3), the past-power channel of ctx is
    replaced by a noisy version (q_sample with t ∈ [50, 200])
  - CFG dropout at 5%
  - Resume from latest checkpoint by default
"""
from __future__ import annotations
import argparse, math, os, sys, time, copy, signal, datetime, threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# Local imports
import src.model.ditArV5 as M
import src.etl.chunk as cE
import src.etl.seq as seqETL
from src.shared.detect_hw import *


# ─────────────────────────────────────────────────────────────────────────────
# DDP setup helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_free_port() -> int:
    """Bind to port 0 to let the OS pick a free ephemeral port, then release it."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _ddp_setup(rank: int, world_size: int) -> None:
    # MASTER_PORT is set by run() before spawning so all workers share the same
    # pre-validated free port. Fall back to a random port if somehow not set.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    # Enable P2P (NVLink) direct GPU-to-GPU copies for NCCL all-reduce on H200.
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")  # no InfiniBand on single node

    # Pre-flight: check GPU memory before touching the device. If a previous
    # run left zombie CUDA contexts, set_device will fail with an opaque OOM.
    # Report free/total memory so the user knows to kill lingering processes.
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(rank)
        # This query uses the NVML C API and does NOT require a CUDA context.
        free, total = torch.cuda.mem_get_info(rank)
        free_gb  = free  / 1024**3
        total_gb = total / 1024**3
        print(f"[{_ts()}][rank {rank}] GPU {rank} ({props.name}): {free_gb:.1f} GB free / {total_gb:.1f} GB total", flush=True)
        if free_gb < 1.0:
            raise RuntimeError(
                f"GPU {rank} has only {free_gb:.2f} GB free — likely a zombie CUDA context "
                f"from a previous run. Run: pkill -9 -u $USER -f python3"
            )

    # 30-minute NCCL timeout — prevents a hung rank from silently blocking
    # the whole job and consuming all remaining wall time.
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=1800),
    )
    torch.cuda.set_device(rank)


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _ts() -> str:
    """Current wall-clock time string for log prefixes."""
    return time.strftime("%H:%M:%S")


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

    # ── CUDA throughput flags ─────────────────────────────────────────────────
    if device.type == "cuda":
        # cuDNN auto-tunes kernels for fixed input shapes (no cost after warmup).
        torch.backends.cudnn.benchmark = True
        # TF32 gives ~3× faster fp32 matmuls on Ampere+ with negligible accuracy
        # loss. Redundant under bf16 autocast but applies to any fp32 fallback.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # High precision for matmul reduces to TF32 on Hopper (H200).
        torch.set_float32_matmul_precision("high")
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
    # Print from every rank so the user can confirm all GPUs are initialised.
    print(f"[{_ts()}][rank {rank}] device={device}, amp_dtype={amp_dtype}, world_size={world_size}", flush=True)

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
        print(f"[{_ts()}][rank {rank}] dataset: {len(ds):,} chunks", flush=True)

    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if is_ddp else None
    )
    n_workers = train_cfg.get("num_workers")
    if n_workers is None:
        # Cap at 16 per rank to stay under OS recommended total of 32
        n_workers = min(16, max(2, (os.cpu_count() or 4) // max(1, world_size)))
    n_workers = int(n_workers)
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2)) if n_workers > 0 else None
    # When DDP is active, NCCL has already been initialized (dist.init_process_group
    # runs before this point). Linux's default DataLoader multiprocessing context
    # is 'fork', which inherits NCCL's sockets and shared-memory handles into the
    # worker processes. Workers don't use NCCL themselves, but closing inherited
    # NCCL file-descriptors on worker exit corrupts NCCL in the parent → deadlock.
    # 'forkserver' starts a clean helper server once (before any NCCL state exists),
    # then forks workers from that server. Workers never see NCCL handles. Startup
    # is faster than 'spawn' (workers clone the server's already-imported modules
    # rather than reimporting from scratch). Cost: one-time dataset pickle per
    # worker at startup (~60 MB, <10 s); free thereafter with persistent_workers.
    mp_ctx = "forkserver" if is_ddp and n_workers > 0 else None
    loader = DataLoader(
        ds, batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=cE._collate_v5,
        persistent_workers=n_workers > 0,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_ctx,
    )
    if main:
        print(f"[{_ts()}][rank {rank}] dataloader: batch_size={train_cfg['batch_size']} "
              f"num_workers={n_workers} prefetch_factor={prefetch_factor}", flush=True)

    # ── Model + optimizer + EMA ──────────────────────────────────────────────
    # raw_net holds the original nn.Module — used for EMA, checkpointing, and
    # optimizer parameters so state dicts stay compatible with/without compile.
    raw_net = M.build_model(cfg["model"]).to(device)
    sched = M.DiffusionSchedule(T=cfg["model"].get("diffusion_T", 1000)).to(device)
    if main:
        print(f"[{_ts()}][rank {rank}] model: {M.count_params(raw_net) / 1e6:.1f}M params", flush=True)

    # Compile before DDP for best kernel fusion; raw_net shares the same weights.
    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        # Print from ALL ranks — user can see both GPUs starting to compile.
        print(f"[{_ts()}][rank {rank}] torch.compile wrapping model (kernel tracing "
              f"happens on the FIRST batch — expect 5-20 min of silence per rank)", flush=True)
        net = torch.compile(raw_net)
        print(f"[{_ts()}][rank {rank}] torch.compile graph captured", flush=True)
    else:
        net = raw_net

    if is_ddp:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    optim = torch.optim.AdamW(
        raw_net.parameters(),
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler() if use_grad_scaler else None
    ema = EMA(raw_net, decay=train_cfg["ema_decay"]) if main else None

    # ── Resume from checkpoint if present ─────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"

    start_epoch, global_step = 0, 0
    if last_ckpt.is_file():
        sd = torch.load(last_ckpt, map_location=device)
        raw_net.load_state_dict(sd["model"])
        optim.load_state_dict(sd["optim"])
        if ema is not None and "ema" in sd:
            ema.load_state_dict(sd["ema"])
        if scaler is not None and "scaler" in sd and sd["scaler"] is not None:
            scaler.load_state_dict(sd["scaler"])
        start_epoch = sd.get("epoch", 0)
        global_step = sd.get("step", 0)
        if main:
            print(f"[{_ts()}][rank {rank}] resumed from {last_ckpt} "
                  f"at step {global_step}, epoch {start_epoch}", flush=True)

    # ── GPU keepalive + DataLoader pre-start ──────────────────────────────────
    # DataLoader workers are launched lazily on the first iter() call. We kick
    # them off NOW so their parquet-loading startup overlaps with GPU warmup
    # instead of counting against the first training step.
    # Without this: GPU sits at 0% for 1–3 min while workers init → job
    # termination risk on schedulers that monitor GPU utilisation.
    if n_workers > 0:
        _pre_iter = iter(loader)   # triggers forkserver to fork workers
        print(f"[{_ts()}][rank {rank}] DataLoader workers launched (pre-start)", flush=True)
    else:
        _pre_iter = None

    if device.type == "cuda":
        _wu_m, _wu_s = cfg["model"], cfg["sequence"]
        _B_wu = min(16, train_cfg["batch_size"])
        _wu_dur = float(train_cfg.get("gpu_warmup_secs", 90))
        print(
            f"[{_ts()}][rank {rank}] GPU keepalive: running dummy forward passes "
            f"for {_wu_dur:.0f}s while DataLoader workers initialize...", flush=True,
        )
        raw_net.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _wu_ctx  = torch.zeros(_B_wu, _wu_m["n_channels"],     _wu_s["W_ctx"],  device=device)
            _wu_aux  = torch.zeros(_B_wu, _wu_m["n_aux_channels"], _wu_s["W_pred"], device=device)
            _wu_xn   = torch.zeros(_B_wu, 1,                       _wu_s["W_pred"], device=device)
            _wu_t    = torch.zeros(_B_wu, device=device, dtype=torch.long)
            _wu_cond = torch.zeros(_B_wu, _wu_m["cond_dim"],                        device=device)
            _wu_end  = time.time() + _wu_dur
            while time.time() < _wu_end:
                raw_net(_wu_ctx, _wu_aux, _wu_xn, _wu_t, _wu_cond)
            torch.cuda.synchronize()
            del _wu_ctx, _wu_aux, _wu_xn, _wu_t, _wu_cond
        raw_net.train()
        torch.cuda.empty_cache()
        print(f"[{_ts()}][rank {rank}] GPU keepalive done — workers should be ready", flush=True)

    # ── Training loop ────────────────────────────────────────────────────────
    n_epochs = train_cfg["n_epochs"]
    # max_steps_per_epoch caps how many batches we consume per epoch.
    # The DataLoader shuffle (seeded by epoch via sampler.set_epoch) gives a
    # fresh random subset each epoch, so the model sees different data every
    # pass even though not all 26 M chunks are visited in one epoch.
    _max_spe = train_cfg.get("max_steps_per_epoch", None)
    steps_per_epoch = min(len(loader), _max_spe) if _max_spe else len(loader)
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

    def _save(tag: str = "last"):
        if not main:
            return
        sd = {
            "model": raw_net.state_dict(),
            "optim": optim.state_dict(),
            "ema":   ema.state_dict() if ema else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": current_epoch,
            "step":  global_step,
            "cfg":   cfg,
        }
        # Write to a temp file first, then rename — guarantees that a crash or
        # SIGKILL mid-write never corrupts the last good checkpoint on disk.
        tmp = ckpt_dir / f"{tag}.tmp.pt"
        dst = ckpt_dir / f"{tag}.pt"
        try:
            torch.save(sd, tmp)
            tmp.replace(dst)   # os.replace — atomic on POSIX when same filesystem
        except Exception as e:
            print(f"[rank 0] WARNING: checkpoint save to {dst} failed: {e}", flush=True)

    # ── Graceful-stop signal handling ────────────────────────────────────────
    _stop_requested = threading.Event()

    def _signal_handler(signum, frame):
        _stop_requested.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGUSR2, _signal_handler)

    # ── Heartbeat thread ──────────────────────────────────────────────────────
    # Prints a "still alive" line every 60 s from every rank so the user can
    # confirm both GPUs are active even during silent torch.compile warmup.
    _hb_stop = threading.Event()

    def _heartbeat():
        while not _hb_stop.wait(timeout=60.0):
            free, total = torch.cuda.mem_get_info(rank)
            used_gb = (total - free) / 1024 ** 3
            total_gb = total / 1024 ** 3
            print(
                f"[{_ts()}][rank {rank}] ♥ alive | step {global_step} | "
                f"GPU {rank}: {used_gb:.1f}/{total_gb:.1f} GB used",
                flush=True,
            )

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True, name=f"heartbeat-rank{rank}")
    _hb_thread.start()

    n_skipped_total = 0
    t_start = time.time()
    current_epoch = start_epoch
    last_loss = float("nan")
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

            # Print epoch start from ALL ranks so user sees both GPUs entering.
            free, total = torch.cuda.mem_get_info(rank)
            used_gb = (total - free) / 1024 ** 3
            print(
                f"[{_ts()}][rank {rank}] ── epoch {current_epoch} started "
                f"| steps_this_epoch={steps_per_epoch:,} "
                f"| GPU {rank}: {used_gb:.1f}/{total/1024**3:.1f} GB used",
                flush=True,
            )

            epoch_stop = False
            first_batch_done = False
            for _epoch_step, batch in enumerate(loader):
                # Both DDP ranks hit this at the same _epoch_step → no desync.
                if _max_spe is not None and _epoch_step >= _max_spe:
                    break
                ctx        = batch["ctx"].to(device, non_blocking=True)         # (B, n_ch, W_ctx)
                future_aux = batch["future_aux"].to(device, non_blocking=True)  # (B, 3, W_pred)
                target     = batch["target"].to(device, non_blocking=True)      # (B, 1, W_pred)
                pred_mask  = batch["pred_mask"].to(device, non_blocking=True)   # (B, W_pred)
                cond       = batch["cond"].to(device, non_blocking=True)        # (B, 24)
                B = ctx.size(0)

                # LR schedule (update before forward so step 0 uses warmup=0 lr)
                for pg in optim.param_groups:
                    pg["lr"] = train_cfg["lr"] * _lr_factor(global_step, warmup, total_steps)

                if device.type == "cuda":
                    cm = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                else:
                    cm = torch.amp.autocast(device_type="cpu", enabled=False)

                # ── OOM-safe forward + backward ──────────────────────────────
                batch_ok = True
                try:
                    # Sample diffusion timestep + noise
                    t = torch.randint(0, sched.T, (B,), device=device)
                    noise = torch.randn_like(target)
                    x_t = sched.q_sample(target, t, noise)
                    v_target = sched.get_v(target, t, noise)

                    # CFG dropout
                    cond_drop = torch.rand(B, device=device) < cfg_dropout_p

                    # Scheduled-noise exposure-bias mitigation
                    if ss_p > 0:
                        do_ss = torch.rand(B, device=device) < ss_p
                        if do_ss.any():
                            ss_t = torch.randint(ss_t_min, ss_t_max, (B,), device=device)
                            ctx_power = ctx[:, 3:4, :]
                            ctx_power_noisy = sched.q_sample(
                                ctx_power, ss_t, torch.randn_like(ctx_power)
                            )
                            ctx = ctx.clone()
                            ctx[:, 3:4, :] = torch.where(
                                do_ss[:, None, None], ctx_power_noisy, ctx_power
                            )

                    with cm:
                        v_pred = net(ctx, future_aux, x_t, t, cond, cond_drop_mask=cond_drop)
                        sq = (v_pred - v_target) ** 2                        # (B, 1, W_pred)
                        sq = sq.squeeze(1) * pred_mask                        # (B, W_pred)
                        valid = pred_mask.sum(dim=1).clamp_min(1.0)           # (B,)
                        per_sample_loss = sq.sum(dim=1) / valid               # (B,)

                    # Outlier guard: ZERO OUT bad per-sample losses instead of
                    # skipping the whole batch. Each rank sees different data, so
                    # independently skipping backward desynchronises the DDP
                    # gradient all_reduce → NCCL timeout after 30 min.
                    bad_mask = ~torch.isfinite(per_sample_loss) | (per_sample_loss > loss_skip_thr)
                    if bad_mask.any():
                        n_skipped_total += int(bad_mask.sum().item())
                        per_sample_loss = per_sample_loss.masked_fill(bad_mask, 0.0)
                    loss = per_sample_loss.mean()

                    # Coordinate NaN/Inf check across ALL DDP ranks. If ANY rank
                    # ends up with a still-NaN loss (e.g. ALL samples were bad),
                    # every rank skips backward together — the only DDP-safe way
                    # to avoid a one-sided gradient all_reduce that would hang.
                    nan_t = torch.tensor(0 if torch.isfinite(loss) else 1,
                                         device=device, dtype=torch.int32)
                    if is_ddp:
                        dist.all_reduce(nan_t, op=dist.ReduceOp.MAX)

                    if nan_t.item():
                        optim.zero_grad(set_to_none=True)
                        batch_ok = False
                    else:
                        # Backward + optimizer step (all ranks participate — DDP-safe)
                        optim.zero_grad(set_to_none=True)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optim)
                            torch.nn.utils.clip_grad_norm_(raw_net.parameters(), grad_clip)
                            scaler.step(optim)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(raw_net.parameters(), grad_clip)
                            optim.step()

                        if ema is not None:
                            ema.update(raw_net)

                        last_loss = loss.item()

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    optim.zero_grad(set_to_none=True)
                    n_skipped_total += 1
                    batch_ok = False
                    if main:
                        print(f"[rank {rank}] CUDA OOM at step {global_step} — skipping batch, cache cleared", flush=True)
                except Exception as exc:
                    optim.zero_grad(set_to_none=True)
                    n_skipped_total += 1
                    batch_ok = False
                    if main:
                        print(f"[rank {rank}] step {global_step} error ({type(exc).__name__}): {exc}", flush=True)

                global_step += 1

                # ── First completed batch announcement (all ranks) ────────────
                if batch_ok and not first_batch_done:
                    first_batch_done = True
                    free, total = torch.cuda.mem_get_info(rank)
                    used_gb = (total - free) / 1024 ** 3
                    print(
                        f"[{_ts()}][rank {rank}] ✓ first batch done "
                        f"(torch.compile warmup complete) "
                        f"| GPU {rank}: {used_gb:.1f}/{total/1024**3:.1f} GB used",
                        flush=True,
                    )

                # ── Stop-signal check (all ranks agree via all_reduce) ────────
                # MUST run on ALL ranks regardless of batch_ok. If placed after
                # the batch_ok `continue`, a rank that skips a bad batch would
                # miss this all_reduce while the other rank enters it → hang.
                if global_step % log_every == 0:
                    stop_t = torch.tensor(
                        1 if _stop_requested.is_set() else 0,
                        device=device, dtype=torch.int32,
                    )
                    if is_ddp:
                        dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
                    if stop_t.item():
                        if main:
                            print(f"[{_ts()}][rank 0] stop signal received — saving checkpoint and exiting", flush=True)
                            _save("last")
                        epoch_stop = True
                        break  # exit batch loop; finally will clean up

                if not batch_ok:
                    continue  # skip logging and checkpoint for bad steps

                # ── Logging (rank 0 only) ─────────────────────────────────────
                if main and global_step % log_every == 0:
                    cur_lr = optim.param_groups[0]["lr"]
                    elapsed = time.time() - t_start
                    rate = global_step / max(1.0, elapsed)
                    eta_h = (total_steps - global_step) / max(1.0, rate) / 3600.0
                    free, total_mem = torch.cuda.mem_get_info(rank)
                    used_gb = (total_mem - free) / 1024 ** 3
                    print(
                        f"[{_ts()}] step {global_step}/{total_steps} ep {current_epoch} "
                        f"lr {cur_lr:.2e} loss {last_loss:.4f} "
                        f"skipped {n_skipped_total} rate {rate:.1f} steps/s "
                        f"eta {eta_h:.1f}h ss_p {ss_p:.2f} "
                        f"GPU0: {used_gb:.1f}GB",
                        flush=True,
                    )
                    with open(log_path, "a") as f:
                        f.write(f"{global_step},{current_epoch},{cur_lr},{last_loss},{n_skipped_total}\n")

                # ── Periodic checkpoint ───────────────────────────────────────
                if main and global_step % ckpt_every == 0:
                    _save("last")

            # Epoch end
            if main and not epoch_stop:
                _save("last")
                print(f"[{_ts()}][rank 0] epoch {current_epoch} done at step {global_step}", flush=True)

            if epoch_stop:
                break  # exit epoch loop

    except KeyboardInterrupt:
        if main:
            print(f"[{_ts()}][rank 0] KeyboardInterrupt — saving checkpoint and exiting", flush=True)
            _save("last")
    finally:
        _hb_stop.set()   # stop heartbeat thread before cleanup
        if main:
            _save("final")
            if ema is not None:
                # Save EMA-only checkpoint for inference convenience
                try:
                    torch.save({"ema": ema.state_dict(), "cfg": cfg},
                               ckpt_dir / "ema.pt")
                except Exception as e:
                    print(f"[rank 0] WARNING: ema.pt save failed: {e}", flush=True)
        _ddp_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(cfg: dict) -> int:
    # Raise the OS file-descriptor limit before spawning workers.
    # DataLoader pin_memory + mmap workers each consume fds; the default soft
    # limit (1024) is far too low for 16 workers × prefetch=4 × mmap cache.
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _target = min(65536, _hard)
    if _soft < _target:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
        print(f"Raised fd limit: {_soft} → {_target}")

    # MKL ships with INTEL threading by default, which is incompatible with
    # libgomp (GNU OpenMP) used by PyTorch on Linux. Switch to GNU threading
    # before spawning so all child processes inherit the correct setting.
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

    # Persist torch.compile / Inductor kernel cache to NFS scratch so it
    # survives across SLURM jobs and different compute nodes.  On a cache hit
    # the compile warmup drops from ~15 min to ~30 s (CUDA context init only).
    _proj_root = Path(__file__).resolve().parent.parent.parent
    _cache_dir = _proj_root / ".torch_compile_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_dir))
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")  # suppress debug dumps

    # Pick a free port once here, before spawning. All worker processes inherit
    # the environment and will agree on this port. Using setdefault lets an
    # externally set MASTER_PORT (e.g. from a SLURM launcher) take precedence.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())
        print(f"DDP rendezvous port: {os.environ['MASTER_PORT']}")

    res = detect_hw()
    print(f"Detected resources: {res}")
    world = max(1, res["gpu_count"])
    if world > 1:
        torch.multiprocessing.spawn(train_one_rank, args=(world, cfg), nprocs=world, join=True)
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
