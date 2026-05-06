"""DiT-AR v5 — Sliding-window autoregressive diffusion transformer.

Per forward pass, the model takes:
  ctx          : (B, n_ch, W_ctx)         past context, all 4 channels
  future_aux   : (B, 3,    W_pred)        future aux conditioning (gpu/mem/MiB)
  noisy_power  : (B, 1,    W_pred)        noisy power being denoised
  cond         : (B, cond_dim)            24-dim SLURM features
  t            : (B,)                     diffusion timestep (long)
  cond_drop_mask: (B,) bool, optional     CFG conditioning dropout

And returns:
  v_pred       : (B, 1, W_pred)           v-prediction target

Sequence layout (in token order):
  [ cond_token | ctx_tokens (W_ctx/P) | aux_tokens (W_pred/P) | pred_tokens (W_pred/P) ]

At inference: DDIM 50 steps, CFG=1.5, sliding-window AR.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion schedule (cosine β, T=1000) + v-prediction utilities
# ─────────────────────────────────────────────────────────────────────────────
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = 1 - ac[1:] / ac[:-1]
    return torch.clip(betas, 1e-4, 0.999).float()


class DiffusionSchedule(nn.Module):
    """Cosine-β schedule with v-prediction reparameterization.

    v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0      (Salimans & Ho 2022)
    x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
    eps = sqrt(alpha_bar) * v + sqrt(1 - alpha_bar) * x_t
    """

    def __init__(self, T: int = 1000):
        super().__init__()
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.T = T
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", ac)
        self.register_buffer("sqrt_ac", ac.sqrt())
        self.register_buffer("sqrt_one_minus_ac", (1 - ac).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa  = self.sqrt_ac[t].view(-1, 1, 1)
        som = self.sqrt_one_minus_ac[t].view(-1, 1, 1)
        return sa * x0 + som * noise

    def get_v(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa  = self.sqrt_ac[t].view(-1, 1, 1)
        som = self.sqrt_one_minus_ac[t].view(-1, 1, 1)
        return sa * noise - som * x0

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sa  = self.sqrt_ac[t].view(-1, 1, 1)
        som = self.sqrt_one_minus_ac[t].view(-1, 1, 1)
        return sa * x_t - som * v

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sa  = self.sqrt_ac[t].view(-1, 1, 1)
        som = self.sqrt_one_minus_ac[t].view(-1, 1, 1)
        return sa * v + som * x_t


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (DiT-style with AdaLN-Zero modulation)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Standard DiT block: multi-head self-attention + MLP, both with AdaLN-Zero
    modulation from the conditioning vector c_vec."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.dropout = dropout

        self.norm1     = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv       = nn.Linear(d, 3 * d, bias=True)
        self.attn_proj = nn.Linear(d, d, bias=True)

        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        d_ff = mlp_ratio * d
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d, bias=True),
        )
        # AdaLN-Zero: zero-init the modulation projection so blocks act as
        # near-identity at initialization. Critical for training stability.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # scaled_dot_product_attention auto-uses Flash Attention on H100/H200,
        # which is essential here — at N=2049, naive O(N²) attention scores
        # would consume ~10s of GB. Flash keeps memory linear in N.
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        o = o.transpose(1, 2).reshape(B, N, D)
        return self.attn_proj(o)

    def forward(self, x: torch.Tensor, c_vec: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c_vec).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self._attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final layer: AdaLN-Zero modulation + linear projection to patch_size.
    Input is (B, N_pred, d), output is (B, N_pred, patch_size).
    Output unpatchifies (reshape) to (B, 1, W_pred) externally."""

    def __init__(self, d: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.norm   = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d, patch_size, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 2 * d, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # Zero-init the output linear so v_pred = 0 at init. With v-prediction
        # this corresponds to x0_pred = sqrt(alpha_bar) * x_t — a stable
        # starting point that training can break symmetry from.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c_vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c_vec).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────
class DiT_AR_v5(nn.Module):
    """DiT-AR v5 sliding-window autoregressive diffusion transformer.

    Forward pass produces a v-prediction for the noisy_power region given:
      - past context (4-channel),
      - future auxiliary signals (3-channel deterministic conditioning),
      - SLURM features (24-dim, both as a token and via AdaLN modulation).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d         = cfg["d_model"]
        self.W_ctx  = cfg["W_ctx"]
        self.W_pred = cfg["W_pred"]
        self.P      = cfg["patch_size"]
        self.cond_dim   = cfg["cond_dim"]
        self.n_ch       = cfg["n_channels"]            # 4 — number of channels in ctx
        self.n_aux      = cfg["n_aux_channels"]        # 3 — gpu_pct, mem_pct, mem_MiB

        assert self.W_ctx  % self.P == 0
        assert self.W_pred % self.P == 0
        self.N_ctx  = self.W_ctx  // self.P
        self.N_pred = self.W_pred // self.P            # used for both aux and pred

        # ── Three separate patch embeddings (different input channel counts) ──
        self.ctx_patch  = nn.Conv1d(self.n_ch,  d, kernel_size=self.P, stride=self.P)
        self.aux_patch  = nn.Conv1d(self.n_aux, d, kernel_size=self.P, stride=self.P)
        self.pred_patch = nn.Conv1d(1,          d, kernel_size=self.P, stride=self.P)

        # ── Positional embeddings (separate per region; learned) ──
        # Tokens are arranged as: [cond | ctx (N_ctx) | aux (N_pred) | pred (N_pred)]
        # Positional embeddings differ per region so the model knows what role
        # each token plays. The cond token has no positional embedding.
        self.pos_ctx  = nn.Parameter(torch.zeros(1, self.N_ctx,  d))
        self.pos_aux  = nn.Parameter(torch.zeros(1, self.N_pred, d))
        self.pos_pred = nn.Parameter(torch.zeros(1, self.N_pred, d))
        for p in (self.pos_ctx, self.pos_aux, self.pos_pred):
            nn.init.trunc_normal_(p, std=0.02)

        # ── Cond token: in-sequence projection (separate from AdaLN path) ──
        self.cond_token_proj = nn.Sequential(
            nn.Linear(cfg["cond_dim"], d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # ── Time embedding for AdaLN ──
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmb(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        # ── SLURM embedding for AdaLN (parallel to time, summed) ──
        self.c_embed = nn.Sequential(
            nn.Linear(cfg["cond_dim"], d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # ── CFG null conditioning (learned) ──
        self.null_cond = nn.Parameter(torch.zeros(1, cfg["cond_dim"]))
        nn.init.trunc_normal_(self.null_cond, std=0.02)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg["n_heads"], cfg["mlp_ratio"], cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])

        # ── Final layer (predicts only on the pred positions) ──
        self.final = FinalLayer(d, self.P)

        # Gradient checkpointing: recomputes each DiT block during backward
        # instead of storing all intermediate activations. Reduces activation
        # VRAM from O(n_layers × B × N × d) to O(n_layers × B × N × d / n_layers)
        # = O(B × N × d) checkpoint boundaries only. Mandatory for B > ~300 on
        # H200 (139 GB), allows B up to ~2048+ on H200.
        self.use_checkpoint = cfg.get("use_checkpoint", False)

        # General init
        self.apply(self._init_weights)
        # Re-zero AdaLN modulation/output after generic init pass
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[-1].weight)
            nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        ctx: torch.Tensor,             # (B, n_ch, W_ctx)
        future_aux: torch.Tensor,      # (B, n_aux, W_pred)
        noisy_power: torch.Tensor,     # (B, 1, W_pred)
        t: torch.Tensor,               # (B,) long
        cond: torch.Tensor,            # (B, cond_dim)
        cond_drop_mask: torch.Tensor | None = None,  # (B,) bool
    ) -> torch.Tensor:
        """Returns v_pred of shape (B, 1, W_pred)."""
        B = ctx.size(0)

        # CFG: replace cond with null where mask is True
        if cond_drop_mask is not None:
            null = self.null_cond.expand(B, -1)
            cond = torch.where(cond_drop_mask.unsqueeze(1), null, cond)

        # ── Patch embeddings ──
        h_ctx  = self.ctx_patch(ctx).transpose(1, 2)               # (B, N_ctx,  d)
        h_aux  = self.aux_patch(future_aux).transpose(1, 2)         # (B, N_pred, d)
        h_pred = self.pred_patch(noisy_power).transpose(1, 2)       # (B, N_pred, d)

        # Add per-region positional embeddings
        h_ctx  = h_ctx  + self.pos_ctx
        h_aux  = h_aux  + self.pos_aux
        h_pred = h_pred + self.pos_pred

        # ── Prepend cond token ──
        ct = self.cond_token_proj(cond).unsqueeze(1)                # (B, 1, d)

        # Concatenate to form the full sequence
        h = torch.cat([ct, h_ctx, h_aux, h_pred], dim=1)            # (B, 1+N_ctx+2*N_pred, d)

        # ── AdaLN conditioning vector (time + SLURM) ──
        c_vec = self.t_embed(t) + self.c_embed(cond)                # (B, d)

        # ── Transformer trunk ──
        # With use_checkpoint=True each block's intermediate activations are
        # discarded and recomputed during backward (O(1) activation storage
        # per block) — essential for large B on H200. use_reentrant=False is
        # required for compatibility with torch.compile and DDP.
        if self.use_checkpoint:
            for blk in self.blocks:
                h = gradient_checkpoint(blk, h, c_vec, use_reentrant=False)
        else:
            for blk in self.blocks:
                h = blk(h, c_vec)

        # ── Slice out only the pred-region tokens ──
        # Layout: [cond(1), ctx(N_ctx), aux(N_pred), pred(N_pred)]
        pred_start = 1 + self.N_ctx + self.N_pred
        h_out = h[:, pred_start:, :]                                # (B, N_pred, d)

        # ── Final projection + unpatchify ──
        out = self.final(h_out, c_vec)                              # (B, N_pred, P)
        out = out.reshape(B, self.N_pred * self.P).unsqueeze(1)     # (B, 1, W_pred)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg: dict) -> DiT_AR_v5:
    """Construct a DiT_AR_v5 from a config dict, validating required keys."""
    required = [
        "d_model", "n_heads", "n_layers", "mlp_ratio", "dropout",
        "patch_size", "W_ctx", "W_pred", "cond_dim",
        "n_channels", "n_aux_channels",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Model config missing required keys: {missing}")
    return DiT_AR_v5(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
