"""
Egocentric Spatial Cross-Attention Temporal Alignment Module.

Aggregates historical BEV latent feature maps into a spatially-aligned context
by conditioning each history frame with its ego-relative odometry, then
cross-attending from the current frame's spatial queries to the conditioned
history key-value tokens.

Tensor flow summary
-------------------
  current_bev   : (B, C, H, W)
  history_bev   : (B, T, C, H, W)
  relative_odom : (B, T, D_odom)

  Step 1 — odom MLP + broadcast  →  conditioned_history : (B, T, C, H, W)
  Step 2 — flatten               →  Q:(B, H*W, C)  KV:(B, T*H*W, C)
  Step 3 — cross-attention       →  attn_output : (B, H*W, C)
  Step 4 — reshape               →  (B, C, H, W)
"""
from __future__ import annotations

import collections
import torch
import torch.nn as nn


class EgoCentricCrossAttentionTemporalModule(nn.Module):
    """
    Odometry-conditioned cross-attention over BEV spatial feature maps.

    Parameters
    ----------
    channels        : C — BEV feature-map channel depth (must equal backbone output channels)
    odom_dim        : D_odom — dimensionality of per-step relative odometry vector
    num_heads       : number of attention heads (must evenly divide channels)
    max_history_len : T — maximum history frames kept in the circular online cache
    dropout         : attention + MLP dropout probability
    """

    def __init__(
        self,
        channels:        int   = 128,
        odom_dim:        int   = 3,
        num_heads:       int   = 8,
        max_history_len: int   = 8,
        dropout:         float = 0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.channels        = channels
        self.odom_dim        = odom_dim
        self.num_heads       = num_heads
        self.max_history_len = max_history_len

        # Step 1 — lightweight MLP: (B, T, D_odom) → (B, T, C)
        self.odom_mlp = nn.Sequential(
            nn.Linear(odom_dim, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

        # Step 3 — cross-attention; batch_first=True → inputs are (B, seq, C)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # Pre-LN style: normalise Q and KV before attention, output after residual
        self.norm_q   = nn.LayerNorm(channels)
        self.norm_kv  = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)

    # ------------------------------------------------------------------
    # Training / batch forward
    # ------------------------------------------------------------------

    def forward(
        self,
        current_bev:   torch.Tensor,   # (B, C, H, W)
        history_bev:   torch.Tensor,   # (B, T, C, H, W)
        relative_odom: torch.Tensor,   # (B, T, D_odom)
    ) -> torch.Tensor:
        """
        Attend current BEV spatial tokens over odometry-conditioned history tokens.

        Returns
        -------
        attended_bev : (B, C, H, W)
        """
        B, C, H, W   = current_bev.shape
        _, T, _, _, _ = history_bev.shape

        # ------------------------------------------------------------------
        # Step 1: Odometry State Embedding (Spatial Conditioning)
        # ------------------------------------------------------------------

        # Project relative pose vectors into channel space
        # (B, T, D_odom) → odom_emb : (B, T, C)
        odom_emb = self.odom_mlp(relative_odom)

        # Broadcast pose embedding across spatial dims H, W
        # (B, T, C) → (B, T, C, 1, 1) → (B, T, C, H, W)
        odom_emb = odom_emb.unsqueeze(-1).unsqueeze(-1).expand(B, T, C, H, W)

        # Residual-fuse: shift each history feature map by its ego-relative pose
        # conditioned_history_bev : (B, T, C, H, W)
        conditioned_history_bev = history_bev + odom_emb

        # ------------------------------------------------------------------
        # Step 2: Dense Feature Flattening
        # ------------------------------------------------------------------

        # Query: current-frame spatial tokens
        # (B, C, H, W) → permute (B, H, W, C) → reshape (B, H*W, C)
        Q = current_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Key & Value: conditioned history — merge T and spatial dims
        # (B, T, C, H, W) → permute (B, T, H, W, C) → reshape (B, T*H*W, C)
        KV = conditioned_history_bev.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)

        # Pre-norm before attention (Pre-LN convention)
        Q  = self.norm_q(Q)    # (B, H*W,   C)
        KV = self.norm_kv(KV)  # (B, T*H*W, C)

        # ------------------------------------------------------------------
        # Step 3: Cross-Attention  Q:(B, H*W, C)  ×  KV:(B, T*H*W, C)
        # ------------------------------------------------------------------

        # attn_output : (B, H*W, C)
        attn_output, _ = self.cross_attn(query=Q, key=KV, value=KV)

        # Residual skip from Q + output norm
        # (preserves current-frame information if history attention is uncertain)
        attn_output = self.norm_out(attn_output + Q)   # (B, H*W, C)

        # ------------------------------------------------------------------
        # Step 4: Spatial Reshaping
        # ------------------------------------------------------------------

        # (B, H*W, C) → (B, H, W, C) → (B, C, H, W)
        out = attn_output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out   # (B, C, H, W)

    # ------------------------------------------------------------------
    # Online KV Cache — fixed-length circular buffer for real-time deployment
    # ------------------------------------------------------------------

    def init_cache(self) -> None:
        """Reset the circular history cache. Call once at episode start."""
        # deque(maxlen=T) auto-evicts the oldest entry when capacity is exceeded
        self._bev_cache  = collections.deque(maxlen=self.max_history_len)
        self._odom_cache = collections.deque(maxlen=self.max_history_len)

    def step(
        self,
        new_bev:            torch.Tensor,   # (B, C, H, W) — newest backbone feature map
        relative_odom_step: torch.Tensor,   # (B, D_odom)  — odom of this frame rel. to current
    ) -> torch.Tensor:
        """
        Push one frame into the circular buffer and return the attended BEV.

        The buffer holds at most `max_history_len` frames; the oldest is evicted
        automatically.  KV sequence length stays bounded at O(max_history_len - 1).

        On the very first call (or when only one frame has been seen), the
        current BEV is returned unchanged because there is no history to attend
        over.

        Returns
        -------
        attended_bev : (B, C, H, W)
        """
        if not hasattr(self, "_bev_cache"):
            self.init_cache()

        # Detach before caching to prevent stale gradient graphs
        self._bev_cache.append(new_bev.detach())               # (B, C, H, W)
        self._odom_cache.append(relative_odom_step.detach())   # (B, D_odom)

        if len(self._bev_cache) < 2:
            return new_bev   # no history yet

        # History = every cached frame except the most recent (current)
        history_bev   = torch.stack(list(self._bev_cache)[:-1],  dim=1)   # (B, T, C, H, W)
        relative_odom = torch.stack(list(self._odom_cache)[:-1], dim=1)   # (B, T, D_odom)

        return self.forward(new_bev, history_bev, relative_odom)


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("EgoCentricCrossAttentionTemporalModule — shape verification")
    print("=" * 64)

    B, T, C, H, W = 2, 6, 128, 5, 5
    D_ODOM   = 3
    HEADS    = 8
    MAX_HIST = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Config : B={B} T={T} C={C} H={H} W={W} D_odom={D_ODOM} heads={HEADS}\n")

    module = EgoCentricCrossAttentionTemporalModule(
        channels        = C,
        odom_dim        = D_ODOM,
        num_heads       = HEADS,
        max_history_len = MAX_HIST,
        dropout         = 0.0,
    ).to(device)

    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable parameters : {n_params:,}")

    # ------------------------------------------------------------------
    # (a) Training forward pass — full batch with history tensor
    # ------------------------------------------------------------------
    print("\n--- (a) Training forward pass ---")
    current_bev   = torch.randn(B, C, H, W,    device=device)
    history_bev   = torch.randn(B, T, C, H, W, device=device)
    relative_odom = torch.randn(B, T, D_ODOM,  device=device)

    out = module(current_bev, history_bev, relative_odom)

    print(f"  current_bev   : {tuple(current_bev.shape)}")
    print(f"  history_bev   : {tuple(history_bev.shape)}")
    print(f"  relative_odom : {tuple(relative_odom.shape)}")
    print(f"  output        : {tuple(out.shape)}")
    assert out.shape == (B, C, H, W), f"FAIL: expected {(B,C,H,W)}, got {out.shape}"
    print("  PASS")

    # ------------------------------------------------------------------
    # (b) Online step-mode — circular KV cache with B=1 (robot deployment)
    # ------------------------------------------------------------------
    print("\n--- (b) Online step-mode (circular KV cache) ---")
    module.eval()
    module.init_cache()

    N_STEPS = MAX_HIST + 2   # push past buffer capacity to test eviction
    for t in range(N_STEPS):
        bev_t  = torch.randn(1, C, H, W,  device=device)
        odom_t = torch.randn(1, D_ODOM,   device=device)

        with torch.no_grad():
            out_t = module.step(bev_t, odom_t)

        cache_sz = len(module._bev_cache)
        assert out_t.shape == (1, C, H, W), f"FAIL at t={t}: {out_t.shape}"
        assert cache_sz <= MAX_HIST,         f"FAIL: cache overflowed at t={t}: {cache_sz}"
        print(
            f"  t={t:2d}  cache={cache_sz:2d}/{MAX_HIST}"
            f"  output={tuple(out_t.shape)}  PASS"
        )

    print("\nAll shape checks passed.")
