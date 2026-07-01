"""
Depth frame encoders.

Spatial-only (per-frame):
  CNNEncoder   (B, T, H, W) → (B, T, D)
  ViTEncoder   (B, T, H, W) → (B, T, D)

No-temporal wrappers (context = current = last-frame spatial embedding):
  CNNOnlyEncoder
  ViTOnlyEncoder

Temporal — spatial + GRU:
  TemporalCNNEncoder
  TemporalViTEncoder

Temporal — spatial + Transformer encoder across time:
  TemporalTransformerCNNEncoder
  TemporalTransformerViTEncoder

All temporal encoders share the same output contract:
  forward(depth: B, T, H, W) -> (context: B, D, current: B, D)
    context : temporal summary (GRU hidden state or Transformer CLS token)
    current : last-frame spatial encoding (no temporal processing)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cross_attn_temporal import EgoCentricCrossAttentionTemporalModule


# ---------------------------------------------------------------------------
# Spatial CNN encoder (per-frame, shared weights across T)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Four-stage strided CNN on single-channel depth images → (B, T, embed_dim)."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),  nn.GELU(),

            nn.Conv2d(32, 64,  kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) → (B, T, embed_dim)"""
        B, T, H, W = depth.shape
        x = self.cnn(depth.view(B * T, 1, H, W)).flatten(1)
        return self.proj(x).view(B, T, self.embed_dim)


# ---------------------------------------------------------------------------
# Spatial ViT encoder (per-frame, shared weights across T)
# ---------------------------------------------------------------------------

class _PatchEmbedding(nn.Module):
    def __init__(self, img_h: int, img_w: int, patch_size: int, embed_dim: int):
        super().__init__()
        assert img_h % patch_size == 0 and img_w % patch_size == 0
        self.n_patches = (img_h // patch_size) * (img_w // patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, H, W) → (N, n_patches, embed_dim)"""
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for single-channel depth frames → (B, T, embed_dim)."""

    def __init__(
        self,
        img_h:      int   = 256,
        img_w:      int   = 256,
        patch_size: int   = 32,
        embed_dim:  int   = 256,
        depth:      int   = 4,
        num_heads:  int   = 8,
        mlp_ratio:  float = 4.0,
        drop:       float = 0.0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_embed = _PatchEmbedding(img_h, img_w, patch_size, embed_dim)
        n_patches        = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=depth,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) → (B, T, embed_dim)"""
        B, T, H, W = depth.shape
        x   = self.patch_embed(depth.view(B * T, 1, H, W))
        cls = self.cls_token.expand(B * T, -1, -1)
        x   = self.norm(self.transformer(torch.cat([cls, x], dim=1) + self.pos_embed))
        return x[:, 0].view(B, T, self.embed_dim)   # CLS token per frame


# ---------------------------------------------------------------------------
# No-temporal encoders  (context = current = last-frame spatial embedding)
# ---------------------------------------------------------------------------

class CNNOnlyEncoder(nn.Module):
    """CNN per-frame, no temporal module.  context = current = last-frame embedding."""

    def __init__(self, embed_dim: int = 256, **_):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = CNNEncoder(embed_dim=embed_dim)

    def forward(self, depth: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        last = self.spatial(depth)[:, -1]   # (B, D)
        return last, last


class ViTOnlyEncoder(nn.Module):
    """ViT per-frame, no temporal module.  context = current = last-frame embedding."""

    def __init__(
        self,
        img_h:      int   = 256,
        img_w:      int   = 256,
        patch_size: int   = 32,
        embed_dim:  int   = 256,
        vit_depth:  int   = 4,
        num_heads:  int   = 8,
        drop:       float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = ViTEncoder(
            img_h=img_h, img_w=img_w, patch_size=patch_size,
            embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads, drop=drop,
        )

    def forward(self, depth: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        last = self.spatial(depth)[:, -1]   # (B, D)
        return last, last


# ---------------------------------------------------------------------------
# Temporal — spatial + GRU
# ---------------------------------------------------------------------------

class TemporalCNNEncoder(nn.Module):
    """
    CNN (per-frame) + GRU (across time).

    context : (B, D)  GRU final hidden state — temporal visual memory
    current : (B, D)  last-frame CNN output  — current observation

    state_delta (B, T, 3) is projected and added to the per-frame visual
    embedding before the GRU.  Without it the GRU sees raw depth frames whose
    appearance changes with every camera pose, making temporal integration
    impossible.  Pass zeros if unavailable.
    """

    def __init__(
        self,
        embed_dim:  int   = 256,
        num_layers: int   = 2,
        drop:       float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.spatial     = CNNEncoder(embed_dim=embed_dim)
        self.motion_proj = nn.Linear(3, embed_dim)
        self.gru         = nn.GRU(
            input_size=embed_dim, hidden_size=embed_dim,
            num_layers=num_layers, batch_first=True,
            dropout=drop if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        depth:       torch.Tensor,
        state_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(depth)                             # (B, T, D)
        if state_delta is not None:
            visual = visual + self.motion_proj(state_delta)      # fuse ego-motion
        _, h_n = self.gru(visual)                                # h_n: (num_layers, B, D)
        return h_n[-1], visual[:, -1]


class TemporalViTEncoder(nn.Module):
    """
    ViT (per-frame) + GRU (across time).

    context : (B, D)  GRU final hidden state — temporal visual memory
    current : (B, D)  last-frame ViT output  — current observation
    """

    def __init__(
        self,
        img_h:      int   = 256,
        img_w:      int   = 256,
        patch_size: int   = 32,
        embed_dim:  int   = 256,
        vit_depth:  int   = 4,
        num_heads:  int   = 8,
        gru_layers: int   = 2,
        drop:       float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.spatial     = ViTEncoder(
            img_h=img_h, img_w=img_w, patch_size=patch_size,
            embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads, drop=drop,
        )
        self.motion_proj = nn.Linear(3, embed_dim)
        self.gru         = nn.GRU(
            input_size=embed_dim, hidden_size=embed_dim,
            num_layers=gru_layers, batch_first=True,
            dropout=drop if gru_layers > 1 else 0.0,
        )

    def forward(
        self,
        depth:       torch.Tensor,
        state_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(depth)
        if state_delta is not None:
            visual = visual + self.motion_proj(state_delta)
        _, h_n = self.gru(visual)
        return h_n[-1], visual[:, -1]


# ---------------------------------------------------------------------------
# Temporal — spatial + Transformer encoder across time
# ---------------------------------------------------------------------------

def _make_temporal_transformer(embed_dim: int, depth: int, heads: int, drop: float):
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop, activation="gelu",
            batch_first=True, norm_first=True,
        ),
        num_layers=depth,
    )


class TemporalTransformerCNNEncoder(nn.Module):
    """
    CNN (per-frame) + Transformer encoder (across time).

    context : (B, D)  CLS token from temporal Transformer — temporal visual memory
    current : (B, D)  last-frame CNN output (raw)         — current observation
    """

    def __init__(
        self,
        embed_dim:      int   = 256,
        seq_len:        int   = 10,
        temporal_depth: int   = 2,
        temporal_heads: int   = 8,
        drop:           float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = CNNEncoder(embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.temporal = _make_temporal_transformer(embed_dim, temporal_depth, temporal_heads, drop)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, depth: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(depth)                      # (B, T, D)
        B, T, _ = visual.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), visual], dim=1)  # (B, T+1, D)
        x = self.norm(self.temporal(x + self.pos_embed[:, :T + 1]))
        return x[:, 0], visual[:, -1]                    # CLS context, raw last-frame current


class TemporalTransformerViTEncoder(nn.Module):
    """
    ViT (per-frame) + Transformer encoder (across time).

    context : (B, D)  CLS token from temporal Transformer — temporal visual memory
    current : (B, D)  last-frame ViT output (raw)         — current observation
    """

    def __init__(
        self,
        img_h:          int   = 256,
        img_w:          int   = 256,
        patch_size:     int   = 32,
        embed_dim:      int   = 256,
        vit_depth:      int   = 4,
        num_heads:      int   = 8,
        seq_len:        int   = 10,
        temporal_depth: int   = 2,
        temporal_heads: int   = 8,
        drop:           float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = ViTEncoder(
            img_h=img_h, img_w=img_w, patch_size=patch_size,
            embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads, drop=drop,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.temporal = _make_temporal_transformer(embed_dim, temporal_depth, temporal_heads, drop)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, depth: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(depth)
        B, T, _ = visual.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), visual], dim=1)
        x = self.norm(self.temporal(x + self.pos_embed[:, :T + 1]))
        return x[:, 0], visual[:, -1]


# ---------------------------------------------------------------------------
# BEV encoders  (input: (B, T, 2, G, G)  — occupancy + height channels)
# ---------------------------------------------------------------------------

class BEVCNNEncoder(nn.Module):
    """
    2-channel BEV grid (occupancy + height) → (B, T, embed_dim).

    Input shape: (B, T, 2, G, G).
    Three stride-2 conv stages shrink (G, G) → (1, 1) via AdaptiveAvgPool2d.
    """

    def __init__(self, grid_size: int = 40, embed_dim: int = 256, **_):
        super().__init__()
        self.embed_dim = embed_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(2,  32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """(B, T, 2, G, G) → (B, T, embed_dim)"""
        B, T, C, G, _ = bev.shape
        x = self.cnn(bev.view(B * T, C, G, G)).flatten(1)
        return self.proj(x).view(B, T, self.embed_dim)


class BEVCNNOnlyEncoder(nn.Module):
    """BEV CNN, no temporal.  context = current = last-frame embedding."""

    def __init__(self, grid_size: int = 40, embed_dim: int = 256, **_):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = BEVCNNEncoder(grid_size=grid_size, embed_dim=embed_dim)

    def forward(self, bev: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Only feed the last frame — avoids wasted compute when seq_len > 1 in the obs buffer.
        last = self.spatial(bev[:, -1:])[:, 0]
        return last, last


class TemporalBEVCNNEncoder(nn.Module):
    """
    BEV CNN (per-frame) + GRU (across time).

    context : (B, D)  GRU final hidden state
    current : (B, D)  last-frame BEV CNN output
    """

    def __init__(
        self,
        grid_size:  int   = 40,
        embed_dim:  int   = 256,
        num_layers: int   = 2,
        drop:       float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.spatial     = BEVCNNEncoder(grid_size=grid_size, embed_dim=embed_dim)
        self.motion_proj = nn.Linear(3, embed_dim)
        self.gru         = nn.GRU(
            input_size=embed_dim, hidden_size=embed_dim,
            num_layers=num_layers, batch_first=True,
            dropout=drop if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        bev:         torch.Tensor,
        state_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(bev)
        if state_delta is not None:
            visual = visual + self.motion_proj(state_delta)
        _, h_n = self.gru(visual)
        return h_n[-1], visual[:, -1]


class BEVViTEncoder(nn.Module):
    """
    Vision Transformer encoder for 2-channel BEV grids → (B, T, embed_dim).

    Input: (B, T, 2, G, G)  occupancy + height channels.
    Patch-embeds each G×G frame, prepends a CLS token, runs a spatial
    Transformer, and returns the CLS token per frame.
    """

    def __init__(
        self,
        grid_size:  int   = 40,
        patch_size: int   = 8,    # must divide grid_size evenly
        embed_dim:  int   = 256,
        depth:      int   = 4,
        num_heads:  int   = 8,
        mlp_ratio:  float = 4.0,
        drop:       float = 0.0,
        **_,
    ):
        super().__init__()
        assert grid_size % patch_size == 0, (
            f"grid_size ({grid_size}) must be divisible by patch_size ({patch_size})"
        )
        self.embed_dim = embed_dim
        n_patches      = (grid_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=depth,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """(B, T, 2, G, G) → (B, T, embed_dim)"""
        B, T, C, G, _ = bev.shape
        x   = self.patch_embed(bev.view(B * T, C, G, G))   # (B*T, D, ph, pw)
        x   = x.flatten(2).transpose(1, 2)                  # (B*T, n_patches, D)
        cls = self.cls_token.expand(B * T, -1, -1)
        x   = self.norm(self.transformer(torch.cat([cls, x], dim=1) + self.pos_embed))
        return x[:, 0].view(B, T, self.embed_dim)           # CLS token per frame


class TemporalBEVViTEncoder(nn.Module):
    """
    BEV ViT (per-frame) + GRU (across time).

    context : (B, D)  GRU final hidden state
    current : (B, D)  last-frame BEV ViT output
    """

    def __init__(
        self,
        grid_size:      int   = 40,
        bev_patch_size: int   = 8,
        embed_dim:      int   = 256,
        vit_depth:      int   = 4,
        num_heads:      int   = 8,
        gru_layers:     int   = 2,
        drop:           float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.spatial     = BEVViTEncoder(
            grid_size=grid_size, patch_size=bev_patch_size,
            embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads, drop=drop,
        )
        self.motion_proj = nn.Linear(3, embed_dim)
        self.gru         = nn.GRU(
            input_size=embed_dim, hidden_size=embed_dim,
            num_layers=gru_layers, batch_first=True,
            dropout=drop if gru_layers > 1 else 0.0,
        )

    def forward(
        self,
        bev:         torch.Tensor,
        state_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(bev)
        if state_delta is not None:
            visual = visual + self.motion_proj(state_delta)
        _, h_n = self.gru(visual)
        return h_n[-1], visual[:, -1]


class BEVViTOnlyEncoder(nn.Module):
    """BEV ViT, no temporal.  context = current = last-frame embedding."""

    def __init__(
        self,
        grid_size:      int   = 40,
        bev_patch_size: int   = 8,
        embed_dim:      int   = 256,
        vit_depth:      int   = 4,
        num_heads:      int   = 8,
        drop:           float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = BEVViTEncoder(
            grid_size=grid_size, patch_size=bev_patch_size,
            embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads, drop=drop,
        )

    def forward(self, bev: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Only feed the last frame — avoids wasted compute when seq_len > 1 in the obs buffer.
        last = self.spatial(bev[:, -1:])[:, 0]
        return last, last


class TemporalTransformerBEVViTEncoder(nn.Module):
    """
    BEV ViT (per-frame) + Transformer encoder (across time).

    context : (B, D)  CLS token from temporal Transformer
    current : (B, D)  last-frame BEV ViT output
    """

    def __init__(
        self,
        grid_size:      int   = 40,
        bev_patch_size: int   = 8,
        embed_dim:      int   = 256,
        vit_depth:      int   = 4,
        num_heads:      int   = 8,
        seq_len:        int   = 10,
        temporal_depth: int   = 2,
        temporal_heads: int   = 8,
        drop:           float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = BEVViTEncoder(
            grid_size=grid_size, patch_size=bev_patch_size,
            embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads, drop=drop,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.temporal  = _make_temporal_transformer(embed_dim, temporal_depth, temporal_heads, drop)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, bev: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(bev)
        B, T, _ = visual.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), visual], dim=1)
        x = self.norm(self.temporal(x + self.pos_embed[:, :T + 1]))
        return x[:, 0], visual[:, -1]


class TemporalTransformerBEVCNNEncoder(nn.Module):
    """
    BEV CNN (per-frame) + Transformer encoder (across time).

    context : (B, D)  CLS token from temporal Transformer
    current : (B, D)  last-frame BEV CNN output
    """

    def __init__(
        self,
        grid_size:      int   = 40,
        embed_dim:      int   = 256,
        seq_len:        int   = 10,
        temporal_depth: int   = 2,
        temporal_heads: int   = 8,
        drop:           float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial   = BEVCNNEncoder(grid_size=grid_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.temporal  = _make_temporal_transformer(embed_dim, temporal_depth, temporal_heads, drop)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, bev: torch.Tensor, state_delta=None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.spatial(bev)
        B, T, _ = visual.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), visual], dim=1)
        x = self.norm(self.temporal(x + self.pos_embed[:, :T + 1]))
        return x[:, 0], visual[:, -1]


# ---------------------------------------------------------------------------
# Cross-attention spatial backbones (preserve H×W, no global pool)
# ---------------------------------------------------------------------------

class BEVCNNSpatialBackbone(nn.Module):
    """
    2-channel BEV grid → per-frame feature maps with spatial dims intact.

    Identical to the first three conv stages of BEVCNNEncoder but omits
    AdaptiveAvgPool2d and the Linear projection, so spatial structure is
    preserved for cross-attention processing.

    Input : (B, T, 2, G, G)
    Output: (B, T, feat_channels, G/8, G/8)  (exact size depends on G)
    """

    def __init__(self, feat_channels: int = 128, **_):
        super().__init__()
        self.feat_channels = feat_channels
        self.cnn = nn.Sequential(
            # (2, G, G) → (32, G/2, G/2)
            nn.Conv2d(2,  32,            kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.GELU(),
            # (32, G/2, G/2) → (64, G/4, G/4)
            nn.Conv2d(32, 64,            kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(),
            # (64, G/4, G/4) → (feat_channels, G/8, G/8)
            nn.Conv2d(64, feat_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels), nn.GELU(),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """(B, T, 2, G, G) → (B, T, feat_channels, H_feat, W_feat)"""
        B, T, C, G, _ = bev.shape
        x = self.cnn(bev.view(B * T, C, G, G))    # (B*T, feat_channels, H_feat, W_feat)
        H_f, W_f = x.shape[-2:]
        return x.view(B, T, self.feat_channels, H_f, W_f)


class BEVViTSpatialBackbone(nn.Module):
    """
    BEV ViT encoder that returns patch tokens as a 2D spatial map
    instead of the global CLS token.

    The full spatial Transformer is applied (same architecture as BEVViTEncoder),
    then the patch tokens are reshaped back to a 2D grid so that downstream
    cross-attention can reason about spatial positions.

    Input : (B, T, 2, G, G)
    Output: (B, T, embed_dim, H_p, W_p)  where H_p = W_p = G // patch_size
    """

    def __init__(
        self,
        grid_size:  int   = 40,
        patch_size: int   = 8,
        embed_dim:  int   = 256,
        depth:      int   = 4,
        num_heads:  int   = 8,
        mlp_ratio:  float = 4.0,
        drop:       float = 0.0,
        **_,
    ):
        super().__init__()
        assert grid_size % patch_size == 0, (
            f"grid_size ({grid_size}) must be divisible by patch_size ({patch_size})"
        )
        self.embed_dim = embed_dim
        self.H_p       = grid_size // patch_size   # spatial height of patch grid
        self.W_p       = grid_size // patch_size   # spatial width  of patch grid
        n_patches      = self.H_p * self.W_p

        self.patch_embed = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=depth,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """(B, T, 2, G, G) → (B, T, embed_dim, H_p, W_p)"""
        B, T, C, G, _ = bev.shape

        # Patch-embed and run spatial Transformer
        x   = self.patch_embed(bev.view(B * T, C, G, G))    # (B*T, D, H_p, W_p)
        x   = x.flatten(2).transpose(1, 2)                   # (B*T, n_patches, D)
        cls = self.cls_token.expand(B * T, -1, -1)
        x   = self.norm(self.transformer(torch.cat([cls, x], dim=1) + self.pos_embed))

        # Drop CLS; reshape patch tokens to 2D feature map
        # (B*T, n_patches, D) → (B*T, D, H_p, W_p)
        patch_tokens = x[:, 1:]                              # (B*T, n_patches, D)
        feat_map = (
            patch_tokens
            .transpose(1, 2)                                 # (B*T, D, n_patches)
            .reshape(B * T, self.embed_dim, self.H_p, self.W_p)
        )
        return feat_map.view(B, T, self.embed_dim, self.H_p, self.W_p)


# ---------------------------------------------------------------------------
# Cross-attention temporal encoders (ReachNet contract: → context, current)
# ---------------------------------------------------------------------------

class TemporalCrossAttnBEVCNNEncoder(nn.Module):
    """
    BEV CNN spatial backbone + EgoCentricCrossAttentionTemporalModule.

    The CNN extracts per-frame feature maps (spatial H×W retained).  The
    cross-attention module conditions history maps with ego-relative odometry
    and attends from the current frame's spatial queries.  A global avg-pool
    + linear projection then produces the (B, embed_dim) context and current
    vectors expected by ReachNet.

    context : (B, embed_dim) — pooled attended feature map (temporal memory)
    current : (B, embed_dim) — pooled raw last-frame features (current obs.)

    state_delta (B, T, 3) provides relative odometry; history frames use
    state_delta[:, :-1] (the T-1 incremental deltas leading up to the current).
    """

    def __init__(
        self,
        grid_size:       int   = 40,
        embed_dim:       int   = 256,
        feat_channels:   int   = 128,
        odom_dim:        int   = 3,
        temporal_heads:  int   = 8,
        max_history_len: int   = 8,
        drop:            float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.odom_dim  = odom_dim

        self.backbone   = BEVCNNSpatialBackbone(feat_channels=feat_channels)
        self.cross_attn = EgoCentricCrossAttentionTemporalModule(
            channels        = feat_channels,
            odom_dim        = odom_dim,
            num_heads       = temporal_heads,
            max_history_len = max_history_len,
            dropout         = drop,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # feat_channels (default 128) ≠ embed_dim (default 256): project after pool
        self.proj = nn.Linear(feat_channels, embed_dim)

    def forward(
        self,
        bev:         torch.Tensor,                # (B, T, 2, G, G)
        state_delta: torch.Tensor | None = None,  # (B, T, D_odom)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # feature_maps : (B, T, feat_channels, H_feat, W_feat)
        feature_maps = self.backbone(bev)
        B, T, C_f, H_f, W_f = feature_maps.shape

        current_bev = feature_maps[:, -1]    # (B, C_f, H_feat, W_feat)
        history_bev = feature_maps[:, :-1]   # (B, T-1, C_f, H_feat, W_feat)

        if state_delta is not None:
            # Incremental odom of the T-1 history frames toward the current
            relative_odom = state_delta[:, :-1]   # (B, T-1, D_odom)
        else:
            T_hist = T - 1
            relative_odom = torch.zeros(
                B, T_hist, self.odom_dim, device=bev.device, dtype=bev.dtype
            )

        if T > 1:
            attended = self.cross_attn(current_bev, history_bev, relative_odom)
        else:
            attended = current_bev   # (B, C_f, H_feat, W_feat)

        # Global avg-pool → (B, C_f) → project → (B, embed_dim)
        context = self.proj(self.pool(attended).flatten(1))      # (B, embed_dim)
        current = self.proj(self.pool(current_bev).flatten(1))   # (B, embed_dim)

        return context, current


class TemporalCrossAttnBEVViTEncoder(nn.Module):
    """
    BEV ViT spatial backbone (patch tokens) + EgoCentricCrossAttentionTemporalModule.

    The ViT patch-embeds each BEV frame and runs a spatial Transformer, keeping
    the full patch token grid (H_p × W_p) as the feature map for cross-attention.
    Because the backbone already outputs embed_dim channels, no final projection
    is needed after global avg-pooling.

    context : (B, embed_dim) — pooled attended patch feature map
    current : (B, embed_dim) — pooled raw last-frame patch features
    """

    def __init__(
        self,
        grid_size:       int   = 40,
        bev_patch_size:  int   = 8,
        embed_dim:       int   = 256,
        vit_depth:       int   = 4,
        num_heads:       int   = 8,
        temporal_heads:  int   = 8,
        max_history_len: int   = 8,
        drop:            float = 0.1,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.odom_dim  = 3   # matches system-wide state_delta dimensionality

        self.backbone = BEVViTSpatialBackbone(
            grid_size  = grid_size,
            patch_size = bev_patch_size,
            embed_dim  = embed_dim,
            depth      = vit_depth,
            num_heads  = num_heads,
            drop       = drop,
        )
        self.cross_attn = EgoCentricCrossAttentionTemporalModule(
            channels        = embed_dim,
            odom_dim        = self.odom_dim,
            num_heads       = temporal_heads,
            max_history_len = max_history_len,
            dropout         = drop,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Backbone output channels == embed_dim; no projection needed

    def forward(
        self,
        bev:         torch.Tensor,                # (B, T, 2, G, G)
        state_delta: torch.Tensor | None = None,  # (B, T, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # feature_maps : (B, T, embed_dim, H_p, W_p)
        feature_maps = self.backbone(bev)
        B, T, D, H_p, W_p = feature_maps.shape

        current_bev = feature_maps[:, -1]    # (B, embed_dim, H_p, W_p)
        history_bev = feature_maps[:, :-1]   # (B, T-1, embed_dim, H_p, W_p)

        if state_delta is not None:
            relative_odom = state_delta[:, :-1]   # (B, T-1, 3)
        else:
            T_hist = T - 1
            relative_odom = torch.zeros(
                B, T_hist, self.odom_dim, device=bev.device, dtype=bev.dtype
            )

        if T > 1:
            attended = self.cross_attn(current_bev, history_bev, relative_odom)
        else:
            attended = current_bev

        # Global avg-pool; backbone dim already == embed_dim, no projection
        context = self.pool(attended).flatten(1)      # (B, embed_dim)
        current = self.pool(current_bev).flatten(1)   # (B, embed_dim)

        return context, current
