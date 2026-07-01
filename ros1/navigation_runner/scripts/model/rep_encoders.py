"""
RepBaseline pretrained encoder wrappers for ReachMap.

Wraps three pretrained visual backbones from RepBaseline as per-frame spatial
encoders, each projecting to a common embed_dim via a learned linear head.
The backbone can then be combined with ReachMap's standard GRU or Transformer
temporal modules, followed by the full set of task heads.

Backbones
---------
  rep_cnn  — vitfly ConvNet conv layers (60×90 → ~840-d → embed_dim)
  rep_vit  — vitfly MixTransformer encoder_blocks + decoder (60×90 → 512 → embed_dim)
  rep_vae  — mavrl VAE encoder μ (256×256 → 64 → embed_dim)

Depth preprocessing
-------------------
  Vitfly (CNN / ViT):
      depth_norm [0,1] × depth_max_m  →  raw metres
      → clamp [0, 5.0 m]
      → bilinear resize to 60 × 90
  VAE:
      depth_norm [0,1] kept as-is
      → bilinear resize to 256 × 256

freeze_backbone (default True)
      When True, pretrained weights are frozen; only the projection layer
      (and temporal GRU/Transformer modules above) are trained.
      Set False to fine-tune the entire backbone.

Model keys added to _ENCODER_REGISTRY
--------------------------------------
  rep_cnn / rep_cnn_gru / rep_cnn_transformer
  rep_vit / rep_vit_gru / rep_vit_transformer
  rep_vae / rep_vae_gru / rep_vae_transformer
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import _make_temporal_transformer

# ── RepBaseline paths ─────────────────────────────────────────────────────────
_REPBASELINE   = Path(__file__).resolve().parents[2] / "RepBaseline"
_VITFLY_MODELS = _REPBASELINE / "vitfly" / "models"
_MAVRL_MODELS  = _REPBASELINE / "mavrl"  / "models"

# Vitfly depth sensor max range (metres); depth is passed raw to the model.
_VITFLY_MAX_DEPTH_M: float = 5.0


# ---------------------------------------------------------------------------
# Import helpers — load RepBaseline files by path to avoid colliding with
# our own 'model' package already cached in sys.modules.
# ---------------------------------------------------------------------------

_vitfly_mod: "object | None" = None
_mavrl_vae_cls: "object | None" = None


def _load_vitfly_module() -> object:
    global _vitfly_mod
    if _vitfly_mod is None:
        # vitfly's model.py does `from ViTsubmodules import *`, so its directory
        # must be on sys.path.  This is safe: 'ViTsubmodules' doesn't clash with
        # our 'model' package (already cached in sys.modules under that name).
        s = str(_VITFLY_MODELS)
        if s not in sys.path:
            sys.path.insert(0, s)
        spec = importlib.util.spec_from_file_location(
            "_vitfly_model", _VITFLY_MODELS / "model.py"
        )
        _vitfly_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_vitfly_mod)   # type: ignore[union-attr]
    return _vitfly_mod


def _import_convnet():
    return _load_vitfly_module().ConvNet       # type: ignore[attr-defined]


def _import_vitfly_vit():
    return _load_vitfly_module().ViT           # type: ignore[attr-defined]


def _import_mavrl_vae():
    global _mavrl_vae_cls
    if _mavrl_vae_cls is None:
        spec = importlib.util.spec_from_file_location(
            "_mavrl_vae", _MAVRL_MODELS / "vae.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)           # type: ignore[union-attr]
        _mavrl_vae_cls = mod.VAE               # type: ignore[attr-defined]
    return _mavrl_vae_cls


def _load_weights(ckpt_path: str) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict):
        return raw.get("state_dict", raw)
    return raw


# ---------------------------------------------------------------------------
# Spatial backbone wrappers  (B, T, H, W) → (B, T, embed_dim)
# ---------------------------------------------------------------------------

class RepBaselineCNNSpatial(nn.Module):
    """
    vitfly ConvNet — conv layers only (no fc head, which mixes in metadata).

    Architecture extracted:
        conv1(1→4, k=3, s=3) → BN → ReLU → MaxPool(2,s=1)
        conv2(4→10, k=3, s=2) → ReLU → AvgPool(3,s=1)
        flatten  →  Linear(visual_dim, embed_dim)

    visual_dim is inferred via a dummy forward (≈840 for 60×90 input).
    """

    def __init__(
        self,
        embed_dim:       int   = 256,
        ckpt_path:       str | None = None,
        freeze_backbone: bool  = True,
        depth_max_m:     float = 10.0,
        **_,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.depth_max_m = depth_max_m

        net = _import_convnet()()
        if ckpt_path:
            net.load_state_dict(_load_weights(ckpt_path))

        # Keep only the visual conv layers — discard fc0 (which mixes des_vel/quat)
        self.conv1   = net.conv1
        self.bn1     = net.bn1
        self.maxpool = net.maxpool
        self.conv2   = net.conv2
        self.avgpool = net.avgpool

        if freeze_backbone:
            for m in (self.conv1, self.bn1, self.conv2):
                for p in m.parameters():
                    p.requires_grad = False

        with torch.no_grad():
            visual_dim = self._conv_fwd(torch.zeros(1, 1, 60, 90)).shape[-1]
        self.proj = nn.Linear(visual_dim, embed_dim)

    def _conv_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 60, 90) → (B, visual_dim)"""
        x = -self.maxpool(-self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(F.relu(self.conv2(x)))
        return torch.flatten(x, 1)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) normalised depth → (B, T, embed_dim)"""
        B, T, H, W = depth.shape
        x = depth.reshape(B * T, 1, H, W) * self.depth_max_m          # → metres
        x = F.interpolate(x, size=(60, 90), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, _VITFLY_MAX_DEPTH_M)
        return self.proj(self._conv_fwd(x)).view(B, T, self.embed_dim)


class RepBaselineViTSpatial(nn.Module):
    """
    vitfly ViT (MixTransformer) — encoder_blocks + pixel-shuffle decoder.

    Architecture extracted:
        2× MixTransformerEncoderLayer
        → PixelShuffle(2) + Upsample(16×24) + Conv2d(48→12, k=3)
        → flatten → Linear(4608, 512)
        → Linear(512, embed_dim)

    Input to the vitfly model is resized to 60×90 raw-depth (metres).
    """

    def __init__(
        self,
        embed_dim:       int   = 256,
        ckpt_path:       str | None = None,
        freeze_backbone: bool  = True,
        depth_max_m:     float = 10.0,
        **_,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.depth_max_m = depth_max_m

        net = _import_vitfly_vit()()
        if ckpt_path:
            net.load_state_dict(_load_weights(ckpt_path))

        self.encoder_blocks = net.encoder_blocks
        self.up_sample      = net.up_sample      # Upsample(16×24, bilinear)
        self.pxShuffle      = net.pxShuffle      # PixelShuffle(2)
        self.down_sample    = net.down_sample     # Conv2d(48→12, k=3)
        self.decoder        = net.decoder        # Linear(4608, 512)

        if freeze_backbone:
            for m in (self.encoder_blocks, self.down_sample, self.decoder):
                for p in m.parameters():
                    p.requires_grad = False

        self.proj = nn.Linear(512, embed_dim)

    def _vit_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 60, 90) → (B, 512)"""
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1)
        return self.decoder(self.down_sample(out).flatten(1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) normalised depth → (B, T, embed_dim)"""
        B, T, H, W = depth.shape
        x = depth.reshape(B * T, 1, H, W) * self.depth_max_m
        x = F.interpolate(x, size=(60, 90), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, _VITFLY_MAX_DEPTH_M)
        return self.proj(self._vit_fwd(x)).view(B, T, self.embed_dim)


class RepBaselineVAESpatial(nn.Module):
    """
    mavrl VAE encoder — uses distribution mean μ as deterministic latent.

    Architecture extracted:
        6× Conv2d(stride=2) → flatten(2×2×256=1024)
        → fc_mu: Linear(1024, latent_size=64)
        → Linear(64, embed_dim)

    Input: depth_norm [0,1] resized to 256×256 (VAE was trained on normalised depth).
    """

    def __init__(
        self,
        embed_dim:       int   = 256,
        rep_vae_latent:  int   = 64,
        ckpt_path:       str | None = None,
        freeze_backbone: bool  = True,
        **_,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        vae = _import_mavrl_vae()(img_channels=1, latent_size=rep_vae_latent)
        if ckpt_path:
            vae.load_state_dict(_load_weights(ckpt_path))

        self.vae_enc = vae.encoder   # mavrl Encoder: returns (mu, logsigma)
        if freeze_backbone:
            for p in self.vae_enc.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(rep_vae_latent, embed_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) normalised depth → (B, T, embed_dim)"""
        B, T, H, W = depth.shape
        x     = depth.reshape(B * T, 1, H, W)
        x     = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        mu, _ = self.vae_enc(x)                         # (B*T, latent_size)
        return self.proj(mu).view(B, T, self.embed_dim)


# ---------------------------------------------------------------------------
# Temporal wrappers — generated for each backbone via factory functions
# ---------------------------------------------------------------------------

def _rep_only(spatial_cls: type, clsname: str) -> type:
    class _Only(nn.Module):
        """No temporal — context = current = last-frame embedding."""
        def __init__(self, **kw):
            super().__init__()
            self.embed_dim = kw.get("embed_dim", 256)
            self.spatial   = spatial_cls(**kw)
        def forward(self, depth: torch.Tensor, state_delta=None):
            last = self.spatial(depth)[:, -1]
            return last, last
    _Only.__name__ = _Only.__qualname__ = clsname
    return _Only


def _rep_gru(spatial_cls: type, clsname: str) -> type:
    class _GRU(nn.Module):
        """Backbone spatial + GRU temporal."""
        def __init__(self, embed_dim: int = 256, gru_layers: int = 2,
                     drop: float = 0.1, **kw):
            super().__init__()
            self.embed_dim   = embed_dim
            self.spatial     = spatial_cls(embed_dim=embed_dim, **kw)
            self.motion_proj = nn.Linear(3, embed_dim)
            self.gru         = nn.GRU(
                input_size=embed_dim, hidden_size=embed_dim,
                num_layers=gru_layers, batch_first=True,
                dropout=drop if gru_layers > 1 else 0.0,
            )
        def forward(self, depth: torch.Tensor, state_delta=None):
            visual = self.spatial(depth)        # (B, T, D)
            if state_delta is not None:
                visual = visual + self.motion_proj(state_delta)
            _, h_n = self.gru(visual)
            return h_n[-1], visual[:, -1]       # context, current
    _GRU.__name__ = _GRU.__qualname__ = clsname
    return _GRU


def _rep_transformer(spatial_cls: type, clsname: str) -> type:
    class _Transformer(nn.Module):
        """Backbone spatial + Transformer temporal."""
        def __init__(self, embed_dim: int = 256, seq_len: int = 30,
                     temporal_depth: int = 2, temporal_heads: int = 8,
                     drop: float = 0.1, **kw):
            super().__init__()
            self.embed_dim = embed_dim
            self.spatial   = spatial_cls(embed_dim=embed_dim, **kw)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.temporal  = _make_temporal_transformer(
                embed_dim, temporal_depth, temporal_heads, drop)
            self.norm      = nn.LayerNorm(embed_dim)
        def forward(self, depth: torch.Tensor, state_delta=None):
            visual = self.spatial(depth)        # (B, T, D)
            B, T, _ = visual.shape
            x = torch.cat([self.cls_token.expand(B, -1, -1), visual], dim=1)
            x = self.norm(self.temporal(x + self.pos_embed[:, :T + 1]))
            return x[:, 0], visual[:, -1]      # context (CLS), current (last frame)
    _Transformer.__name__ = _Transformer.__qualname__ = clsname
    return _Transformer


# ── Instantiate all 9 encoder classes ────────────────────────────────────────

RepCNNOnlyEncoder        = _rep_only(RepBaselineCNNSpatial,  "RepCNNOnlyEncoder")
RepCNNGRUEncoder         = _rep_gru(RepBaselineCNNSpatial,   "RepCNNGRUEncoder")
RepCNNTransformerEncoder = _rep_transformer(RepBaselineCNNSpatial, "RepCNNTransformerEncoder")

RepViTOnlyEncoder        = _rep_only(RepBaselineViTSpatial,  "RepViTOnlyEncoder")
RepViTGRUEncoder         = _rep_gru(RepBaselineViTSpatial,   "RepViTGRUEncoder")
RepViTTransformerEncoder = _rep_transformer(RepBaselineViTSpatial, "RepViTTransformerEncoder")

RepVAEOnlyEncoder        = _rep_only(RepBaselineVAESpatial,  "RepVAEOnlyEncoder")
RepVAEGRUEncoder         = _rep_gru(RepBaselineVAESpatial,   "RepVAEGRUEncoder")
RepVAETransformerEncoder = _rep_transformer(RepBaselineVAESpatial, "RepVAETransformerEncoder")
