"""
ReachNet: triple-head prediction model on top of a temporal encoder.

Architecture
------------

  depth (B, T, H, W)
      └─► TemporalCNNEncoder / TemporalViTEncoder
               CNN/ViT (per-frame) + GRU (across time)
               └─► context : (B, D)  GRU hidden state — temporal visual memory
               └─► current : (B, D)  last-frame feature — current observation
                        │               │                        │
                   reach head     raycast head             policy head
                MLP(D→D→G²)     MLP(D→D→R)      current + state_proj(3→D) + goal_proj(2→D)
                        │               │                        │
           reachability map (B,G,G)  ray dists (B,R)    action logits (B,A)

Usage
-----
  from model import build_cnn_model, build_vit_model

  model = build_cnn_model(grid_size=32, embed_dim=256, num_actions=3, num_rays=180)
  out   = model(depth, state_delta, goal_delta)
  # out["reach_mask"] : (B, G, G)  binary logits
  # out["reach_step"] : (B, G, G)  step-count regression
  # out["raycast"]    : (B, R)
  # out["policy"]     : (B, A)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import (
    CNNOnlyEncoder, ViTOnlyEncoder,
    TemporalCNNEncoder, TemporalViTEncoder,
    TemporalTransformerCNNEncoder, TemporalTransformerViTEncoder,
    BEVCNNOnlyEncoder,
    TemporalBEVCNNEncoder,
    TemporalTransformerBEVCNNEncoder,
    BEVViTOnlyEncoder,
    TemporalBEVViTEncoder,
    TemporalTransformerBEVViTEncoder,
    TemporalCrossAttnBEVCNNEncoder,
    TemporalCrossAttnBEVViTEncoder,
)
from .rep_encoders import (
    RepCNNOnlyEncoder, RepCNNGRUEncoder, RepCNNTransformerEncoder,
    RepViTOnlyEncoder, RepViTGRUEncoder, RepViTTransformerEncoder,
    RepVAEOnlyEncoder, RepVAEGRUEncoder, RepVAETransformerEncoder,
)


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, drop: float) -> nn.Sequential:
    """MLP with n_layers hidden layers (LayerNorm + GELU + Dropout), then linear output."""
    if n_layers == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(cur, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(drop)]
        cur = hidden_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class ReachNet(nn.Module):
    """
    Parameters
    ----------
    encoder     : TemporalCNNEncoder or TemporalViTEncoder
    grid_size   : G, side of the square reachability output grid
    embed_dim   : D, must match encoder.embed_dim
    num_actions : number of discrete actions for the policy head
    drop        : dropout on the reach head MLP
    """

    def __init__(
        self,
        encoder:         nn.Module,
        grid_size:       int   = 64,
        embed_dim:       int   = 256,
        num_actions:          int   = 3,
        num_rays:             int   = 180,
        drop:                 float = 0.1,
        use_reach_mask:       bool  = True,
        use_reach_step:       bool  = True,
        use_raycast:          bool  = True,
        use_policy:           bool  = True,
        use_known_mask:       bool  = False,
        mask_head_layers:     int   = 2,
        step_head_layers:     int   = 1,
        raycast_head_layers:  int   = 1,
        policy_head_layers:   int   = 0,
        known_head_layers:    int   = 1,
    ):
        super().__init__()
        assert embed_dim == encoder.embed_dim, (
            f"embed_dim mismatch: ReachNet={embed_dim}, encoder={encoder.embed_dim}"
        )

        self.encoder        = encoder
        self.grid_size      = grid_size
        self.embed_dim      = embed_dim
        self.num_rays       = num_rays
        self.use_reach_mask = use_reach_mask
        self.use_reach_step = use_reach_step
        self.use_raycast    = use_raycast
        self.use_policy     = use_policy
        self.use_known_mask = use_known_mask

        G = grid_size

        if use_reach_mask:
            self.mask_head = _make_mlp(embed_dim, 512, G * G, mask_head_layers, drop)

        if use_reach_step:
            self.step_head = _make_mlp(embed_dim, embed_dim, G * G, step_head_layers, drop)

        if use_raycast:
            self.raycast_head = _make_mlp(embed_dim, embed_dim, num_rays, raycast_head_layers, drop)

        if use_policy:
            self.state_proj  = nn.Linear(3, embed_dim)
            self.goal_proj   = nn.Linear(2, embed_dim)
            self.policy_head = _make_mlp(embed_dim, embed_dim, num_actions, policy_head_layers, drop)

        if use_known_mask:
            self.known_head = _make_mlp(embed_dim, embed_dim, G * G, known_head_layers, drop)

        # Uncertainty-weighting log-variances (Kendall et al. 2018).
        # One scalar per active head; optimised jointly with the rest of the network.
        if use_reach_mask:
            self.log_var_mask   = nn.Parameter(torch.zeros(1))
        if use_reach_step:
            self.log_var_step   = nn.Parameter(torch.zeros(1))
        if use_raycast:
            self.log_var_ray    = nn.Parameter(torch.zeros(1))
        if use_policy:
            self.log_var_policy = nn.Parameter(torch.zeros(1))
        if use_known_mask:
            self.log_var_known  = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def forward(
        self,
        depth:       torch.Tensor,                 # (B, T, H, W)
        state_delta: torch.Tensor,                 # (B, T, 3)
        goal_delta:  torch.Tensor | None = None,   # (B, 2)
    ) -> dict[str, torch.Tensor]:
        B = depth.shape[0]

        context, current = self.encoder(depth, state_delta)   # (B, D), (B, D)
        out = {}

        if self.use_reach_mask:
            out["reach_mask"] = self.mask_head(context).view(B, self.grid_size, self.grid_size)

        if self.use_reach_step:
            out["reach_step"] = self.step_head(context).view(B, self.grid_size, self.grid_size)

        if self.use_raycast:
            out["raycast"] = self.raycast_head(current)

        if self.use_policy:
            policy_feat = current + self.state_proj(state_delta[:, -1])
            if goal_delta is not None:
                policy_feat = policy_feat + self.goal_proj(goal_delta)
            out["policy"] = self.policy_head(policy_feat)

        if self.use_known_mask:
            out["known_mask"] = self.known_head(context).view(B, self.grid_size, self.grid_size)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

#: Mapping from encoder_type string to encoder class.
_ENCODER_REGISTRY = {
    "cnn":                 CNNOnlyEncoder,
    "vit":                 ViTOnlyEncoder,
    "cnn_gru":             TemporalCNNEncoder,
    "vit_gru":             TemporalViTEncoder,
    "cnn_transformer":     TemporalTransformerCNNEncoder,
    "vit_transformer":     TemporalTransformerViTEncoder,
    # BEV input variants — take (B, T, 2, G, G) instead of (B, T, H, W)
    "bev_cnn":             BEVCNNOnlyEncoder,
    "bev_cnn_gru":         TemporalBEVCNNEncoder,
    "bev_cnn_transformer": TemporalTransformerBEVCNNEncoder,
    "bev_vit":             BEVViTOnlyEncoder,
    "bev_vit_gru":         TemporalBEVViTEncoder,
    "bev_vit_transformer": TemporalTransformerBEVViTEncoder,
    # BEV encoders — cross-attention temporal (odometry-conditioned spatial attention)
    "bev_cnn_crossattn":   TemporalCrossAttnBEVCNNEncoder,
    "bev_vit_crossattn":   TemporalCrossAttnBEVViTEncoder,
    # RepBaseline pretrained backbone + ReachMap temporal + task heads
    "rep_cnn":             RepCNNOnlyEncoder,
    "rep_cnn_gru":         RepCNNGRUEncoder,
    "rep_cnn_transformer": RepCNNTransformerEncoder,
    "rep_vit":             RepViTOnlyEncoder,
    "rep_vit_gru":         RepViTGRUEncoder,
    "rep_vit_transformer": RepViTTransformerEncoder,
    "rep_vae":             RepVAEOnlyEncoder,
    "rep_vae_gru":         RepVAEGRUEncoder,
    "rep_vae_transformer": RepVAETransformerEncoder,
}


def _rep_ckpt_path(encoder_type: str, cfg: dict) -> "str | None":
    if not encoder_type.startswith("rep_"):
        return None
    if "vae" in encoder_type:
        return cfg.get("rep_vae_ckpt")
    if "vit" in encoder_type:
        return cfg.get("rep_vit_ckpt")
    if "cnn" in encoder_type:
        return cfg.get("rep_cnn_ckpt")
    return None


def build_model(
    encoder_type: str,
    cfg:          dict,
    img_h:        int = 256,
    img_w:        int = 256,
) -> "ReachNet":
    """
    Build a ReachNet for the given encoder type.

    encoder_type
    ------------
    "cnn"             — CNN only (no temporal); context = current = last-frame CNN
    "vit"             — ViT only (no temporal); context = current = last-frame ViT
    "cnn_gru"         — CNN per-frame + GRU across time
    "vit_gru"         — ViT per-frame + GRU across time
    "cnn_transformer" — CNN per-frame + Transformer encoder across time
    "vit_transformer" — ViT per-frame + Transformer encoder across time

    Relevant cfg keys
    -----------------
    embed_dim, grid_size, num_actions, num_rays, drop
    num_layers      (GRU layers,          cnn_gru / vit_gru)
    patch_size      (ViT patch size,       vit / vit_gru / vit_transformer)
    vit_depth       (ViT internal layers,  vit / vit_gru / vit_transformer)
    num_heads       (ViT attention heads,  vit / vit_gru / vit_transformer)
    seq_len         (positional emb size,  cnn_transformer / vit_transformer)
    temporal_depth  (temporal Transformer layers, cnn_transformer / vit_transformer)
    temporal_heads  (temporal Transformer heads,  cnn_transformer / vit_transformer)
    """
    if encoder_type not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder_type {encoder_type!r}. "
            f"Choose from: {list(_ENCODER_REGISTRY)}"
        )

    embed_dim   = cfg.get("embed_dim",       256)
    grid_size   = cfg.get("grid_size",        64)
    num_actions = cfg.get("num_actions",       3)
    num_rays    = cfg.get("num_rays",        180)
    drop        = cfg.get("drop",            0.1)

    enc_kwargs = dict(
        embed_dim      = embed_dim,
        drop           = drop,
        # BEV spatial
        grid_size      = grid_size,
        bev_patch_size = cfg.get("bev_patch_size",   8),
        # ViT spatial (depth)
        img_h          = img_h,
        img_w          = img_w,
        patch_size     = cfg.get("patch_size",      32),
        vit_depth      = cfg.get("vit_depth",        4),
        num_heads      = cfg.get("num_heads",         8),
        # GRU temporal
        num_layers     = cfg.get("num_layers",        2),
        gru_layers     = cfg.get("num_layers",        2),
        # Transformer temporal
        seq_len        = cfg.get("seq_len",          10),
        temporal_depth = cfg.get("temporal_depth",    2),
        temporal_heads = cfg.get("temporal_heads",    8),
        # Cross-attention temporal settings
        feat_channels    = cfg.get("feat_channels",    128),
        odom_dim         = cfg.get("odom_dim",           3),
        max_history_len  = cfg.get("max_history_len",    8),
        # RepBaseline pretrained backbone settings
        ckpt_path       = _rep_ckpt_path(encoder_type, cfg),
        freeze_backbone = cfg.get("rep_freeze_backbone", True),
        depth_max_m     = cfg.get("rep_depth_max_m", 10.0),
        rep_vae_latent  = cfg.get("rep_vae_latent",  64),
    )

    enc_cls = _ENCODER_REGISTRY[encoder_type]
    enc     = enc_cls(**enc_kwargs)

    return ReachNet(
        enc,
        grid_size=grid_size,
        embed_dim=embed_dim,
        num_actions=num_actions,
        num_rays=num_rays,
        drop=drop,
        use_reach_mask       = cfg.get("use_reach_mask",       True),
        use_reach_step       = cfg.get("use_reach_step",       True),
        use_raycast          = cfg.get("use_raycast",          True),
        use_policy           = cfg.get("use_policy",           True),
        use_known_mask       = cfg.get("use_known_mask",       False),
        mask_head_layers     = cfg.get("mask_head_layers",     2),
        step_head_layers     = cfg.get("step_head_layers",     1),
        raycast_head_layers  = cfg.get("raycast_head_layers",  1),
        policy_head_layers   = cfg.get("policy_head_layers",   0),
        known_head_layers    = cfg.get("known_head_layers",    1),
    )
