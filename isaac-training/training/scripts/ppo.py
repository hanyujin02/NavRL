import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor, GRUModule
from torchrl.envs.transforms import CatTensors
from utils import (
    ValueNorm, make_mlp, build_conv_stack, IndependentNormal, Actor, GAE, make_batch,
    make_batch_recurrent, IndependentBeta, BetaActor, vec_to_world, BuildTokens, TransformerBackbone,
)

# ---------------------------------------------------------------------------
# ReachMap encoder integration
# ---------------------------------------------------------------------------

# Path to ReachMap root: scripts/ → training/ → isaac-training/ → NavRL/ → ReachMap/
_REACHMAP_ROOT = Path(__file__).resolve().parents[4]

# Encoder types available from ReachMap (depth-input and BEV variants)
_REACHMAP_DEPTH_ENCODERS = [
    "cnn", "vit",
    "cnn_gru", "vit_gru",
    "cnn_transformer", "vit_transformer",
    "rep_cnn", "rep_cnn_gru", "rep_cnn_transformer",
    "rep_vit", "rep_vit_gru", "rep_vit_transformer",
    "rep_vae", "rep_vae_gru", "rep_vae_transformer",
    "bev_cnn", "bev_vit",
    "bev_cnn_gru", "bev_vit_gru",
    "bev_cnn_transformer", "bev_vit_transformer",
]


def _reachmap_registry() -> dict:
    """Import ReachMap encoder classes, adding the repo to sys.path if needed."""
    root = str(_REACHMAP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.encoder import (
        CNNOnlyEncoder, ViTOnlyEncoder,
        TemporalCNNEncoder, TemporalViTEncoder,
        TemporalTransformerCNNEncoder, TemporalTransformerViTEncoder,
    )
    from model.rep_encoders import (
        RepCNNOnlyEncoder,        RepCNNGRUEncoder,        RepCNNTransformerEncoder,
        RepViTOnlyEncoder,        RepViTGRUEncoder,        RepViTTransformerEncoder,
        RepVAEOnlyEncoder,        RepVAEGRUEncoder,        RepVAETransformerEncoder,
    )
    from model.encoder import (
        BEVCNNOnlyEncoder,            BEVViTOnlyEncoder,
        TemporalBEVCNNEncoder,        TemporalBEVViTEncoder,
        TemporalTransformerBEVCNNEncoder, TemporalTransformerBEVViTEncoder,
    )
    return {
        "cnn":                     CNNOnlyEncoder,
        "vit":                     ViTOnlyEncoder,
        "cnn_gru":                 TemporalCNNEncoder,
        "vit_gru":                 TemporalViTEncoder,
        "cnn_transformer":         TemporalTransformerCNNEncoder,
        "vit_transformer":         TemporalTransformerViTEncoder,
        "rep_cnn":                 RepCNNOnlyEncoder,
        "rep_cnn_gru":             RepCNNGRUEncoder,
        "rep_cnn_transformer":     RepCNNTransformerEncoder,
        "rep_vit":                 RepViTOnlyEncoder,
        "rep_vit_gru":             RepViTGRUEncoder,
        "rep_vit_transformer":     RepViTTransformerEncoder,
        "rep_vae":                 RepVAEOnlyEncoder,
        "rep_vae_gru":             RepVAEGRUEncoder,
        "rep_vae_transformer":     RepVAETransformerEncoder,
        "bev_cnn":                 BEVCNNOnlyEncoder,
        "bev_vit":                 BEVViTOnlyEncoder,
        "bev_cnn_gru":             TemporalBEVCNNEncoder,
        "bev_vit_gru":             TemporalBEVViTEncoder,
        "bev_cnn_transformer":     TemporalTransformerBEVCNNEncoder,
        "bev_vit_transformer":     TemporalTransformerBEVViTEncoder,
    }


class ReachMapEncoderWrapper(nn.Module):
    """
    Adapts a ReachMap encoder for NavRL's PPO feature extractor.

    ReachMap encoders expect (B, T, H, W) and return (context: B, D), (current: B, D).
    NavRL feeds (N, 1, H, W) depth images — the channel-1 dim acts as T=1, so no
    reshape is needed; the encoder interprets it as a single-frame sequence.

    The context vector (temporal summary, or last-frame feature for non-temporal
    encoders) is projected through a LayerNorm and returned as (N, embed_dim).
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, is_bev: bool = False,
                 freeze_encoder: bool = False, projection_mlp: bool = True,
                 freeze_bn: bool = True):
        super().__init__()
        self.encoder        = encoder
        self.embed_dim      = embed_dim
        if projection_mlp:
            self.projection = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
            )
        else:
            self.projection = nn.LayerNorm(embed_dim)
        self.is_bev         = is_bev
        self._freeze_encoder = freeze_encoder
        # freeze_bn is only consulted when freeze_encoder=True (with
        # freeze_encoder=False the encoder is fully trainable, BN included,
        # regardless of this flag). freeze_bn=False lets a weight-frozen
        # encoder's BatchNorm running_mean/var keep adapting to the live
        # rollout data distribution during training, while conv/linear
        # weights stay fixed (requires_grad=False below) — the opposite of
        # the "_bnfrozen" naming convention used for earlier runs, which is
        # freeze_bn=True (the default, and the only behavior this wrapper
        # supported before this option existed).
        self._freeze_bn = freeze_bn
        if freeze_encoder and freeze_bn:
            # Must happen at construction, not only in train() below.
            # requires_grad=False freezes parameters but NOT BatchNorm's
            # running_mean/var — those are buffers updated in forward() whenever
            # the module is in train mode. Modules default to training=True, and
            # SyncDataCollector deepcopies the whole policy at construction time
            # (torchrl collectors.py: `self.policy = deepcopy(policy)`): params
            # and buffers are shared tensors, but the `training` flag is per
            # instance. The collector's copy therefore keeps whatever mode it was
            # deepcopied in, and nothing ever calls .train()/.eval() on it again —
            # so without this line the rollout path silently updated the "frozen"
            # encoder's BN stats once per env step (measured: +64/iteration,
            # running_var drifting 45-70% off the pretrained checkpoint by 50k).
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_encoder and self._freeze_bn:
            self.encoder.eval()   # keep frozen encoder in eval regardless of outer mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_bev:
            # x: (N, 2, G, G) BEV — add T=1 dim for ReachMap BEV encoders → (N, 1, 2, G, G)
            x = x.unsqueeze(1)
        else:
            # x: (N, 1, H, W) depth — treated as (B=N, T=1, H, W)
            pass

        # For ViT encoders, mean-pool patch tokens instead of the CLS token.
        # Patch tokens retain local spatial structure even in a frozen encoder,
        # giving the projection layer meaningful obstacle-proximity features.
        spatial = getattr(self.encoder, "spatial", None)
        if spatial is not None and hasattr(spatial, "cls_token"):
            B, T, H, W = x.shape
            patches = spatial.patch_embed(x.view(B * T, 1, H, W))       # (B*T, n_patches, D)
            cls     = spatial.cls_token.expand(B * T, -1, -1)
            tokens  = spatial.norm(spatial.transformer(
                torch.cat([cls, patches], dim=1) + spatial.pos_embed
            ))                                                            # (B*T, 1+n_patches, D)
            context = tokens[:, 1:].mean(dim=1).view(B, T, self.embed_dim)[:, -1]  # (B, D)
        else:
            context, _ = self.encoder(x)

        return self.projection(context)      # (N, embed_dim // 2) if projection_mlp else (N, embed_dim)


class FlattenLeadingDims(nn.Module):
    """Let a per-timestep encoder accept more than one leading batch dim.

    The ReachMap encoders (and the dynamic-obstacle Rearrange) take exactly one
    batch dim: (B, T, H, W) for depth/lidar, (n, c, w, h) for dyn-obs. The
    cnn/transformer paths never hand them more, because they only call the
    feature extractor either on a single-batch-dim rollout tensordict or
    through torch.vmap (which strips the env dim) in train()'s next-value
    bootstrap. The gru path has no such luxury: GRUModule's recurrent mode
    needs a real (num_env, num_frames) time axis, so the extractor is called
    directly on a 2-batch-dim tensordict -- both for the bootstrap and for
    make_batch_recurrent's minibatches -- which would reach the encoder as a
    5-dim tensor.

    These upstream modules are per-timestep and order-independent, so folding
    the extra dims into the batch dim and restoring them afterwards is exact,
    not an approximation. Only the GRU itself needs the (batch, time) split,
    and it sits downstream of this.
    """

    def __init__(self, module: nn.Module, n_feature_dims: int = 3):
        super().__init__()
        self.module = module
        self.n_feature_dims = n_feature_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_lead = x.dim() - self.n_feature_dims
        if n_lead <= 1:
            return self.module(x)          # ordinary single-batch-dim call
        lead = x.shape[:n_lead]
        out = self.module(x.reshape(-1, *x.shape[n_lead:]))
        return out.reshape(*lead, *out.shape[1:])


def _load_encoder_ckpt(encoder: nn.Module, ckpt_path: str, submodule: str = "") -> None:
    """Load encoder weights from a ReachNet checkpoint (best.pt / latest.pt).

    submodule: if non-empty, only load keys under encoder.<submodule>.*
               (e.g. "spatial" to load the backbone only, matching ReachMap's
               freeze_spatial pattern in scripts/train.py).
    """
    raw     = torch.load(ckpt_path, map_location="cpu")
    full_sd = raw.get("state_dict", raw)

    # Strip "encoder." prefix, then optionally restrict to one submodule.
    # Mirrors ReachMap's `k.startswith("encoder.spatial.")` backbone-only filter.
    submodule_prefix = f"{submodule}." if submodule else ""
    enc_sd = {
        k[len("encoder."):]: v
        for k, v in full_sd.items()
        if k.startswith(f"encoder.{submodule_prefix}")
    }

    # Drop keys whose shapes don't match the current model (e.g. pos_embed when
    # the checkpoint was trained at a different resolution).  Those params are
    # left at their randomly-initialised values.
    model_sd = encoder.state_dict()
    skipped, filtered = [], {}
    for k, v in enc_sd.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}")
        else:
            filtered[k] = v
    if skipped:
        print(f"[NavRL] encoder ckpt skipped (shape mismatch): {skipped}")
    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[NavRL] encoder ckpt missing keys: {missing[:5]}")
    if unexpected:
        print(f"[NavRL] encoder ckpt unexpected keys: {unexpected[:5]}")
    sub_desc = f" (submodule: {submodule})" if submodule else ""
    print(f"[NavRL] Loaded encoder weights{sub_desc} from {ckpt_path}")


def build_depth_encoder(cfg) -> nn.Module:
    """
    Build the depth-image feature extractor.

    encoder_type = 'scratch'  (default)
        Simple 3-layer strided CNN trained from scratch.

    encoder_type = '<reachmap_type>'  (e.g. 'cnn_gru', 'vit', 'rep_cnn_gru', ...)
        ReachMap encoder loaded from cfg.encoder_ckpt if provided (a ReachNet
        best.pt / latest.pt).  encoder weights are frozen when cfg.freeze_encoder=True.

    All variants return a module: (N, 1, H, W) -> (N, embed_dim).
    """
    encoder_type = getattr(cfg, "encoder_type", "scratch")
    embed_dim    = getattr(cfg, "embed_dim",    256)

    # ── Scratch CNN (default) ─────────────────────────────────────────────────
    if encoder_type == "scratch":
        return nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=1), nn.ELU(),
            Rearrange("n c h w -> n (c h w)"),
            nn.LazyLinear(embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
        )

    # ── NavRL-plus-plus's exact static-obstacle (lidar) CNN ────────────────────
    # Ported verbatim from NavRL-plus-plus's ppo.py "transformer" branch's
    # static_obstacle_network, for a like-for-like architecture comparison
    # against this repo's generic 'scratch' CNN on the lidar+transformer path.
    # Outputs 128-d (not embed_dim) -- matches NavRL++ exactly; BuildTokens'
    # own static_adapter (a LazyLinear) maps whatever width this returns to
    # d_model, so it doesn't need to be embed_dim-sized.
    if encoder_type == "lidar_navrlpp":
        return nn.Sequential(
            build_conv_stack(
                channels=[4, 16, 16],
                kernel_size=[[5, 3], [5, 3], [5, 3]],
                stride=[[1, 1], [2, 1], [2, 1]],
                padding=[[2, 1], [2, 1], [2, 1]],
            ),
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128]),
        )

    # ── ReachMap encoder ─────────────────────────────────────────────────────
    registry = _reachmap_registry()
    if encoder_type not in registry:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose 'scratch' or one of: {_REACHMAP_DEPTH_ENCODERS}"
        )

    # For rep_* encoders, resolve the RepBaseline backbone checkpoint path exactly
    # as ReachMap's build_model does: pick by backbone family (cnn/vit/vae).
    rep_backbone_ckpt: str | None = None
    if encoder_type.startswith("rep_"):
        if "vae" in encoder_type:
            rep_backbone_ckpt = getattr(cfg, "rep_vae_ckpt", None)
        elif "vit" in encoder_type:
            rep_backbone_ckpt = getattr(cfg, "rep_vit_ckpt", None)
        elif "cnn" in encoder_type:
            rep_backbone_ckpt = getattr(cfg, "rep_cnn_ckpt", None)

    is_bev = encoder_type.startswith("bev_")

    if is_bev:
        enc_kwargs = dict(
            embed_dim        = embed_dim,
            norm_type        = getattr(cfg, "norm_type",        "batchnorm"),
            groupnorm_groups = getattr(cfg, "groupnorm_groups", 8),
            grid_size      = getattr(cfg, "bev_grid_size",    50),
            bev_patch_size = getattr(cfg, "bev_patch_size",   10),
            vit_depth      = getattr(cfg, "vit_depth",         4),
            num_heads      = getattr(cfg, "num_heads",          8),
            num_layers     = getattr(cfg, "gru_layers",         2),
            gru_layers     = getattr(cfg, "gru_layers",         2),
            seq_len        = getattr(cfg, "seq_len",           10),
            temporal_depth = getattr(cfg, "temporal_depth",     2),
            temporal_heads = getattr(cfg, "temporal_heads",     8),
            drop           = getattr(cfg, "drop",             0.1),
        )
    else:
        enc_kwargs = dict(
            embed_dim        = embed_dim,
            norm_type        = getattr(cfg, "norm_type",        "batchnorm"),
            groupnorm_groups = getattr(cfg, "groupnorm_groups", 8),
            img_h          = getattr(cfg, "img_h",          64),
            img_w          = getattr(cfg, "img_w",          64),
            patch_size     = getattr(cfg, "patch_size",      8),
            vit_depth      = getattr(cfg, "vit_depth",       4),
            num_heads      = getattr(cfg, "num_heads",        8),
            num_layers     = getattr(cfg, "gru_layers",       2),
            gru_layers     = getattr(cfg, "gru_layers",       2),
            seq_len        = getattr(cfg, "seq_len",         10),
            temporal_depth = getattr(cfg, "temporal_depth",   2),
            temporal_heads = getattr(cfg, "temporal_heads",   8),
            drop           = getattr(cfg, "drop",            0.1),
            # RepBaseline-specific (ignored by non-rep encoders via **_)
            ckpt_path       = rep_backbone_ckpt,
            freeze_backbone = getattr(cfg, "rep_freeze_backbone", True),
            depth_max_m     = getattr(cfg, "rep_depth_max_m",     5.0),
            rep_vae_latent  = getattr(cfg, "rep_vae_latent",       64),
        )

    enc = registry[encoder_type](**enc_kwargs)

    # Optionally load a full ReachNet checkpoint (encoder.* keys) on top.
    # Works for all encoder types: non-rep encoders trained in ReachMap,
    # or rep encoders fine-tuned end-to-end via ReachMap.
    reachnet_ckpt = getattr(cfg, "encoder_ckpt", None)
    if reachnet_ckpt:
        submodule = getattr(cfg, "encoder_ckpt_submodule", "")
        _load_encoder_ckpt(enc, reachnet_ckpt, submodule=submodule)

    freeze = getattr(cfg, "freeze_encoder", False)
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
        print(f"[NavRL] Encoder weights frozen.")

    # Only meaningful when freeze_encoder=True — see ReachMapEncoderWrapper's
    # freeze_bn docstring/comment. Defaults to True (the only behavior this
    # wrapper supported before this option existed) so existing configs are
    # unaffected unless they explicitly set freeze_bn: false.
    freeze_bn = getattr(cfg, "freeze_bn", True)
    if freeze and not freeze_bn:
        print(f"[NavRL] Encoder weights frozen, BatchNorm running stats NOT frozen (freeze_bn=false).")

    projection_mlp = getattr(cfg, "projection_mlp", True)
    return ReachMapEncoderWrapper(enc, embed_dim, is_bev=is_bev, freeze_encoder=freeze,
                                   projection_mlp=projection_mlp, freeze_bn=freeze_bn)



class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device, disturbance_cfg=None, sim_dt=0.016):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Sim-to-real domain-randomization config (cfg/disturbance.yaml) — a
        # sibling of `cfg.algo` in the full config tree, so it's passed in
        # separately rather than via self.cfg (which stays cfg.algo, matching
        # every existing self.cfg.* access below). None (the default, used by
        # any caller that doesn't pass it — e.g. ROS deployment) means fully
        # disabled, identical to disturbance.enabled: false.
        self.disturbance_cfg = disturbance_cfg
        self.sim_dt = sim_dt
        # Action-latency ring buffer — allocated lazily on first __call__ once
        # the actual rollout batch size (num_envs) is known.
        self._action_history = None
        self._action_write_ptr = 0


        # Depth encoder: scratch CNN or pretrained ReachMap encoder
        depth_encoder = build_depth_encoder(cfg.feature_extractor).to(self.device)

        # Image observation key:
        #   bev_* encoders always use "bev" (env_depth sets this key when use_bev=True)
        #   all others use obs_image_key from config ("depth" for depth env, "lidar" for lidar env)
        _enc_type = getattr(cfg.feature_extractor, "encoder_type", "scratch")
        if _enc_type.startswith("bev_"):
            img_key = "bev"
        else:
            img_key = getattr(cfg.feature_extractor, "obs_image_key", "lidar")

        # network_type selects the WHOLE feature-extractor architecture that
        # turns depth_encoder's output + the other observation streams into
        # "_feature". "cnn" (default) is the original CNN+MLP-concat path,
        # unchanged below. "transformer" (ported from NavRL++) tokenizes
        # static(vision)/dynamic-obstacle/state streams through a small
        # Transformer instead — see utils.py's BuildTokens/TransformerBackbone
        # docstrings for the single-frame adaptation. "gru" concatenates the
        # same streams as "cnn" into one vector, then runs a GRU that carries
        # real hidden state across env steps (genuine temporal memory, unlike
        # the other two, which are both per-timestep-only). Either way this
        # block's only job is to assign self.feature_extractor; actor/critic
        # below are architecture-agnostic (they only read "_feature").
        network_type = getattr(cfg, "network_type", "cnn")
        self.network_type = network_type

        if network_type == "cnn":
            use_dyn_obs_net = getattr(cfg.feature_extractor, "use_dyn_obs_net", True)

            if use_dyn_obs_net:
                dynamic_obstacle_network = nn.Sequential(
                    Rearrange("n c w h -> n (c w h)"),
                    make_mlp([128, 64])
                ).to(self.device)
                self.feature_extractor = TensorDictSequential(
                    TensorDictModule(depth_encoder, [("agents", "observation", img_key)], ["_cnn_feature"]),
                    TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
                    CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False),
                    TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
                ).to(self.device)
            else:
                # Named separately (not inlined) so disturbance.asymmetric_critic
                # below can reuse this exact module (shared weights) for the
                # critic-only clean-observation pass.
                _fusion_mlp = make_mlp([256, 256])
                self.feature_extractor = TensorDictSequential(
                    TensorDictModule(depth_encoder, [("agents", "observation", img_key)], ["_cnn_feature"]),
                    CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
                    TensorDictModule(_fusion_mlp, ["_feature"], ["_feature"]),
                ).to(self.device)
        elif network_type == "transformer":
            tcfg = cfg.transformer
            # Drop the dynamic-obstacle token entirely when there are none to
            # sense (dyn_obs_num=0, e.g. train_depth_robust.yaml with
            # env_dyn.num_obstacles=0) — otherwise it's a dead, all-zero
            # token with its own unused adapter weights. Automatically comes
            # back once dyn_obs_num > 0 (dynamic obstacles turned on).
            dyn_obs_num = getattr(cfg.feature_extractor, "dyn_obs_num", 0)
            use_dynamic_token = dyn_obs_num > 0
            build_tokens = BuildTokens(
                d_model=tcfg.d_model, max_T=getattr(tcfg, "max_T", 8), use_dynamic=use_dynamic_token,
            ).to(self.device)
            transformer_backbone = TransformerBackbone(
                d_model=tcfg.d_model, nhead=tcfg.nhead, num_layers=tcfg.num_layers,
                dim_feedforward=tcfg.dim_feedforward, dropout=tcfg.dropout,
            ).to(self.device)
            token_in_keys = ["_cnn_feature", ("agents", "observation", "state")]
            if use_dynamic_token:
                token_in_keys.append(("agents", "observation", "dynamic_obstacle"))
            self.feature_extractor = TensorDictSequential(
                TensorDictModule(depth_encoder, [("agents", "observation", img_key)], ["_cnn_feature"]),
                TensorDictModule(build_tokens, token_in_keys, ["_token"]),
                TensorDictModule(transformer_backbone, ["_token"], ["_feature"]),
                TensorDictModule(make_mlp(list(tcfg.feature_extractor_mlp)), ["_feature"], ["_feature"]),
            ).to(self.device)
        elif network_type == "gru":
            gcfg = cfg.gru
            use_dyn_obs_net = getattr(cfg.feature_extractor, "use_dyn_obs_net", True)

            # FlattenLeadingDims: this branch's extractor is called on
            # (num_env, num_frames)-shaped tensordicts (see its docstring), so
            # the per-timestep encoder / dyn-obs net need the extra leading dim
            # folded away. cnn/transformer wrap nothing and are unaffected.
            upstream_modules = [
                TensorDictModule(
                    FlattenLeadingDims(depth_encoder, n_feature_dims=3),
                    [("agents", "observation", img_key)], ["_cnn_feature"],
                ),
            ]
            if use_dyn_obs_net:
                dynamic_obstacle_network = nn.Sequential(
                    Rearrange("n c w h -> n (c w h)"),
                    make_mlp([128, 64])
                ).to(self.device)
                upstream_modules.append(
                    TensorDictModule(
                        FlattenLeadingDims(dynamic_obstacle_network, n_feature_dims=3),
                        [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"],
                    )
                )
                cat_in_keys = ["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"]
            else:
                cat_in_keys = ["_cnn_feature", ("agents", "observation", "state")]
            upstream_modules.append(CatTensors(cat_in_keys, "_feature", del_keys=False))
            # nn.GRU has no Lazy variant, so this MUST project to a concrete,
            # fixed width == gcfg.input_size (unlike cnn's make_mlp([256,256])
            # above, which only needs to be LazyLinear-consistent).
            upstream_modules.append(TensorDictModule(make_mlp([gcfg.input_size]), ["_feature"], ["_feature"]))

            # python_based=True: nn.GRU's cuDNN backend is not vmap-compatible,
            # and train() below calls torch.vmap(self.feature_extractor)(...)
            # to bootstrap next-state values — this MUST stay True or that
            # call breaks (see torchrl's own test_gru_vmap_complex_model).
            self.gru_module = GRUModule(
                input_size=gcfg.input_size,
                hidden_size=gcfg.hidden_size,
                num_layers=getattr(gcfg, "num_layers", 1),
                dropout=getattr(gcfg, "dropout", 0.0),
                python_based=True,
                device=self.device,
                in_key="_feature",
                out_key="_feature",
            )
            # Rollout/inference: step-by-step, one env-step per call.
            self.feature_extractor = TensorDictSequential(*upstream_modules, self.gru_module).to(self.device)
            # Training/BPTT: same upstream module OBJECTS (weight-shared with
            # the rollout extractor above) + the GRU in recurrent mode, used
            # only by _update() over make_batch_recurrent's time-ordered
            # minibatches (see train()/_update() below).
            self.feature_extractor_train = TensorDictSequential(
                *upstream_modules, self.gru_module.set_recurrent_mode(True)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown algo.network_type '{network_type}'. Choose 'cnn', 'transformer', or 'gru'.")

        # _update() reads through this — equals self.feature_extractor for
        # cnn/transformer (same object, so the call in _update() is a no-op
        # change for those two), and the recurrent-mode extractor for gru.
        self._train_feature_extractor = (
            self.feature_extractor_train if network_type == "gru" else self.feature_extractor
        )

        # Disturbance: asymmetric actor-critic. When on, the actor keeps
        # seeing exactly what a real sensor would give it (noisy/delayed
        # depth+state, unchanged above); the critic instead reads the
        # undistorted depth_clean/state_clean the env only emits when
        # disturbance.enabled (env_depth.py's self._emit_clean_obs) — value
        # estimation gets ground truth even though the policy never does.
        # feature_extractor_critic reuses the SAME depth_encoder/_fusion_mlp
        # module objects as feature_extractor (shared weights, just a second
        # TensorDictModule wrapper reading different keys) — one backward()
        # call accumulates gradients from both passes onto those shared
        # parameters, so no separate optimizer/grad-clip wiring is needed.
        # Only wired up for the cnn/use_dyn_obs_net=false path (this
        # pipeline's actual deploy config); anything else raises rather than
        # silently ignoring the flag.
        self.asymmetric_critic = bool(
            disturbance_cfg is not None
            and getattr(disturbance_cfg, "enabled", False)
            and getattr(disturbance_cfg, "asymmetric_critic", False)
        )
        if self.asymmetric_critic:
            if network_type != "cnn" or use_dyn_obs_net:
                raise NotImplementedError(
                    "disturbance.asymmetric_critic is only implemented for "
                    "algo.network_type=cnn with feature_extractor.use_dyn_obs_net=false "
                    f"(got network_type={network_type!r}, use_dyn_obs_net={use_dyn_obs_net!r})"
                )
            self.feature_extractor_critic = TensorDictSequential(
                TensorDictModule(depth_encoder, [("agents", "observation", "depth_clean")], ["_cnn_feature_critic"]),
                CatTensors(["_cnn_feature_critic", ("agents", "observation", "state_clean")], "_feature_critic", del_keys=False),
                TensorDictModule(_fusion_mlp, ["_feature_critic"], ["_feature_critic"]),
            ).to(self.device)
        _critic_in_key = "_feature_critic" if self.asymmetric_critic else "_feature"

        # Actor etwork
        # actor.hidden_layers / critic.hidden_layers ([] by default -- see
        # ppo.yaml) let each head grow a shared nonlinear trunk (make_mlp)
        # before its final linear output, instead of NavRL++'s original bare
        # linear readout off `_feature`. Cheap, orthogonal probe for whether
        # the collision-dominated SR ceiling is a head-capacity limitation.
        self.n_agents, self.action_dim = action_spec.shape
        actor_hidden = list(getattr(cfg.actor, "hidden_layers", []) or [])
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim, hidden_layers=actor_hidden), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        critic_hidden = list(getattr(cfg.critic, "hidden_layers", []) or [])
        critic_module = (
            nn.Sequential(make_mlp(critic_hidden), nn.LazyLinear(1))
            if critic_hidden else nn.LazyLinear(1)
        )
        self.critic = TensorDictModule(
            critic_module, [_critic_in_key], ["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        #
        # The GRU trunk lives inside self.feature_extractor, so by default it
        # inherits cfg.feature_extractor.learning_rate (1e-5) -- a rate chosen
        # for a FROZEN pretrained encoder, not for a recurrent trunk that has
        # to learn what to remember from scratch. cnn/transformer put only a
        # fusion MLP there and tolerate it; the gru branch adds ~460k
        # recurrent params on the same 1e-5, i.e. 1/5 the actor's lr and 1/30
        # the critic's. algo.gru.learning_rate breaks that coupling by giving
        # the GRUModule its own param group.
        #
        # Scope is the GRUModule only -- the pre-GRU fusion MLP stays on the
        # feature_extractor lr, matching cnn's own fusion MLP, so a run with
        # this set differs from the cnn baseline in the recurrence alone.
        # null (the default) => single param group, byte-identical to before.
        gru_lr = getattr(getattr(cfg, "gru", None), "learning_rate", None) if network_type == "gru" else None
        if gru_lr is not None:
            gru_param_ids = {id(p) for p in self.gru_module.parameters()}
            self.feature_extractor_optim = torch.optim.Adam([
                {"params": [p for p in self.feature_extractor.parameters() if id(p) not in gru_param_ids],
                 "lr": cfg.feature_extractor.learning_rate},
                # feature_extractor_train shares these parameter OBJECTS (see
                # the gru branch above), so one group covers both extractors.
                {"params": list(self.gru_module.parameters()), "lr": float(gru_lr)},
            ])
            print(f"[NavRL] GRU trunk lr={float(gru_lr):g} (rest of feature_extractor: {cfg.feature_extractor.learning_rate:g})")
        else:
            self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic.learning_rate)

        # LR decay: factor schedule over training iterations (one per train() call).
        # Note: the iteration counter restarts at 0 when resuming from a checkpoint
        # (only module weights are saved), so a resumed run re-warms from base lr.
        self.lr_decay            = getattr(cfg, "lr_decay", None)
        self.lr_decay_iters      = int(float(getattr(cfg, "lr_decay_iters", 40000)))  # float() first: YAML "4e4" parses as str
        self.lr_decay_min_factor = float(getattr(cfg, "lr_decay_min_factor", 0.1))
        self._lr_iter = 0
        self._lr_optims = [self.feature_extractor_optim, self.actor_optim, self.critic_optim]
        self._lr_base = [[g["lr"] for g in opt.param_groups] for opt in self._lr_optims]
        if self.lr_decay:
            print(f"[NavRL] LR decay: {self.lr_decay} over {self.lr_decay_iters} iters "
                  f"(floor {self.lr_decay_min_factor}x)")

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)

        # GRUModule.forward() unconditionally reads "is_init" (no default,
        # raises if missing); observation_spec.zero() won't have it until
        # InitTracker is wired into the env (train.py/eval.py, only when
        # network_type=="gru"). Gated strictly behind "gru" so cnn/transformer's
        # warmup call is byte-for-byte unchanged -- no new key injected at all.
        if network_type == "gru" and "is_init" not in dummy_input.keys():
            dummy_input.set(
                "is_init",
                torch.zeros(*dummy_input.batch_size, 1, dtype=torch.bool, device=self.device),
            )

        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        if self.asymmetric_critic:
            # env_depth.py only populates depth_clean/state_clean while
            # env.training is True — never during an eval rollout (see its
            # _emit_clean_obs comment: eval doesn't read state_value at all,
            # so it isn't worth doubling the depth image through a ~2200-step
            # eval buffer). Fall back to the actor's own (noisy) "_feature"
            # for the critic in that case rather than erroring on a missing
            # key — the resulting state_value is simply unused by eval.
            if tensordict.get(("agents", "observation", "depth_clean"), None) is not None:
                self.feature_extractor_critic(tensordict)
            else:
                tensordict["_feature_critic"] = tensordict["_feature"]
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])

        actions_world = self._apply_action_disturbance(actions_world)

        tensordict["agents", "action"] = actions_world
        return tensordict

    def _apply_action_disturbance(self, actions_world: torch.Tensor) -> torch.Tensor:
        """Action-latency + Gaussian action noise, ported from NavRL++ (ppo.py
        __call__). No-op unless disturbance_cfg.enabled and the relevant
        field is non-zero — see disturbance.yaml for the exact gating.

        Simplification vs. NavRL++: skips the per-env "not enough history
        yet" fallback (there, borrowed from an env-side history-length
        counter this pipeline doesn't track) — the ring buffer is
        zero-initialized and read unconditionally once latency is enabled,
        which only affects the first ~0.3s of a run/episode.
        """
        dcfg = self.disturbance_cfg
        if dcfg is None or not getattr(dcfg, "enabled", False):
            return actions_world

        action_latency = getattr(dcfg, "action_latency", 0.0)
        has_latency = (action_latency != 0.0) if isinstance(action_latency, float) else any(action_latency)
        if has_latency:
            B = actions_world.shape[0]
            action_dim = actions_world.shape[-1]
            hist_length = int(0.3 / self.sim_dt) + 1
            if self._action_history is None or self._action_history.shape[0] != B:
                self._action_history = torch.zeros(
                    B, hist_length, action_dim, device=actions_world.device
                )
                self._action_write_ptr = 0

            j = self._action_write_ptr % hist_length
            self._action_history[:, j, :] = actions_world.detach().reshape(B, action_dim)
            self._action_write_ptr += 1
            base = torch.arange(hist_length, device=actions_world.device)
            idx = (base + (self._action_write_ptr % hist_length)) % hist_length  # oldest to newest

            if isinstance(action_latency, float):
                action_latency_step = int(action_latency / self.sim_dt) + 1
            else:
                low = int(action_latency[0] / self.sim_dt) + 1
                high = int(action_latency[1] / self.sim_dt) + 1
                action_latency_step = np.random.randint(low=low, high=high + 1)
            action_latency_step = min(action_latency_step, hist_length)

            delayed_idx = idx[-action_latency_step]
            actions_world = self._action_history[:, delayed_idx, :].reshape(actions_world.shape)

        action_gaussian_std = getattr(dcfg, "action_gaussian_std", 0.0)
        if action_gaussian_std != 0.0:
            action_noise_threshold = getattr(dcfg, "action_noise_threshold", 0.0)
            actions_world_norm = torch.norm(actions_world, dim=-1, keepdim=True)
            mask = actions_world_norm > action_noise_threshold
            noise = torch.normal(
                mean=torch.zeros_like(actions_world),
                std=action_gaussian_std * torch.ones_like(actions_world),
            )
            actions_world = actions_world + mask * noise

        return actions_world

    def _lr_decay_factor(self) -> float:
        if not self.lr_decay:
            return 1.0
        t = min(self._lr_iter / max(self.lr_decay_iters, 1), 1.0)
        m = self.lr_decay_min_factor
        if self.lr_decay == "linear":
            return 1.0 - (1.0 - m) * t
        if self.lr_decay == "cosine":
            import math
            return m + 0.5 * (1.0 - m) * (1.0 + math.cos(math.pi * t))
        raise ValueError(f"Unknown lr_decay '{self.lr_decay}' (use null, 'linear' or 'cosine')")

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames

        # Apply LR decay before this iteration's updates
        lr_factor = self._lr_decay_factor()
        if self.lr_decay:
            for opt, base_lrs in zip(self._lr_optims, self._lr_base):
                for group, base in zip(opt.param_groups, base_lrs):
                    group["lr"] = base * lr_factor
        self._lr_iter += 1

        next_tensordict = tensordict["next"]
        with torch.no_grad():
            if self.network_type == "gru":
                # vmap(self.feature_extractor) (the cnn/transformer path below)
                # doesn't work here: GRUModule's step mode only supports a
                # single batch dim, and next_tensordict is (num_env,
                # num_frames) -- both an explicit outer vmap AND GRUModule's
                # own internal vmap-over-extra-dims fallback hit the same
                # "data-dependent control flow" error on its `is_init.any()`
                # check (confirmed empirically, python_based=True does not
                # help — that flag only affects the cuDNN-vs-python GRU cell,
                # not this). Recurrent mode has no such restriction: it
                # natively consumes a (num_env, num_frames) sequence using
                # next_tensordict's own recorded is_init/recurrent_state.
                # Chunked along the env axis only (num_frames axis stays intact
                # per chunk -- GRUModule's recurrent mode needs the full time
                # axis to correctly replay is_init/recurrent_state, same as
                # make_batch_recurrent's minibatches below). Unlike _update(),
                # this call used to run on the WHOLE (num_envs, num_frames)
                # tensordict in one shot -- FlattenLeadingDims folds both dims
                # into one batch dim for the per-frame CNN encoder, so at
                # num_envs=300+/training_frame_num=128 that's 38k+ images in a
                # single forward. Confirmed empirically: this exact call OOM'd
                # (model/encoder.py:99, CNN forward) at num_envs=300 and 350,
                # while _update()'s own per-minibatch BPTT pass at this same
                # per-chunk size never has. Reuses num_minibatches as the chunk
                # count so both passes share one memory budget.
                self.feature_extractor_train.eval()
                num_chunks = max(1, int(self.cfg.num_minibatches))
                chunk_size = -(-next_tensordict.shape[0] // num_chunks)  # ceil div
                next_tensordict = torch.cat([
                    self.feature_extractor_train(next_tensordict[i:i + chunk_size])
                    for i in range(0, next_tensordict.shape[0], chunk_size)
                ], dim=0)
                self.feature_extractor_train.train()
            else:
                # Chunked along the env axis for the same reason as the gru
                # branch above (see its comment) -- torch.vmap batches the
                # WHOLE (num_envs, num_frames) tensordict through the
                # per-frame CNN encoder in one shot just like an unchunked
                # direct call would; vmap is a vectorization wrapper, not a
                # memory-saving one. Confirmed empirically: this exact line
                # OOM'd at num_envs=512 and 1024 (ppo.py:812, 65k+ images in
                # one CNN forward at 512x128). Reuses num_minibatches as the
                # chunk count, same as the gru branch.
                num_chunks = max(1, int(self.cfg.num_minibatches))
                chunk_size = -(-next_tensordict.shape[0] // num_chunks)  # ceil div
                self.feature_extractor.eval()
                next_tensordict = torch.cat([
                    torch.vmap(self.feature_extractor)(next_tensordict[i:i + chunk_size])
                    for i in range(0, next_tensordict.shape[0], chunk_size)
                ], dim=0)
                self.feature_extractor.train()
                if self.asymmetric_critic:
                    self.feature_extractor_critic.eval()
                    next_tensordict = torch.cat([
                        torch.vmap(self.feature_extractor_critic)(next_tensordict[i:i + chunk_size])
                        for i in range(0, next_tensordict.shape[0], chunk_size)
                    ], dim=0)
                    self.feature_extractor_critic.train()
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states
        # Same "physics crash near reset" hazard as rewards/values above, but for GAE's
        # `not_done` term: GAE recurses backward through time, so a single NaN here
        # poisons every earlier advantage in the same trajectory, not just one step.
        # 0.0 (= "not done") is the safe default, matching the neutral treatment already
        # given to a corrupted reward at that step.
        dones = dones.nan_to_num(0.0)

        # Sanitize rewards and values: a physics crash on the final step before reset can
        # leave NaN in one transition; replace with 0 so GAE and ValueNorm stay clean.
        rewards = rewards.nan_to_num(0.0)

        # Sanitize stored state_value (b_value): if the critic produced NaN during
        # collection (e.g., NaN observation escaped env sanitization), b_value in
        # _update would be NaN → critic_loss NaN. Replace with 0.
        raw_state_val = tensordict["state_value"]
        if torch.isnan(raw_state_val).any():
            n_nan = torch.isnan(raw_state_val).sum().item()
            print(f"[NavRL] {n_nan}/{raw_state_val.numel()} NaN state_values in buffer — sanitizing")
            tensordict["state_value"] = raw_state_val.nan_to_num(0.0)

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)
        values = values.nan_to_num(0.0)
        next_values = next_values.nan_to_num(0.0)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        if not torch.isfinite(adv).all():

            def _rawstats(name, t):
                t = t.detach()
                finite = torch.isfinite(t)
                print(f"[NavRL]   {name}: min={t[finite].min().item() if finite.any() else float('nan'):.4g} "
                      f"max={t[finite].max().item() if finite.any() else float('nan'):.4g} "
                      f"n_nan={torch.isnan(t).sum().item()} n_inf={torch.isinf(t).sum().item()} / {t.numel()}")

            print("[NavRL] raw GAE output (pre-normalization) is not finite -- inputs:")
            _rawstats("rewards", rewards)
            _rawstats("dones", dones)
            _rawstats("values (denormalized)", values)
            _rawstats("next_values (denormalized)", next_values)
            _rawstats("adv (raw GAE output)", adv)
            _rawstats("ret (raw GAE output)", ret)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        # ValueNorm running stats become permanently NaN once corrupted (EMA: NaN*β + …=NaN).
        # Detect and reset before the update so future batches are not poisoned.
        if (torch.isnan(self.value_norm.running_mean).any()
                or torch.isnan(self.value_norm.running_mean_sq).any()
                or torch.isnan(self.value_norm.debiasing_term).any()):
            print("[NavRL] ValueNorm running stats corrupted (NaN) — resetting")
            self.value_norm.reset_parameters()

        # Value-function diagnostics. MUST be computed here, while `ret` is still
        # the raw GAE return: `values` above was denormalize()d into raw return
        # units, and two lines below `ret` gets normalize()d into unit scale. Taking
        # the residual across that boundary subtracts a raw-scale prediction from a
        # normalized target, which is not a residual at all -- it reads as a value
        # function thousands of times worse than it is.
        #
        # explained_var = 1 - Var(ret - V) / Var(ret). Read it together with
        # return_var: this reward carries a constant +1 alive bonus, so a large part
        # of every return is identical across envs and the denominator is small. Low
        # explained_var with a small return_var is a property of the reward; low
        # explained_var with a large return_var is a broken critic.
        with torch.no_grad():
            _ret = ret.detach().float()
            _res = _ret - values.detach().float()
            _rv = _ret.var()
            # NOTE: no "explained_var" here. _update() already logs a correct one
            # (ppo.py:1043) computed on the NORMALIZED ret against the freshly
            # predicted value -- same units, the right quantity. These keys are
            # prefixed raw_ because they are the only view of the return in real
            # reward units, which the normalized metric cannot show: it divides by
            # ret.var() ~= 1 by construction, so it can never tell you that the
            # denominator is small because the alive bonus dominates the return.
            self._value_diag = {
                "raw_return_var": _rv.item(),
                "raw_return_mean": _ret.mean().item(),
                "raw_value_residual_var": _res.var().item(),
            }

        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        ret = ret.nan_to_num(0.0)  # guard: normalize() can return NaN if stats were just reset
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)


        # Training
        # gru needs each minibatch's per-env time axis kept in order (BPTT
        # through set_recurrent_mode(True)); cnn/transformer only need
        # per-timestep independence, so they keep the original shuffle.
        batch_fn = make_batch_recurrent if self.network_type == "gru" else make_batch
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = batch_fn(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        out = {k: v.item() for k, v in infos.items()}
        out.update(getattr(self, "_value_diag", {}))
        out["lr_factor"] = lr_factor
        out["actor_lr"] = self.actor_optim.param_groups[0]["lr"]
        return out    

    
    def _update(self, tensordict): # tensordict shape (batch_size, )
        self._train_feature_extractor(tensordict)
        if self.asymmetric_critic:
            self.feature_extractor_critic(tensordict)

        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution

        # Beta(alpha, beta) sharpness. A collapsed policy shows up here long before
        # it shows up in the returns: concentration = alpha + beta grows without
        # bound as the distribution spikes, and action_std is that spike expressed
        # in the units the drone actually flies in. Var[u] = ab/((a+b)^2 (a+b+1))
        # on [0,1]; ppo.py maps u -> 2*L*u - L, so the commanded velocity std is
        # 2*L*sqrt(Var[u]) with L = actor.action_limit.
        # Read the parameters off the distribution rather than the tensordict:
        # IndependentBeta wraps torch.distributions.Beta (utils.py:325-330), whose
        # concentration1/concentration0 ARE alpha/beta, so this cannot KeyError if
        # get_dist ever stops writing those keys back.
        with torch.no_grad():
            _bd = action_dist.base_dist
            _a = _bd.concentration1.detach().float()
            _b = _bd.concentration0.detach().float()
            _conc = _a + _b
            _ustd = (_a * _b / (_conc.pow(2) * (_conc + 1.0))).sqrt()
            beta_alpha = _a.mean()
            beta_conc = _conc.mean()
            action_std = 2.0 * float(self.cfg.actor.action_limit) * _ustd.mean()
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        # If action_normalized hit a Beta boundary (0 or 1), log_prob = -inf.
        # -inf - (-inf) = NaN; nan_to_num(0) → ratio=1 (neutral, skip that transition).
        # Then clamp to guard against large-but-finite divergence.
        log_ratio = (log_probs - tensordict["sample_log_prob"]).nan_to_num(0.0).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio).unsqueeze(-1)
        # Sanity signal: for this minibatch's FIRST optimizer step on freshly
        # collected data (training_epoch_num=1 -> every minibatch is seen
        # exactly once), no parameter has moved since rollout yet, so ratio
        # should be ~1 and approx_kl ~0 by construction. A large deviation
        # here (before any gradient has been applied) points at a replay
        # mismatch between rollout-time and update-time forward passes
        # (hidden state / is_init handling, BN running stats, dropout, input
        # normalization) rather than at the optimizer being too aggressive.
        with torch.no_grad():
            approx_kl = torch.mean((ratio.squeeze(-1) - 1) - log_ratio)
            ratio_mean = ratio.mean()
            ratio_std = ratio.std()
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        # Skip update if loss is NaN/Inf (e.g., caused by a bad batch)
        if not torch.isfinite(loss).item():
            _dev = loss.device
            print(f"[NavRL] NaN/Inf loss at update step, skipping optimizer (actor={actor_loss.item():.4f}, critic={critic_loss.item():.4f}, entropy={entropy_loss.item():.4f})")

            def _stats(name, t):
                t = t.detach()
                finite = torch.isfinite(t)
                print(f"[NavRL]   {name}: min={t[finite].min().item() if finite.any() else float('nan'):.4g} "
                      f"max={t[finite].max().item() if finite.any() else float('nan'):.4g} "
                      f"n_nan={torch.isnan(t).sum().item()} n_inf={torch.isinf(t).sum().item()} / {t.numel()}")

            _stats("_feature", tensordict["_feature"])
            _stats("log_probs (current policy)", log_probs)
            _stats("sample_log_prob (behavior policy, stored)", tensordict["sample_log_prob"])
            _stats("advantage", advantage)
            _stats("ratio", ratio)
            return TensorDict({
                "actor_loss": actor_loss.detach(),
                "critic_loss": critic_loss.detach(),
                "entropy": entropy_loss.detach(),
                "actor_grad_norm": torch.tensor(float("nan"), device=_dev),
                "critic_grad_norm": torch.tensor(float("nan"), device=_dev),
                "explained_var": torch.tensor(float("nan"), device=_dev),
                "approx_kl": approx_kl.detach(),
                "ratio_mean": ratio_mean.detach(),
                "ratio_std": ratio_std.detach(),
                "beta_alpha": beta_alpha.detach(),
                "beta_conc": beta_conc.detach(),
                "action_std": action_std.detach(),
            }, [])

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var().clamp(min=1e-8)
        out = TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
            "approx_kl": approx_kl,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "beta_alpha": beta_alpha,
            "beta_conc": beta_conc,
            "action_std": action_std,
        }, [])
        return out