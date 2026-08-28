import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import (
    ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor,
    vec_to_world, BuildTokens, TransformerBackbone,
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
            embed_dim      = embed_dim,
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
            embed_dim      = embed_dim,
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
        # docstrings for the single-frame adaptation. Either way this block's
        # only job is to assign self.feature_extractor; actor/critic below are
        # architecture-agnostic (they only read "_feature").
        network_type = getattr(cfg, "network_type", "cnn")

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
                self.feature_extractor = TensorDictSequential(
                    TensorDictModule(depth_encoder, [("agents", "observation", img_key)], ["_cnn_feature"]),
                    CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
                    TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
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
        else:
            raise ValueError(f"Unknown algo.network_type '{network_type}'. Choose 'cnn' or 'transformer'.")

        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
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
            self.feature_extractor.eval()
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            self.feature_extractor.train()
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

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

        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        ret = ret.nan_to_num(0.0)  # guard: normalize() can return NaN if stats were just reset
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # Training
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        out = {k: v.item() for k, v in infos.items()}
        out["lr_factor"] = lr_factor
        out["actor_lr"] = self.actor_optim.param_groups[0]["lr"]
        return out    

    
    def _update(self, tensordict): # tensordict shape (batch_size, )
        self.feature_extractor(tensordict)

        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
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
            return TensorDict({
                "actor_loss": actor_loss.detach(),
                "critic_loss": critic_loss.detach(),
                "entropy": entropy_loss.detach(),
                "actor_grad_norm": torch.tensor(float("nan"), device=_dev),
                "critic_grad_norm": torch.tensor(float("nan"), device=_dev),
                "explained_var": torch.tensor(float("nan"), device=_dev),
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
        }, [])
        return out