import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world

# ---------------------------------------------------------------------------
# ReachMap encoder integration
# ---------------------------------------------------------------------------

# model/ is bundled alongside this script
_REACHMAP_ROOT = Path(__file__).resolve().parent

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
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder        = encoder
        self.embed_dim      = embed_dim
        self.projection     = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
        )
        self.is_bev         = is_bev
        self._freeze_encoder = freeze_encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_encoder:
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

        return self.projection(context)      # (N, embed_dim // 2)


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

    return ReachMapEncoderWrapper(enc, embed_dim, is_bev=is_bev, freeze_encoder=freeze)



class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        
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

        # Aux head: predict FOV-masked LiDAR distances from encoder features.
        # Supervises the encoder (or projection when frozen) to retain obstacle info.
        fov_key = ("agents", "observation", "lidar_fov")
        if fov_key in list(observation_spec.keys(True, True)):
            num_fov_rays = observation_spec[fov_key].shape[-1]
            embed_out = getattr(cfg.feature_extractor, "embed_dim", 256) // 2
            self.aux_head = nn.Linear(embed_out, num_fov_rays).to(self.device)
            self.aux_loss_coeff = getattr(cfg.feature_extractor, "aux_loss_coeff", 0.1)
            self.aux_optim = torch.optim.Adam(
                self.aux_head.parameters(), lr=cfg.feature_extractor.learning_rate
            )
            self.has_aux_head = True
            print(f"[NavRL] Aux LiDAR head: {embed_out} → {num_fov_rays} FOV rays (coeff={self.aux_loss_coeff})")
        else:
            self.has_aux_head = False

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic.learning_rate)

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

    def forward(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def update(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
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
        return {k: v.item() for k, v in infos.items()}    

    
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

        # Aux loss: predict FOV LiDAR distances from _cnn_feature (128-d encoder output).
        # Gradient flows through the projection (always) and encoder backbone (if not frozen).
        if self.has_aux_head:
            cnn_feat = tensordict["_cnn_feature"]          # (batch, 1, embed_dim//2)
            lidar_target = tensordict["agents", "observation", "lidar_fov"]  # (batch, 1, num_fov_rays)
            aux_loss = self.aux_loss_coeff * F.mse_loss(self.aux_head(cnn_feat), lidar_target)
        else:
            aux_loss = 0.0

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss + aux_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        if self.has_aux_head:
            self.aux_optim.zero_grad()
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
        if self.has_aux_head:
            self.aux_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var().clamp(min=1e-8)
        out = TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
        }, [])
        if self.has_aux_head:
            out["aux_loss"] = aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else torch.tensor(0.)
        return out