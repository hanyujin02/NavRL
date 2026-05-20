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

# Path to ReachMap root: scripts/ → training/ → isaac-training/ → NavRL/ → ReachMap/
_REACHMAP_ROOT = Path(__file__).resolve().parents[4]

# Depth-input encoder types available from ReachMap (excludes BEV variants)
_REACHMAP_DEPTH_ENCODERS = [
    "cnn", "vit",
    "cnn_gru", "vit_gru",
    "cnn_transformer", "vit_transformer",
    "rep_cnn", "rep_cnn_gru", "rep_cnn_transformer",
    "rep_vit", "rep_vit_gru", "rep_vit_transformer",
    "rep_vae", "rep_vae_gru", "rep_vae_transformer",
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
    return {
        "cnn":                 CNNOnlyEncoder,
        "vit":                 ViTOnlyEncoder,
        "cnn_gru":             TemporalCNNEncoder,
        "vit_gru":             TemporalViTEncoder,
        "cnn_transformer":     TemporalTransformerCNNEncoder,
        "vit_transformer":     TemporalTransformerViTEncoder,
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


class ReachMapEncoderWrapper(nn.Module):
    """
    Adapts a ReachMap encoder for NavRL's PPO feature extractor.

    ReachMap encoders expect (B, T, H, W) and return (context: B, D), (current: B, D).
    NavRL feeds (N, 1, H, W) depth images — the channel-1 dim acts as T=1, so no
    reshape is needed; the encoder interprets it as a single-frame sequence.

    The context vector (temporal summary, or last-frame feature for non-temporal
    encoders) is projected through a LayerNorm and returned as (N, embed_dim).
    """

    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder   = encoder
        self.embed_dim = embed_dim
        self.out_norm  = nn.LayerNorm(embed_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: (N, 1, H, W) — treated as (B=N, T=1, H, W) by ReachMap encoders
        context, _ = self.encoder(depth)
        return self.out_norm(context)        # (N, embed_dim)


def _load_encoder_ckpt(encoder: nn.Module, ckpt_path: str) -> None:
    """Load encoder weights from a ReachNet checkpoint (best.pt / latest.pt)."""
    raw    = torch.load(ckpt_path, map_location="cpu")
    full_sd = raw.get("state_dict", raw)
    enc_sd  = {
        k[len("encoder."):]: v
        for k, v in full_sd.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"[NavRL] encoder ckpt missing keys: {missing[:5]}")
    if unexpected:
        print(f"[NavRL] encoder ckpt unexpected keys: {unexpected[:5]}")
    print(f"[NavRL] Loaded encoder weights from {ckpt_path}")


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
            nn.LazyLinear(embed_dim), nn.LayerNorm(embed_dim),
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
        depth_max_m     = getattr(cfg, "rep_depth_max_m",     4.0),
        rep_vae_latent  = getattr(cfg, "rep_vae_latent",       64),
    )

    enc = registry[encoder_type](**enc_kwargs)

    # Optionally load a full ReachNet checkpoint (encoder.* keys) on top.
    # Works for all encoder types: non-rep encoders trained in ReachMap,
    # or rep encoders fine-tuned end-to-end via ReachMap.
    reachnet_ckpt = getattr(cfg, "encoder_ckpt", None)
    if reachnet_ckpt:
        _load_encoder_ckpt(enc, reachnet_ckpt)

    if getattr(cfg, "freeze_encoder", False):
        for p in enc.parameters():
            p.requires_grad = False
        print(f"[NavRL] Encoder weights frozen.")

    return ReachMapEncoderWrapper(enc, embed_dim)



class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        
        # Depth encoder: scratch CNN or pretrained ReachMap encoder
        depth_encoder = build_depth_encoder(cfg.feature_extractor).to(self.device)

        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Image observation key: 'depth' for depth camera env, 'lidar' for lidar env
        img_key = getattr(cfg.feature_extractor, "obs_image_key", "lidar")

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(depth_encoder, [("agents", "observation", img_key)], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
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

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

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
        tensordict["agents", "action"] = actions_world
        return tensordict

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
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
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
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

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])