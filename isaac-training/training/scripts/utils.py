import gc
import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type


# ── Depth image noise for closed-loop noisy rollout ──────────────────────────

def apply_depth_noise(depth: torch.Tensor, noise_type: str, **kwargs) -> torch.Tensor:
    """Apply image-level noise to a depth tensor in [0, 1].
    depth: (..., H, W) — supports any leading batch/channel dims.
    """
    if noise_type == "gaussian":
        return (depth + torch.randn_like(depth) * kwargs.get("sigma", 0.05)).clamp(0.0, 1.0)
    if noise_type == "dropout":
        # fill_value=0.0 (default) mimics a false near-obstacle reading;
        # fill_value=1.0 mimics a lost/no-return ray, which this env's own
        # sensor pipeline (env_depth.py: nan_to_num(nan=depth_range), missed
        # rays -> max range) treats as max range, not zero.
        fill_value = kwargs.get("fill_value", 0.0)
        mask = (torch.rand_like(depth) > kwargs.get("rate", 0.15)).float()
        return depth * mask + fill_value * (1.0 - mask)
    if noise_type == "cutout":
        out = depth.clone()
        H, W = depth.shape[-2], depth.shape[-1]
        ph = kwargs.get("patch_h", 20)
        pw = kwargs.get("patch_w", 20)
        fill_value = kwargs.get("fill_value", 0.0)
        for _ in range(kwargs.get("n_holes", 1)):
            y0 = int(torch.randint(0, max(1, H - ph + 1), (1,)))
            x0 = int(torch.randint(0, max(1, W - pw + 1), (1,)))
            out[..., y0:y0 + ph, x0:x0 + pw] = fill_value
        return out
    if noise_type == "quantization":
        lvl = kwargs.get("levels", 16)
        return (depth * lvl).floor().clamp(0, lvl) / lvl
    raise ValueError(f"Unknown noise_type '{noise_type}'")


class NoisyPolicyWrapper:
    """Wraps a NavRL PPO policy and injects depth noise at each rollout step.

    Usage:
        noisy_policy = NoisyPolicyWrapper(policy, noise_type="gaussian", sigma=0.05)
        trajs = env.rollout(..., policy=noisy_policy, ...)
    """

    def __init__(self, policy, noise_type: str, img_key: str = "depth", **noise_kwargs):
        self._policy     = policy
        self._noise_type = noise_type
        self._img_key    = img_key
        self._noise_kw   = noise_kwargs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        obs_key = ("agents", "observation", self._img_key)
        if obs_key in list(tensordict.keys(True, True)):
            noisy = apply_depth_noise(tensordict[obs_key], self._noise_type, **self._noise_kw)
            tensordict[obs_key] = noisy
        return self._policy(tensordict)

    def __getattr__(self, name):
        # Proxy attribute access to the underlying policy
        return getattr(self._policy, name)

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int = 0,
    exploration_type: ExplorationType = ExplorationType.MEAN,
    noise_cfg: dict = None,
):

    # eval_video=false skips the RTX viewer/recording pipeline entirely —
    # saves GPU render buffers + a ~2.5 GB CPU frame buffer per eval and
    # several seconds of encode time. Depth sensors are unaffected (they use
    # their own render products regardless of enable_render).
    eval_video = bool(getattr(cfg, "eval_video", True))
    if eval_video:
        env.enable_render(True)
    env.eval()
    env.set_seed(seed)

    # Optionally wrap policy with noise injection
    if noise_cfg is not None:
        noise_type = noise_cfg.get("type")
        if noise_type:
            img_key = noise_cfg.get("img_key", "depth")
            noise_kwargs = {k: v for k, v in noise_cfg.items()
                           if k not in ("type", "img_key")}
            eval_policy = NoisyPolicyWrapper(policy, noise_type, img_key, **noise_kwargs)
        else:
            eval_policy = policy
    else:
        eval_policy = policy

    render_callback = RenderCallback(interval=2) if eval_video else None

    # No gradients needed for an eval rollout — without this, every policy
    # forward pass across all num_envs x max_episode_length steps builds and
    # retains a full autograd graph (saved activations for backprop that will
    # never happen), which at num_envs=350 x 2200 steps is tens of GB on top
    # of the raw observation/action tensors and isn't reliably reclaimed by
    # del/gc.collect()/empty_cache() alone between successive eval() calls —
    # OOMs a 31GB GPU by the second noise condition in eval.py's loop.
    # return_contiguous=False deliberately: True forces one single contiguous
    # allocation for the whole trajectory (~14.8GB of depth alone at
    # num_envs=350), which needs one contiguous free region and OOMs sooner
    # under a fragmented pool than many smaller per-step allocations do —
    # confirmed empirically (OOM moved from the 2nd condition to inside the
    # 1st when tried at num_envs=350).
    with torch.no_grad(), set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,
            policy=eval_policy,
            callback=render_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    # base_env.enable_render(not cfg.headless)
    env.enable_render(not cfg.headless)
    env.reset()
    
    done = trajs.get(("next", "done"))
    first_done = torch.argmax(done.long(), dim=1).cpu() # idx of first done will be return for each trajs

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }
    # Free the large rollout buffer (holds all obs/depth for all envs×steps) ASAP.
    # gc.collect() before empty_cache() matters here: at large num_envs (e.g.
    # 350) x full episode length, this buffer is tens of GB, and without a
    # hard collect, lingering Python-level refs (TensorDict internals) can
    # keep the CUDA allocator from actually reclaiming it before the caller's
    # next rollout starts.
    del trajs, done, first_done
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    info = {
        "eval/stats." + k: torch.mean(v.float()).item()
        for k, v in traj_stats.items()
    }

    # log video then free frame buffer
    if eval_video:
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4"
        )
        render_callback.frames.clear()

    env.train()
    # env.reset()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x (clamp norm to avoid NaN when goal_direction is zero)
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)

    # goal direction y (cross product is zero when goal is vertical; clamp to avoid NaN)
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"

