"""
One-off diagnostic: verify the GRU feature_extractor's hidden state is
actually carried across env steps during rollout (not silently reset to
zero every step), and that it resets to zero exactly at episode boundaries.

Not part of the normal train/eval pipeline -- standalone probe, small
num_envs, no wandb, no video. Loads the same checkpoint/config as the live
GRU training run (train_depth.yaml, network_type: gru) to test the actual
hidden-state wiring (InitTracker + make_batched_gru_primer + GRUModule) on
real trained weights.

Usage:
  python training/scripts/probe_gru_hidden_state.py --config-name=train_depth \
      env.num_envs=6
"""
import os
import hydra
import torch
import isaacsim  # must come before omni.isaac.kit on Isaac Sim 4.x
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from utils import make_batched_gru_primer

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="train_depth", version_base=None)
def main(cfg):
    assert getattr(cfg.algo, "network_type", "cnn") == "gru", \
        "this probe only makes sense for network_type: gru"

    sim_app = SimulationApp({"headless": True, "anti_aliasing": 0})

    _env_module = __import__(getattr(cfg, "env_script", "env_lidar"))
    env = _env_module.NavigationEnv(cfg)

    vel_gain_factor = getattr(getattr(cfg, "disturbance", None), "vel_gain_factor", 0.0)
    controller = LeePositionController(9.81, env.drone.params, vel_gain_factor).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
    transformed_env.set_seed(cfg.seed)

    policy = PPO(
        cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device,
        disturbance_cfg=getattr(cfg, "disturbance", None), sim_dt=cfg.sim.dt,
    )
    if getattr(cfg, "checkpoint", None):
        missing, unexpected = policy.load_state_dict(
            torch.load(cfg.checkpoint, map_location=cfg.device), strict=False)
        print(f"[probe] loaded checkpoint: {cfg.checkpoint}")
        print(f"[probe] missing={list(missing)[:8]} unexpected={list(unexpected)[:4]}")

    transformed_env.append_transform(InitTracker())
    transformed_env.append_transform(
        make_batched_gru_primer(policy.gru_module, cfg.env.num_envs, cfg.device)
    )

    for m in [policy.feature_extractor, policy.critic_feature_extractor, policy.actor, policy.critic]:
        m.eval()

    hidden_key = policy.gru_module.in_keys[1]     # e.g. "recurrent_state"
    hidden_out_key = policy.gru_module.out_keys[1]  # e.g. ("next", "recurrent_state")
    print(f"[probe] hidden_key={hidden_key!r} hidden_out_key={hidden_out_key!r}")

    transformed_env.reset()

    def rollout_and_report(n_steps, label):
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            trajs = transformed_env.rollout(
                max_steps=n_steps, policy=policy, auto_reset=True,
                break_when_any_done=False, return_contiguous=True,
            )
        is_init = trajs.get("is_init").squeeze(-1)          # (num_envs, n_steps)
        hidden_next = trajs.get(("next", *((hidden_out_key[1:]) if isinstance(hidden_out_key, tuple) else (hidden_out_key,))))
        # hidden_next shape: (num_envs, n_steps, num_layers, hidden_size)
        hidden_norm = hidden_next.flatten(2).norm(dim=-1)    # (num_envs, n_steps)

        print(f"\n===== {label} =====")
        for env_idx in range(min(3, hidden_norm.shape[0])):
            row_init = is_init[env_idx].tolist()
            row_norm = [round(x, 3) for x in hidden_norm[env_idx].tolist()]
            print(f"  env {env_idx}: is_init={row_init}")
            print(f"  env {env_idx}: hidden_norm={row_norm}")
        return trajs, is_init, hidden_norm

    # Phase 1: normal rollout, no manual reset -- expect is_init all False
    # (after the initial transformed_env.reset() above) and hidden_norm
    # evolving (not constant, not ~0) across steps for every env.
    rollout_and_report(20, "phase 1: continuous rollout (no forced reset)")

    # Phase 2: force a full reset, then roll out again -- expect is_init=True
    # at the first step of this phase for every env, and hidden_norm ~0 at
    # that same first step, then growing again over subsequent steps.
    transformed_env.reset()
    rollout_and_report(20, "phase 2: right after transformed_env.reset()")

    # Phase 3: force a persistent, stark difference in observed depth for two
    # envs (one "wall right in front" = all-zero depth, one "wide open" =
    # all-max depth), leave a third env's depth untouched (natural sim), and
    # check whether their hidden_norm trajectories diverge over a longer
    # horizon. If they stay near-identical despite radically different input
    # for 100+ consecutive steps, the GRU's gates are effectively saturated
    # (hidden state barely depends on input), not genuinely encoding
    # per-env information -- consistent with, though not proof of, the
    # near-identical curves seen in phase 1/2.
    depth_key = ("agents", "observation", "depth")
    close_env, far_env, natural_env = 0, 1, 2
    n_steps = 120

    td = transformed_env.reset()
    norms = {i: [] for i in (close_env, far_env, natural_env)}
    depth_means = {i: [] for i in (close_env, far_env, natural_env)}
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        for t in range(n_steps):
            depth = td.get(depth_key).clone()
            depth[close_env] = 0.0   # normalized depth ~0 => obstacle right at sensor
            depth[far_env] = 1.0     # normalized depth ~1 => nothing in range
            td.set(depth_key, depth)
            for i in (close_env, far_env, natural_env):
                depth_means[i].append(round(depth[i].mean().item(), 3))

            td = policy(td)
            td = transformed_env.step(td)
            hidden = td.get(("next", *hidden_out_key[1:])) if isinstance(hidden_out_key, tuple) else td.get(("next", hidden_out_key))
            for i in (close_env, far_env, natural_env):
                norms[i].append(round(hidden[i].flatten().norm().item(), 3))
            td = step_mdp(td)

    print("\n===== phase 3: forced-divergent depth input, 120 steps =====")
    for i, label in [(close_env, "close_env (depth forced=0)"),
                      (far_env, "far_env (depth forced=1)"),
                      (natural_env, "natural_env (unmodified)")]:
        print(f"  {label}: depth_mean[:5]={depth_means[i][:5]} depth_mean[-5:]={depth_means[i][-5:]}")
        print(f"  {label}: hidden_norm[:10]={norms[i][:10]}")
        print(f"  {label}: hidden_norm[-10:]={norms[i][-10:]}")
    max_diff_early = max(abs(norms[close_env][k] - norms[far_env][k]) for k in range(10))
    max_diff_late = max(abs(norms[close_env][k] - norms[far_env][k]) for k in range(n_steps - 10, n_steps))
    print(f"  |close_env - far_env| hidden_norm: max over first 10 steps = {max_diff_early:.3f}, "
          f"max over last 10 steps = {max_diff_late:.3f}")

    # Phase 4: the decisive test. close_env and far_env now carry 120 steps
    # of maximally different history (from phase 3). At this final step,
    # force EVERY observation key of far_env to exactly equal close_env's
    # (so the two envs' current-timestep input to the policy is byte-for-byte
    # identical), then run the policy once and compare the actor's output
    # distribution (alpha/beta) between the two envs. Their hidden states
    # differ (different history); their current observation does not. If
    # alpha/beta come out identical (or near machine precision) despite this,
    # the memory pathway has zero causal effect on the action distribution --
    # a direct behavioral proof, independent of the hidden_norm heuristics in
    # phase 1-3.
    obs_keys = [
        ("agents", "observation", "depth"),
        ("agents", "observation", "direction"),
        ("agents", "observation", "state"),
        ("agents", "observation", "lidar_fov"),
        ("agents", "observation", "dynamic_obstacle"),
    ]
    print("\n===== phase 4: identical current obs, divergent history -> compare action dist =====")
    for k in obs_keys:
        val = td.get(k).clone()
        val[far_env] = val[close_env]
        td.set(k, val)
        assert torch.equal(td.get(k)[close_env], td.get(k)[far_env])
    print("  confirmed: close_env/far_env observation keys are now byte-identical")
    print(f"  hidden_norm going into this step: close_env={norms[close_env][-1]:.3f} "
          f"far_env={norms[far_env][-1]:.3f}  (still different -- different history)")

    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td_out = policy(td.clone())
    print("  td_out keys:", sorted(str(k) for k in td_out.keys(True, True)
                                    if isinstance(k, str) or (isinstance(k, tuple) and len(k) == 1)))

    for dist_key in ("alpha", "beta", "loc", "scale"):
        if dist_key in td_out.keys():
            v = td_out.get(dist_key)
            diff = (v[close_env] - v[far_env]).abs()
            print(f"  {dist_key}: close_env={v[close_env].tolist()} far_env={v[far_env].tolist()} "
                  f"max_abs_diff={diff.max().item():.6g}")

    action_key = ("agents", "action")
    if action_key in td_out.keys(True, True):
        a = td_out.get(action_key)
        adiff = (a[close_env] - a[far_env]).abs()
        print(f"  sampled action: close_env={a[close_env].tolist()} far_env={a[far_env].tolist()} "
              f"max_abs_diff={adiff.max().item():.6g}")

    print("\n[probe] done.")
    sim_app.close()


if __name__ == "__main__":
    main()
