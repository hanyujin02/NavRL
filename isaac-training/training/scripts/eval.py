"""
Closed-loop NavRL evaluation with optional depth-image noise injection.

Runs the policy in Isaac Sim for one full episode length per noise condition,
logs all results to wandb, and saves a video per condition.

Usage
-----
  # Clean evaluation only:
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt

  # Clean + all noise conditions defined in cfg.eval.noise_conditions:
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt \
      eval.run_noise=true

  # Single noise condition override:
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt \
      eval.run_noise=true \
      "eval.noise_conditions=[{name: gaussian_high, type: gaussian, sigma: 0.10}]"
"""
import os
import datetime
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import isaacsim  # must come before omni.isaac.kit on Isaac Sim 4.x
from omni.isaac.kit import SimulationApp

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="eval_noise", version_base=None)
def main(cfg: DictConfig):
    # ── Simulation App ────────────────────────────────────────────────────────
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 0})

    from ppo import PPO
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from torchrl.envs.transforms import TransformedEnv, Compose
    from torchrl.envs.utils import ExplorationType
    from utils import evaluate

    # ── Wandb ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        mode=cfg.wandb.mode,
    )

    # ── Environment ───────────────────────────────────────────────────────────
    _env_module = __import__(getattr(cfg, "env_script", "env_depth"))
    env = _env_module.NavigationEnv(cfg)

    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
    transformed_env.set_seed(cfg.seed)

    # ── Policy ────────────────────────────────────────────────────────────────
    policy = PPO(cfg.algo, transformed_env.observation_spec,
                 transformed_env.action_spec, cfg.device)

    if not getattr(cfg, "checkpoint", None):
        raise ValueError("Set checkpoint=<path> on the command line.")

    state_dict = torch.load(cfg.checkpoint, map_location=cfg.device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[eval] Skipped keys (training-only): {unexpected[:4]}"
              + (" …" if len(unexpected) > 4 else ""))
    print(f"[eval] Loaded checkpoint: {cfg.checkpoint}")

    # ── Build noise conditions list ───────────────────────────────────────────
    # Always run clean first; append configured noise conditions if enabled.
    conditions = [{"name": "clean", "type": None}]
    if cfg.eval.get("run_noise", True):
        for nc in cfg.eval.get("noise_conditions", []):
            conditions.append(OmegaConf.to_container(nc, resolve=True))

    # ── Evaluate each condition ───────────────────────────────────────────────
    for cond in conditions:
        name      = cond["name"]
        noise_cfg = {k: v for k, v in cond.items() if k != "name"} or None
        print(f"\n[eval] === condition: {name} ===")

        env.eval()
        eval_info = evaluate(
            env=transformed_env,
            policy=policy,
            cfg=cfg,
            seed=cfg.seed,
            exploration_type=ExplorationType.MEAN,
            noise_cfg=noise_cfg,
        )
        env.train()
        env.reset()
        torch.cuda.empty_cache()

        # Log with condition prefix, e.g. "clean/stats.reach_goal"
        prefixed = {f"{name}/{k}": v for k, v in eval_info.items()}
        run.log(prefixed)
        print(f"[eval] {name}  "
              + "  ".join(f"{k.split('.')[-1]}={v:.3f}"
                          for k, v in eval_info.items()
                          if isinstance(v, float)))

    # ── Done ──────────────────────────────────────────────────────────────────
    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
