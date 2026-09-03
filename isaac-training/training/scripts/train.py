import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
import isaacsim  # must come before omni.isaac.kit on Isaac Sim 4.x
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
from utils import evaluate, make_batched_gru_primer
from torchrl.envs.utils import ExplorationType




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Use Wandb to monitor training
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    wandb_entity = cfg.wandb.entity if cfg.wandb.entity else None
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=wandb_entity,
            config=cfg_dict,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=wandb_entity,
            config=cfg_dict,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Save config files to the wandb run directory for reproducibility
    import shutil, glob as _glob, yaml as _yaml
    # 1. Save all raw cfg/*.yaml files
    for _yf in _glob.glob(os.path.join(FILE_PATH, "*.yaml")):
        shutil.copy(_yf, run.dir)
    # 2. Save the fully resolved config (all overrides merged in)
    with open(os.path.join(run.dir, "config_resolved.yaml"), "w") as _f:
        _yaml.dump(cfg_dict, _f, default_flow_style=False)
    wandb.save(os.path.join(run.dir, "*.yaml"), base_path=run.dir, policy="now")

    # Navigation Training Environment
    # env_script selects env_lidar (default) or env_depth
    _env_module = __import__(getattr(cfg, "env_script", "env_lidar"))
    env = _env_module.NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    # vel_gain_factor simulates a mismatched/imperfectly-tuned velocity controller
    # for sim-to-real robustness training (see cfg/disturbance.yaml). Not every
    # config that reaches this shared entrypoint defines `disturbance` (e.g.
    # train.yaml/env_lidar), so default to 0.0 (no-op) via getattr rather than
    # assuming the key exists.
    vel_gain_factor = getattr(getattr(cfg, "disturbance", None), "vel_gain_factor", 0.0)
    controller = LeePositionController(9.81, env.drone.params, vel_gain_factor).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(
        cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device,
        disturbance_cfg=getattr(cfg, "disturbance", None), sim_dt=cfg.sim.dt,
    )

    if getattr(cfg, "checkpoint", None):
        policy.load_state_dict(torch.load(cfg.checkpoint, map_location=cfg.device))
        print(f"[NavRL]: resumed from checkpoint: {cfg.checkpoint}")

    # network_type: gru needs the env to carry an "is_init" reset flag and a
    # "recurrent_state" hidden-state key across steps -- InitTracker sets the
    # former, the primer wires the latter (including zeroing it for envs whose
    # episode just ended). make_batched_gru_primer instead of the module's own
    # make_tensordict_primer(): IsaacEnv is batch-locked, so the spec needs the
    # num_envs leading dim (see that helper's docstring). No-op for cnn/transformer.
    if getattr(cfg.algo, "network_type", "cnn") == "gru":
        transformed_env.append_transform(InitTracker())
        transformed_env.append_transform(
            make_batched_gru_primer(policy.gru_module, cfg.env.num_envs, cfg.device)
        )
        # Critic's independent GRU (see ppo.py's asymmetric actor/critic
        # extractor) has its own hidden-state key ("critic_recurrent_state")
        # -- needs its own primer alongside the actor's above.
        transformed_env.append_transform(
            make_batched_gru_primer(policy.critic_gru_module, cfg.env.num_envs, cfg.device)
        )

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    # Best-checkpoint tracking: single eval points are noisy (adjacent evals can
    # swing 0.2-0.6 in reach_goal on this fixed hard-eval scenario — see 2026-07
    # log analysis), so "best" is judged on an EMA of eval reach_goal rather than
    # the raw single-point score, to avoid crowning a lucky outlier eval.
    best_eval_sr_ema = None
    best_eval_sr_raw = float("-inf")
    eval_sr_ema_alpha = float(getattr(cfg, "best_ckpt_ema_alpha", 0.3))

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            torch.cuda.empty_cache()
            if bool(getattr(cfg, "eval_video", True)):
                env.enable_render(True)
            env.eval()
            eval_info = evaluate(
                env=transformed_env,
                policy=policy,
                seed=cfg.seed,
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            )
            env.enable_render(not cfg.headless)
            env.train()
            env.reset()
            torch.cuda.empty_cache()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")

            # Track best checkpoint by EMA-smoothed eval reach_goal
            eval_sr = eval_info.get("eval/stats.reach_goal")
            if eval_sr is not None:
                best_eval_sr_ema = eval_sr if best_eval_sr_ema is None \
                    else eval_sr_ema_alpha * eval_sr + (1 - eval_sr_ema_alpha) * best_eval_sr_ema
                info["eval/stats.reach_goal_ema"] = best_eval_sr_ema
                if best_eval_sr_ema > best_eval_sr_raw:
                    best_eval_sr_raw = best_eval_sr_ema
                    best_ckpt_path = os.path.join(run.dir, "best.pt")
                    torch.save(policy.state_dict(), best_ckpt_path)
                    print(f"[NavRL]: new best checkpoint (eval SR ema={best_eval_sr_ema:.3f}) "
                          f"saved at training step {i}")

        # Update wand info
        run.log(info)


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)

    # Flush collected data to disk (no-op when collect_data: false)
    if getattr(cfg, "collect_data", False):
        env.save_data()

    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    