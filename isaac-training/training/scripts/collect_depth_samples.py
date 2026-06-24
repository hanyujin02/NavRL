"""Run one env for a handful of steps and save depth + BEV sample images.

Usage (from the isaac-training directory, inside the NavRL conda env):
    python training/scripts/collect_depth_samples.py \
        env.num_envs=1 headless=True

Images land in ./depth_samples/ relative to the cwd.
"""

import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="train_depth", version_base=None)
def main(cfg: DictConfig):
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 0})

    import env_depth
    import torch as _torch

    class FixedSpawnEnv(env_depth.NavigationEnv):
        """Spawn all drones at (0, 0, 2) regardless of env_id."""
        def _reset_idx(self, env_ids):
            # Call parent to handle target/lidar/stats resets, then override pose.
            super()._reset_idx(env_ids)
            pos = _torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 2] = 2.0  # 2 m above ground
            from omni_drones.utils.torch import euler_to_quaternion
            rpy = _torch.zeros(len(env_ids), 1, 3, device=self.device)
            self.drone.set_world_poses(pos, euler_to_quaternion(rpy), env_ids)
            self.drone.set_velocities(self.init_vels[env_ids], env_ids)

    env = FixedSpawnEnv(cfg)

    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from torchrl.envs.transforms import TransformedEnv, Compose

    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    tenv = TransformedEnv(env, Compose(vel_transform))
    tenv.set_seed(0)

    td = tenv.reset()
    print("[collect] reset done — stepping 20 times to trigger image save ...")

    action_spec = tenv.action_spec["agents", "action"]
    for step in range(20):
        td["agents", "action"] = torch.zeros(action_spec.shape, device=cfg.device)
        td = tenv.step(td)["next"]
        if env._bev_saved or env._depth_saved:
            print(f"[collect] Images saved after step {step + 1}. Done.")
            break

    sim_app.close()


if __name__ == "__main__":
    main()
