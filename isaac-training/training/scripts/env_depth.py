import torch
import einops
import numpy as np
import heapq
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, RayCasterCamera, RayCasterCameraCfg, patterns
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
import os
import re
from pathlib import Path
from PIL import Image

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # Depth camera params
        self.depth_range = cfg.sensor.depth_range
        self.depth_height = cfg.sensor.depth_height
        self.depth_width = cfg.sensor.depth_width

        # FOV mask — must be computed before super().__init__() because _set_specs() uses it
        import math
        hfov_half_deg = math.degrees(
            math.atan2(cfg.sensor.horizontal_aperture / 2.0, cfg.sensor.focal_length)
        )
        lidar_hbeams = int(360 / cfg.sensor.lidar_hres)
        h_angles = torch.arange(0, 360, cfg.sensor.lidar_hres, dtype=torch.float32)
        h_angles_signed = ((h_angles + 180.0) % 360.0) - 180.0
        self._lidar_fov_mask_h = h_angles_signed.abs() <= hfov_half_deg  # (lidar_hbeams,)
        self._lidar_fov_num_rays = int(self._lidar_fov_mask_h.sum().item()) * cfg.sensor.lidar_vbeams

        # BEV mode: back-project depth to top-down grid for bev_* encoders
        _enc_type = getattr(cfg.algo.feature_extractor, "encoder_type", "scratch")
        self.use_bev = _enc_type.startswith("bev_")
        if self.use_bev:
            self.bev_grid_size  = getattr(cfg.algo.feature_extractor, "bev_grid_size",  50)
            self.bev_map_range  = getattr(cfg.algo.feature_extractor, "bev_map_range",  5.0)
            self.bev_height_clip = getattr(cfg.algo.feature_extractor, "bev_height_clip", 3.0)
            self.bev_no_height_clip = getattr(cfg.algo.feature_extractor, "bev_no_height_clip", False)
            self.bev_cell_size  = self.bev_map_range / (self.bev_grid_size / 2.0)
            # Per-pixel direction vectors for pinhole back-projection
            # fx = focal_length * width / horizontal_aperture (standard pinhole formula)
            # fy = fx because Isaac Sim uses square pixels by default
            fx = cfg.sensor.focal_length * cfg.sensor.depth_width / cfg.sensor.horizontal_aperture
            fy = fx
            cx, cy = cfg.sensor.depth_width / 2.0, cfg.sensor.depth_height / 2.0
            H, W = cfg.sensor.depth_height, cfg.sensor.depth_width
            uu = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
            vv = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
            # x_dir: lateral (right) displacement per unit depth; y_dir: downward per unit depth
            self.bev_x_dir = ((uu - cx) / fx).reshape(-1).to(cfg.device)  # (H*W,)
            self.bev_y_dir = ((vv - cy) / fy).reshape(-1).to(cfg.device)  # (H*W,)

        super().__init__(cfg, cfg.headless)

        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # Depth Camera Initialization (raycasting-based, no render pipeline)
        depth_camera_cfg = RayCasterCameraCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            attach_yaw_only=False,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=cfg.sensor.focal_length,
                horizontal_aperture=cfg.sensor.horizontal_aperture,
                height=self.depth_height,
                width=self.depth_width,
            ),
            data_types=["distance_to_image_plane"],
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.depth_camera = RayCasterCamera(depth_camera_cfg)
        self.depth_camera._initialize_impl()

        # LiDAR sensor — used only for safety reward (not observation)
        self.lidar_range = cfg.sensor.lidar_range
        lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        lidar_vbeams = cfg.sensor.lidar_vbeams
        lidar_hbeams = int(360 / cfg.sensor.lidar_hres)
        self.lidar_resolution = (lidar_hbeams, lidar_vbeams)

        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=cfg.sensor.lidar_hres,
                vertical_ray_angles=torch.linspace(*lidar_vfov, lidar_vbeams),
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()

        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1, 3)
            self.prev_drone_pos = torch.zeros(self.num_envs, 1, 3)
        self._depth_saved = False
        self._bev_saved = False

        self.use_dijkstra_reward = getattr(cfg.env, "use_dijkstra_reward", False)
        if self.use_dijkstra_reward:
            self._init_dijkstra_reward()
        self.use_cbf_safety_reward = getattr(cfg.env, "use_cbf_safety_reward", False)
        if self.use_cbf_safety_reward:
            self._init_cbf_safety_reward()

        # Data collection (enabled via cfg.collect_data)
        if getattr(cfg, "collect_data", False):
            self.collect_timestep = 0
            self._collect_run_dir = None
            self._col_pos       = []
            self._col_vel       = []
            self._col_target    = []
            self._col_depth     = []   # raw metric depth (N, 1, H, W)
            self._col_bev       = []   # BEV grid (N, 2, G, G); empty list if not use_bev
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        self.static_obstacle_cfg = HfDiscreteObstaclesTerrainCfg(
            horizontal_scale=0.1,
            vertical_scale=0.1,
            border_width=0.0,
            num_obstacles=self.cfg.env.num_obstacles,
            obstacle_height_mode="range",
            obstacle_width_range=(0.4, 1.1),
            obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
            obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
            platform_width=0.0,
        )

        # Patch the terrain function BEFORE terrain_cfg is created so the
        # deep-copy inside TerrainImporterCfg/__post_init__ propagates our wrapper.
        _orig_fn = self.static_obstacle_cfg.function

        def _capture_terrain(difficulty, cfg):
            self._terrain_height_field = _orig_fn.__wrapped__(difficulty, cfg).copy()
            return _orig_fn(difficulty, cfg)

        self.static_obstacle_cfg.function = _capture_terrain

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2),
                border_width=5.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": self.static_obstacle_cfg,
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)
        self.static_obstacle_cfg.function = _orig_fn

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    def _init_dijkstra_reward(self):
        self.dijkstra_grid_resolution = float(getattr(self.cfg.env, "dijkstra_grid_resolution", 0.4))
        self.dijkstra_reward_scale = float(getattr(self.cfg.env, "dijkstra_reward_scale", 1.0))
        self.dijkstra_reward_clip = float(getattr(self.cfg.env, "dijkstra_reward_clip", 2.0))
        self.dijkstra_extent = float(getattr(self.cfg.env, "dijkstra_map_extent", max(self.map_range[0], self.map_range[1])))
        self.dijkstra_shape = (
            int(np.ceil((self.dijkstra_extent * 2.0) / self.dijkstra_grid_resolution)),
            int(np.ceil((self.dijkstra_extent * 2.0) / self.dijkstra_grid_resolution)),
        )
        self.dijkstra_occupancy = self._build_dijkstra_occupancy(
            inflate_radius=float(getattr(self.cfg.env, "dijkstra_inflate_radius", 0.3))
        )
        self.dijkstra_potential = torch.zeros(
            (self.num_envs, *self.dijkstra_shape), dtype=torch.float32, device=self.device
        )
        print(
            "[NavRL] Dijkstra reward enabled: "
            f"grid={self.dijkstra_shape}, extent=+/-{self.dijkstra_extent:.1f}m, "
            f"resolution={self.dijkstra_grid_resolution:.2f}m"
        )

    def _init_cbf_safety_reward(self):
        if not hasattr(self, "dijkstra_shape"):
            self.dijkstra_grid_resolution = float(getattr(self.cfg.env, "dijkstra_grid_resolution", 0.4))
            self.dijkstra_extent = float(getattr(self.cfg.env, "dijkstra_map_extent", max(self.map_range[0], self.map_range[1])))
            self.dijkstra_shape = (
                int(np.ceil((self.dijkstra_extent * 2.0) / self.dijkstra_grid_resolution)),
                int(np.ceil((self.dijkstra_extent * 2.0) / self.dijkstra_grid_resolution)),
            )

        self.cbf_safe_margin = float(getattr(self.cfg.env, "cbf_safe_margin", 0.6))
        self.cbf_gamma = float(getattr(self.cfg.env, "cbf_gamma", 2.0))
        self.cbf_reward_scale = float(getattr(self.cfg.env, "cbf_reward_scale", 1.0))
        self.cbf_reward_clip = float(getattr(self.cfg.env, "cbf_reward_clip", 5.0))
        cbf_inflate_radius = float(getattr(self.cfg.env, "cbf_obstacle_inflate_radius", 0.0))
        self.cbf_occupancy = self._build_dijkstra_occupancy(inflate_radius=cbf_inflate_radius)

        dist = self._compute_obstacle_distance_field_np()
        grad_x, grad_y = np.gradient(dist, self.dijkstra_grid_resolution, self.dijkstra_grid_resolution)
        grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2).clip(min=1e-6)
        grad = np.stack([grad_x / grad_norm, grad_y / grad_norm], axis=-1).astype(np.float32)

        self.cbf_distance_field = torch.as_tensor(dist, dtype=torch.float32, device=self.device)
        self.cbf_grad_field = torch.as_tensor(grad, dtype=torch.float32, device=self.device)
        print(
            "[NavRL] CBF safety reward enabled: "
            f"margin={self.cbf_safe_margin:.2f}m, gamma={self.cbf_gamma:.2f}"
        )

    def _build_dijkstra_occupancy(self, inflate_radius: float):
        # Use the heightfield captured during _design_scene — guaranteed to match
        # the simulator's obstacle layout regardless of the TerrainGenerator's
        # internal random call sequence.
        height_m = self._terrain_height_field.astype(np.float32) * self.static_obstacle_cfg.vertical_scale
        blocked_hi = height_m > float(getattr(self.cfg.env, "dijkstra_obstacle_height", 0.2))

        grid_h, grid_w = self.dijkstra_shape
        hi_h, hi_w = blocked_hi.shape
        xs_world = (np.arange(grid_h, dtype=np.float32) + 0.5) * self.dijkstra_grid_resolution - self.dijkstra_extent
        ys_world = (np.arange(grid_w, dtype=np.float32) + 0.5) * self.dijkstra_grid_resolution - self.dijkstra_extent
        x_inside = (xs_world >= -self.map_range[0]) & (xs_world <= self.map_range[0])
        y_inside = (ys_world >= -self.map_range[1]) & (ys_world <= self.map_range[1])
        x_idx = np.clip(((xs_world + self.map_range[0]) / (2.0 * self.map_range[0]) * hi_h).astype(np.int64), 0, hi_h - 1)
        y_idx = np.clip(((ys_world + self.map_range[1]) / (2.0 * self.map_range[1]) * hi_w).astype(np.int64), 0, hi_w - 1)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        inside = np.outer(x_inside, y_inside)
        blocked[inside] = blocked_hi[np.ix_(x_idx, y_idx)][inside]

        inflate_cells = int(np.ceil(inflate_radius / self.dijkstra_grid_resolution))
        if inflate_cells > 0:
            padded = np.pad(blocked, inflate_cells, mode="constant", constant_values=False)
            inflated = np.zeros_like(blocked, dtype=bool)
            for dx in range(-inflate_cells, inflate_cells + 1):
                for dy in range(-inflate_cells, inflate_cells + 1):
                    if dx * dx + dy * dy <= inflate_cells * inflate_cells:
                        x0 = inflate_cells + dx
                        y0 = inflate_cells + dy
                        inflated |= padded[x0 : x0 + grid_h, y0 : y0 + grid_w]
            blocked = inflated
        return blocked

    def _world_xy_to_dijkstra_index_np(self, xy):
        ix = np.floor((xy[..., 0] + self.dijkstra_extent) / self.dijkstra_grid_resolution).astype(np.int64)
        iy = np.floor((xy[..., 1] + self.dijkstra_extent) / self.dijkstra_grid_resolution).astype(np.int64)
        ix = np.clip(ix, 0, self.dijkstra_shape[0] - 1)
        iy = np.clip(iy, 0, self.dijkstra_shape[1] - 1)
        return ix, iy

    def _nearest_free_dijkstra_cell(self, start):
        sx, sy = start
        if not self.dijkstra_occupancy[sx, sy]:
            return sx, sy
        max_radius = max(self.dijkstra_shape)
        for radius in range(1, max_radius):
            x0, x1 = max(0, sx - radius), min(self.dijkstra_shape[0] - 1, sx + radius)
            y0, y1 = max(0, sy - radius), min(self.dijkstra_shape[1] - 1, sy + radius)
            candidates = []
            for x in range(x0, x1 + 1):
                candidates.append((x, y0))
                candidates.append((x, y1))
            for y in range(y0 + 1, y1):
                candidates.append((x0, y))
                candidates.append((x1, y))
            for x, y in candidates:
                if not self.dijkstra_occupancy[x, y]:
                    return x, y
        return sx, sy

    def _compute_dijkstra_field_np(self, target_xy):
        from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
        grid_h, grid_w = self.dijkstra_shape
        target_idx = self._world_xy_to_dijkstra_index_np(target_xy.reshape(1, 2))
        tx, ty = self._nearest_free_dijkstra_cell((int(target_idx[0][0]), int(target_idx[1][0])))

        # Build sparse adjacency graph (once could be cached, but occupancy is static so fine here)
        n = grid_h * grid_w
        free = ~self.dijkstra_occupancy  # (H, W) bool
        rows, cols, weights = [], [], []
        neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
        )
        xs, ys = np.where(free)
        for dx, dy, w in neighbors:
            nx_, ny_ = xs + dx, ys + dy
            valid = (nx_ >= 0) & (nx_ < grid_h) & (ny_ >= 0) & (ny_ < grid_w) & free[nx_.clip(0, grid_h-1), ny_.clip(0, grid_w-1)]
            src = xs[valid] * grid_w + ys[valid]
            dst = nx_[valid] * grid_w + ny_[valid]
            rows.extend(src); cols.extend(dst)
            weights.extend([w * self.dijkstra_grid_resolution] * int(valid.sum()))

        from scipy.sparse import csr_matrix
        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        target_node = tx * grid_w + ty
        dist_flat = scipy_dijkstra(graph, indices=target_node, directed=False)
        dist = dist_flat.reshape(grid_h, grid_w).astype(np.float32)

        finite = np.isfinite(dist)
        if not finite.all():
            max_finite = float(dist[finite].max()) if finite.any() else 0.0
            xs_ = (np.arange(grid_h, dtype=np.float32) + 0.5) * self.dijkstra_grid_resolution - self.dijkstra_extent
            ys_ = (np.arange(grid_w, dtype=np.float32) + 0.5) * self.dijkstra_grid_resolution - self.dijkstra_extent
            xx, yy = np.meshgrid(xs_, ys_, indexing="ij")
            fallback = np.sqrt((xx - target_xy[0]) ** 2 + (yy - target_xy[1]) ** 2)
            dist[~finite] = max_finite + fallback[~finite]
        return dist

    def _compute_obstacle_distance_field_np(self):
        grid_h, grid_w = self.dijkstra_shape
        dist = np.full((grid_h, grid_w), np.inf, dtype=np.float32)
        heap = []
        obstacle_cells = np.argwhere(self.cbf_occupancy)
        if obstacle_cells.size == 0:
            return np.full((grid_h, grid_w), self.dijkstra_extent * 2.0, dtype=np.float32)

        for x, y in obstacle_cells:
            dist[x, y] = 0.0
            heapq.heappush(heap, (0.0, int(x), int(y)))

        neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0)),
        )
        while heap:
            curr, x, y = heapq.heappop(heap)
            if curr != dist[x, y]:
                continue
            for dx, dy, step in neighbors:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= grid_h or ny < 0 or ny >= grid_w:
                    continue
                new_dist = curr + step * self.dijkstra_grid_resolution
                if new_dist < dist[nx, ny]:
                    dist[nx, ny] = new_dist
                    heapq.heappush(heap, (new_dist, nx, ny))
        return dist

    def _update_dijkstra_fields(self, env_ids: torch.Tensor):
        if not self.use_dijkstra_reward:
            return
        if env_ids.numel() == 0:
            return
        env_ids_cpu = env_ids.detach().cpu().numpy().astype(np.int64)
        target_xy = self.target_pos[env_ids, 0, :2].detach().cpu().numpy()
        fields = [self._compute_dijkstra_field_np(target_xy[i]) for i in range(len(env_ids_cpu))]
        fields_t = torch.as_tensor(np.stack(fields, axis=0), dtype=torch.float32, device=self.device)
        self.dijkstra_potential[env_ids] = fields_t

    def _sample_dijkstra_potential(self, pos: torch.Tensor) -> torch.Tensor:
        xy = pos[..., :2].squeeze(1)
        ix = torch.floor((xy[:, 0] + self.dijkstra_extent) / self.dijkstra_grid_resolution).long()
        iy = torch.floor((xy[:, 1] + self.dijkstra_extent) / self.dijkstra_grid_resolution).long()
        in_bounds = (
            (ix >= 0) & (ix < self.dijkstra_shape[0]) &
            (iy >= 0) & (iy < self.dijkstra_shape[1])
        )
        ix = ix.clamp(0, self.dijkstra_shape[0] - 1)
        iy = iy.clamp(0, self.dijkstra_shape[1] - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        potential = self.dijkstra_potential[env_idx, ix, iy].unsqueeze(-1)
        euclidean = (self.target_pos - pos).norm(dim=-1)
        return torch.where(in_bounds.unsqueeze(-1), potential, euclidean)

    def _sample_cbf_distance_and_grad(self, pos: torch.Tensor):
        xy = pos[..., :2].squeeze(1)
        ix = torch.floor((xy[:, 0] + self.dijkstra_extent) / self.dijkstra_grid_resolution).long()
        iy = torch.floor((xy[:, 1] + self.dijkstra_extent) / self.dijkstra_grid_resolution).long()
        in_bounds = (
            (ix >= 0) & (ix < self.dijkstra_shape[0]) &
            (iy >= 0) & (iy < self.dijkstra_shape[1])
        )
        ix = ix.clamp(0, self.dijkstra_shape[0] - 1)
        iy = iy.clamp(0, self.dijkstra_shape[1] - 1)
        distance = self.cbf_distance_field[ix, iy].unsqueeze(-1)
        grad = self.cbf_grad_field[ix, iy]
        far_distance = torch.full_like(distance, self.dijkstra_extent * 2.0)
        zero_grad = torch.zeros_like(grad)
        return (
            torch.where(in_bounds.unsqueeze(-1), distance, far_distance),
            torch.where(in_bounds.unsqueeze(-1), grad, zero_grad),
        )


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        if self.use_bev:
            img_spec_key = "bev"
            img_spec_val = UnboundedContinuousTensorSpec(
                (2, self.bev_grid_size, self.bev_grid_size), device=self.device
            )
        else:
            img_spec_key = "depth"
            img_spec_val = UnboundedContinuousTensorSpec(
                (1, self.depth_height, self.depth_width), device=self.device
            )

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    img_spec_key: img_spec_val,
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                    "lidar_fov": UnboundedContinuousTensorSpec((1, self._lidar_fov_num_rays), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.prev_drone_pos[env_ids] = pos
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self._update_dijkstra_fields(env_ids)

        self.stats[env_ids] = 0.  
        
    # ------------------------------------------------------------------ data collection
    def _collect_init_dir(self):
        """Create a timestamped output directory on the first collection step."""
        base = Path(getattr(self.cfg, "data_path", "./collected_data"))
        base.mkdir(parents=True, exist_ok=True)
        # find next free index
        rx = re.compile(r"^run_(\d{3})$")
        max_idx = 0
        for p in base.iterdir():
            m = rx.match(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        run_dir = base / f"run_{max_idx + 1:03d}"
        run_dir.mkdir()
        print(f"[NavRL] Data collection → {run_dir}")
        return run_dir

    def _collect_step(self, depth_data: torch.Tensor, img_obs):
        """Append one snapshot to the in-memory buffers.

        depth_data : (N, 1, H, W)  — raw metric depth, same tensor as the encoder input
        img_obs    : (N, 2, G, G)  — BEV grid when use_bev=True, else None
        """
        if self._collect_run_dir is None:
            self._collect_run_dir = self._collect_init_dir()

        self._col_pos.append(self.root_state[..., :3].detach().cpu().numpy())      # (N, 1, 3)
        self._col_vel.append(self.root_state[..., 7:10].detach().cpu().numpy())    # (N, 1, 3)
        self._col_target.append(self.target_pos.detach().cpu().numpy())            # (N, 1, 3)
        self._col_depth.append(depth_data.detach().cpu().numpy())                  # (N, 1, H, W)
        if img_obs is not None:
            self._col_bev.append(img_obs.detach().cpu().numpy())                   # (N, 2, G, G)

    def save_data(self):
        """Flush all collection buffers to disk as .npy files."""
        if not getattr(self.cfg, "collect_data", False) or self._collect_run_dir is None:
            return
        d = self._collect_run_dir
        np.save(d / "pos.npy",        np.array(self._col_pos))     # (T, N, 1, 3)
        np.save(d / "vel.npy",        np.array(self._col_vel))     # (T, N, 1, 3)
        np.save(d / "target_pos.npy", np.array(self._col_target))  # (T, N, 1, 3)
        np.save(d / "depth.npy",      np.array(self._col_depth))   # (T, N, 1, H, W)
        if self._col_bev:
            np.save(d / "bev.npy",    np.array(self._col_bev))     # (T, N, 2, G, G)
        print(f"[NavRL] Saved {len(self._col_pos)} steps to {d}")
    # --------------------------------------------------------------------------

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.depth_camera.update(self.dt)
        self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        # Sanitize physics state: Isaac Sim can produce NaN velocities/positions for one
        # step before the episode resets when a drone clips through geometry. Replacing
        # NaN with 0 prevents it from propagating into observations, rewards, and ValueNorm.
        self.root_state = self.drone.get_state(env_frame=False).nan_to_num(0.0)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: Depth image-------------------
        # RayCasterCamera output: (num_envs, H, W, 1) → (num_envs, 1, H, W)
        depth_data = self.depth_camera.data.output["distance_to_image_plane"]
        # RayCasterCamera stores distance_to_image_plane as (N, H, W) — no channel dim
        depth_data = depth_data.nan_to_num(nan=self.depth_range, posinf=self.depth_range)
        depth_data = depth_data.unsqueeze(1).clamp(min=0.0, max=self.depth_range)

        # Lidar scan for safety reward and collision — same formula as env_lidar.py
        # Depth camera is observation-only; reward/collision use LiDAR exclusively.
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .nan_to_num(nan=self.lidar_range, posinf=self.lidar_range)  # missed rays → max range
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        )

        # Encoder input: BEV or normalized depth
        if self.use_bev:
            # BEV encoders need raw metric depth (metres) — project to top-down grid
            z_drone = self.root_state[:, 0, 2]   # (N,) drone altitude above ground
            img_obs = self._depth_to_bev(depth_data, z_drone)   # (N, 2, G, G)

            if not self._bev_saved:
                try:
                    from hydra.utils import get_original_cwd
                    save_dir = Path(get_original_cwd()) / "depth_samples"
                except Exception:
                    save_dir = Path("depth_samples")
                save_dir.mkdir(exist_ok=True)
                n_save = min(4, self.num_envs)
                for i in range(n_save):
                    # Raw depth
                    raw = depth_data[i, 0].cpu().float().numpy()
                    raw_uint8 = (255 * (1.0 - raw / self.depth_range)).clip(0, 255).astype(np.uint8)
                    Image.fromarray(raw_uint8).save(save_dir / f"env{i}_depth_raw.png")
                    # BEV occupancy channel (map 1→white, -1→black)
                    occ = img_obs[i, 0].cpu().float().numpy()
                    occ_uint8 = ((occ + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(occ_uint8).save(save_dir / f"env{i}_bev_occ.png")
                    # BEV height channel (0…height_clip → 0…255, -1 → 0)
                    ht = img_obs[i, 1].cpu().float().numpy()
                    ht_max = float(ht[ht >= 0].max()) if (ht >= 0).any() else self.bev_height_clip
                    ht_norm_max = ht_max if self.bev_no_height_clip else self.bev_height_clip
                    ht_uint8 = (255 * (ht.clip(0, ht_norm_max) / max(ht_norm_max, 1e-6))).clip(0, 255).astype(np.uint8)
                    Image.fromarray(ht_uint8).save(save_dir / f"env{i}_bev_height.png")
                np.save(save_dir / "bev_batch.npy", img_obs[:n_save].cpu().float().numpy())
                print(f"[NavRL] Saved {n_save} depth+BEV samples to {save_dir.resolve()}")
                self._bev_saved = True
        else:
            # All depth encoders (cnn, vit, rep_*) receive depth normalised to [0, 1]
            # using the sensor maximum range (depth_range = 5.0 m).
            depth_norm = depth_data / self.depth_range   # (N, 1, H, W) in [0, 1]
            img_obs = depth_norm

            if not self._depth_saved:
                try:
                    from hydra.utils import get_original_cwd
                    save_dir = Path(get_original_cwd()) / "depth_samples"
                except Exception:
                    save_dir = Path("depth_samples")
                save_dir.mkdir(exist_ok=True)
                n_save = min(4, self.num_envs)
                for i in range(n_save):
                    raw = depth_data[i, 0].cpu().float().numpy()
                    raw_uint8 = (255 * (1.0 - raw / self.depth_range)).clip(0, 255).astype(np.uint8)
                    Image.fromarray(raw_uint8).save(save_dir / f"env{i}_raw.png")
                    norm = depth_norm[i, 0].cpu().float().numpy()
                    norm_uint8 = (255 * norm).clip(0, 255).astype(np.uint8)
                    Image.fromarray(norm_uint8).save(save_dir / f"env{i}_norm.png")
                print(f"[NavRL] Saved {n_save} depth samples to {save_dir.resolve()}")
                self._depth_saved = True

        # Data collection: capture depth + BEV at the configured frequency,
        # using the exact same tensors that are fed to the encoder during training.
        if getattr(self.cfg, "collect_data", False):
            save_interval = max(1, round((1.0 / self.cfg.save_freq) / self.cfg.sim.dt))
            if self.collect_timestep % save_interval == 0:
                self._collect_step(depth_data, img_obs if self.use_bev else None)
            self.collect_timestep += 1

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        
        # c. velocity in the goal frame (root_state already sanitized via nan_to_num at top)
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            
        # -----------------Network Input Final--------------
        img_key = "bev" if self.use_bev else "depth"

        # FOV-masked LiDAR: normalized distances [0,1] for beams within camera HFOV.
        # lidar_scan = lidar_range - distance, so scan/lidar_range gives obstacle proximity
        # (0 = nothing in range, 1 = obstacle touching sensor).
        fov_mask = self._lidar_fov_mask_h.to(self.device)
        lidar_fov = (self.lidar_scan[:, 0, fov_mask, :] / self.lidar_range
                     ).reshape(self.num_envs, 1, -1)  # (N, 1, num_fov_rays)

        obs = {
            "state": drone_state,
            img_key: img_obs,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states,
            "lidar_fov": lidar_fov,
        }


        # -----------------Reward Calculation-----------------
        # a. velocity reward for goal direction (sanitize vel in case physics gives NaN)
        vel_w_safe = self.drone.vel_w[..., :3].nan_to_num(0.0)
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (vel_w_safe * vel_direction).sum(-1)#.clip(max=2.0)

        # b. CBF-style static obstacle safety reward.
        # h = d_obstacle - margin; safe if h_dot + gamma * h >= 0.
        # The reward only penalizes violations, matching CBF reward shaping as a soft constraint.
        if self.use_cbf_safety_reward:
            cbf_dist, cbf_grad = self._sample_cbf_distance_and_grad(self.root_state[..., :3])
            cbf_h = cbf_dist - self.cbf_safe_margin
            cbf_hdot = (cbf_grad * vel_w_safe[..., :2].squeeze(1)).sum(dim=-1, keepdim=True)
            cbf_condition = cbf_hdot + self.cbf_gamma * cbf_h
            reward_safety_static = self.cbf_reward_scale * cbf_condition.clamp(
                min=-self.cbf_reward_clip, max=0.0
            )
        else:
            reward_safety_static = torch.log((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))

        # c. safety reward for dynamic obstacles
        if (self.cfg.env_dyn.num_obstacles != 0):
            if self.use_cbf_safety_reward:
                dyn_grad = -closest_dyn_obs_rpos / closest_dyn_obs_rpos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                dyn_rel_vel = vel_w_safe - closest_dyn_obs_vel
                dyn_h = closest_dyn_obs_distance_reward - self.cbf_safe_margin
                dyn_hdot = (dyn_grad * dyn_rel_vel).sum(dim=-1)
                dyn_condition = dyn_hdot + self.cbf_gamma * dyn_h
                reward_safety_dynamic = self.cbf_reward_scale * dyn_condition.clamp(
                    min=-self.cbf_reward_clip, max=0.0
                ).mean(dim=-1, keepdim=True)
            else:
                reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)

        # g. goal-approach reward: Dijkstra potential progress when enabled,
        # otherwise the original Euclidean distance progress.
        if self.use_dijkstra_reward:
            prev_potential = self._sample_dijkstra_potential(self.prev_drone_pos)
            curr_potential = self._sample_dijkstra_potential(self.root_state[..., :3])
            reward_goal = self.dijkstra_reward_scale * (prev_potential - curr_potential).clamp(
                -self.dijkstra_reward_clip, self.dijkstra_reward_clip
            )
        else:
            prev_distance = (self.target_pos - self.prev_drone_pos).norm(dim=-1)  # (N, 1)
            reward_goal = prev_distance - distance.squeeze(-1)                     # (N, 1)

        # d. smoothness reward for action smoothness
        penalty_smooth = (vel_w_safe - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]


        # f. Collision condition with its penalty
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (self.lidar_range - 0.3)
        collision = static_collision | dynamic_collision
        
        # Final reward calculation
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = reward_vel + reward_goal + 1. + reward_safety_static * 1.0 + reward_safety_dynamic * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0
        else:
            self.reward = reward_vel + reward_goal + 1. + reward_safety_static * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0


        # Terminal reward
        # self.reward[collision] -= 50. # collision

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = below_bound | above_bound | collision
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # update previous state for next-step calculations
        self.prev_drone_vel_w = vel_w_safe.clone()
        self.prev_drone_pos = self.root_state[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _depth_to_bev(self, depth: torch.Tensor, z_drone: torch.Tensor) -> torch.Tensor:
        """
        GPU-vectorised depth → BEV projection.

        depth   : (N, 1, H, W) raw metric depth in metres (clamped to depth_range)
        z_drone : (N,) drone altitude above ground in metres

        Returns : (N, 2, G, G)
                   channel 0 — occupancy:  1.0 if cell observed, -1.0 if not
                   channel 1 — height map: max obstacle height (m) in cell, -1.0 if unobserved
        """
        N = depth.shape[0]
        G = self.bev_grid_size
        half = G / 2.0
        cell_size = self.bev_cell_size

        d_flat = depth[:, 0].reshape(N, -1)          # (N, H*W)
        x3d = self.bev_x_dir.unsqueeze(0) * d_flat   # (N, H*W) lateral (right) displacement
        hgt = z_drone.unsqueeze(1) - self.bev_y_dir.unsqueeze(0) * d_flat  # (N, H*W) height above ground

        valid = (
            (d_flat > 0) &
            (d_flat <= self.bev_map_range) &
            (x3d.abs() <= self.bev_map_range) &
            (hgt >= 0.0) &
            (self.bev_no_height_clip | (hgt <= self.bev_height_clip))
        )  # (N, H*W)

        row = (half - d_flat / cell_size).floor().long().clamp(0, G - 1)   # forward → row
        col = (half + x3d / cell_size).floor().long().clamp(0, G - 1)      # lateral → col
        flat_idx = row * G + col                                             # (N, H*W)

        # Global flat index across all envs for scatter_reduce_
        env_offset = torch.arange(N, device=depth.device).unsqueeze(1) * (G * G)
        global_idx = (flat_idx + env_offset).reshape(-1)   # (N*H*W,)
        valid_flat  = valid.reshape(-1)
        hgt_flat    = hgt.reshape(-1)

        # Max-height per cell; unoccupied cells stay at -1.0
        bev_h_flat = torch.full((N * G * G,), -1.0, device=depth.device, dtype=torch.float32)
        # All valid heights are >= 0 > -1, so include_self=True never pulls a cell below 0
        bev_h_flat.scatter_reduce_(
            0, global_idx[valid_flat], hgt_flat[valid_flat], reduce="amax", include_self=True
        )
        bev_h = bev_h_flat.reshape(N, G, G)

        bev_occ = torch.where(bev_h >= 0.0,
                              bev_h.new_ones(1),
                              bev_h.new_full((1,), -1.0))
        return torch.stack([bev_occ, bev_h], dim=1)   # (N, 2, G, G)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
