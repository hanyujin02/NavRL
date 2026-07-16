import rospy
import numpy as np
import torch
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Quaternion, Vector3
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from ppo import PPO
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils import vec_to_new_frame
import math
import tf.transformations
import time
import threading
import os


class Navigation:
    def __init__(self, cfg):
        self.cfg = cfg

        # Depth camera
        self.depth_height = cfg.sensor.depth_height
        self.depth_width = cfg.sensor.depth_width
        self.depth_range = cfg.sensor.depth_range
        self.depth_image = None
        self.depth_received = False
        try:
            from cv_bridge import CvBridge
            self._cv_bridge = CvBridge()
        except ImportError:
            self._cv_bridge = None
            rospy.logwarn("[navRunner]: cv_bridge not found — depth images will not be processed.")
        depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
        rospy.loginfo(f"[navRunner]: Depth topic: {depth_topic}.")
        rospy.Subscriber(depth_topic, Image, self.depth_callback)

        self.goal = None
        self.goal_received = False
        self.target_dir = None
        self.stable_times = 0
        self.has_action = False

        self.height_control = rospy.get_param('~height_control', False)
        self.takeoff_height = rospy.get_param('~takeoff_height', 1.0)
        self.px4_control = rospy.get_param('rl/use_px4', True)

        self.odom_received = False
        if self.px4_control:
            odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
            self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
            self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
            self.action_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
            self.pose_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
            self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
            self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

            self.mavros_state = None
            self.offb_set_mode = SetModeRequest()
            self.offb_set_mode.custom_mode = 'OFFBOARD'
            self.arm_cmd = CommandBoolRequest()
            self.arm_cmd.value = True
        else:
            odom_topic = rospy.get_param('~odom_topic', '/CERLAB/quadcopter/odom')
            self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
            self.action_pub = rospy.Publisher("/CERLAB/quadcopter/cmd_vel", TwistStamped, queue_size=10)
            self.pose_pub = rospy.Publisher("/CERLAB/quadcopter/setpoint_pose", PoseStamped, queue_size=10)

        goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        self.goal_sub = rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback)
        self.cmd_vis_pub = rospy.Publisher("/rl_navigation/cmd", MarkerArray, queue_size=10)
        self.goal_vis_pub = rospy.Publisher("rl_navigation/goal", MarkerArray, queue_size=10)

        self.policy = self.init_model()
        self.policy.eval()

        # safety thread
        self.safety_stop = False
        safety_thread = threading.Thread(target=self.safety_check)
        safety_thread.start()

        self.takeoff()

    def init_model(self):
        # must match training env.attitude_obs: appends body roll/pitch to the
        # state input (8 -> 10 dims). Checkpoints are dim-specific.
        self.attitude_obs = bool(getattr(self.cfg.env, "attitude_obs", False))
        observation_dim = 10 if self.attitude_obs else 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.cfg.device),
                    "depth": UnboundedContinuousTensorSpec((1, self.depth_height, self.depth_width), device=self.cfg.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.cfg.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.cfg.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.cfg.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.cfg.device),
            })
        }).expand(1, action_dim).to(self.cfg.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.cfg.device)

        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = rospy.get_param('~checkpoint', 'checkpoint_34000(best).pt')
        missing, unexpected = policy.load_state_dict(
            torch.load(os.path.join(file_dir, checkpoint), map_location=self.cfg.device),
            strict=False
        )
        if missing:
            rospy.logwarn(f"[navRunner]: checkpoint missing keys: {missing}")
        if unexpected:
            rospy.loginfo(f"[navRunner]: checkpoint extra keys (ignored): {unexpected}")
        return policy

    def takeoff(self):
        takeoff_height = self.takeoff_height
        r = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.odom_received:
            print("[nav-ros]: Wait for robot odom...")
            r.sleep()

        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = self.odom.pose.pose.position.x
        takeoff_pose.pose.position.y = self.odom.pose.pose.position.y
        takeoff_pose.pose.position.z = takeoff_height
        takeoff_pose.pose.orientation = self.odom.pose.pose.orientation
        self.takeoff_pose = takeoff_pose
        if self.px4_control:
            pose = PoseStamped()
            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 2
            rate = rospy.Rate(20)
            for i in range(100):
                if rospy.is_shutdown():
                    break
                self.pose_pub.publish(pose)
                rate.sleep()
            last_req = rospy.Time.now()
        while not rospy.is_shutdown() and not (np.abs(self.odom.pose.pose.position.z - takeoff_height) <= 0.2):
            if self.px4_control:
                if self.mavros_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    if self.set_mode_client.call(self.offb_set_mode).mode_sent == True:
                        print("[nav-ros]: OFFBOARD enabled.")
                    last_req = rospy.Time.now()
                else:
                    if not self.mavros_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                        if self.arming_client.call(self.arm_cmd).success == True:
                            print("[nav-ros]: Vehicle armed.")
                        last_req = rospy.Time.now()
            self.pose_pub.publish(takeoff_pose)
            r.sleep()
        print("[nav-ros]: take off completed at height: ", takeoff_height)

    def safety_check(self):
        while not rospy.is_shutdown():
            if not self.safety_stop:
                input("[nav-ros]: Press Enter to STOP motion!\n")
                self.safety_stop = True
                self.stop_pose = PoseStamped()
                self.stop_pose.pose = self.odom.pose.pose
            else:
                input("[nav-ros]: Press Enter to CONTINUE motion!\n")
                self.safety_stop = False

    def depth_callback(self, msg):
        if self._cv_bridge is None:
            return
        import cv2
        if msg.encoding in ('32FC1', '32FC'):
            depth_np = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        else:  # 16UC1 — millimetres
            depth_np = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1').astype(np.float32) / 1000.0
        if depth_np.shape != (self.depth_height, self.depth_width):
            depth_np = cv2.resize(depth_np, (self.depth_width, self.depth_height),
                                  interpolation=cv2.INTER_NEAREST)
        depth_np = np.nan_to_num(depth_np, nan=self.depth_range, posinf=self.depth_range, neginf=0.0)
        depth_np = np.clip(depth_np, 0.0, self.depth_range)
        depth_t = torch.tensor(depth_np, dtype=torch.float32, device=self.cfg.device)
        self.depth_image = (depth_t / self.depth_range).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.depth_received = True

    def odom_callback(self, odom):
        self.odom = odom
        self.odom_received = True

    def state_callback(self, state):
        self.mavros_state = state

    def goal_callback(self, goal):
        if not self.odom_received:
            return
        self.goal = goal
        self.goal.pose.position.z = self.takeoff_pose.pose.position.z
        dir_x = self.goal.pose.position.x - self.odom.pose.pose.position.x
        dir_y = self.goal.pose.position.y - self.odom.pose.pose.position.y
        dir_z = self.goal.pose.position.z - self.odom.pose.pose.position.z
        self.target_dir = torch.tensor([dir_x, dir_y, dir_z], device=self.cfg.device)
        self.goal_received = True
        self.stable_times = 0
        rospy.loginfo(f"[nav-ros]: Goal received: ({self.goal.pose.position.x:.2f}, {self.goal.pose.position.y:.2f}, {self.goal.pose.position.z:.2f})")
        rospy.loginfo(f"[nav-ros]: Drone pos:     ({self.odom.pose.pose.position.x:.2f}, {self.odom.pose.pose.position.y:.2f}, {self.odom.pose.pose.position.z:.2f})")
        rospy.loginfo(f"[nav-ros]: Target dir:    ({dir_x:.2f}, {dir_y:.2f}, {dir_z:.2f})")
        rospy.loginfo(f"[nav-ros]: Takeoff z:     {self.takeoff_pose.pose.position.z:.2f}")

    def quaternion_to_rotation_matrix(self, quaternion):
        w = quaternion.w
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        xx, xy, xz = x**2, x*y, x*z
        yy, yz = y**2, y*z
        zz = z**2
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])

    def get_action(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor):
        rpos = goal - pos
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)

        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.
        # Guard: if goal is directly above/below (x,y identical), use current rpos direction
        if target_dir_2d.norm() < 1e-4:
            target_dir_2d = rpos.clone()
            target_dir_2d[2] = 0.
            if target_dir_2d.norm() < 1e-4:
                target_dir_2d = torch.tensor([1., 0., 0.], device=self.cfg.device)

        rpos_clipped = rpos / distance.clamp(1e-6)
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d).squeeze(0).squeeze(0)
        vel_g = vec_to_new_frame(vel, target_dir_2d).squeeze(0).squeeze(0)
        state_parts = [rpos_clipped_g, distance_2d, distance_z, vel_g]
        if self.attitude_obs:
            # body roll/pitch (rad) from odometry, matching training env.attitude_obs=true
            q = self.odom.pose.pose.orientation
            roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
            state_parts.append(torch.tensor([roll, pitch], dtype=rpos_clipped_g.dtype, device=self.cfg.device))
        drone_state = torch.cat(state_parts, dim=-1).unsqueeze(0)

        # Dynamic obstacles disabled (dyn_obs_num=0 in config); always zeros.
        dyn_obs_states = torch.zeros(
            1, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10,
            device=self.cfg.device
        )

        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "depth": self.depth_image,  # (1, 1, H, W) normalized [0, 1]
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states,
                })
            })
        })

        with set_exploration_type(ExplorationType.MEAN):
            output = self.policy(obs)
        return output["agents", "action"]

    def control_callback(self, event):
        if not self.odom_received:
            return

        if not self.goal_received or not self.depth_received:
            self.pose_pub.publish(self.takeoff_pose)
            return

        if self.safety_stop:
            self.pose_pub.publish(self.stop_pose)
            return

        goal_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        _, _, curr_angle = tf.transformations.euler_from_quaternion([
            self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w
        ])
        angle_diff = np.abs(goal_angle - curr_angle)
        if angle_diff > math.pi:
            angle_diff = np.abs(angle_diff - math.pi * 2)
        if angle_diff >= 0.1:
            pose_msg = PoseStamped()
            pose_msg.pose.position = self.odom.pose.pose.position  # position only (safe ref)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, goal_angle)
            pose_msg.pose.orientation.w = quaternion[3]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            self.pose_pub.publish(pose_msg)
            rospy.loginfo_throttle(1.0, f"[nav-ros]: Aligning yaw — curr: {np.degrees(curr_angle):.1f} deg, goal: {np.degrees(goal_angle):.1f} deg, diff: {np.degrees(angle_diff):.1f} deg")
            return
        else:
            self.stable_times += 1
            if self.stable_times <= 10:
                return

        pos = torch.tensor([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z
        ], device=self.cfg.device)
        goal = torch.tensor([
            self.goal.pose.position.x,
            self.goal.pose.position.y,
            self.goal.pose.position.z
        ], device=self.cfg.device)
        rot = self.quaternion_to_rotation_matrix(self.odom.pose.pose.orientation)
        vel_body = np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z
        ])
        vel_world = torch.tensor(rot @ vel_body, device=self.cfg.device, dtype=torch.float)

        rospy.loginfo_throttle(1.0, f"[nav-ros]: pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f})  dist={float((pos-goal).norm()):.2f}m  depth_ok={self.depth_received}")

        _t0 = time.time()
        cmd_vel_world = self.get_action(pos, vel_world, goal).squeeze(0).squeeze(0).detach().cpu().numpy()
        rospy.loginfo_throttle(1.0, f"[nav-ros]: policy inference {(time.time()-_t0)*1000:.1f} ms")
        self.cmd_vel_world = cmd_vel_world.copy()
        rospy.loginfo_throttle(1.0, f"[nav-ros]: cmd_vel_world=({cmd_vel_world[0]:.2f},{cmd_vel_world[1]:.2f},{cmd_vel_world[2]:.2f})")

        quat_no_tilt = tf.transformations.quaternion_from_euler(0, 0, curr_angle)
        quat_msg = Quaternion()
        quat_msg.w = quat_no_tilt[3]
        quat_msg.x = quat_no_tilt[0]
        quat_msg.y = quat_no_tilt[1]
        quat_msg.z = quat_no_tilt[2]
        rot_no_tilt = self.quaternion_to_rotation_matrix(quat_msg)
        cmd_vel_local = np.linalg.inv(rot_no_tilt) @ cmd_vel_world

        # Goal slowdown
        distance = (pos - goal).norm()
        if 0.3 < distance <= 3.:
            if np.linalg.norm(cmd_vel_local) != 0:
                cmd_vel_local = 0.5 * cmd_vel_local / np.linalg.norm(cmd_vel_local)
                cmd_vel_world = 0.5 * cmd_vel_world / np.linalg.norm(cmd_vel_world)
        elif distance <= 1.0:
            cmd_vel_local = cmd_vel_local * 0.
            cmd_vel_world = cmd_vel_world * 0.

        if self.px4_control:
            final_cmd_vel = PositionTarget()
            final_cmd_vel.coordinate_frame = final_cmd_vel.FRAME_LOCAL_NED
            final_cmd_vel.header.stamp = rospy.Time.now()
            final_cmd_vel.header.frame_id = "map"
            if self.height_control:
                final_cmd_vel.velocity.x = cmd_vel_world[0]
                final_cmd_vel.velocity.y = cmd_vel_world[1]
                final_cmd_vel.velocity.z = cmd_vel_world[2]
                final_cmd_vel.yaw = goal_angle
                final_cmd_vel.type_mask = (
                    final_cmd_vel.IGNORE_PX + final_cmd_vel.IGNORE_PY + final_cmd_vel.IGNORE_PZ +
                    final_cmd_vel.IGNORE_AFX + final_cmd_vel.IGNORE_AFY + final_cmd_vel.IGNORE_AFZ +
                    final_cmd_vel.IGNORE_YAW_RATE
                )
            else:
                final_cmd_vel.velocity.x = cmd_vel_world[0]
                final_cmd_vel.velocity.y = cmd_vel_world[1]
                final_cmd_vel.position.z = self.takeoff_pose.pose.position.z
                final_cmd_vel.yaw = goal_angle
                final_cmd_vel.type_mask = (
                    final_cmd_vel.IGNORE_PX + final_cmd_vel.IGNORE_PY + final_cmd_vel.IGNORE_VZ +
                    final_cmd_vel.IGNORE_AFX + final_cmd_vel.IGNORE_AFY + final_cmd_vel.IGNORE_AFZ +
                    final_cmd_vel.IGNORE_YAW_RATE
                )
        else:
            final_cmd_vel = TwistStamped()
            final_cmd_vel.header.stamp = rospy.Time.now()
            final_cmd_vel.twist.linear.x = cmd_vel_local[0]
            final_cmd_vel.twist.linear.y = cmd_vel_local[1]
            if self.height_control:
                final_cmd_vel.twist.linear.z = cmd_vel_world[2]
            else:
                final_cmd_vel.twist.linear.z = 0
        self.action_pub.publish(final_cmd_vel)
        self.has_action = True

    def run(self):
        rospy.Timer(rospy.Duration(0.05), self.control_callback)
        rospy.Timer(rospy.Duration(0.05), self.goal_vis_callback)
        rospy.Timer(rospy.Duration(0.05), self.cmd_vis_callback)

    def goal_vis_callback(self, event):
        if not self.goal_received:
            return
        msg = MarkerArray()
        goal_point = Marker()
        goal_point.header.frame_id = "map"
        goal_point.header.stamp = rospy.get_rostime()
        goal_point.ns = "goal_point"
        goal_point.id = 1
        goal_point.type = goal_point.SPHERE
        goal_point.action = goal_point.ADD
        goal_point.pose.position.x = self.goal.pose.position.x
        goal_point.pose.position.y = self.goal.pose.position.y
        goal_point.pose.position.z = self.goal.pose.position.z
        goal_point.lifetime = rospy.Time(0.1)
        goal_point.scale.x = 0.3
        goal_point.scale.y = 0.3
        goal_point.scale.z = 0.3
        goal_point.color.r = 1.0
        goal_point.color.b = 1.0
        goal_point.color.a = 1.0
        msg.markers.append(goal_point)
        self.goal_vis_pub.publish(msg)

    def cmd_vis_callback(self, event):
        if not self.has_action:
            return
        msg = MarkerArray()
        arrow = Marker()
        arrow.header.frame_id = "map"
        arrow.header.stamp = rospy.get_rostime()
        arrow.ns = "rl_action"
        arrow.id = 0
        arrow.type = arrow.ARROW
        arrow.action = arrow.ADD

        agent_pos = Point()
        agent_pos.x = self.odom.pose.pose.position.x
        agent_pos.y = self.odom.pose.pose.position.y
        agent_pos.z = self.odom.pose.pose.position.z

        vel_end = Point()
        vel_end.x = self.cmd_vel_world[0] + agent_pos.x
        vel_end.y = self.cmd_vel_world[1] + agent_pos.y
        vel_end.z = self.cmd_vel_world[2] + agent_pos.z

        arrow.points.append(agent_pos)
        arrow.points.append(vel_end)
        arrow.lifetime = rospy.Duration(0.1)
        arrow.scale.x = 0.06
        arrow.scale.y = 0.06
        arrow.scale.z = 0.06
        arrow.color.a = 1.0
        arrow.color.r = 1.0
        arrow.color.g = 0.0
        arrow.color.b = 0.0
        msg.markers.append(arrow)
        self.cmd_vis_pub.publish(msg)
