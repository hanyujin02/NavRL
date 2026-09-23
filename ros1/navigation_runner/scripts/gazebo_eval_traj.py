#!/usr/bin/env python3
"""Trajectory-recording variant of gazebo_eval.py: drives the already-running
navigation_node across the same num_trials start-goal legs (see gazebo_eval.py
for the success_rate/collision_rate eval this is based on), but additionally
records the drone's position + world-frame velocity during each leg and, at
the end, plots all recorded trajectories on top of the Gazebo world's obstacle
layout.

Usage (after `roslaunch uav_simulator start.launch` and
      `roslaunch navigation_runner navigation.launch` are both up):
    rosrun navigation_runner gazebo_eval_traj.py

Recording only happens inside a trial leg (from goal-published to
success/timeout) — the one-off, unscored reposition to trial 0's start is
never recorded. Trials otherwise chain directly from wherever the previous one
ended (see gazebo_eval.py's docstring), so there is no separate "goal -> next
start" transit to exclude: consecutive trial legs already connect head-to-tail.
"""
import sys
# Strip ROS remapping args (__name:=, __log:=, etc.) before Hydra parses sys.argv
sys.argv = [a for a in sys.argv if not a.startswith('__')]

import os
import re
import time
import json
import datetime
import numpy as np
import rospy
import rospkg
import hydra
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class GazeboTrajEval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.odom = None
        self.odom_sub = rospy.Subscriber(cfg.odom_topic, Odometry, self._odom_cb)
        self.goal_pub = rospy.Publisher(cfg.goal_topic, PoseStamped, queue_size=10)
        self.pcd = self._load_pcd(self._resolve_pcd_path())
        rospy.loginfo(f"[gazebo_eval_traj]: loaded {self.pcd.shape[0]} obstacle points for collision checks.")
        self.z = self._resolve_z()
        rospy.loginfo(f"[gazebo_eval_traj]: using z={self.z:.2f}m for all start/goal poses.")
        self.checkpoint_name = self._resolve_checkpoint_name()
        rospy.loginfo(f"[gazebo_eval_traj]: tagging outputs with checkpoint '{self.checkpoint_name}'.")
        self.trajectories = []

    def _resolve_z(self):
        if self.cfg.z is not None:
            return float(self.cfg.z)
        return float(rospy.get_param("/navigation_node/takeoff_height", 1.0))

    @staticmethod
    def _resolve_checkpoint_name():
        """Read navigation_node's running ~checkpoint param (set from
        navigation.launch's checkpoint arg) and turn it into a filesystem-safe
        tag for naming trajectory outputs — e.g.
        'cnn_reach_ae_deploy/reach_ae_400.pt' -> 'cnn_reach_ae_deploy_reach_ae_400'."""
        ckpt = rospy.get_param("/navigation_node/checkpoint", "unknown_ckpt")
        name = os.path.splitext(ckpt)[0]  # drop .pt, keep any subdirectory
        name = name.replace(os.sep, "_").replace("/", "_")
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def _resolve_pcd_path(self):
        if self.cfg.pcd_path:
            return self.cfg.pcd_path
        pkg_path = rospkg.RosPack().get_path("uav_simulator")
        return os.path.join(pkg_path, "worlds", "generated_env", "generated_env_static.pcd")

    def _resolve_world_path(self):
        if self.cfg.world_path:
            return self.cfg.world_path
        pkg_path = rospkg.RosPack().get_path("uav_simulator")
        return os.path.join(pkg_path, "worlds", "generated_env", "generated_env_static.world")

    @staticmethod
    def _load_pcd(path):
        with open(path, "r") as f:
            lines = f.readlines()
        data_idx = next(i for i, l in enumerate(lines) if l.strip().upper() == "DATA ASCII") + 1
        return np.loadtxt(lines[data_idx:], dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _load_world_obstacles(path):
        """Parse cylinder/box <model> blocks out of a Gazebo .world file for a
        top-down (XY) obstacle layout — used only to draw the plot background."""
        with open(path, "r") as f:
            content = f.read()

        circles = []  # (x, y, radius)
        for m in re.finditer(
            r"<model name='cylinder_[^']*'>.*?<pose>([^<]+)</pose>.*?<radius>([^<]+)</radius>",
            content, re.S,
        ):
            x, y = (float(v) for v in m.group(1).split()[:2])
            circles.append((x, y, float(m.group(2))))

        rects = []  # (x, y, width, depth, yaw)
        for m in re.finditer(
            r"<model name='box_[^']*'>.*?<pose>([^<]+)</pose>.*?<size>([^<]+)</size>",
            content, re.S,
        ):
            pose_vals = m.group(1).split()
            x, y, yaw = float(pose_vals[0]), float(pose_vals[1]), float(pose_vals[5])
            w, d = (float(v) for v in m.group(2).split()[:2])
            rects.append((x, y, w, d, yaw))

        return circles, rects

    def _odom_cb(self, msg):
        self.odom = msg

    def _wait_for_odom(self, timeout=30.0):
        rospy.loginfo("[gazebo_eval_traj]: waiting for odom …")
        t0 = time.time()
        while not rospy.is_shutdown() and self.odom is None:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for odom on {self.cfg.odom_topic}.")
            rospy.sleep(0.1)

    def _drone_pos(self):
        p = self.odom.pose.pose.position
        return np.array([p.x, p.y, p.z])

    def _drone_vel_world(self):
        # odom.twist is body-frame (matches navigation.py's own vel_world computation).
        q = self.odom.pose.pose.orientation
        rot = self._quat_to_rot(q)
        vel_body = np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ])
        return rot @ vel_body

    @staticmethod
    def _quat_to_rot(q):
        w, x, y, z = q.w, q.x, q.y, q.z
        xx, xy, xz = x * x, x * y, x * z
        yy, yz = y * y, y * z
        zz = z * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ])

    def _min_obstacle_dist(self, pos):
        return float(np.linalg.norm(self.pcd - pos, axis=1).min())

    def _publish_goal(self, xyz):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = xyz
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _reposition_to(self, xyz, tol, timeout_s):
        """One-off, unscored, unrecorded move to the exact start pose before
        trial 0 — only arrival or timeout ends it."""
        self._publish_goal(xyz)
        target = np.array(xyz)
        t0 = time.time()
        rate = rospy.Rate(self.cfg.rate_hz)
        while not rospy.is_shutdown():
            elapsed = time.time() - t0
            dist = float(np.linalg.norm(self._drone_pos() - target))
            if dist < tol:
                return "success", elapsed, dist
            if elapsed > timeout_s:
                return "timeout", elapsed, dist
            rate.sleep()
        raise RuntimeError("rospy shut down mid-reposition.")

    def _fly_leg(self, xyz, timeout_s):
        """Publish xyz as the goal and fly it to completion, recording
        (t, x, y, z, vx, vy, vz, speed) at rate_hz the whole way. A collision is
        recorded but never cuts the leg short; only reaching the goal (success)
        or timeout ends it. Returns (outcome, collided, elapsed_s, final_dist, traj).
        """
        self._publish_goal(xyz)
        target = np.array(xyz)
        t0 = time.time()
        rate = rospy.Rate(self.cfg.rate_hz)
        collided = False
        traj = []
        while not rospy.is_shutdown():
            elapsed = time.time() - t0
            pos = self._drone_pos()
            vel_world = self._drone_vel_world()
            traj.append({
                "t": elapsed,
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "vx": float(vel_world[0]), "vy": float(vel_world[1]), "vz": float(vel_world[2]),
                "speed": float(np.linalg.norm(vel_world)),
            })
            dist = float(np.linalg.norm(pos - target))
            if elapsed >= self.cfg.min_check_delay_s:
                if not collided and self._min_obstacle_dist(pos) < self.cfg.collision_radius:
                    collided = True
                    rospy.logwarn(f"[gazebo_eval_traj]: collision at elapsed={elapsed:.1f}s — continuing to goal.")
                if dist < self.cfg.success_radius:
                    return "success", collided, elapsed, dist, traj
            if elapsed > timeout_s:
                return "timeout", collided, elapsed, dist, traj
            rate.sleep()
        raise RuntimeError("rospy shut down mid-trial.")

    def run(self):
        self._wait_for_odom()
        num_trials = int(self.cfg.num_trials)
        xs = np.linspace(self.cfg.x_range[0], self.cfg.x_range[1], num_trials)
        results = []
        self.trajectories = []
        run_tag = f"{self.checkpoint_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        start0 = (float(xs[0]), float(self.cfg.y_start), self.z)
        rospy.loginfo(f"[gazebo_eval_traj]: initial repositioning to {start0} …")
        outcome0, elapsed0, dist0 = self._reposition_to(
            start0, tol=self.cfg.reposition_tol, timeout_s=self.cfg.reposition_timeout_s,
        )
        if outcome0 != "success":
            rospy.logwarn(
                f"[gazebo_eval_traj]: initial reposition ended with '{outcome0}' "
                f"after {elapsed0:.1f}s (dist={dist0:.2f}m) — proceeding anyway."
            )

        for i, x in enumerate(xs):
            goal_y = self.cfg.y_goal if i % 2 == 0 else self.cfg.y_start
            goal = (float(x), float(goal_y), self.z)

            rospy.loginfo(f"[gazebo_eval_traj]: trial {i + 1}/{num_trials} — flying to {goal} …")
            outcome, collided, elapsed, final_dist, traj = self._fly_leg(goal, timeout_s=self.cfg.trial_timeout_s)
            rospy.loginfo(
                f"[gazebo_eval_traj]: trial {i + 1}/{num_trials} -> {outcome} "
                f"(collided={collided}, elapsed={elapsed:.1f}s, final_dist={final_dist:.2f}m, "
                f"{len(traj)} samples recorded)"
            )
            results.append({
                "trial": i, "x": float(x), "goal_y": float(goal_y), "outcome": outcome,
                "collided": collided, "elapsed_s": elapsed, "final_dist_to_goal": final_dist,
            })
            self.trajectories.append(traj)

        self._report(results, run_tag)
        self._plot(results, run_tag)

    def _report(self, results, run_tag):
        n = len(results)
        success = sum(r["outcome"] == "success" and not r["collided"] for r in results)
        timeout = sum(r["outcome"] == "timeout" for r in results)
        collision = sum(r["collided"] for r in results)
        summary = {
            "num_trials": n,
            "success_rate": success / n,
            "collision_rate": collision / n,
            "timeout_rate": timeout / n,
        }
        rospy.loginfo(f"[gazebo_eval_traj]: === summary over {n} trials ===")
        rospy.loginfo(
            f"[gazebo_eval_traj]: success_rate={summary['success_rate']:.2f}  "
            f"collision_rate={summary['collision_rate']:.2f}  "
            f"timeout_rate={summary['timeout_rate']:.2f}"
        )

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg.results_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gazebo_eval_traj_{run_tag}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "trials": results, "trajectories": self.trajectories}, f, indent=2)
        rospy.loginfo(f"[gazebo_eval_traj]: results + trajectories saved to {out_path}")

    def _plot(self, results, run_tag):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Rectangle
            from matplotlib.transforms import Affine2D
        except ImportError:
            rospy.logerr("[gazebo_eval_traj]: matplotlib not available — skipping plot (raw data was still saved).")
            return

        circles, rects = self._load_world_obstacles(self._resolve_world_path())

        fig, (ax_xy, ax_v) = plt.subplots(2, 1, figsize=(9, 11), gridspec_kw={"height_ratios": [2, 1]})

        for x, y, r in circles:
            ax_xy.add_patch(Circle((x, y), r, facecolor="0.55", edgecolor="0.3", linewidth=0.5, alpha=0.6, zorder=1))
        for x, y, w, d, yaw in rects:
            rect = Rectangle((-w / 2., -d / 2.), w, d, facecolor="0.55", edgecolor="0.3", linewidth=0.5, alpha=0.6, zorder=1)
            rect.set_transform(Affine2D().rotate(yaw).translate(x, y) + ax_xy.transData)
            ax_xy.add_patch(rect)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.trajectories)))
        for i, traj in enumerate(self.trajectories):
            if not traj:
                continue
            xs = [p["x"] for p in traj]
            ys = [p["y"] for p in traj]
            ts = [p["t"] for p in traj]
            speeds = [p["speed"] for p in traj]
            outcome = results[i]["outcome"]
            failed = outcome != "success" or results[i]["collided"]
            style = "--" if failed else "-"

            ax_xy.plot(xs, ys, style, color=colors[i], linewidth=1.8,
                       label=f"trial {i} ({outcome}{', collision' if results[i]['collided'] else ''})", zorder=3)
            ax_xy.scatter([xs[0]], [ys[0]], color=colors[i], marker="o", s=40, zorder=4)
            ax_xy.scatter([xs[-1]], [ys[-1]], color=colors[i], marker="*", s=120, zorder=4)

            ax_v.plot(ts, speeds, color=colors[i], linewidth=1.2, label=f"trial {i}")

        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.set_title(f"{len(self.trajectories)} start-goal trajectories over generated_env_static")
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(alpha=0.3)
        ax_xy.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

        ax_v.set_xlabel("time since goal published [s]")
        ax_v.set_ylabel("speed [m/s]")
        ax_v.set_title("commanded speed over time (per trial, matching colors above)")
        ax_v.grid(alpha=0.3)

        fig.tight_layout()

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg.results_dir)
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f"gazebo_eval_traj_{run_tag}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        rospy.loginfo(f"[gazebo_eval_traj]: trajectory plot saved to {plot_path}")


FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="gazebo_eval_traj", version_base=None)
def main(cfg):
    rospy.init_node("gazebo_eval_traj")
    node = GazeboTrajEval(cfg)
    node.run()


if __name__ == "__main__":
    main()
