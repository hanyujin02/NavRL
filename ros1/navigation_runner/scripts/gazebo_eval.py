#!/usr/bin/env python3
"""Closed-loop Gazebo eval: drives the already-running navigation_node back and
forth across the obstacle field (mirroring isaac-training/training/scripts/
eval.py's y=-12 <-> y=12 crossing convention) and reports success_rate /
collision_rate.

Usage (after `roslaunch uav_simulator start.launch` and
      `roslaunch navigation_runner navigation.launch` are both up):
    rosrun navigation_runner gazebo_eval.py

This script makes no changes to navigation_node/navigation.py — it only
publishes to the same goal_topic navigation.py already subscribes to, and reads
odom for ground truth. There's a single one-off reposition to the first start
point; every trial after that just continues from wherever the previous one
ended, alternating which side of the field it flies to next (no flying back to
a shared start each time — there is no Gazebo teleport/respawn wired up in this
repo anyway; see the plan doc for that tradeoff). A collision never cuts a
trial short — the drone keeps flying regardless and the leg only ends on
reaching the goal or timing out; whether a collision happened along the way is
recorded separately.
"""
import sys
# Strip ROS remapping args (__name:=, __log:=, etc.) before Hydra parses sys.argv
sys.argv = [a for a in sys.argv if not a.startswith('__')]

import os
import time
import json
import datetime
import numpy as np
import rospy
import rospkg
import hydra
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class GazeboEval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.odom = None
        self.odom_sub = rospy.Subscriber(cfg.odom_topic, Odometry, self._odom_cb)
        self.goal_pub = rospy.Publisher(cfg.goal_topic, PoseStamped, queue_size=10)
        self.pcd = self._load_pcd(self._resolve_pcd_path())
        rospy.loginfo(f"[gazebo_eval]: loaded {self.pcd.shape[0]} obstacle points for collision checks.")
        self.z = self._resolve_z()
        rospy.loginfo(f"[gazebo_eval]: using z={self.z:.2f}m for all start/goal poses.")

    def _resolve_z(self):
        # This is a 2D policy: height isn't actively controlled, the drone just
        # sits at navigation_node's takeoff_height for the whole flight. If our
        # goal z doesn't match that exactly, the 3D distance-to-goal never drops
        # below success_radius even once xy is essentially at the goal (e.g.
        # 0.11m xy + 0.5m z mismatch -> 0.51m > success_radius). Auto-read the
        # live value instead of hardcoding one that can drift out of sync with
        # whatever takeoff_height navigation.launch was actually started with.
        if self.cfg.z is not None:
            return float(self.cfg.z)
        return float(rospy.get_param("/navigation_node/takeoff_height", 1.0))

    def _resolve_pcd_path(self):
        if self.cfg.pcd_path:
            return self.cfg.pcd_path
        pkg_path = rospkg.RosPack().get_path("uav_simulator")
        return os.path.join(pkg_path, "worlds", "generated_env", "generated_env_static.pcd")

    @staticmethod
    def _load_pcd(path):
        with open(path, "r") as f:
            lines = f.readlines()
        data_idx = next(i for i, l in enumerate(lines) if l.strip().upper() == "DATA ASCII") + 1
        return np.loadtxt(lines[data_idx:], dtype=np.float64).reshape(-1, 3)

    def _odom_cb(self, msg):
        self.odom = msg

    def _wait_for_odom(self, timeout=30.0):
        rospy.loginfo("[gazebo_eval]: waiting for odom …")
        t0 = time.time()
        while not rospy.is_shutdown() and self.odom is None:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for odom on {self.cfg.odom_topic}.")
            rospy.sleep(0.1)

    def _drone_pos(self):
        p = self.odom.pose.pose.position
        return np.array([p.x, p.y, p.z])

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
        """One-off, unscored move to the exact start pose before trial 0 — only
        arrival or timeout ends it; proximity to obstacles is irrelevant here."""
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
        """Publish xyz as the goal and fly it to completion — a collision is
        recorded but never cuts the leg short (the drone keeps flying regardless);
        only reaching the goal (success) or timeout ends it. Returns
        (outcome, collided, elapsed_s, final_dist) with outcome in
        {"success", "timeout"}.
        """
        self._publish_goal(xyz)
        target = np.array(xyz)
        t0 = time.time()
        rate = rospy.Rate(self.cfg.rate_hz)
        collided = False
        while not rospy.is_shutdown():
            elapsed = time.time() - t0
            pos = self._drone_pos()
            dist = float(np.linalg.norm(pos - target))
            # Skip checks during the grace period — otherwise a collision still
            # lingering from the tail end of the previous leg (same odom, ~0s
            # elapsed here) gets misattributed to this new leg.
            if elapsed >= self.cfg.min_check_delay_s:
                if not collided and self._min_obstacle_dist(pos) < self.cfg.collision_radius:
                    collided = True
                    rospy.logwarn(f"[gazebo_eval]: collision at elapsed={elapsed:.1f}s — continuing to goal.")
                if dist < self.cfg.success_radius:
                    return "success", collided, elapsed, dist
            if elapsed > timeout_s:
                return "timeout", collided, elapsed, dist
            rate.sleep()
        raise RuntimeError("rospy shut down mid-trial.")

    def run(self):
        self._wait_for_odom()
        num_trials = int(self.cfg.num_trials)
        xs = np.linspace(self.cfg.x_range[0], self.cfg.x_range[1], num_trials)
        results = []

        # One-off reposition to the very first start point. Every trial after
        # that starts wherever the previous one ended — no back-and-forth: trial
        # i alternates which side (y_goal / y_start) it flies to, so trial i+1
        # just continues from there instead of flying back to a shared start.
        start0 = (float(xs[0]), float(self.cfg.y_start), self.z)
        rospy.loginfo(f"[gazebo_eval]: initial repositioning to {start0} …")
        outcome0, elapsed0, dist0 = self._reposition_to(
            start0, tol=self.cfg.reposition_tol, timeout_s=self.cfg.reposition_timeout_s,
        )
        if outcome0 != "success":
            rospy.logwarn(
                f"[gazebo_eval]: initial reposition ended with '{outcome0}' "
                f"after {elapsed0:.1f}s (dist={dist0:.2f}m) — proceeding anyway."
            )

        for i, x in enumerate(xs):
            goal_y = self.cfg.y_goal if i % 2 == 0 else self.cfg.y_start
            goal = (float(x), float(goal_y), self.z)

            rospy.loginfo(f"[gazebo_eval]: trial {i + 1}/{num_trials} — flying to {goal} …")
            outcome, collided, elapsed, final_dist = self._fly_leg(goal, timeout_s=self.cfg.trial_timeout_s)
            rospy.loginfo(
                f"[gazebo_eval]: trial {i + 1}/{num_trials} -> {outcome} "
                f"(collided={collided}, elapsed={elapsed:.1f}s, final_dist={final_dist:.2f}m)"
            )
            results.append({
                "trial": i, "x": float(x), "goal_y": float(goal_y), "outcome": outcome,
                "collided": collided, "elapsed_s": elapsed, "final_dist_to_goal": final_dist,
            })

        self._report(results)

    def _report(self, results):
        n = len(results)
        # A trial only counts as success if it reached the goal AND never
        # collided along the way — reaching the goal after grazing an obstacle
        # is not a clean success.
        success = sum(r["outcome"] == "success" and not r["collided"] for r in results)
        timeout = sum(r["outcome"] == "timeout" for r in results)
        collision = sum(r["collided"] for r in results)
        summary = {
            "num_trials": n,
            "success_rate": success / n,
            "collision_rate": collision / n,
            "timeout_rate": timeout / n,
        }
        rospy.loginfo(f"[gazebo_eval]: === summary over {n} trials ===")
        rospy.loginfo(
            f"[gazebo_eval]: success_rate={summary['success_rate']:.2f}  "
            f"collision_rate={summary['collision_rate']:.2f}  "
            f"timeout_rate={summary['timeout_rate']:.2f}"
        )

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg.results_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gazebo_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "trials": results}, f, indent=2)
        rospy.loginfo(f"[gazebo_eval]: results saved to {out_path}")


FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="gazebo_eval", version_base=None)
def main(cfg):
    rospy.init_node("gazebo_eval")
    node = GazeboEval(cfg)
    node.run()


if __name__ == "__main__":
    main()
