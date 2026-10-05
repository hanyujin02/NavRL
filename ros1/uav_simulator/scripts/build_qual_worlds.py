#!/usr/bin/env python3
"""Hand-designed (deterministic, non-random) Gazebo worlds for qualitative
policy comparison, as opposed to world_generator.py's randomized fields.

Each scenario below is built from explicit box obstacles so the geometry is
exact and reproducible. For every scenario this script writes both:
  worlds/<name>/<name>_static.world   (Gazebo SDF)
  worlds/<name>/<name>_static.pcd     (matching point cloud, same sampling
                                        convention as world_generator.py, so
                                        it drops into gazebo_eval.py's
                                        collision checker unchanged)

Robot footprint reference (from urdf/px4_quadcopter.sdf): base_link box is
0.47 x 0.47 m, rotors add ~0.128 m radius each side -> effective span ~0.73 m.
Gap widths below are chosen relative to that.

Run with: python3 build_qual_worlds.py
"""
import os
import numpy as np

WALL_HEIGHT = 2.5
WALL_THICK = 0.2
POINT_STEP = 0.1

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
WORLDS_DIR = os.path.join(os.path.dirname(CURR_DIR), "worlds")


def box(name, cx, cy, z0, sx, sy, sz):
    """One static SDF box model + the point-cloud samples that fill its volume
    (same dense-fill convention world_generator.py uses for its own boxes)."""
    sdf = f"""
        <model name='{name}'>
        <static>true</static>
        <pose>{cx} {cy} {z0 + sz / 2.0} 0 0 0</pose>
        <link name='link'>
            <collision name='collision'>
            <geometry>
                <box><size>{sx} {sy} {sz}</size></box>
            </geometry>
            </collision>
            <visual name='visual'>
            <geometry>
                <box><size>{sx} {sy} {sz}</size></box>
            </geometry>
            </visual>
        </link>
        </model>
        """
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    pts = []
    for px in np.arange(x0, x1 + POINT_STEP, POINT_STEP):
        for py in np.arange(y0, y1 + POINT_STEP, POINT_STEP):
            for pz in np.arange(z0, z0 + sz + POINT_STEP, POINT_STEP):
                pts.append((px, py, pz))
    return sdf, pts


def wall_segment(name, x0, x1, y, z0=0.0, height=WALL_HEIGHT, thick=WALL_THICK):
    """Wall running along x from x0 to x1, centered at y, full height."""
    cx = (x0 + x1) / 2.0
    sx = x1 - x0
    return box(name, cx, y, z0, sx, thick, height)


def side_wall(name, x, y0, y1, z0=0.0, height=WALL_HEIGHT, thick=WALL_THICK):
    """Wall running along y from y0 to y1, centered at x, full height."""
    cy = (y0 + y1) / 2.0
    sy = y1 - y0
    return box(name, x, cy, z0, thick, sy, height)


WORLD_TEMPLATE = """
<sdf version='1.7'>
  <world name='default'>
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose>0 0 10 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
      <spot><inner_angle>0</inner_angle><outer_angle>0</outer_angle><falloff>0</falloff></spot>
    </light>
    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface>
            <contact><collide_bitmask>65535</collide_bitmask><ode/></contact>
            <friction><ode><mu>100</mu><mu2>50</mu2></ode><torsional><ode/></torsional></friction>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Grey</name></script></material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene><ambient>0.4 0.4 0.4 1</ambient><background>0.7 0.7 0.7 1</background><shadows>1</shadows></scene>
    <wind/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>0</latitude_deg><longitude_deg>0</longitude_deg>
      <elevation>0</elevation><heading_deg>0</heading_deg>
    </spherical_coordinates>
    {models}
  </world>
</sdf>
"""


def write_scenario(name, parts):
    """parts: list of (sdf_str, points) tuples."""
    sdfs = [p[0] for p in parts]
    pts = np.array([pt for p in parts for pt in p[1]], dtype=np.float64)

    out_dir = os.path.join(WORLDS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    world_path = os.path.join(out_dir, f"{name}_static.world")
    with open(world_path, "w") as f:
        f.write(WORLD_TEMPLATE.format(models="".join(sdfs)))

    pcd_path = os.path.join(out_dir, f"{name}_static.pcd")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(pts)}\nDATA ascii\n"
    )
    with open(pcd_path, "w") as f:
        f.write(header)
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"[{name}] wrote {world_path}")
    print(f"[{name}] wrote {pcd_path} ({len(pts)} points)")


# ---------------------------------------------------------------------------
# Scenario 1: clearance threshold
# One continuous corridor (half-width 3m) with 4 sequential cross-wall gates,
# all the same marginal gap width (0.9m -> ~0.2m clearance total) but NOT
# aligned with each other: each gate's gap center alternates left/right
# (a staggered slalom), so the drone must reposition laterally between gates
# instead of flying straight through a lined-up sequence of apertures.
# (An impassable-gap tier was dropped: the policy's known failure to actually
# stop/refuse at a true dead end is a separate reward/training issue, not
# something this scene can isolate cleanly.)
# Single flight: start (0,-5,z) -> goal (0,20,z), threading all 4 gates.
# ---------------------------------------------------------------------------
def build_clearance_test():
    gap = 0.9
    gate_offset = 1.0  # how far each gate's gap center is thrown off the corridor centerline
    gate_ys = [0.0, 5.0, 10.0, 15.0]
    gate_offsets = [-gate_offset, gate_offset, -gate_offset, gate_offset]  # zigzag
    half_width = 3.0
    y0, y1 = -6.0, 20.0
    parts = [
        side_wall("clearance_corridor_left", -half_width, y0, y1),
        side_wall("clearance_corridor_right", half_width, y0, y1),
    ]
    for i, (gy, gx) in enumerate(zip(gate_ys, gate_offsets)):
        parts.append(wall_segment(f"clearance_gate{i}_left", -half_width, gx - gap / 2.0, gy))
        parts.append(wall_segment(f"clearance_gate{i}_right", gx + gap / 2.0, half_width, gy))
    write_scenario("clearance_test", parts)
    print(f"[clearance_test] start (0,-5) -> goal (0,20); {len(gate_ys)} staggered 0.9m gates at y={gate_ys}, offsets={gate_offsets}")


# ---------------------------------------------------------------------------
# Scenario 2: blocked direct path
# Start (0,-5) -> goal (0,5). A wall blocks x in [-3,1] at y=0 (covers the
# straight-ahead line at x=0), leaving a 2m opening on the right (x in [1,3])
# next to the right side wall. The policy must give up the direct heading and
# detour right through the opening.
# ---------------------------------------------------------------------------
def build_blocked_path():
    half_width = 3.0
    y0, y1 = -6.0, 6.0
    parts = [
        side_wall("blocked_left", -half_width, y0, y1),
        side_wall("blocked_right", half_width, y0, y1),
        wall_segment("blocked_center_obstacle", -half_width, 1.0, 0.0),
    ]
    write_scenario("blocked_path", parts)
    print("[blocked_path] start (0,-5) -> goal (0,5); only opening is x in [1,3]")


# ---------------------------------------------------------------------------
# Scenario 3: visible fork
# A single trunk corridor forks at y=-1 into two branches of matching width
# and appearance. The right branch dead-ends 5m in (y=4); the left branch is
# open all the way to the goal. From the fork, both branches look alike until
# the drone is close enough to sense the right branch's dead-end wall.
# ---------------------------------------------------------------------------
def build_visible_fork():
    trunk_x = 1.2
    divider_x = 0.15
    branch_outer_x = 2.55
    fork_y = -1.0
    trunk_y0 = -6.0
    branch_y1 = 9.0
    dead_end_y = 4.0

    parts = [
        side_wall("fork_trunk_left", -trunk_x, trunk_y0, fork_y),
        side_wall("fork_trunk_right", trunk_x, trunk_y0, fork_y),
        side_wall("fork_divider", 0.0, fork_y, branch_y1, thick=2 * divider_x),
        side_wall("fork_left_outer", -branch_outer_x, fork_y, branch_y1),
        side_wall("fork_right_outer", branch_outer_x, fork_y, branch_y1),
        wall_segment("fork_right_deadend", divider_x, branch_outer_x, dead_end_y),
    ]
    write_scenario("visible_fork", parts)
    print(
        f"[visible_fork] start (0,-5) -> goal ({-(divider_x + branch_outer_x) / 2.0:.2f},{branch_y1 - 1}); "
        f"right branch dead-ends at y={dead_end_y}"
    )


if __name__ == "__main__":
    build_clearance_test()
    build_blocked_path()
    build_visible_fork()
