"""Preview the oriented-obstacle training environment WITHOUT launching Isaac Sim.

Generates the exact same obstacle layout as training (same cfg/env.yaml, same
obstacle seed, same sampling code) and renders:
  - topdown.png : obstacle footprints colored by top height + the 2D
                  Dijkstra/CBF occupancy grid (conservative XY projection)
  - scene.glb   : full 3D mesh — open with any glTF viewer, or pass --show
                  for an interactive trimesh window

Usage (inside the NavRL conda env):
    python preview_env.py                        # uses ../cfg/env.yaml
    python preview_env.py --seed 3 --show
    python preview_env.py --cfg /path/to/env.yaml --out /tmp/preview
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import yaml

import oriented_obstacles as oo


def load_env_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)["env"]


def make_terrain_cfg(env_cfg, seed_override=None):
    obs = env_cfg["obstacle"]
    map_range = env_cfg["map_range"]
    seed = obs.get("seed", 0) if seed_override is None else seed_override
    return SimpleNamespace(
        size=(2.0 * map_range[0], 2.0 * map_range[1]),
        obstacle_seed=int(seed),
        num_obstacles=int(env_cfg["num_obstacles"]),
        obstacle_width_range=tuple(obs["width_range"]),
        obstacle_length_range=tuple(obs["length_range"]),
        obstacle_height_range=list(obs["height_range"]),
        obstacle_height_probability=list(obs["height_probability"]),
        yaw_range=tuple(obs["yaw_range"]),
        roll_range=tuple(obs["roll_range"]),
        pitch_range=tuple(obs["pitch_range"]),
        tilt_distribution=str(obs.get("tilt_distribution", "uniform")),
        tilt_std=float(obs.get("tilt_std", 10.0)),
        min_center_distance=float(obs["min_center_distance"]),
        border_margin=float(obs["border_margin"]),
    )


def footprint_polygon(pos, rot, size):
    """XY convex hull of the 8 box corners (world frame)."""
    from scipy.spatial import ConvexHull

    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    corners = pos + (0.5 * size * signs) @ rot.T
    xy = corners[:, :2]
    hull = ConvexHull(xy)
    return xy[hull.vertices], corners[:, 2].max()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default=os.path.join(script_dir, "..", "cfg", "env.yaml"))
    parser.add_argument("--seed", type=int, default=None, help="override env.obstacle.seed")
    parser.add_argument("--out", default=os.path.join(script_dir, "..", "env_preview"))
    parser.add_argument("--resolution", type=float, default=0.4, help="occupancy grid resolution (m)")
    parser.add_argument("--z-min", type=float, default=0.2, help="lower bound of the flight height band (m)")
    parser.add_argument("--inflate", type=float, default=0.3, help="occupancy inflation radius (m)")
    parser.add_argument("--show", action="store_true", help="open an interactive 3D window")
    args = parser.parse_args()

    env_cfg = load_env_cfg(args.cfg)
    map_range = env_cfg["map_range"]
    tcfg = make_terrain_cfg(env_cfg, args.seed)

    meshes, _ = oo.oriented_obstacles_terrain(0.0, tcfg)
    info = oo.last_generated_info

    import trimesh

    scene = trimesh.util.concatenate(meshes)
    # shift local frame [0, size] -> centered world frame (same as TerrainGenerator)
    T = np.eye(4)
    T[:2, 3] = -0.5 * tcfg.size[0], -0.5 * tcfg.size[1]
    scene.apply_transform(T)

    os.makedirs(args.out, exist_ok=True)
    glb_path = os.path.join(args.out, "scene.glb")
    scene.export(glb_path)

    # occupancy grid over the map extent (same math as env_depth._build_dijkstra_occupancy)
    extent = max(map_range[0], map_range[1])
    res = args.resolution
    n = int(np.ceil(2.0 * extent / res))
    xs = (np.arange(n) + 0.5) * res - extent
    ys = (np.arange(n) + 0.5) * res - extent
    blocked = oo.compute_column_occupancy(info, xs, ys, (args.z_min, map_range[2]))
    inflated = oo.inflate_occupancy(blocked, int(np.ceil(args.inflate / res)))

    # top-down figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax = axes[0]
    footprints = [footprint_polygon(info["pos"][i], info["rot"][i], info["size"][i]) for i in range(len(info["pos"]))]
    max_h = max(max(top_z for _, top_z in footprints), 1e-6)
    cmap = plt.get_cmap("viridis")
    for poly, top_z in footprints:
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=cmap(top_z / max_h), edgecolor="k", linewidth=0.3, alpha=0.9))
    ax.set_xlim(-map_range[0], map_range[0])
    ax.set_ylim(-map_range[1], map_range[1])
    ax.set_aspect("equal")
    ax.set_title(f"footprints (color = top height, max {max_h:.1f} m)\n"
                 f"{len(info['pos'])} obstacles, seed {tcfg.obstacle_seed}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1]
    grid_img = blocked.astype(float) + 0.5 * (inflated & ~blocked)
    ax.imshow(grid_img.T, origin="lower", cmap="Reds",
              extent=[-extent, extent, -extent, extent], vmin=0, vmax=1.5)
    ax.set_aspect("equal")
    ax.set_title(f"2D occupancy (Dijkstra/CBF view, z in [{args.z_min}, {map_range[2]}] m)\n"
                 f"dark = blocked, light = inflated ({args.inflate} m) — conservative XY projection")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    png_path = os.path.join(args.out, "topdown.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    tilt = np.abs(info["euler_deg"][:, :2])
    print(f"[preview] {len(info['pos'])} obstacles | "
          f"roll/pitch |max| = {tilt.max():.1f} deg | "
          f"heights {info['size'][:, 2].min():.2f}–{info['size'][:, 2].max():.2f} m")
    print(f"[preview] wrote {png_path}")
    print(f"[preview] wrote {glb_path}")

    if args.show:
        trimesh.Scene(scene).show()


if __name__ == "__main__":
    main()
