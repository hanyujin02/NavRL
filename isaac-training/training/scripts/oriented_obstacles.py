"""Oriented static-obstacle generation for the NavRL depth environment.

Pure numpy + trimesh — no Isaac Sim imports — so the exact same code path is
shared by the training env (via a SubTerrainBaseCfg wrapper in env_depth.py)
and the standalone preview script (preview_env.py).

All boxes are merged into ONE trimesh by the orbit TerrainGenerator because
the orbit RayCaster (depth camera + lidar) only supports a single mesh prim.

Coordinate conventions:
- The terrain function returns meshes in the orbit sub-terrain local frame
  [0, size] x [0, size]; the TerrainGenerator's transform chain shifts this
  by -size/2, so `last_generated_info` is stored directly in the final WORLD
  frame (map centered at the origin).
- Obstacle sampling uses its own np.random.RandomState(cfg.obstacle_seed) so
  the layout is reproducible regardless of the caller's global RNG state.
"""

import numpy as np
import trimesh

# World-frame info of the most recent generation. The training env captures
# this right after TerrainGenerator invokes the terrain function.
last_generated_info = None


def _euler_deg_to_matrix(roll, pitch, yaw):
    """Vectorized R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Angles in degrees, shape (N,) -> (N, 3, 3)."""
    r, p, y = np.radians(roll), np.radians(pitch), np.radians(yaw)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.empty((len(r), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def _sample_truncated_normal(rng, num, std, lo, hi):
    """Normal centered at the midpoint of [lo, hi], rejection-sampled into the range."""
    if hi - lo < 1e-9:
        return np.full(num, lo)
    mean = 0.5 * (lo + hi)
    out = np.empty(num)
    remaining = np.arange(num)
    while len(remaining):
        s = rng.normal(mean, std, size=len(remaining))
        ok = (s >= lo) & (s <= hi)
        out[remaining[ok]] = s[ok]
        remaining = remaining[~ok]
    return out


def _sample_heights(rng, num, edges, probs):
    """Mirror the hf 'range' mode: edges define intervals, probs pick one, uniform inside."""
    edges = np.asarray(edges, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    bins = rng.choice(len(probs), size=num, p=probs)
    return rng.uniform(edges[bins], edges[bins + 1])


def sample_oriented_obstacles(cfg):
    """Sample oriented boxes in the centered world frame.

    Reads (duck-typed) from cfg: obstacle_seed, num_obstacles, size,
    obstacle_width_range, obstacle_length_range, obstacle_height_range,
    obstacle_height_probability, yaw_range, roll_range, pitch_range,
    min_center_distance, border_margin.

    Returns a dict with world-frame arrays:
        pos (N, 3), rot (N, 3, 3), size (N, 3), euler_deg (N, 3)
    """
    rng = np.random.RandomState(int(getattr(cfg, "obstacle_seed", 0)))
    num = int(cfg.num_obstacles)
    half_x = 0.5 * float(cfg.size[0]) - float(cfg.border_margin)
    half_y = 0.5 * float(cfg.size[1]) - float(cfg.border_margin)

    sizes = np.empty((num, 3))
    sizes[:, 0] = rng.uniform(*cfg.obstacle_width_range, size=num)
    sizes[:, 1] = rng.uniform(*cfg.obstacle_length_range, size=num)
    sizes[:, 2] = _sample_heights(rng, num, cfg.obstacle_height_range, cfg.obstacle_height_probability)

    # roll/pitch tilt distribution: "uniform" (flat over the range) or
    # "normal" (truncated normal centered at the range midpoint, std=tilt_std
    # in degrees — mostly-upright obstacles with occasional strong tilts).
    # yaw is always uniform: there is no preferred horizontal direction.
    tilt_distribution = str(getattr(cfg, "tilt_distribution", "uniform"))
    euler = np.empty((num, 3))
    if tilt_distribution == "normal":
        tilt_std = float(getattr(cfg, "tilt_std", 10.0))
        euler[:, 0] = _sample_truncated_normal(rng, num, tilt_std, *cfg.roll_range)
        euler[:, 1] = _sample_truncated_normal(rng, num, tilt_std, *cfg.pitch_range)
    elif tilt_distribution == "uniform":
        euler[:, 0] = rng.uniform(*cfg.roll_range, size=num)
        euler[:, 1] = rng.uniform(*cfg.pitch_range, size=num)
    else:
        raise ValueError(f"Unknown tilt_distribution '{tilt_distribution}'. Must be 'uniform' or 'normal'.")
    euler[:, 2] = rng.uniform(*cfg.yaw_range, size=num)
    rot = _euler_deg_to_matrix(euler[:, 0], euler[:, 1], euler[:, 2])

    # xy placement with a min-center-distance check; the threshold relaxes if
    # the sampler struggles (same spirit as the dynamic-obstacle placement).
    min_dist = float(cfg.min_center_distance)
    centers = np.empty((num, 2))
    for i in range(num):
        dist = min_dist
        attempts = 0
        while True:
            c = np.array([rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)])
            if i == 0 or np.min(np.linalg.norm(centers[:i] - c, axis=1)) >= dist:
                break
            attempts += 1
            if attempts % 200 == 0:
                dist *= 0.8
        centers[i] = c

    # z placement: drop each tilted box so its lowest corner sits slightly
    # below the ground plane (no floating gaps under tilted boxes).
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    corners_local = 0.5 * sizes[:, None, :] * signs[None, :, :]           # (N, 8, 3)
    corners_z = np.einsum("nij,nkj->nki", rot, corners_local)[..., 2]     # (N, 8)
    embed = 0.05
    pos = np.empty((num, 3))
    pos[:, :2] = centers
    pos[:, 2] = -corners_z.min(axis=1) - embed

    return {"pos": pos, "rot": rot, "size": sizes, "euler_deg": euler}


def build_obstacle_meshes(info, xy_offset=(0.0, 0.0)):
    """Create one trimesh box per obstacle, translated by xy_offset."""
    meshes = []
    for i in range(len(info["pos"])):
        T = np.eye(4)
        T[:3, :3] = info["rot"][i]
        T[:3, 3] = info["pos"][i] + np.array([xy_offset[0], xy_offset[1], 0.0])
        meshes.append(trimesh.creation.box(extents=info["size"][i], transform=T))
    return meshes


def oriented_obstacles_terrain(difficulty, cfg):
    """Orbit sub-terrain function: returns (list[trimesh.Trimesh], origin) in local frame [0, size]."""
    global last_generated_info
    info = sample_oriented_obstacles(cfg)
    last_generated_info = info

    # ground slab (top face at z=0) so sensor rays hit the floor
    thickness = 0.2
    ground_T = np.eye(4)
    ground_T[:3, 3] = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * thickness]
    ground = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], thickness), transform=ground_T)

    meshes = [ground] + build_obstacle_meshes(info, xy_offset=(0.5 * cfg.size[0], 0.5 * cfg.size[1]))
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    return meshes, origin


def compute_column_occupancy(info, xs_world, ys_world, z_band):
    """Exact 2D occupancy: cell (i, j) is blocked iff the vertical segment
    x=xs[i], y=ys[j], z in [z0, z1] intersects any oriented box.

    NOTE: this is the conservative 2D projection used by the Dijkstra/CBF
    rewards — 3D (height-aware) distance fields are NOT implemented.
    """
    xs_world = np.asarray(xs_world, dtype=np.float64)
    ys_world = np.asarray(ys_world, dtype=np.float64)
    z0, z1 = float(z_band[0]), float(z_band[1])
    blocked = np.zeros((len(xs_world), len(ys_world)), dtype=bool)
    eps = 1e-9

    for pos, rot, size in zip(info["pos"], info["rot"], info["size"]):
        half = 0.5 * size
        # world-axis-aligned bounding half-extents of the box -> candidate cells
        ext = np.abs(rot) @ half
        ix = np.nonzero(np.abs(xs_world - pos[0]) <= ext[0] + eps)[0]
        iy = np.nonzero(np.abs(ys_world - pos[1]) <= ext[1] + eps)[0]
        if len(ix) == 0 or len(iy) == 0:
            continue

        px, py = np.meshgrid(xs_world[ix], ys_world[iy], indexing="ij")
        # local(z) = R^T ([px, py, z] - c) = a + b * z
        d0 = np.stack([px - pos[0], py - pos[1], np.full_like(px, -pos[2])], axis=-1)  # (m, k, 3)
        a = d0 @ rot          # R^T applied to each row vector
        b = rot[2, :]         # R^T @ e_z

        z_lo = np.full(px.shape, z0)
        z_hi = np.full(px.shape, z1)
        feasible = np.ones(px.shape, dtype=bool)
        for k in range(3):
            if abs(b[k]) < eps:
                feasible &= np.abs(a[..., k]) <= half[k] + eps
            else:
                lo = (-half[k] - a[..., k]) / b[k]
                hi = (half[k] - a[..., k]) / b[k]
                if b[k] < 0:
                    lo, hi = hi, lo
                z_lo = np.maximum(z_lo, lo)
                z_hi = np.minimum(z_hi, hi)
        hit = feasible & (z_lo <= z_hi)
        blocked[np.ix_(ix, iy)] |= hit

    return blocked


def inflate_occupancy(blocked, inflate_cells):
    """Disc-shaped binary dilation (same logic as env_depth's inflate loop)."""
    if inflate_cells <= 0:
        return blocked
    grid_h, grid_w = blocked.shape
    padded = np.pad(blocked, inflate_cells, mode="constant", constant_values=False)
    inflated = np.zeros_like(blocked, dtype=bool)
    for dx in range(-inflate_cells, inflate_cells + 1):
        for dy in range(-inflate_cells, inflate_cells + 1):
            if dx * dx + dy * dy <= inflate_cells * inflate_cells:
                x0 = inflate_cells + dx
                y0 = inflate_cells + dy
                inflated |= padded[x0 : x0 + grid_h, y0 : y0 + grid_w]
    return inflated
