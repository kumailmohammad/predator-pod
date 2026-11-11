"""What each fish can sense. Brains decide using only this.

Everything is expressed in the fish's own frame of reference ("forward" and
"side"), so the same brain works no matter which way the fish is facing.
"""

from dataclasses import dataclass

import numpy as np

N_FEATURES = 18

WALL_SENSE_RANGE = 200.0      # px, walls farther than this read as 1.0
POD_SENSE_RANGE = 200.0
NEIGHBOR_SENSE_RANGE = 50.0
SHARK_CLOSE_SCALE = 60.0      # px, closeness = 1 / (1 + d / this)

FEATURE_NAMES = [
    "wall_ahead", "wall_left", "wall_right", "wall_behind",
    "shark_dir_fwd", "shark_dir_side", "shark_dist", "shark_closeness",
    "shark_heading_fwd", "shark_heading_side", "shark_aiming_at_me",
    "pod_dir_fwd", "pod_dir_side", "pod_dist",
    "neighbor_dir_fwd", "neighbor_dir_side", "neighbor_dist",
    "pod_alignment",
]


@dataclass
class Observation:
    heading: object          # pygame.Vector2, fish's current heading
    features: np.ndarray     # shape (N_FEATURES,)


def _ray_to_wall(px, py, ux, uy, rect):
    """Distance from (px, py) along unit direction (ux, uy) to the arena border."""
    t = float("inf")
    if ux > 1e-9:
        t = min(t, (rect.right - px) / ux)
    elif ux < -1e-9:
        t = min(t, (rect.left - px) / ux)
    if uy > 1e-9:
        t = min(t, (rect.bottom - py) / uy)
    elif uy < -1e-9:
        t = min(t, (rect.top - py) / uy)
    return max(t, 0.0)


def observe_all(fishes, shark, arena):
    """Build an Observation for every live fish."""
    n = len(fishes)
    if n == 0:
        return []
    rect = arena.rect
    scale = float(max(rect.width, rect.height))

    pos = np.array([[f.pos.x, f.pos.y] for f in fishes], dtype=np.float64)
    heads = np.array([[f.heading.x, f.heading.y] for f in fishes], dtype=np.float64)
    pos_sum, head_sum = pos.sum(0), heads.sum(0)

    # Nearest neighbor for every fish.
    diff = pos[None, :, :] - pos[:, None, :]          # diff[i, j] = pos[j] - pos[i]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nearest = d2.argmin(1) if n > 1 else None

    sx, sy = shark.pos.x, shark.pos.y
    shx, shy = shark.heading.x, shark.heading.y

    out = []
    for i, f in enumerate(fishes):
        px, py = pos[i]
        hx, hy = heads[i]

        def local(vx, vy):
            # (forward, side) components relative to the fish heading
            return vx * hx + vy * hy, -vx * hy + vy * hx

        def unit_local(vx, vy):
            d = (vx * vx + vy * vy) ** 0.5
            if d < 1e-9:
                return 0.0, 0.0, 0.0
            fwd, side = local(vx / d, vy / d)
            return fwd, side, d

        walls = [
            _ray_to_wall(px, py, hx, hy, rect),      # ahead
            _ray_to_wall(px, py, hy, -hx, rect),     # left
            _ray_to_wall(px, py, -hy, hx, rect),     # right
            _ray_to_wall(px, py, -hx, -hy, rect),    # behind
        ]
        walls = [min(w / WALL_SENSE_RANGE, 1.0) for w in walls]

        s_fwd, s_side, s_d = unit_local(sx - px, sy - py)
        sh_fwd, sh_side = local(shx, shy)
        aiming = 0.0
        if s_d > 1e-9:
            aiming = (shx * (px - sx) + shy * (py - sy)) / s_d   # 1 = shark heading straight at me

        if n > 1:
            cx, cy = (pos_sum - pos[i]) / (n - 1)
            p_fwd, p_side, p_d = unit_local(cx - px, cy - py)
            j = nearest[i]
            nb_fwd, nb_side, nb_d = unit_local(pos[j, 0] - px, pos[j, 1] - py)
            ox, oy = head_sum - heads[i]
            on = (ox * ox + oy * oy) ** 0.5
            align = (ox * hx + oy * hy) / on if on > 1e-9 else 0.0
        else:
            p_fwd = p_side = p_d = nb_fwd = nb_side = 0.0
            nb_d = NEIGHBOR_SENSE_RANGE
            align = 0.0

        features = np.array(
            walls
            + [
                s_fwd, s_side, s_d / scale, 1.0 / (1.0 + s_d / SHARK_CLOSE_SCALE),
                sh_fwd, sh_side, aiming,
                p_fwd, p_side, min(p_d / POD_SENSE_RANGE, 1.0),
                nb_fwd, nb_side, min(nb_d / NEIGHBOR_SENSE_RANGE, 1.0),
                align,
            ],
            dtype=np.float64,
        )
        out.append(Observation(heading=f.heading, features=features))
    return out
