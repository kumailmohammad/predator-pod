"""What each fish can sense: the 31 inputs to its neural network.

Directions are in world (screen) terms (right/left/down/up) because the
network's movement outputs are world directions too.

The first 18 inputs are the original senses; the rest were added later and
are appended at the end, so a brain trained on the 18 can be widened
without losing what it learned (see Learner.load).
"""

from dataclasses import dataclass

import numpy as np

from predator_pod import config

N_FEATURES_V1 = 18
N_FEATURES = 31

SHARK_CLOSE_SCALE = 60.0      # px, closeness = 1 / (1 + d / this)
WALL_SENSE_RANGE = 150.0      # px, walls farther than this read as 1.0
POD_SENSE_RANGE = 200.0
NEIGHBOR_SENSE_RANGE = 60.0
CROWD_RANGE = 60.0            # px, fish closer than this count as "my group"
CROWD_FULL = 6                # this many neighbors reads as 1.0
HERD_RANGE = 100.0            # px, the "nearby fish" a selfish-herd move aims at
FREE_SPACE_RANGE = 200.0      # px, open water farther than this reads as 1.0
LANE_HALF_WIDTH = 15.0        # px, a fish this close to a ray blocks it
CORNER_RANGE = 60.0           # px, two walls closer than this = in a corner

FEATURE_NAMES = [
    # --- v1 (original 18) ---
    "shark_dx", "shark_dy", "shark_dist", "shark_closeness",
    "shark_heading_x", "shark_heading_y",
    "wall_left", "wall_right", "wall_top", "wall_bottom",
    "pod_dx", "pod_dy", "pod_dist",
    "neighbor_dx", "neighbor_dy", "neighbor_dist",
    "heading_x", "heading_y",
    # --- v2 ---
    "is_target", "crowd",
    "free_up", "free_down", "free_left", "free_right",
    "corner", "in_shark_path", "closing_speed",
    "school_hx", "school_hy",
    "shark_wall", "burst_ready",
]
assert len(FEATURE_NAMES) == N_FEATURES


@dataclass
class Senses:
    features: np.ndarray     # (n, N_FEATURES) network inputs
    shark_dist: np.ndarray   # (n,) px, used for rewards
    pod_dist: np.ndarray     # (n,) px to the center of the other fish
    wall_dist: np.ndarray    # (n,) px to the nearest wall
    crowd_count: np.ndarray  # (n,) other fish within CROWD_RANGE
    corner: np.ndarray       # (n,) bool, two walls within CORNER_RANGE
    away_dir: np.ndarray     # (n, 2) unit vector pointing away from the shark
    school_dir: np.ndarray   # (n, 2) average heading of the group (0 if alone)
    herd_dir: np.ndarray     # (n, 2) toward the middle of nearby fish (or the pod)
    shark_heading: np.ndarray  # (2,)


def _unit(v):
    d = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(d > 1e-9, v / np.maximum(d, 1e-9), 0.0), d[:, 0]


def sense(fishes, shark, arena):
    n = len(fishes)
    rect = arena.rect
    scale = float(max(rect.width, rect.height))
    pos = np.array([[f.pos.x, f.pos.y] for f in fishes], dtype=np.float64)
    heads = np.array([[f.heading.x, f.heading.y] for f in fishes], dtype=np.float64)
    shark_pos = np.array([shark.pos.x, shark.pos.y])
    shark_head = np.array([shark.heading.x, shark.heading.y])

    shark_dir, shark_dist = _unit(shark_pos - pos)
    shark_heading = np.tile(shark_head, (n, 1))

    walls = np.stack([
        pos[:, 0] - rect.left, rect.right - pos[:, 0],
        pos[:, 1] - rect.top, rect.bottom - pos[:, 1],
    ], axis=1)

    # Pairwise offsets: diff[i, j] = pos[j] - pos[i]
    diff = pos[None, :, :] - pos[:, None, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(d2)

    if n > 1:
        others_center = (pos.sum(0) - pos) / (n - 1)
        pod_dir, pod_dist = _unit(others_center - pos)
        nearest = d2.argmin(1)
        nb_dir, nb_dist = _unit(pos[nearest] - pos)
    else:
        pod_dir, pod_dist = np.zeros((n, 2)), np.zeros(n)
        nb_dir, nb_dist = np.zeros((n, 2)), np.full(n, NEIGHBOR_SENSE_RANGE)

    # --- v2 senses ---------------------------------------------------------
    rank = shark_dist.argsort().argsort()            # 0 = closest to the shark
    is_target = 1.0 / (1.0 + rank)

    in_group = dist < CROWD_RANGE
    crowd_count = in_group.sum(1)
    school_sum = in_group.astype(np.float64) @ heads
    school_dir, _ = _unit(school_sum)

    near = dist < HERD_RANGE
    near_n = near.sum(1)
    local_center = np.where(
        near_n[:, None] > 0,
        (near.astype(np.float64) @ pos) / np.maximum(near_n, 1)[:, None],
        pos + pod_dir,
    )
    herd_dir, _ = _unit(local_center - pos)

    # Free water along up, down, left, right: nearest wall or fish in that lane.
    free = np.stack([walls[:, 2], walls[:, 3], walls[:, 0], walls[:, 1]], axis=1)
    dx, dy = diff[..., 0], diff[..., 1]
    lane_gap = 2 * config.FISH_RADIUS
    for k, (along, across) in enumerate([(-dy, dx), (dy, dx), (-dx, dy), (dx, dy)]):
        block = (along > 0) & (np.abs(across) < LANE_HALF_WIDTH)
        np.fill_diagonal(block, False)
        hit = np.where(block, np.maximum(along - lane_gap, 0.0), np.inf).min(1)
        free[:, k] = np.minimum(free[:, k], hit)

    corner = (walls < CORNER_RANGE).sum(1) >= 2
    away_dir = -shark_dir
    in_path = np.clip(away_dir @ shark_head, 0.0, 1.0)

    fish_vel = heads * config.FISH_SPEED
    shark_vel = np.array([shark.velocity.x, shark.velocity.y])
    closing = ((shark_vel[None, :] - fish_vel) * away_dir).sum(1) / config.SHARK_SPEED

    s_walls = np.array([shark.pos.x - rect.left, rect.right - shark.pos.x,
                        shark.pos.y - rect.top, rect.bottom - shark.pos.y])
    shark_wall = min(s_walls.min() / WALL_SENSE_RANGE, 1.0)

    full = config.FISH_BURST_TIME + config.FISH_BURST_COOLDOWN
    burst_ready = np.array([1.0 if f.burst_ready else 1.0 - max(f.cooldown, 0.0) / full
                            for f in fishes])

    features = np.concatenate([
        shark_dir,
        (shark_dist / scale)[:, None],
        (1.0 / (1.0 + shark_dist / SHARK_CLOSE_SCALE))[:, None],
        shark_heading,
        np.minimum(walls / WALL_SENSE_RANGE, 1.0),
        pod_dir,
        np.minimum(pod_dist / POD_SENSE_RANGE, 1.0)[:, None],
        nb_dir,
        np.minimum(nb_dist / NEIGHBOR_SENSE_RANGE, 1.0)[:, None],
        heads,
        # v2
        is_target[:, None],
        np.minimum(crowd_count / CROWD_FULL, 1.0)[:, None],
        np.minimum(free / FREE_SPACE_RANGE, 1.0),
        corner.astype(np.float64)[:, None],
        in_path[:, None],
        np.clip(closing, -1.0, 1.0)[:, None],
        school_dir,
        np.full((n, 1), shark_wall),
        burst_ready[:, None],
    ], axis=1)
    return Senses(features, shark_dist, pod_dist, walls.min(1), crowd_count, corner,
                  away_dir, school_dir, herd_dir, shark_head)
