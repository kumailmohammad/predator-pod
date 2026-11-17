"""What each fish can sense: the 18 inputs to its neural network.

Directions are in world (screen) terms (right/left/down/up) because the
network's outputs are world directions too: up, down, left, right.
"""

from dataclasses import dataclass

import numpy as np

N_FEATURES = 18

SHARK_CLOSE_SCALE = 60.0      # px, closeness = 1 / (1 + d / this)
WALL_SENSE_RANGE = 150.0      # px, walls farther than this read as 1.0
POD_SENSE_RANGE = 200.0
NEIGHBOR_SENSE_RANGE = 60.0

FEATURE_NAMES = [
    "shark_dx", "shark_dy", "shark_dist", "shark_closeness",
    "shark_heading_x", "shark_heading_y",
    "wall_left", "wall_right", "wall_top", "wall_bottom",
    "pod_dx", "pod_dy", "pod_dist",
    "neighbor_dx", "neighbor_dy", "neighbor_dist",
    "heading_x", "heading_y",
]


@dataclass
class Senses:
    features: np.ndarray     # (n, N_FEATURES) network inputs
    shark_dist: np.ndarray   # (n,) px, used for rewards
    pod_dist: np.ndarray     # (n,) px to the center of the other fish
    wall_dist: np.ndarray    # (n,) px to the nearest wall


def _unit(v):
    d = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(d > 1e-9, v / np.maximum(d, 1e-9), 0.0), d[:, 0]


def sense(fishes, shark, arena):
    n = len(fishes)
    rect = arena.rect
    scale = float(max(rect.width, rect.height))
    pos = np.array([[f.pos.x, f.pos.y] for f in fishes], dtype=np.float64)
    heads = np.array([[f.heading.x, f.heading.y] for f in fishes], dtype=np.float64)

    shark_dir, shark_dist = _unit(np.array([shark.pos.x, shark.pos.y]) - pos)
    shark_heading = np.tile([shark.heading.x, shark.heading.y], (n, 1))

    walls = np.stack([
        pos[:, 0] - rect.left, rect.right - pos[:, 0],
        pos[:, 1] - rect.top, rect.bottom - pos[:, 1],
    ], axis=1)

    if n > 1:
        others_center = (pos.sum(0) - pos) / (n - 1)
        pod_dir, pod_dist = _unit(others_center - pos)
        diff = pos[None, :, :] - pos[:, None, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nearest = d2.argmin(1)
        nb_dir, nb_dist = _unit(pos[nearest] - pos)
    else:
        pod_dir, pod_dist = np.zeros((n, 2)), np.zeros(n)
        nb_dir, nb_dist = np.zeros((n, 2)), np.full(n, NEIGHBOR_SENSE_RANGE)

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
    ], axis=1)
    return Senses(features, shark_dist, pod_dist, walls.min(1))
