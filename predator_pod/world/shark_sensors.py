"""What the AI shark can sense: the 29 inputs to its neural network.

Everything is measured in the shark's own frame ("ahead" and "to my right"),
because the shark's outputs are turn commands.
"""

import numpy as np

from predator_pod import config

N_SHARK_FEATURES = 29

RANGE = 600.0                 # px, distances are divided by this (capped at 1)
WALL_RAY_RANGE = 300.0
CROWD_RANGE = 60.0            # same "group" radius the fish use
ISOLATED_SEARCH = 300.0       # px, look for lone fish this close
N_NEAREST = 3
RAY_ANGLES = (0.0, -45.0, 45.0, -90.0, 90.0)   # ahead, ahead-left, ahead-right, left, right

FEATURE_NAMES = (
    ["speed", "turn_speed"]
    + [f"wall_ray_{a:+.0f}" for a in RAY_ANGLES]
    + [f"fish{k}_{c}" for k in range(N_NEAREST) for c in ("ahead", "right", "dist")]
    + ["nearest_heading_ahead", "nearest_heading_right", "nearest_crowd",
       "nearest_wall", "nearest_burst_ready"]
    + ["lone_ahead", "lone_right", "lone_dist"]
    + ["pod_ahead", "pod_right", "pod_dist", "pod_spread"]
    + ["fish_left"]
)
assert len(FEATURE_NAMES) == N_SHARK_FEATURES


def _ray_to_wall(pos, d, rect):
    t = np.inf
    if d[0] > 1e-9:
        t = min(t, (rect.right - pos[0]) / d[0])
    elif d[0] < -1e-9:
        t = min(t, (rect.left - pos[0]) / d[0])
    if d[1] > 1e-9:
        t = min(t, (rect.bottom - pos[1]) / d[1])
    elif d[1] < -1e-9:
        t = min(t, (rect.top - pos[1]) / d[1])
    return max(t, 0.0)


def shark_sense(shark, fishes, arena, total):
    """Returns (features (29,), distance to the nearest fish in px)."""
    rect = arena.rect
    spos = np.array([shark.pos.x, shark.pos.y])
    fwd = np.array([shark.heading.x, shark.heading.y])
    right = np.array([-fwd[1], fwd[0]])          # screen y points down
    local = lambda v: np.stack([v @ fwd, v @ right], axis=-1)

    f = [shark.speed / config.SHARK_SPEED, shark.turn_speed / config.SHARK_TURN_RATE]
    for a in RAY_ANGLES:
        d = shark.heading.rotate(a)
        f.append(min(_ray_to_wall(spos, (d.x, d.y), rect) / WALL_RAY_RANGE, 1.0))

    n = len(fishes)
    if n == 0:
        return np.array(f + [0.0, 0.0, 1.0] * N_NEAREST + [0.0] * 5 + [0.0, 0.0, 1.0]
                        + [0.0, 0.0, 1.0, 0.0, 0.0]), RANGE

    pos = np.array([[x.pos.x, x.pos.y] for x in fishes])
    heads = np.array([[x.heading.x, x.heading.y] for x in fishes])
    rel = pos - spos
    dist = np.linalg.norm(rel, axis=1)
    order = dist.argsort()
    unit = lambda v, d: v / max(d, 1e-9)

    for k in range(N_NEAREST):
        if k < n:
            i = order[k]
            f += list(local(unit(rel[i], dist[i]))) + [min(dist[i] / RANGE, 1.0)]
        else:
            f += [0.0, 0.0, 1.0]

    pair = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
    np.fill_diagonal(pair, np.inf)
    crowd = (pair < CROWD_RANGE).sum(1)
    i = order[0]
    p = fishes[i].pos
    wall = min(p.x - rect.left, rect.right - p.x, p.y - rect.top, rect.bottom - p.y)
    f += list(local(heads[i])) + [min(crowd[i] / 6, 1.0), min(wall / 150.0, 1.0),
                                  1.0 if fishes[i].burst_ready else 0.0]

    near = np.where(dist < ISOLATED_SEARCH)[0]
    if len(near):
        j = near[np.lexsort((dist[near], crowd[near]))[0]]   # fewest neighbors, then closest
        f += list(local(unit(rel[j], dist[j]))) + [min(dist[j] / RANGE, 1.0)]
    else:
        f += [0.0, 0.0, 1.0]

    center = pos.mean(0)
    to_c = center - spos
    dc = np.linalg.norm(to_c)
    spread = np.linalg.norm(pos - center, axis=1).mean()
    f += list(local(unit(to_c, dc))) + [min(dc / RANGE, 1.0), min(spread / 200.0, 1.0)]
    f.append(n / max(total, 1))
    return np.array(f, dtype=np.float64), float(dist[order[0]])
