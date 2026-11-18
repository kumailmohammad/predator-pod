"""Headless rounds for automatic training (run inside worker processes).

A task describes one round: who the shark is (the learning AI, an older AI
snapshot, or a scripted human-style hunter), how many fish, and the brains'
weights. The result carries both sides' experience and the round's stats.
"""

import os
import random
import signal

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # tiny matrices: one thread per process is fastest

import numpy as np  # noqa: E402

from predator_pod.brains import learner as fish_mod  # noqa: E402
from predator_pod.brains import shark_brain as shark_mod  # noqa: E402
from predator_pod.world.scripted_sharks import ScriptedShark  # noqa: E402
from predator_pod.world.simulation import World  # noqa: E402

DT = 1.0 / 30.0               # fixed physics step for training (the game runs at 60 FPS)
ROUND_CAP = 60.0              # s, a round ends here; survivors count as surviving 60 s


def worker_init():
    # Ctrl+C is handled by the main process, which then stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_round(task):
    seed = task["seed"]
    random.seed(seed)
    fish = fish_mod.Learner.from_weights(task["fish"], seed=seed)
    kind = task["opponent"]
    if kind in ScriptedShark.KINDS:
        ctrl = ScriptedShark(kind)
    else:
        shark = shark_mod.SharkBrain.from_weights(task["shark"], seed=seed + 1)
        ctrl = shark_mod.AISharkController(shark, greedy=task.get("shark_greedy", False),
                                           record=task.get("record_shark", False))
    world = World(fish, task["fish_count"], ctrl, greedy_fish=task.get("fish_greedy", False))
    while world.fishes and world.elapsed < ROUND_CAP:
        world.step(DT)

    eaten = world.total - len(world.fishes)
    out = {
        "opponent": kind,
        "fish_count": world.total,
        "elapsed": world.elapsed,
        "eaten": eaten,
        "survival": float(np.mean(world.survival_times())),
        "stats": {k: v for k, v in world.stats().items() if k != "top_actions"},
        "actions": world.action_counts,
        "fish_batch": world.episode.batch(fish_mod.GAMMA) if task.get("record_fish") else None,
        "shark_batch": None,
        "shark_reward": None,
    }
    if isinstance(ctrl, shark_mod.AISharkController):
        out["shark_reward"] = ctrl.total_reward / max(ctrl.decisions, 1)
        if ctrl.traj is not None:
            out["shark_batch"] = ctrl.traj.batch(shark_mod.GAMMA)
    return out
