"""Train the pod headless (no window) against the computer shark.

    python train.py                 # continue from models/pod_brain.npz
    python train.py --reset         # start from a fresh random brain

Each generation tests 2*pairs candidate brains. In each round the whole pod
uses one candidate, and the score is the pod's mean survival time. A +/- pair
of candidates gets the exact same starting layout so their comparison is fair.

Stops automatically when the learner reports convergence (or at --generations).
The brain is saved after every generation, so Ctrl+C is safe anytime.
"""

import argparse
import random
import time

import numpy as np

from predator_pod.brains.evolution import PodLearner
from predator_pod.brains.neural import NeuralBrain
from predator_pod.world.shark_ai import ai_target
from predator_pod.world.simulation import World


def pod_score(params, fish_count, max_seconds, dt, seed):
    """Mean seconds survived when every fish uses `params`."""
    random.seed(seed)
    world = World([NeuralBrain(params) for _ in range(fish_count)], random_shark_start=True)
    while world.fishes and world.elapsed < max_seconds:
        world.step(dt, ai_target(world.shark, world.fishes))
    return float(np.mean(world.fitness()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fish", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=5, help="+/- candidate pairs per generation")
    ap.add_argument("--generations", type=int, default=1000, help="max generations this run")
    ap.add_argument("--round-seconds", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=1 / 30)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    learner = PodLearner.load_or_new()
    if args.reset:
        learner.reset()
    if learner.converged:
        print(f"Already converged at generation {learner.generation}. Use --reset to retrain.")
        return

    start = time.time()
    for _ in range(args.generations):
        cands = learner.candidates(args.pairs)
        seeds = [random.randrange(1 << 30) for _ in range(args.pairs)] * 2   # same layout per pair
        scores = [pod_score(c, args.fish, args.round_seconds, args.dt, s) for c, s in zip(cands, seeds)]
        learner.step_candidates(scores)
        recent = learner.history[-20:]
        print(
            f"gen {learner.generation:4d} | pod survival {learner.history[-1]:5.1f}s"
            f" | rolling(20) {sum(recent) / len(recent):5.1f}s | best candidate {max(scores):5.1f}s"
            f" | sigma {learner.sigma:.3f} | stale {learner.stale:2d} | {time.time() - start:5.0f}s",
            flush=True,
        )
        if learner.converged:
            print(f"Converged at generation {learner.generation}. Brain saved to {learner.path}")
            break


if __name__ == "__main__":
    main()
