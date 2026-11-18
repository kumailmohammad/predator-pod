"""Automatic self-play training: the AI shark hunts, the fish learn to escape.

    python train.py                 # 30 minutes (resumes where the last run stopped)
    python train.py --minutes 120   # longer run
    python train.py --fresh-shark   # start the AI shark from scratch

Each iteration plays a batch of headless rounds in parallel. Opponents:
the current AI shark (which learns too), older AI sharks from a snapshot pool
(so the fish don't forget old tricks), and scripted human-style hunters.
Both brains are updated with PPO after every iteration.

Every few iterations the fish take a fixed benchmark (fixed seeds) against the
scripted hunters. models/pod_brain.npz, the brain the game uses, is replaced
only when that benchmark improves. Ctrl+C stops cleanly; run again to resume.
Watch progress with:  python progress.py
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # tiny matrices: one thread per process is fastest

import numpy as np  # noqa: E402

from predator_pod.brains import learner as fish_mod  # noqa: E402
from predator_pod.brains.ppo import merge_batches  # noqa: E402
from predator_pod.brains.shark_brain import SharkBrain  # noqa: E402
from predator_pod.selfplay import run_round, worker_init  # noqa: E402
from predator_pod.world.scripted_sharks import ScriptedShark  # noqa: E402

MODELS = fish_mod.MODELS_DIR
CHECKPOINTS = MODELS / "checkpoints"
POOL_DIR = CHECKPOINTS / "shark_pool"
LOGS = MODELS / "logs"
STATE_FILE = CHECKPOINTS / "state.json"
FISH_LATEST = CHECKPOINTS / "fish_latest.npz"
SHARK_LATEST = CHECKPOINTS / "shark_latest.npz"

ROUNDS_PER_WORKER = 3         # rounds per worker per iteration
MIX = {"current": 0.6, "pool": 0.2, "scripted": 0.2}
FISH_EPOCHS, FISH_BATCH = 3, 4096
SHARK_EPOCHS, SHARK_BATCH = 4, 1024
BENCH_EVERY = 10              # iterations
BENCH_SEEDS = [101, 202, 303, 404, 505, 606]
BENCH_FISH = 20
POOL_EVERY = 20               # iterations between shark snapshots
POOL_SIZE = 10

TRAIN_COLUMNS = ["iteration", "minutes", "rounds_total", "rounds_per_sec",
                 "fish_survival", "fish_survival_vs_ai", "fish_survival_vs_scripted",
                 "school_pct", "wall_deaths", "bursts",
                 "shark_catch_per_min", "shark_reward", "pool_size"]
BENCH_COLUMNS = ["iteration", "minutes", "nearest", "isolated", "intercept",
                 "scripted_mean", "vs_ai", "best_score", "saved"]


def append_csv(path, columns, row):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k]) for k in columns})


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"iteration": 0, "minutes": 0.0, "rounds_total": 0, "stale_evals": 0}


def save_state(state):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def load_fish():
    """Latest training checkpoint, unless the game's brain is newer (you played since)."""
    game = fish_mod.DEFAULT_PATH
    use_latest = FISH_LATEST.exists() and (
        not game.exists() or FISH_LATEST.stat().st_mtime >= game.stat().st_mtime)
    fish = fish_mod.Learner.load_or_new(game)
    if use_latest:
        fish.load(FISH_LATEST)
        print("Fish: resuming from the last training checkpoint")
    else:
        print(f"Fish: starting from the game's brain ({fish.rounds} human rounds, "
              f"{fish.selfplay_rounds:,} self-play rounds)")
    return fish


def load_shark(fresh):
    shark = SharkBrain(path=SHARK_LATEST)
    if not fresh:
        for p in (SHARK_LATEST, fish_mod.MODELS_DIR / "shark_brain.npz"):
            if p.exists():
                try:
                    shark.load(p)
                    print(f"Shark: resuming ({shark.rounds:,} rounds learned)")
                    return shark
                except (OSError, KeyError, ValueError):
                    pass
    print("Shark: starting from scratch")
    return shark


def load_pool():
    pool = []
    for p in sorted(POOL_DIR.glob("shark_*.npz")):
        try:
            s = SharkBrain(path=p)
            s.load()
            pool.append(s.weights())
        except (OSError, KeyError, ValueError):
            pass
    return pool


def benchmark(pool, fish, shark):
    """Fixed test: the same seeds every time, so scores are comparable."""
    fw, sw = fish.weights(), shark.weights()
    tasks = []
    for kind in ScriptedShark.KINDS + ("ai",):
        for seed in BENCH_SEEDS:
            tasks.append({"seed": seed, "fish": fw, "fish_count": BENCH_FISH, "opponent": kind,
                          "shark": sw, "shark_greedy": True, "fish_greedy": fish.converged})
    results = pool.map_async(run_round, tasks).get(timeout=3600)
    score = {}
    for kind in ScriptedShark.KINDS + ("ai",):
        score[kind] = float(np.mean([r["survival"] for r in results if r["opponent"] == kind]))
    score["scripted_mean"] = float(np.mean([score[k] for k in ScriptedShark.KINDS]))
    return score


def touch_lock():
    fish_mod.TRAINING_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fish_mod.TRAINING_LOCK.write_text(str(os.getpid()), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30.0, help="training time (default 30)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--patience", type=int, default=15,
                    help="stop after this many benchmarks without improvement (default 15)")
    ap.add_argument("--fresh-shark", action="store_true", help="start the AI shark from scratch")
    args = ap.parse_args()

    touch_lock()
    state = load_state()
    fish = load_fish()
    shark = load_shark(args.fresh_shark)
    snapshots = load_pool()
    rng = np.random.default_rng()
    pool = mp.Pool(args.workers, initializer=worker_init)
    start = time.time()
    run_minutes0 = state["minutes"]
    best = float("nan")
    print(f"Training for up to {args.minutes:g} min with {args.workers} workers "
          f"(Ctrl+C to stop; progress: python progress.py)")

    try:
        base = benchmark(pool, fish_mod.Learner.load_or_new(), shark)
        best = base["scripted_mean"]
        print(f"Benchmark of the game's current fish: {best:.1f} s average survival "
              f"vs scripted sharks (vs AI shark {base['ai']:.1f} s)")
        append_csv(LOGS / "benchmark_log.csv", BENCH_COLUMNS, {
            "iteration": state["iteration"], "minutes": run_minutes0, **{k: base[k] for k in (
                "nearest", "isolated", "intercept", "scripted_mean")},
            "vs_ai": base["ai"], "best_score": best, "saved": 0})
        stale = 0

        while True:
            elapsed_min = (time.time() - start) / 60
            if elapsed_min >= args.minutes:
                print("Time budget reached.")
                break
            t0 = time.time()
            state["iteration"] += 1
            it = state["iteration"]
            fw, sw = fish.weights(), shark.weights()
            tasks = []
            for _ in range(args.workers * ROUNDS_PER_WORKER):
                u = rng.random()
                task = {"seed": int(rng.integers(1 << 30)), "fish": fw,
                        "fish_count": int(rng.integers(10, 31)), "record_fish": True}
                if u < MIX["current"] or (u < MIX["current"] + MIX["pool"] and not snapshots):
                    task.update(opponent="current", shark=sw, record_shark=True)
                elif u < MIX["current"] + MIX["pool"]:
                    task.update(opponent="pool", shark=snapshots[rng.integers(len(snapshots))])
                else:
                    task.update(opponent=str(rng.choice(ScriptedShark.KINDS)))
                tasks.append(task)
            results = pool.map_async(run_round, tasks).get(timeout=3600)

            fish_batch = merge_batches([r["fish_batch"] for r in results])
            if fish_batch is not None:
                fish.ppo_update(fish_batch, FISH_EPOCHS, FISH_BATCH)
            shark_batch = merge_batches([r["shark_batch"] for r in results])
            if shark_batch is not None:
                shark.ppo_update(shark_batch, SHARK_EPOCHS, SHARK_BATCH)
            n_current = sum(r["opponent"] == "current" for r in results)
            fish.selfplay_rounds += len(results)
            shark.rounds += n_current
            state["rounds_total"] += len(results)

            cur = [r for r in results if r["opponent"] == "current"]
            scr = [r for r in results if r["opponent"] in ScriptedShark.KINDS]
            mean = lambda rs, f: float(np.mean([f(r) for r in rs])) if rs else float("nan")
            row = {
                "iteration": it, "minutes": run_minutes0 + (time.time() - start) / 60,
                "rounds_total": state["rounds_total"],
                "rounds_per_sec": len(results) / max(time.time() - t0, 1e-6),
                "fish_survival": mean(results, lambda r: r["survival"]),
                "fish_survival_vs_ai": mean(cur, lambda r: r["survival"]),
                "fish_survival_vs_scripted": mean(scr, lambda r: r["survival"]),
                "school_pct": mean(results, lambda r: r["stats"]["school_pct"]),
                "wall_deaths": mean(results, lambda r: r["stats"]["wall_deaths"]),
                "bursts": mean(results, lambda r: r["stats"]["bursts"]),
                "shark_catch_per_min": mean(cur, lambda r: 60 * r["eaten"] / max(r["elapsed"], 1e-6)),
                "shark_reward": mean(cur, lambda r: r["shark_reward"]),
                "pool_size": len(snapshots),
            }
            append_csv(LOGS / "train_log.csv", TRAIN_COLUMNS, row)
            print(f"it {it:5d} | {row['minutes']:6.1f} min | {row['rounds_per_sec']:5.1f} rounds/s | "
                  f"fish survive {row['fish_survival']:5.1f}s (vs AI {row['fish_survival_vs_ai']:5.1f}s) | "
                  f"group {row['school_pct']:4.0f}% | shark {row['shark_catch_per_min']:5.1f} catches/min | "
                  f"best {best:5.1f}s")

            if it % POOL_EVERY == 0:
                POOL_DIR.mkdir(parents=True, exist_ok=True)
                shark.save(POOL_DIR / f"shark_{it:06d}.npz")
                snapshots = load_pool()
                for old in sorted(POOL_DIR.glob("shark_*.npz"))[:-POOL_SIZE]:
                    old.unlink()
                snapshots = snapshots[-POOL_SIZE:]

            if it % BENCH_EVERY == 0:
                b = benchmark(pool, fish, shark)
                saved = b["scripted_mean"] > best
                if saved:
                    best = b["scripted_mean"]
                    stale = 0
                    fish.save(fish_mod.DEFAULT_PATH)        # the game's brain
                else:
                    stale += 1
                shark.save(MODELS / "shark_brain.npz")      # the game's "Watch AI" shark
                append_csv(LOGS / "benchmark_log.csv", BENCH_COLUMNS, {
                    "iteration": it, "minutes": row["minutes"],
                    **{k: b[k] for k in ("nearest", "isolated", "intercept", "scripted_mean")},
                    "vs_ai": b["ai"], "best_score": best, "saved": int(saved)})
                print(f"   benchmark: {b['scripted_mean']:.1f}s vs scripted, {b['ai']:.1f}s vs AI shark"
                      f" -> {'NEW BEST, saved to the game' if saved else f'no improvement ({stale}/{args.patience})'}")
                fish.save(FISH_LATEST)
                shark.save(SHARK_LATEST)
                state["minutes"] = row["minutes"]
                save_state(state)
                touch_lock()
                if stale >= args.patience:
                    print("The fish stopped improving: training converged.")
                    break
            if it % 5 == 0:
                touch_lock()
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    finally:
        pool.terminate()
        pool.join()
        fish.save(FISH_LATEST)
        shark.save(SHARK_LATEST)
        if not (MODELS / "shark_brain.npz").exists():
            shutil.copyfile(SHARK_LATEST, MODELS / "shark_brain.npz")
        state["minutes"] = run_minutes0 + (time.time() - start) / 60
        save_state(state)
        try:
            fish_mod.TRAINING_LOCK.unlink()
        except OSError:
            pass
        print(f"Saved. Best fish benchmark: {best:.1f} s. "
              f"The game uses models/pod_brain.npz. Run again to continue training.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
