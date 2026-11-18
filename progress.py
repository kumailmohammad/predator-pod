"""Graphs of how the fish (prey) and the AI shark are learning.

    python progress.py            # open the graphs (also saved as a PNG)
    python progress.py --live     # refresh every 30 s while train.py runs
    python progress.py --no-show  # only save models/logs/training_progress.png
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
LOGS = MODELS / "logs"
PNG = LOGS / "training_progress.png"


def read_csv(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(math.nan)
        out[k] = np.array(vals)
    return out


def smooth(y, n=10):
    """Trailing moving average that skips missing values."""
    if len(y) < 2:
        return y
    ok = ~np.isnan(y)
    s = np.convolve(np.where(ok, y, 0.0), np.ones(n))[:len(y)]
    c = np.convolve(ok.astype(float), np.ones(n))[:len(y)]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def human_history():
    path = MODELS / "pod_brain.npz"
    if not path.exists():
        return np.array([])
    with np.load(path) as d:
        return d["history"].astype(float) if "history" in d else np.array([])


def draw(fig):
    tr = read_csv(LOGS / "train_log.csv")
    bm = read_csv(LOGS / "benchmark_log.csv")
    human = human_history()
    fig.clear()
    axes = fig.subplots(2, 3)
    fig.suptitle("Predator-Pod training progress", fontsize=14, fontweight="bold")
    for ax in axes.flat:
        ax.grid(alpha=0.3)

    # ---------------- prey ----------------
    ax = axes[0, 0]
    ax.set_title("PREY - benchmark survival (fixed test)")
    if bm:
        for k, style in (("nearest", ":"), ("isolated", ":"), ("intercept", ":"), ("vs_ai", "--")):
            ax.plot(bm["minutes"], bm[k], style, label=f"vs {k.replace('vs_', '')} shark")
        ax.plot(bm["minutes"], bm["scripted_mean"], "-", lw=2.5, color="black", label="score (scripted avg)")
        ax.step(bm["minutes"], bm["best_score"], where="post", color="green", lw=1, label="best saved")
        saved = bm["saved"] > 0
        ax.plot(bm["minutes"][saved], bm["scripted_mean"][saved], "g*", ms=11)
        ax.legend(fontsize=7)
    ax.set_xlabel("training minutes")
    ax.set_ylabel("avg fish survival (s, max 60)")

    ax = axes[0, 1]
    ax.set_title("PREY - survival during training")
    if tr:
        m = tr["minutes"]
        ax.plot(m, smooth(tr["fish_survival_vs_ai"]), label="vs learning AI shark")
        ax.plot(m, smooth(tr["fish_survival_vs_scripted"]), label="vs scripted sharks")
        ax.plot(m, smooth(tr["fish_survival"]), lw=2.5, color="black", label="all rounds")
        ax.legend(fontsize=7)
    ax.set_xlabel("training minutes")
    ax.set_ylabel("avg fish survival (s)")

    ax = axes[0, 2]
    ax.set_title("PREY - schooling and wall deaths")
    if tr:
        ax.plot(tr["minutes"], smooth(tr["school_pct"]), color="tab:blue", label="in a group (%)")
        ax.set_ylabel("in a group (%)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(tr["minutes"], smooth(tr["wall_deaths"]), color="tab:red", label="eaten at walls")
        ax2.set_ylabel("eaten at walls per round", color="tab:red")
    ax.set_xlabel("training minutes")

    # ---------------- shark ----------------
    ax = axes[1, 0]
    ax.set_title("SHARK - catches per minute")
    if tr:
        ax.plot(tr["minutes"], tr["shark_catch_per_min"], alpha=0.3, color="tab:red")
        ax.plot(tr["minutes"], smooth(tr["shark_catch_per_min"]), lw=2.5, color="tab:red")
    ax.set_xlabel("training minutes")
    ax.set_ylabel("fish eaten per minute")

    ax = axes[1, 1]
    ax.set_title("SHARK - average reward per decision")
    if tr:
        ax.plot(tr["minutes"], tr["shark_reward"], alpha=0.3, color="tab:purple")
        ax.plot(tr["minutes"], smooth(tr["shark_reward"]), lw=2.5, color="tab:purple")
    ax.set_xlabel("training minutes")
    ax.set_ylabel("reward")

    ax = axes[1, 2]
    ax.set_title("YOUR GAMES - fish survival per human round")
    if len(human):
        x = np.arange(1, len(human) + 1)
        ax.plot(x, human, "o-", ms=3, alpha=0.5, color="gray", label="per round")
        ax.plot(x, smooth(human, 5), lw=2.5, color="black", label="5-round average")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no human rounds yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("round you played")
    ax.set_ylabel("avg fish survival (s)")

    if tr:
        last = {k: v[-1] for k, v in tr.items()}
        fig.text(0.5, 0.005,
                 f"{int(last['rounds_total']):,} self-play rounds in {last['minutes']:.0f} min  |  "
                 f"{last['rounds_per_sec']:.1f} rounds/s  |  shark pool {int(last['pool_size'])}  |  "
                 f"best benchmark {bm['best_score'][-1]:.1f} s" if bm else "",
                 ha="center", fontsize=9)
    elif not bm:
        fig.text(0.5, 0.5, "No training logs yet - run:  python train.py", ha="center", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    LOGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="refresh every 30 s")
    ap.add_argument("--no-show", action="store_true", help="only save the PNG")
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    draw(fig)
    print(f"Saved {PNG}")
    if args.no_show:
        return
    if args.live:
        plt.ion()
        plt.show()
        while plt.fignum_exists(fig.number):
            t = time.time()
            while time.time() - t < 30 and plt.fignum_exists(fig.number):
                plt.pause(0.5)
            if plt.fignum_exists(fig.number):
                draw(fig)
                fig.canvas.draw_idle()
    else:
        plt.show()


if __name__ == "__main__":
    main()
