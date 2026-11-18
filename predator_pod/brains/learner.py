"""The fish brain: what it senses, what it can do, what it is rewarded for.

Network (shared by every fish):
    31 senses -> 32 hidden (tanh) -> two heads
      movement head: softmax over 14 escape behaviors (see ACTION_NAMES)
      fast-start head: sigmoid, "fire a C-start burst now?"

The behaviors are modeled on what real prey fish do against predators:
plain swimming in 8 directions, cruising, fleeing, aligning with the school,
the selfish herd (moving into the middle of the group), the fountain dodge
(splitting sideways around the attacker), protean zigzags, and the C-start
fast escape (an instant turn plus a short burst of speed, then a cooldown).

It learns two ways, both with PPO (see ppo.py):
  * automatically, from self-play against the AI shark (train.py)
  * from every round a human plays (Learner.learn, called by the game)

Rewards (per decision, about 7 per second):
    + being far from the shark          (the main goal)
    + being near the school's center    (small, so it can wander and rejoin)
    + being in a group, most of all when the shark is close
    - being alone while the shark is close
    - being near a wall, or in a corner (worse while chased)
    - being eaten                       (big penalty)

Human-play stopping: once the rolling average survival time of human rounds
stops improving for PATIENCE rounds (after at least MIN_ROUNDS rounds), the
brain stops learning from human play and fish always pick their best behavior.

Upgrades: a brain saved by an older version (fewer senses or behaviors) is
widened on load. New inputs get zero weights and new behaviors start at about
NEW_ACTION_SHARE of the choices, so the fish keep everything they learned.
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np

from predator_pod.brains.ppo import PPOPolicy, Trajectories
from predator_pod.world.sensors import N_FEATURES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_PATH = MODELS_DIR / "pod_brain.npz"
TRAINING_LOCK = MODELS_DIR / ".training.lock"
LOCK_FRESH_SECONDS = 300      # a lock not touched for this long is treated as stale
VERSION = 2

N_INPUTS = N_FEATURES
N_HIDDEN = 32
ACTION_NAMES = [
    "up", "down", "left", "right",
    "up-left", "up-right", "down-left", "down-right",
    "cruise", "flee", "align", "selfish herd", "fountain", "zigzag",
]
N_ACTIONS = len(ACTION_NAMES)
N_DIRECTION_ACTIONS = 8
_D = np.sqrt(0.5)
ACTION_DIRS = np.array([
    [0, -1], [0, 1], [-1, 0], [1, 0],
    [-_D, -_D], [_D, -_D], [-_D, _D], [_D, _D],
], dtype=np.float64)
CRUISE, FLEE, ALIGN, HERD, FOUNTAIN, ZIGZAG = range(8, 14)

DECISION_INTERVAL = 0.15      # s between decisions (direction is held in between)

# Rewards (per decision)
R_FAR_FROM_SHARK = 1.0        # x min(distance / FAR_DISTANCE, 1)
FAR_DISTANCE = 300.0          # px, farther than this counts as fully safe
R_NEAR_POD = 0.25             # x (1 - min(pod distance / POD_RANGE, 1))
POD_RANGE = 150.0
THREAT_RANGE = 250.0          # px, threat = 1 - min(shark distance / this, 1)
R_GROUP = 0.4                 # x min(neighbors / GROUP_FULL, 1) x (0.5 + threat)
GROUP_FULL = 4
R_ALONE = -0.5                # x threat, when no fish within 60 px
R_WALL = -0.6                 # x (1 - wall distance / WALL_DANGER) x (1 + threat)
WALL_DANGER = 60.0
R_CORNER = -0.4               # x (1 + threat)
R_EATEN = -10.0

GAMMA = 0.95                  # how much future rewards count (per decision)
LEARNING_RATE = 0.003
HUMAN_EPOCHS = 10             # passes over one human round's data
HUMAN_BATCH = 256

NEW_ACTION_SHARE = 0.10       # after an upgrade, new behaviors together start at ~10%
NEW_BURST_BIAS = -4.0         # bursts start rare (~2%)

ROLLING_WINDOW = 5            # rounds
IMPROVE_TOLERANCE = 0.03      # rolling average must beat the best by 3%
PATIENCE = 10                 # rounds without improvement -> converged
MIN_ROUNDS = 20
MIN_ROUND_SECONDS = 5.0       # shorter rounds (a quick Esc) are ignored

Episode = Trajectories        # one trajectory per fish


def threat_level(shark_dist):
    return 1.0 - np.minimum(shark_dist / THREAT_RANGE, 1.0)


def reward(shark_dist, pod_dist, wall_dist, crowd_count, corner):
    """Reward for one decision, from the state right after it."""
    threat = threat_level(shark_dist)
    r = R_FAR_FROM_SHARK * np.minimum(shark_dist / FAR_DISTANCE, 1.0)
    r = r + R_NEAR_POD * (1.0 - np.minimum(pod_dist / POD_RANGE, 1.0))
    r = r + R_GROUP * np.minimum(crowd_count / GROUP_FULL, 1.0) * (0.5 + threat)
    r = r + np.where(crowd_count == 0, R_ALONE * threat, 0.0)
    r = r + R_WALL * np.maximum(1.0 - wall_dist / WALL_DANGER, 0.0) * (1.0 + threat)
    r = r + np.where(corner, R_CORNER * (1.0 + threat), 0.0)
    return r


def training_running():
    """True while train.py is running (it keeps the lock file fresh)."""
    try:
        return time.time() - TRAINING_LOCK.stat().st_mtime < LOCK_FRESH_SECONDS
    except OSError:
        return False


class Learner(PPOPolicy):
    """The fish brain plus its human-play learning record and file storage."""

    def __init__(self, path=DEFAULT_PATH, seed=None):
        super().__init__(N_INPUTS, N_HIDDEN, N_ACTIONS, yes_no_head=True,
                         learning_rate=LEARNING_RATE, seed=seed)
        self.path = Path(path) if path else None
        self.upgraded_from = None
        self.reset(save=False)

    @classmethod
    def from_weights(cls, weights, seed=None):
        """A file-less brain for worker processes."""
        brain = cls(path=None, seed=seed)
        brain.set_weights(weights)
        return brain

    def act(self, features, ready, greedy=None):
        """Pick a behavior and a burst decision for every fish."""
        return self.sample(features, ready, self.converged if greedy is None else greedy)

    # ----- learning from a human round --------------------------------------
    def learn(self, episode, survival_times, round_seconds, stats=None):
        """One update from a finished human round. Returns True if it learned."""
        if self.converged or round_seconds < MIN_ROUND_SECONDS or training_running():
            return False
        batch = episode.batch(GAMMA)
        if batch is None:
            return False
        self.ppo_update(batch, HUMAN_EPOCHS, HUMAN_BATCH)
        self.rounds += 1
        self.history.append(float(np.mean(survival_times)))
        stats = stats or {}
        self.school_history.append(float(stats.get("school_pct", np.nan)))
        self.wall_death_history.append(float(stats.get("wall_deaths", np.nan)))
        self._check_convergence()
        self.save()
        return True

    def _check_convergence(self):
        if len(self.history) < ROLLING_WINDOW:
            return
        rolling = float(np.mean(self.history[-ROLLING_WINDOW:]))
        if rolling > self.best_rolling * (1 + IMPROVE_TOLERANCE):
            self.best_rolling = rolling
            self.stale = 0
        else:
            self.stale += 1
        if self.stale >= PATIENCE and self.rounds >= MIN_ROUNDS:
            self.converged = True

    # ----- save / load -----------------------------------------------------
    def reset(self, save=True):
        self.init_params(NEW_BURST_BIAS)
        self.rounds = 0
        self.history = []
        self.school_history = []
        self.wall_death_history = []
        self.best_rolling = 0.0
        self.stale = 0
        self.converged = False
        self.selfplay_rounds = 0
        self.upgraded_from = None
        if save:
            self.save()

    def backup_and_reset(self):
        """The landing page's Reset brain: keep a copy first, then start over."""
        if self.path and self.path.exists():
            backup = MODELS_DIR / "backups" / "pod_brain_before_reset.npz"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path, backup)
        self.reset()

    @classmethod
    def load_or_new(cls, path=DEFAULT_PATH):
        learner = cls(path)
        if learner.path.exists():
            try:
                learner.load()
            except (OSError, KeyError, ValueError):
                # Never silently overwrite a brain we couldn't read: keep a copy.
                keep = learner.path.with_name(learner.path.stem + ".unreadable.npz")
                try:
                    shutil.copyfile(learner.path, keep)
                except OSError:
                    pass
                learner.reset(save=False)
        return learner

    def load(self, path=None):
        with np.load(path or self.path) as f:      # read everything, then close (Windows locks)
            d = {k: f[k] for k in f.files}
        n_in, n_hid, n_act = (int(v) for v in d["layers"])
        if n_hid != N_HIDDEN or n_in > N_INPUTS or n_act > N_ACTIONS:
            raise ValueError("saved brain has an incompatible network shape")
        for k in ("w1", "b1", "w2", "b2") + (("w3", "b3") if "w3" in d else ()):
            setattr(self, k, d[k].astype(np.float64))
            self.m[k] = d["m_" + k].astype(np.float64)
            self.v[k] = d["v_" + k].astype(np.float64)
        self.adam_t = int(d["adam_t"])
        self.rounds = int(d["rounds"])
        self.history = list(d["history"].astype(np.float64))
        pad = [np.nan] * len(self.history)
        self.school_history = list(d["school_history"]) if "school_history" in d else list(pad)
        self.wall_death_history = (list(d["wall_death_history"]) if "wall_death_history" in d
                                   else list(pad))
        self.best_rolling = float(d["best_rolling"])
        self.stale = int(d["stale"])
        self.converged = bool(d["converged"])
        self.selfplay_rounds = int(d["selfplay_rounds"]) if "selfplay_rounds" in d else 0
        if n_in < N_INPUTS or n_act < N_ACTIONS or "w3" not in d:
            self._widen(n_in, n_act, has_burst="w3" in d)

    def _widen(self, n_in, n_act, has_burst):
        """Upgrade an older, smaller brain in place, keeping everything it learned."""
        self.upgraded_from = (n_in, n_act)
        add_in, add_act = N_INPUTS - n_in, N_ACTIONS - n_act
        rows = lambda a: np.vstack([a, np.zeros((add_in,) + a.shape[1:])])
        cols = lambda a, fill=0.0: np.hstack([a, np.full(a.shape[:-1] + (add_act,), fill)])
        self.w1 = rows(self.w1)
        self.m["w1"], self.v["w1"] = rows(self.m["w1"]), rows(self.v["w1"])
        old_w2, old_b2 = self.w2, self.b2
        self.w2 = cols(self.w2)
        self.b2 = cols(self.b2, self._new_action_bias(old_w2, old_b2, add_act))
        for k in ("w2", "b2"):
            self.m[k], self.v[k] = cols(self.m[k]), cols(self.v[k])
        if not has_burst:
            self.w3 = np.zeros(N_HIDDEN)
            self.b3 = np.array([NEW_BURST_BIAS])
            self.m["w3"], self.v["w3"] = np.zeros(N_HIDDEN), np.zeros(N_HIDDEN)
            self.m["b3"], self.v["b3"] = np.zeros(1), np.zeros(1)
        # The goal changed, so give the brain a fresh patience window.
        self.stale = 0
        self.converged = False
        self.save()

    def _new_action_bias(self, old_w2, old_b2, add_act):
        """Bias for new behaviors so together they get ~NEW_ACTION_SHARE of the
        choices. The old behaviors keep their exact ratios to each other."""
        if add_act <= 0:
            return 0.0
        x = self.rng.uniform(-1, 1, (1024, N_INPUTS))
        h = np.tanh(x @ self.w1 + self.b1)
        z = h @ old_w2 + old_b2
        lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
        lo, hi = -30.0, 30.0
        for _ in range(60):
            c = (lo + hi) / 2
            share = (add_act / (add_act + np.exp(lse - c))).mean()
            lo, hi = (c, hi) if share < NEW_ACTION_SHARE else (lo, c)
        return (lo + hi) / 2

    def save(self, path=None):
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp, version=VERSION, rounds=self.rounds, history=np.array(self.history),
            school_history=np.array(self.school_history, dtype=np.float64),
            wall_death_history=np.array(self.wall_death_history, dtype=np.float64),
            best_rolling=self.best_rolling, stale=self.stale, converged=self.converged,
            selfplay_rounds=self.selfplay_rounds, **self.state_arrays(),
        )
        os.replace(tmp, path)

    def status_text(self):
        auto = f", {self.selfplay_rounds:,} self-play rounds" if self.selfplay_rounds else ""
        if self.converged:
            return f"Fish brain: trained - human learning finished after {self.rounds} rounds{auto}"
        if self.rounds == 0:
            return f"Fish brain: learns after every round you play{auto}"
        return (f"Fish brain: learning - {self.rounds} human rounds{auto}, "
                f"no improvement for {self.stale}/{PATIENCE}")
