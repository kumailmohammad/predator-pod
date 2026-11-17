"""The fish brain and how it learns from your games.

Network (shared by every fish):
    31 senses -> 32 hidden (tanh) -> two heads
      movement head: softmax over 14 escape behaviors (see ACTION_NAMES)
      fast-start head: sigmoid, "fire a C-start burst now?"

The behaviors are modeled on what real prey fish do against predators:
plain swimming in 8 directions, cruising, fleeing, aligning with the school,
the selfish herd (moving into the middle of the group), the fountain dodge
(splitting sideways around the attacker), protean zigzags, and the C-start
fast escape (an instant turn plus a short burst of speed, then a cooldown).

Learning: reinforcement learning (policy gradient, PPO-style updates).
    About 7 times a second each fish picks a behavior (and maybe a burst).
    After each decision it gets a reward:
        + being far from the shark          (the main goal)
        + being near the school's center    (small, so it can wander and rejoin)
        + being in a group, most of all when the shark is close
        - being alone while the shark is close
        - being near a wall, or in a corner (worse while chased)
        - being eaten                       (big penalty)
    When your round ends, every decision is scored by the rewards that
    followed it ("return"). Decisions that did better than average become
    more likely, worse ones less likely. The round's data is reused for a few
    passes of small, clipped updates (PPO) so one round teaches a lot without
    the network changing too wildly.

Stopping: once the rolling average survival time stops improving for
PATIENCE rounds (after at least MIN_ROUNDS rounds), the brain is frozen and
saved. From then on fish always pick their best behavior.

Upgrades: a brain saved by an older version (fewer senses or behaviors) is
widened on load. New inputs get zero weights and new behaviors start very
unlikely, so the fish keep everything they learned and discover the new
abilities gradually.
"""

import os
import shutil
from pathlib import Path

import numpy as np

from predator_pod.world.sensors import N_FEATURES

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "pod_brain.npz"
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
ENTROPY_BONUS = 0.01          # keeps some exploration so it doesn't lock in early
GRAD_CLIP = 1.0
EPOCHS = 10                   # passes over one round's data
BATCH_SIZE = 256
CLIP = 0.2                    # PPO: max change in action probability per update

NEW_ACTION_SHARE = 0.10       # after an upgrade, new behaviors together start at ~10%
NEW_BURST_BIAS = -4.0         # bursts start rare (~2%)

ROLLING_WINDOW = 5            # rounds
IMPROVE_TOLERANCE = 0.03      # rolling average must beat the best by 3%
PATIENCE = 10                 # rounds without improvement -> converged
MIN_ROUNDS = 20
MIN_ROUND_SECONDS = 5.0       # shorter rounds (a quick Esc) are ignored


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


def _sigmoid(y):
    return 1.0 / (1.0 + np.exp(-np.clip(y, -30, 30)))


class Episode:
    """Everything the fish did in one round, for learning at the end."""

    def __init__(self, n_fish):
        self.x = [[] for _ in range(n_fish)]
        self.a = [[] for _ in range(n_fish)]
        self.b = [[] for _ in range(n_fish)]      # burst chosen (0/1)
        self.ready = [[] for _ in range(n_fish)]  # burst was available
        self.p = [[] for _ in range(n_fish)]      # probability of the whole choice
        self.r = [[] for _ in range(n_fish)]

    def decide(self, i, x, a, b, ready, p):
        self.x[i].append(x)
        self.a[i].append(a)
        self.b[i].append(b)
        self.ready[i].append(ready)
        self.p[i].append(p)
        self.r[i].append(0.0)

    def reward(self, i, r):
        if self.r[i]:
            self.r[i][-1] += r


class Learner:
    PARAMS = ("w1", "b1", "w2", "b2", "w3", "b3")

    def __init__(self, path=DEFAULT_PATH, seed=None):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.upgraded_from = None
        self.reset(save=False)

    # ----- network ---------------------------------------------------------
    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        y = h @ self.w3 + self.b3[0]
        return h, p, y, _sigmoid(y)

    def act(self, features, ready):
        """Pick a behavior (0..13) and a burst decision for every fish.

        ready: bool array, whether each fish's fast-start is available.
        Returns (actions, bursts, probability of that whole choice).
        """
        _, p, _, q = self.forward(features)
        n = len(p)
        if self.converged:
            a = p.argmax(1)
            b = (q > 0.5) & ready
        else:
            u = self.rng.random((n, 1))
            a = np.minimum((p.cumsum(1) < u).sum(1), N_ACTIONS - 1)
            b = (self.rng.random(n) < q) & ready
        pb = np.where(ready, np.where(b, q, 1.0 - q), 1.0)
        return a, b, p[np.arange(n), a] * pb

    # ----- learning --------------------------------------------------------
    def learn(self, episode, survival_times, round_seconds, stats=None):
        """One update from a finished round. Returns True if it learned."""
        if self.converged or round_seconds < MIN_ROUND_SECONDS:
            return False
        xs, acts, bs, rds, olds, rets = [], [], [], [], [], []
        for x, a, b, rd, p, r in zip(episode.x, episode.a, episode.b, episode.ready,
                                     episode.p, episode.r):
            if not r:
                continue
            g, run = np.zeros(len(r)), 0.0
            for t in range(len(r) - 1, -1, -1):
                run = r[t] + GAMMA * run
                g[t] = run
            xs += x; acts += a; bs += b; rds += rd; olds += p; rets.append(g)
        if not xs:
            return False
        x = np.array(xs)
        a = np.array(acts)
        b = np.array(bs, dtype=np.float64)
        rd = np.array(rds, dtype=np.float64)
        old_p = np.array(olds)
        g = np.concatenate(rets)
        adv = (g - g.mean()) / (g.std() + 1e-8)

        n = len(x)
        for _ in range(EPOCHS):
            order = self.rng.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                self._ppo_step(x[idx], a[idx], b[idx], rd[idx], old_p[idx], adv[idx])

        self.rounds += 1
        self.history.append(float(np.mean(survival_times)))
        stats = stats or {}
        self.school_history.append(float(stats.get("school_pct", np.nan)))
        self.wall_death_history.append(float(stats.get("wall_deaths", np.nan)))
        self._check_convergence()
        self.save()
        return True

    def _ppo_step(self, x, a, b, ready, old_p, adv):
        h, p, y, q = self.forward(x)
        n = len(a)
        onehot = np.eye(N_ACTIONS)[a]
        pb = np.where(ready > 0, np.where(b > 0, q, 1.0 - q), 1.0)
        ratio = p[np.arange(n), a] * pb / (old_p + 1e-12)
        # PPO clip: stop pushing a choice once its probability moved > CLIP.
        active = ~(((adv > 0) & (ratio > 1 + CLIP)) | ((adv < 0) & (ratio < 1 - CLIP)))
        coef = active * adv * ratio
        # Movement head: gradient of ratio*adv + ENTROPY_BONUS*entropy w.r.t. logits.
        logp = np.log(p + 1e-12)
        entropy = -(p * logp).sum(1, keepdims=True)
        gz = coef[:, None] * (onehot - p) + ENTROPY_BONUS * (-p * (logp + entropy))
        # Burst head (only where a burst was possible).
        gy = ready * (coef * (b - q) + ENTROPY_BONUS * (-q * (1 - q) * np.clip(y, -30, 30)))
        gz /= n
        gy /= n
        grads = {"w2": h.T @ gz, "b2": gz.sum(0), "w3": h.T @ gy, "b3": np.array([gy.sum()])}
        gh = (gz @ self.w2.T + gy[:, None] * self.w3[None, :]) * (1 - h * h)
        grads["w1"] = x.T @ gh
        grads["b1"] = gh.sum(0)
        self._adam_ascent(grads)

    def _adam_ascent(self, grads, b1=0.9, b2=0.999):
        norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
        scale = min(1.0, GRAD_CLIP / (norm + 1e-12))
        self.adam_t += 1
        for k, g in grads.items():
            g = g * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.adam_t)
            vh = self.v[k] / (1 - b2 ** self.adam_t)
            setattr(self, k, getattr(self, k) + LEARNING_RATE * mh / (np.sqrt(vh) + 1e-8))

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
        self.w1 = self.rng.normal(0, 1 / np.sqrt(N_INPUTS), (N_INPUTS, N_HIDDEN))
        self.b1 = np.zeros(N_HIDDEN)
        self.w2 = self.rng.normal(0, 0.01, (N_HIDDEN, N_ACTIONS))   # starts ~uniform
        self.b2 = np.zeros(N_ACTIONS)
        self.w3 = np.zeros(N_HIDDEN)
        self.b3 = np.array([NEW_BURST_BIAS])
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self.adam_t = 0
        self.rounds = 0
        self.history = []
        self.school_history = []
        self.wall_death_history = []
        self.best_rolling = 0.0
        self.stale = 0
        self.converged = False
        self.upgraded_from = None
        if save:
            self.save()

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

    def load(self):
        with np.load(self.path) as f:          # read everything, then close (Windows locks)
            d = {k: f[k] for k in f.files}
        n_in, n_hid, n_act = (int(v) for v in d["layers"])
        if n_hid != N_HIDDEN or n_in > N_INPUTS or n_act > N_ACTIONS:
            raise ValueError("saved brain has an incompatible network shape")
        for k in ("w1", "b1", "w2", "b2"):
            setattr(self, k, d[k].astype(np.float64))
            self.m[k] = d["m_" + k].astype(np.float64)
            self.v[k] = d["v_" + k].astype(np.float64)
        if "w3" in d:
            for k in ("w3", "b3"):
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

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        data = {k: getattr(self, k) for k in self.PARAMS}
        data.update({"m_" + k: self.m[k] for k in self.PARAMS})
        data.update({"v_" + k: self.v[k] for k in self.PARAMS})
        np.savez(
            tmp, version=VERSION, layers=np.array([N_INPUTS, N_HIDDEN, N_ACTIONS]),
            adam_t=self.adam_t, rounds=self.rounds, history=np.array(self.history),
            school_history=np.array(self.school_history, dtype=np.float64),
            wall_death_history=np.array(self.wall_death_history, dtype=np.float64),
            best_rolling=self.best_rolling, stale=self.stale, converged=self.converged, **data,
        )
        os.replace(tmp, self.path)

    def status_text(self):
        if self.converged:
            return f"Fish brain: trained - converged after {self.rounds} rounds (frozen)"
        if self.rounds == 0:
            return "Fish brain: untrained - it learns after every round you play"
        return (f"Fish brain: learning - {self.rounds} rounds played, "
                f"no improvement for {self.stale}/{PATIENCE}")
