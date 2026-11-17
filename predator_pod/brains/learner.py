"""The fish brain and how it learns from your games.

Network (shared by every fish):
    18 senses -> 32 hidden (tanh) -> 4 outputs (softmax)
    outputs = probability of swimming UP, DOWN, LEFT, RIGHT

Learning: reinforcement learning (policy gradient, PPO-style updates).
    About 7 times a second each fish picks a direction. After each decision
    it gets a reward:
        + being far from the shark          (the main goal)
        + being near the school's center    (small, so it can wander and rejoin)
        - hugging a wall                    (corners are traps)
        - being eaten                       (big penalty)
    When your round ends, every decision is scored by the rewards that
    followed it ("return"). Decisions that did better than average become
    more likely, worse ones less likely. The round's data is reused for a few
    passes of small, clipped updates (PPO) so one round teaches a lot without
    the network changing too wildly.

Stopping: once the rolling average survival time stops improving for
PATIENCE rounds (after at least MIN_ROUNDS rounds), the brain is frozen and
saved. From then on fish always pick their best direction.
"""

import os
from pathlib import Path

import numpy as np

from predator_pod.world.sensors import N_FEATURES

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "pod_brain.npz"

N_INPUTS = N_FEATURES
N_HIDDEN = 32
N_ACTIONS = 4
ACTION_NAMES = ["up", "down", "left", "right"]
ACTION_DIRS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.float64)

DECISION_INTERVAL = 0.15      # s between decisions (direction is held in between)

# Rewards (per decision)
R_FAR_FROM_SHARK = 1.0        # x min(distance / FAR_DISTANCE, 1)
FAR_DISTANCE = 300.0          # px, farther than this counts as fully safe
R_NEAR_POD = 0.25             # x (1 - min(pod distance / POD_RANGE, 1))
POD_RANGE = 150.0
R_WALL = -0.3                 # when closer than WALL_DANGER to a wall
WALL_DANGER = 20.0
R_EATEN = -10.0

GAMMA = 0.95                  # how much future rewards count (per decision)
LEARNING_RATE = 0.003
ENTROPY_BONUS = 0.01          # keeps some exploration so it doesn't lock in early
GRAD_CLIP = 1.0
EPOCHS = 10                   # passes over one round's data
BATCH_SIZE = 256
CLIP = 0.2                    # PPO: max change in action probability per update

ROLLING_WINDOW = 5            # rounds
IMPROVE_TOLERANCE = 0.03      # rolling average must beat the best by 3%
PATIENCE = 10                 # rounds without improvement -> converged
MIN_ROUNDS = 20
MIN_ROUND_SECONDS = 5.0       # shorter rounds (a quick Esc) are ignored


def reward(shark_dist, pod_dist, wall_dist):
    """Reward for one decision, from the state right after it."""
    r = R_FAR_FROM_SHARK * np.minimum(shark_dist / FAR_DISTANCE, 1.0)
    r = r + R_NEAR_POD * (1.0 - np.minimum(pod_dist / POD_RANGE, 1.0))
    r = r + np.where(wall_dist < WALL_DANGER, R_WALL, 0.0)
    return r


class Episode:
    """Everything the fish did in one round, for learning at the end."""

    def __init__(self, n_fish):
        self.x = [[] for _ in range(n_fish)]
        self.a = [[] for _ in range(n_fish)]
        self.p = [[] for _ in range(n_fish)]   # probability of the chosen action
        self.r = [[] for _ in range(n_fish)]

    def decide(self, i, x, a, p):
        self.x[i].append(x)
        self.a[i].append(a)
        self.p[i].append(p)
        self.r[i].append(0.0)

    def reward(self, i, r):
        if self.r[i]:
            self.r[i][-1] += r


class Learner:
    def __init__(self, path=DEFAULT_PATH, seed=None):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.reset(save=False)

    # ----- network ---------------------------------------------------------
    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return h, p

    def act(self, features):
        """Pick a direction (0..3) for every fish. Returns (actions, their probabilities)."""
        _, p = self.forward(features)
        if self.converged:
            a = p.argmax(1)
        else:
            u = self.rng.random((len(p), 1))
            a = np.minimum((p.cumsum(1) < u).sum(1), N_ACTIONS - 1)
        return a, p[np.arange(len(a)), a]

    # ----- learning --------------------------------------------------------
    def learn(self, episode, survival_times, round_seconds):
        """One update from a finished round. Returns True if it learned."""
        if self.converged or round_seconds < MIN_ROUND_SECONDS:
            return False
        xs, acts, olds, rets = [], [], [], []
        for x, a, p, r in zip(episode.x, episode.a, episode.p, episode.r):
            if not r:
                continue
            g, run = np.zeros(len(r)), 0.0
            for t in range(len(r) - 1, -1, -1):
                run = r[t] + GAMMA * run
                g[t] = run
            xs += x; acts += a; olds += p; rets.append(g)
        if not xs:
            return False
        x = np.array(xs)
        a = np.array(acts)
        old_p = np.array(olds)
        g = np.concatenate(rets)
        adv = (g - g.mean()) / (g.std() + 1e-8)

        n = len(x)
        for _ in range(EPOCHS):
            order = self.rng.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                self._ppo_step(x[idx], a[idx], old_p[idx], adv[idx])

        self.rounds += 1
        self.history.append(float(np.mean(survival_times)))
        self._check_convergence()
        self.save()
        return True

    def _ppo_step(self, x, a, old_p, adv):
        h, p = self.forward(x)
        onehot = np.eye(N_ACTIONS)[a]
        ratio = p[np.arange(len(a)), a] / (old_p + 1e-12)
        # PPO clip: stop pushing an action once its probability moved > CLIP.
        active = ~(((adv > 0) & (ratio > 1 + CLIP)) | ((adv < 0) & (ratio < 1 - CLIP)))
        logp = np.log(p + 1e-12)
        entropy = -(p * logp).sum(1, keepdims=True)
        # gradient of  ratio * adv  +  ENTROPY_BONUS * entropy   w.r.t. the logits
        gz = (active * adv * ratio)[:, None] * (onehot - p) + ENTROPY_BONUS * (-p * (logp + entropy))
        gz /= len(x)
        grads = {"w2": h.T @ gz, "b2": gz.sum(0)}
        gh = (gz @ self.w2.T) * (1 - h * h)
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
    PARAMS = ("w1", "b1", "w2", "b2")

    def reset(self, save=True):
        self.w1 = self.rng.normal(0, 1 / np.sqrt(N_INPUTS), (N_INPUTS, N_HIDDEN))
        self.b1 = np.zeros(N_HIDDEN)
        self.w2 = self.rng.normal(0, 0.01, (N_HIDDEN, N_ACTIONS))   # starts ~uniform: random swimming
        self.b2 = np.zeros(N_ACTIONS)
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self.adam_t = 0
        self.rounds = 0
        self.history = []
        self.best_rolling = 0.0
        self.stale = 0
        self.converged = False
        if save:
            self.save()

    @classmethod
    def load_or_new(cls, path=DEFAULT_PATH):
        learner = cls(path)
        if learner.path.exists():
            try:
                learner.load()
            except (OSError, KeyError, ValueError):
                learner.reset(save=False)
        return learner

    def load(self):
        d = np.load(self.path)
        if tuple(d["layers"]) != (N_INPUTS, N_HIDDEN, N_ACTIONS):
            raise ValueError("saved brain has a different network shape")
        for k in self.PARAMS:
            setattr(self, k, d[k].astype(np.float64))
            self.m[k] = d["m_" + k]
            self.v[k] = d["v_" + k]
        self.adam_t = int(d["adam_t"])
        self.rounds = int(d["rounds"])
        self.history = list(d["history"].astype(np.float64))
        self.best_rolling = float(d["best_rolling"])
        self.stale = int(d["stale"])
        self.converged = bool(d["converged"])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        data = {k: getattr(self, k) for k in self.PARAMS}
        data.update({"m_" + k: self.m[k] for k in self.PARAMS})
        data.update({"v_" + k: self.v[k] for k in self.PARAMS})
        np.savez(
            tmp, layers=np.array([N_INPUTS, N_HIDDEN, N_ACTIONS]), adam_t=self.adam_t,
            rounds=self.rounds, history=np.array(self.history), best_rolling=self.best_rolling,
            stale=self.stale, converged=self.converged, **data,
        )
        os.replace(tmp, self.path)

    def status_text(self):
        if self.converged:
            return f"Fish brain: trained - converged after {self.rounds} rounds (frozen)"
        if self.rounds == 0:
            return "Fish brain: untrained - it learns after every round you play"
        return (f"Fish brain: learning - {self.rounds} rounds played, "
                f"no improvement for {self.stale}/{PATIENCE}")
