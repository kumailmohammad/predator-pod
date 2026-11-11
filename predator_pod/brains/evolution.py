"""Online learning for the pod with an evolution strategy (ES).

One shared brain (theta) is being learned. Every round, each fish gets
theta + sigma * noise (in +/- pairs). Fitness = seconds survived. After the
round, theta moves toward the noise directions whose fish survived longest.

Learning stops by itself ("converged") when the rolling average survival
time hasn't improved for PATIENCE generations. The brain is then frozen and
every fish uses the exact learned weights.
"""

from pathlib import Path

import numpy as np

from predator_pod.brains.neural import N_HIDDEN, N_INPUTS, N_OUTPUTS, N_PARAMS, init_params

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "pod_brain.npz"

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.005
SIGMA_START = 0.15
SIGMA_DECAY = 0.997
SIGMA_MIN = 0.03
ROLLING_WINDOW = 20
IMPROVE_TOLERANCE = 0.01       # rolling average must beat the best by 1%
PATIENCE = 60                  # generations without improvement -> converged
MIN_GENERATIONS = 150


def centered_ranks(values):
    """Map values to [-0.5, 0.5] by rank, averaging ranks of ties."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks / max(n - 1, 1) - 0.5


class PodLearner:
    def __init__(self, path=DEFAULT_PATH, seed=None):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.reset(save=False)

    # ----- persistence ---------------------------------------------------
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
        data = np.load(self.path)
        if tuple(data["layers"]) != (N_INPUTS, N_HIDDEN, N_OUTPUTS):
            raise ValueError("saved brain has a different network shape")
        self.theta = data["theta"].astype(np.float64)
        self.sigma = float(data["sigma"])
        self.generation = int(data["generation"])
        self.history = list(data["history"].astype(np.float64))
        self.best_rolling = float(data["best_rolling"])
        self.stale = int(data["stale"])
        self.converged = bool(data["converged"])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            layers=np.array([N_INPUTS, N_HIDDEN, N_OUTPUTS]),
            theta=self.theta,
            sigma=self.sigma,
            generation=self.generation,
            history=np.array(self.history),
            best_rolling=self.best_rolling,
            stale=self.stale,
            converged=self.converged,
        )

    def reset(self, save=True):
        self.theta = init_params(self.rng)
        self.sigma = SIGMA_START
        self.generation = 0
        self.history = []
        self.best_rolling = 0.0
        self.stale = 0
        self.converged = False
        self._eps = None
        self._pending = []
        if save:
            self.save()

    # ----- learning ------------------------------------------------------
    def sample(self, n):
        """Weights for n fish this round."""
        if self.converged:
            self._eps = None
            return [self.theta.copy() for _ in range(n)]
        half = self.rng.standard_normal((n // 2, N_PARAMS))
        eps = [e for pair in half for e in (pair, -pair)]
        if n % 2:
            eps.append(np.zeros(N_PARAMS))      # one fish uses the current brain exactly
        self._eps = np.array(eps)
        return [self.theta + self.sigma * e for e in self._eps]

    def record(self, fitness):
        """Store one finished round. fitness[i] = seconds fish i survived."""
        if self.converged or self._eps is None:
            return
        self._pending.append((self._eps, np.asarray(fitness, dtype=np.float64)))
        self._eps = None

    def step(self):
        """One generation: move theta using every recorded round, then save."""
        if self.converged or not self._pending:
            return
        grad = np.zeros(N_PARAMS)
        n_eff = 0
        means = []
        for eps, fitness in self._pending:
            # Rank inside each round, so a lucky/unlucky round doesn't dominate.
            weights = centered_ranks(fitness)
            noisy = np.any(eps != 0, axis=1)
            grad += (weights[noisy, None] * eps[noisy]).sum(0)
            n_eff += int(noisy.sum())
            means.append(fitness.mean())
        self._pending = []
        self._apply(grad / (max(n_eff, 1) * self.sigma), float(np.mean(means)))

    # ----- headless training: one candidate brain per round ---------------
    def candidates(self, pairs):
        """2*pairs candidate brains (theta +/- noise). The whole pod uses one per round."""
        half = self.rng.standard_normal((pairs, N_PARAMS))
        self._cand_eps = np.concatenate([half, -half])
        return [self.theta + self.sigma * e for e in self._cand_eps]

    def step_candidates(self, scores):
        """scores[i] = mean pod survival with candidate i."""
        if self.converged:
            return
        weights = centered_ranks(scores)
        grad = (weights[:, None] * self._cand_eps).sum(0) / (len(scores) * self.sigma)
        self._apply(grad, float(np.mean(scores)))

    def _apply(self, grad, metric):
        self.theta = self.theta * (1 - LEARNING_RATE * WEIGHT_DECAY) + LEARNING_RATE * grad
        self.sigma = max(SIGMA_MIN, self.sigma * SIGMA_DECAY)
        self.generation += 1
        self.history.append(metric)
        self._check_convergence()
        self.save()

    def update(self, fitness):
        """Record one round and learn from it immediately (used while you play)."""
        self.record(fitness)
        self.step()

    def _check_convergence(self):
        if len(self.history) < ROLLING_WINDOW:
            return
        rolling = float(np.mean(self.history[-ROLLING_WINDOW:]))
        if rolling > self.best_rolling * (1 + IMPROVE_TOLERANCE):
            self.best_rolling = rolling
            self.stale = 0
        else:
            self.stale += 1
        if self.stale >= PATIENCE and self.generation >= MIN_GENERATIONS:
            self.converged = True

    def status_text(self):
        if self.converged:
            return f"Fish brain: converged at generation {self.generation} (frozen)"
        if self.generation == 0:
            return "Fish brain: untrained - it learns every round you play"
        return f"Fish brain: generation {self.generation}, learning"
