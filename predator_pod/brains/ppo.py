"""A small policy network trained with PPO, shared by the fish and the shark.

    inputs -> hidden (tanh) -> softmax over actions
                            -> optional sigmoid "yes/no" head (the fish's fast-start)

Pure NumPy: forward pass, sampling, the clipped PPO update and Adam.
"""

import numpy as np

ENTROPY_BONUS = 0.01          # keeps some exploration so it doesn't lock in early
GRAD_CLIP = 1.0
CLIP = 0.2                    # PPO: max change in a choice's probability per update


def _sigmoid(y):
    return 1.0 / (1.0 + np.exp(-np.clip(y, -30, 30)))


def discounted_returns(rewards, gamma):
    g, run = np.zeros(len(rewards)), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        run = rewards[t] + gamma * run
        g[t] = run
    return g


class Trajectories:
    """Decisions of one or more agents during a round (one list per agent)."""

    def __init__(self, n_agents):
        self.x = [[] for _ in range(n_agents)]
        self.a = [[] for _ in range(n_agents)]
        self.b = [[] for _ in range(n_agents)]      # yes/no head choice (0/1)
        self.ready = [[] for _ in range(n_agents)]  # yes/no head was available
        self.p = [[] for _ in range(n_agents)]      # probability of the whole choice
        self.r = [[] for _ in range(n_agents)]

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

    def batch(self, gamma):
        """Flatten into arrays (x, a, b, ready, old_p, returns), or None if empty."""
        keep = [i for i in range(len(self.r)) if self.r[i]]
        if not keep:
            return None
        cat = lambda lists, dt: np.concatenate([np.asarray(lists[i], dtype=dt) for i in keep])
        x = np.concatenate([np.asarray(self.x[i], dtype=np.float32) for i in keep])
        g = np.concatenate([discounted_returns(self.r[i], gamma) for i in keep])
        return (x, cat(self.a, np.int64), cat(self.b, np.float64), cat(self.ready, np.float64),
                cat(self.p, np.float64), g)


def merge_batches(batches):
    batches = [b for b in batches if b is not None]
    if not batches:
        return None
    return tuple(np.concatenate(parts) for parts in zip(*batches))


class PPOPolicy:
    def __init__(self, n_inputs, n_hidden, n_actions, yes_no_head=False,
                 learning_rate=0.003, seed=None):
        self.n_inputs, self.n_hidden, self.n_actions = n_inputs, n_hidden, n_actions
        self.yes_no_head = yes_no_head
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.init_params()

    @property
    def PARAMS(self):
        return ("w1", "b1", "w2", "b2", "w3", "b3") if self.yes_no_head else ("w1", "b1", "w2", "b2")

    def init_params(self, yes_no_bias=-4.0):
        self.w1 = self.rng.normal(0, 1 / np.sqrt(self.n_inputs), (self.n_inputs, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.w2 = self.rng.normal(0, 0.01, (self.n_hidden, self.n_actions))   # ~uniform start
        self.b2 = np.zeros(self.n_actions)
        if self.yes_no_head:
            self.w3 = np.zeros(self.n_hidden)
            self.b3 = np.array([yes_no_bias])
        self.reset_optimizer()

    def reset_optimizer(self):
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.PARAMS}
        self.adam_t = 0

    # ----- network ---------------------------------------------------------
    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        if self.yes_no_head:
            y = h @ self.w3 + self.b3[0]
            return h, p, y, _sigmoid(y)
        return h, p, None, None

    def sample(self, features, ready=None, greedy=False):
        """Returns (actions, yes/no choices, probability of each whole choice)."""
        _, p, _, q = self.forward(features)
        n = len(p)
        if greedy:
            a = p.argmax(1)
        else:
            u = self.rng.random((n, 1))
            a = np.minimum((p.cumsum(1) < u).sum(1), self.n_actions - 1)
        prob = p[np.arange(n), a]
        if not self.yes_no_head:
            return a, np.zeros(n, dtype=bool), prob
        ready = np.ones(n, dtype=bool) if ready is None else ready
        b = ((q > 0.5) if greedy else (self.rng.random(n) < q)) & ready
        prob = prob * np.where(ready, np.where(b, q, 1.0 - q), 1.0)
        return a, b, prob

    # ----- learning --------------------------------------------------------
    def ppo_update(self, batch, epochs, batch_size):
        x, a, b, ready, old_p, g = batch
        adv = (g - g.mean()) / (g.std() + 1e-8)
        n = len(x)
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self._ppo_step(x[idx].astype(np.float64), a[idx], b[idx], ready[idx],
                               old_p[idx], adv[idx])

    def _ppo_step(self, x, a, b, ready, old_p, adv):
        h, p, y, q = self.forward(x)
        n = len(a)
        onehot = np.eye(self.n_actions)[a]
        prob = p[np.arange(n), a]
        if self.yes_no_head:
            prob = prob * np.where(ready > 0, np.where(b > 0, q, 1.0 - q), 1.0)
        ratio = prob / (old_p + 1e-12)
        # PPO clip: stop pushing a choice once its probability moved > CLIP.
        active = ~(((adv > 0) & (ratio > 1 + CLIP)) | ((adv < 0) & (ratio < 1 - CLIP)))
        coef = active * adv * ratio
        logp = np.log(p + 1e-12)
        entropy = -(p * logp).sum(1, keepdims=True)
        gz = (coef[:, None] * (onehot - p) + ENTROPY_BONUS * (-p * (logp + entropy))) / n
        grads = {"w2": h.T @ gz, "b2": gz.sum(0)}
        back = gz @ self.w2.T
        if self.yes_no_head:
            gy = ready * (coef * (b - q) + ENTROPY_BONUS * (-q * (1 - q) * np.clip(y, -30, 30))) / n
            grads["w3"] = h.T @ gy
            grads["b3"] = np.array([gy.sum()])
            back = back + gy[:, None] * self.w3[None, :]
        gh = back * (1 - h * h)
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
            setattr(self, k, getattr(self, k) + self.learning_rate * mh / (np.sqrt(vh) + 1e-8))

    # ----- weights as plain arrays (for files and worker processes) --------
    def weights(self):
        return {k: getattr(self, k).copy() for k in self.PARAMS}

    def set_weights(self, w):
        for k in self.PARAMS:
            setattr(self, k, np.asarray(w[k], dtype=np.float64))

    def state_arrays(self):
        """Weights plus optimizer state, keyed as stored in .npz files."""
        d = self.weights()
        d.update({"m_" + k: self.m[k] for k in self.PARAMS})
        d.update({"v_" + k: self.v[k] for k in self.PARAMS})
        d["adam_t"] = self.adam_t
        d["layers"] = np.array([self.n_inputs, self.n_hidden, self.n_actions])
        return d

    def load_state_arrays(self, d):
        for k in self.PARAMS:
            setattr(self, k, np.asarray(d[k], dtype=np.float64))
            self.m[k] = np.asarray(d["m_" + k], dtype=np.float64)
            self.v[k] = np.asarray(d["v_" + k], dtype=np.float64)
        self.adam_t = int(d["adam_t"])
