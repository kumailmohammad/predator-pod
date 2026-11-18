"""The AI shark's brain: a sparring partner that trains the fish.

Network: 29 senses -> 32 hidden (tanh) -> softmax over 5 turn commands
    hard left, left, straight, right, hard right

It steers through exactly the same physics as a human with the keyboard
(same speed, turn rate, slow-down in sharp turns and wall sliding), so it
can't do anything a human player can't.

Rewards (per decision, 10 per second):
    + 10 for every fish eaten                (the goal)
    + getting closer to the nearest fish     (potential-based shaping, can't be farmed)
    - 0.02 per decision                      (hunt fast, don't dawdle)
"""

import os
from pathlib import Path

import numpy as np

from predator_pod.brains.ppo import PPOPolicy, Trajectories
from predator_pod.world.shark_sensors import N_SHARK_FEATURES, RANGE, shark_sense

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_PATH = MODELS_DIR / "shark_brain.npz"

TURN_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
ACTION_NAMES = ["hard left", "left", "straight", "right", "hard right"]
N_HIDDEN = 32
DECISION_INTERVAL = 0.10
GAMMA = 0.97
LEARNING_RATE = 0.003

R_EAT = 10.0
R_STEP = -0.02
SHAPING = 2.0                 # x (gamma * phi' - phi), phi = -nearest distance / 300
SHAPING_SCALE = 300.0


class SharkBrain(PPOPolicy):
    def __init__(self, path=DEFAULT_PATH, seed=None):
        super().__init__(N_SHARK_FEATURES, N_HIDDEN, len(TURN_LEVELS),
                         learning_rate=LEARNING_RATE, seed=seed)
        self.path = Path(path) if path else None
        self.rounds = 0

    @classmethod
    def from_weights(cls, weights, seed=None):
        brain = cls(path=None, seed=seed)
        brain.set_weights(weights)
        return brain

    @classmethod
    def load_or_none(cls, path=DEFAULT_PATH):
        brain = cls(path)
        if not brain.path.exists():
            return None
        try:
            brain.load()
        except (OSError, KeyError, ValueError):
            return None
        return brain

    def load(self, path=None):
        with np.load(path or self.path) as f:
            d = {k: f[k] for k in f.files}
        if tuple(int(v) for v in d["layers"]) != (self.n_inputs, self.n_hidden, self.n_actions):
            raise ValueError("saved shark brain has a different network shape")
        self.load_state_arrays(d)
        self.rounds = int(d["rounds"]) if "rounds" in d else 0

    def save(self, path=None):
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, rounds=self.rounds, **self.state_arrays())
        os.replace(tmp, path)


class AISharkController:
    """Drives the shark with a SharkBrain; optionally records its experience."""

    def __init__(self, brain, greedy=False, record=True):
        self.brain = brain
        self.greedy = greedy
        self.traj = Trajectories(1) if record else None
        self.timer = 0.0
        self.turn = 0.0
        self.phi = None
        self.total_reward = 0.0
        self.decisions = 0

    def _phi(self, nearest):
        return -min(nearest, RANGE) / SHAPING_SCALE

    def command(self, world, dt):
        self.timer -= dt
        if self.timer <= 0 and world.fishes:
            self.timer += DECISION_INTERVAL
            x, nearest = shark_sense(world.shark, world.fishes, world.arena, world.total)
            phi = self._phi(nearest)
            if self.phi is not None:
                self._reward(R_STEP + SHAPING * (GAMMA * phi - self.phi))
            self.phi = phi
            a, _, p = self.brain.sample(x[None, :], greedy=self.greedy)
            self.turn = float(TURN_LEVELS[a[0]])
            self.decisions += 1
            if self.traj is not None:
                self.traj.decide(0, x, int(a[0]), 0, False, float(p[0]))
        return None, self.turn

    def on_eat(self, count):
        self._reward(R_EAT * count)
        self.phi = None        # nearest fish changed: restart the shaping chain

    def _reward(self, r):
        self.total_reward += r
        if self.traj is not None:
            self.traj.reward(0, r)
