"""A small feed-forward network: 18 senses -> 16 hidden (tanh) -> 4 actions."""

import numpy as np

from predator_pod.brains.base import Brain, action_to_direction
from predator_pod.world.sensors import N_FEATURES

N_INPUTS = N_FEATURES
N_HIDDEN = 16
N_OUTPUTS = 4      # forward, backward, left, right
N_PARAMS = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS


def init_params(rng):
    w1 = rng.normal(0, 1 / np.sqrt(N_INPUTS), (N_INPUTS, N_HIDDEN))
    b1 = np.zeros(N_HIDDEN)
    w2 = rng.normal(0, 1 / np.sqrt(N_HIDDEN), (N_HIDDEN, N_OUTPUTS))
    b2 = np.zeros(N_OUTPUTS)
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])


def unpack(params):
    i = 0
    w1 = params[i:i + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN); i += N_INPUTS * N_HIDDEN
    b1 = params[i:i + N_HIDDEN]; i += N_HIDDEN
    w2 = params[i:i + N_HIDDEN * N_OUTPUTS].reshape(N_HIDDEN, N_OUTPUTS); i += N_HIDDEN * N_OUTPUTS
    b2 = params[i:i + N_OUTPUTS]
    return w1, b1, w2, b2


class NeuralBrain(Brain):
    def __init__(self, params):
        self.w1, self.b1, self.w2, self.b2 = unpack(np.asarray(params, dtype=np.float64))

    def act(self, obs, dt):
        hidden = np.tanh(obs.features @ self.w1 + self.b1)
        scores = hidden @ self.w2 + self.b2
        return action_to_direction(int(np.argmax(scores)), obs.heading)
