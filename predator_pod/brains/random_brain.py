"""Milestone 1 brain: wander randomly while loosely staying with the pod.

It ignores the shark on purpose. The trained neural network replaces this.
"""

import random

import pygame

from predator_pod import config
from predator_pod.brains.base import Brain


class RandomBrain(Brain):
    WANDER_JITTER = 180.0   # deg/s of random drift in the wander direction
    W_WANDER = 1.0
    W_COHESION = 0.5
    W_ALIGN = 0.4
    W_WALL = 3.0

    def __init__(self):
        self.wander_angle = random.uniform(0, 360)

    def act(self, obs, dt):
        self.wander_angle += random.uniform(-1, 1) * self.WANDER_JITTER * dt
        desired = pygame.Vector2(1, 0).rotate(self.wander_angle) * self.W_WANDER

        # Stay with the pod: pull harder the farther away the fish is.
        dist = obs.to_pod.length()
        if dist > 1e-6:
            desired += obs.to_pod / dist * self.W_COHESION * min(dist / 120.0, 1.5)

        # Swim roughly the same way as the pod.
        desired += obs.pod_heading * self.W_ALIGN

        # Steer away from walls before touching them.
        m = config.FISH_WALL_MARGIN
        if obs.wall_left < m:
            desired.x += self.W_WALL * (1 - obs.wall_left / m)
        if obs.wall_right < m:
            desired.x -= self.W_WALL * (1 - obs.wall_right / m)
        if obs.wall_top < m:
            desired.y += self.W_WALL * (1 - obs.wall_top / m)
        if obs.wall_bottom < m:
            desired.y -= self.W_WALL * (1 - obs.wall_bottom / m)

        # Keep the wander direction in sync so walls don't fight it forever.
        if desired.length_squared() > 1e-9:
            self.wander_angle = _blend_angle(
                self.wander_angle, desired.as_polar()[1], 0.05
            )
        return desired


def _blend_angle(a, b, t):
    diff = (b - a + 180) % 360 - 180
    return a + diff * t
