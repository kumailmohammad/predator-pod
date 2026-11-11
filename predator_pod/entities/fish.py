import random

import pygame

from predator_pod import config
from predator_pod.entities.drawing import draw_fish


class Fish:
    def __init__(self, pos, brain):
        self.pos = pygame.Vector2(pos)
        self.radius = config.FISH_RADIUS
        self.speed = config.FISH_SPEED          # constant, never changes
        self.heading = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        self.brain = brain
        self.tail_phase = random.uniform(0, 6.28)

    def update(self, obs, dt, arena):
        desired = self.brain.act(obs, dt)
        if desired.length_squared() > 1e-9:
            # Turn toward the desired direction, limited by the turn rate.
            diff = self.heading.angle_to(desired)
            diff = (diff + 180) % 360 - 180
            max_turn = config.FISH_TURN_RATE * dt
            self.heading = self.heading.rotate(max(-max_turn, min(max_turn, diff)))

        self.pos += self.heading * self.speed * dt

        hit_x, hit_y = arena.clamp(self.pos, self.radius)
        if hit_x:
            self.heading.x = -self.heading.x
        if hit_y:
            self.heading.y = -self.heading.y

        self.tail_phase += dt * 14

    def draw(self, surface):
        draw_fish(
            surface,
            self.pos,
            self.heading.as_polar()[1],
            self.radius,
            config.COLOR_FISH,
            config.COLOR_FISH_TAIL,
            config.COLOR_EYE,
            self.tail_phase,
        )
