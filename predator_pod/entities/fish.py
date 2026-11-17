import random

import pygame

from predator_pod import config
from predator_pod.entities.drawing import draw_fish


class Fish:
    """A prey fish. It swims at a constant speed and turns toward the
    direction its brain last picked (up / down / left / right)."""

    def __init__(self, pos, index):
        self.index = index
        self.pos = pygame.Vector2(pos)
        self.radius = config.FISH_RADIUS
        self.speed = config.FISH_SPEED          # constant, never changes
        self.heading = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        self.desired = pygame.Vector2(self.heading)
        self.tail_phase = random.uniform(0, 6.28)

    def update(self, dt, arena):
        diff = (self.heading.angle_to(self.desired) + 180) % 360 - 180
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
            surface, self.pos, self.heading.as_polar()[1], self.radius,
            config.COLOR_FISH, config.COLOR_FISH_TAIL, config.COLOR_EYE, self.tail_phase,
        )
