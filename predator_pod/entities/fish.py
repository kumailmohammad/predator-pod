import random

import pygame

from predator_pod import config
from predator_pod.entities.drawing import draw_fish


class Fish:
    """A prey fish. It cruises at a constant speed and turns toward the
    direction its brain last picked. It can fire a fast-start (C-start):
    an instant turn and a short burst of speed, followed by a cooldown."""

    def __init__(self, pos, index):
        self.index = index
        self.pos = pygame.Vector2(pos)
        self.radius = config.FISH_RADIUS
        self.speed = config.FISH_SPEED          # cruising speed, constant
        self.heading = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        self.desired = pygame.Vector2(self.heading)
        self.tail_phase = random.uniform(0, 6.28)
        self.burst_timer = 0.0                  # > 0 while bursting
        self.cooldown = 0.0                     # > 0 while recovering
        self.zig = 1                            # side of the next protean zigzag

    @property
    def burst_ready(self):
        return self.cooldown <= 0 and self.burst_timer <= 0

    def burst(self):
        """Fast-start: snap to the desired direction and sprint briefly."""
        if not self.burst_ready:
            return False
        if self.desired.length_squared() > 1e-9:
            self.heading = self.desired.normalize()
        self.burst_timer = config.FISH_BURST_TIME
        self.cooldown = config.FISH_BURST_TIME + config.FISH_BURST_COOLDOWN
        return True

    def update(self, dt, arena):
        diff = (self.heading.angle_to(self.desired) + 180) % 360 - 180
        max_turn = config.FISH_TURN_RATE * dt
        self.heading = self.heading.rotate(max(-max_turn, min(max_turn, diff)))

        speed = self.speed
        if self.burst_timer > 0:
            speed *= config.FISH_BURST_FACTOR
            self.burst_timer -= dt
        if self.cooldown > 0:
            self.cooldown -= dt

        self.pos += self.heading * speed * dt
        hit_x, hit_y = arena.clamp(self.pos, self.radius)
        if hit_x:
            self.heading.x = -self.heading.x
        if hit_y:
            self.heading.y = -self.heading.y
        self.tail_phase += dt * (30 if self.burst_timer > 0 else 14)

    def draw(self, surface):
        bursting = self.burst_timer > 0
        draw_fish(
            surface, self.pos, self.heading.as_polar()[1],
            self.radius * (1.25 if bursting else 1.0),
            config.COLOR_FISH, config.COLOR_FISH_TAIL, config.COLOR_EYE, self.tail_phase,
        )
