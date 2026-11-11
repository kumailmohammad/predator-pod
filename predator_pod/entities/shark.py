import pygame

from predator_pod import config
from predator_pod.entities.drawing import draw_shark


class Shark:
    """Player-controlled shark that chases the mouse/touchpad cursor."""

    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.radius = config.SHARK_RADIUS
        self.heading = pygame.Vector2(1, 0)
        self.tail_phase = 0.0

    def update(self, target, dt, arena):
        target = pygame.Vector2(target)
        arena.clamp(target, self.radius)

        to_target = target - self.pos
        dist = to_target.length()
        step = config.SHARK_MAX_SPEED * dt
        moving = dist > 1.0
        if dist <= step:
            self.pos = target
        else:
            self.pos += to_target / dist * step
        arena.clamp(self.pos, self.radius)

        if moving and dist > 4.0:
            diff = (self.heading.angle_to(to_target) + 180) % 360 - 180
            max_turn = config.SHARK_TURN_RATE * dt
            self.heading = self.heading.rotate(max(-max_turn, min(max_turn, diff)))
            self.tail_phase += dt * 8

    def draw(self, surface):
        draw_shark(
            surface,
            self.pos,
            self.heading.as_polar()[1],
            self.radius,
            config.COLOR_SHARK,
            config.COLOR_SHARK_BELLY,
            config.COLOR_EYE,
            self.tail_phase,
        )
