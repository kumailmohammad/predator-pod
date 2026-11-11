"""Brain interface. Every fish has one; swap implementations to change behavior."""

import pygame

# Discrete actions a learned brain will choose from (milestone 2).
FORWARD, BACKWARD, LEFT, RIGHT = range(4)


def action_to_direction(action, heading):
    """Turn a discrete action into a desired world direction.

    The fish always swims at the same speed; actions only pick where it heads.
    """
    if action == FORWARD:
        return pygame.Vector2(heading)
    if action == BACKWARD:
        return -heading
    if action == LEFT:
        return heading.rotate(-90)
    if action == RIGHT:
        return heading.rotate(90)
    raise ValueError(f"unknown action {action}")


class Brain:
    def act(self, obs, dt):
        """Return the desired swim direction (Vector2, any length)."""
        raise NotImplementedError
