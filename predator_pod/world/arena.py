import pygame

from predator_pod import config


class Arena:
    """The bordered rectangle that fish and shark live inside."""

    def __init__(self):
        self.rect = pygame.Rect(
            config.ARENA_X, config.ARENA_Y, config.ARENA_WIDTH, config.ARENA_HEIGHT
        )

    def wall_distances(self, pos):
        """Distances from a point to the left, right, top and bottom walls."""
        return (
            pos.x - self.rect.left,
            self.rect.right - pos.x,
            pos.y - self.rect.top,
            self.rect.bottom - pos.y,
        )

    def clamp(self, pos, radius):
        """Keep a circle fully inside the arena. Returns (hit_x, hit_y)."""
        hit_x = hit_y = False
        if pos.x < self.rect.left + radius:
            pos.x, hit_x = self.rect.left + radius, True
        elif pos.x > self.rect.right - radius:
            pos.x, hit_x = self.rect.right - radius, True
        if pos.y < self.rect.top + radius:
            pos.y, hit_y = self.rect.top + radius, True
        elif pos.y > self.rect.bottom - radius:
            pos.y, hit_y = self.rect.bottom - radius, True
        return hit_x, hit_y

    def draw(self, surface):
        pygame.draw.rect(surface, config.COLOR_ARENA, self.rect)
        pygame.draw.rect(surface, config.COLOR_BORDER, self.rect, config.ARENA_BORDER)
