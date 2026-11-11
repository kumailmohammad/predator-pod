"""Tiny UI helpers: text and buttons."""

import pygame

from predator_pod import config

_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(None, size)
    return _fonts[size]


def draw_text(surface, text, size, color, center=None, topleft=None):
    img = font(size).render(text, True, color)
    rect = img.get_rect()
    if center is not None:
        rect.center = center
    elif topleft is not None:
        rect.topleft = topleft
    surface.blit(img, rect)
    return rect


class Button:
    def __init__(self, text, center, size=(200, 50), font_size=34):
        self.text = text
        self.rect = pygame.Rect(0, 0, *size)
        self.rect.center = center
        self.font_size = font_size

    def clicked(self, event):
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, surface):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = config.COLOR_BUTTON_HOVER if hover else config.COLOR_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, config.COLOR_BORDER, self.rect, 2, border_radius=10)
        draw_text(surface, self.text, self.font_size, config.COLOR_TEXT, center=self.rect.center)
