import pygame

from predator_pod import config
from predator_pod.scenes.play import format_time
from predator_pod.scenes.ui import Button, draw_text


class GameOverScene:
    def __init__(self, game, fish_count, elapsed):
        self.game = game
        self.fish_count = fish_count
        self.elapsed = elapsed
        cx = config.WINDOW_WIDTH // 2
        self.again = Button("Play again", (cx, 380), size=(240, 50))
        self.menu = Button("Menu", (cx, 445), size=(240, 50))

    def handle_event(self, event):
        if self.again.clicked(event) or (
            event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE)
        ):
            self.game.start_play(self.fish_count)
        elif self.menu.clicked(event) or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            self.game.show_landing(self.fish_count)

    def update(self, dt):
        pass

    def draw(self, surface):
        surface.fill(config.COLOR_BG)
        cx = config.WINDOW_WIDTH // 2
        draw_text(surface, "The pod is gone!", 80, config.COLOR_ACCENT, center=(cx, 200))
        draw_text(
            surface,
            f"You ate all {self.fish_count} fish in {format_time(self.elapsed)}",
            38, config.COLOR_TEXT, center=(cx, 280),
        )
        self.again.draw(surface)
        self.menu.draw(surface)
