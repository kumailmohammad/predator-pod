import pygame

from predator_pod import config
from predator_pod.scenes.play import format_time
from predator_pod.scenes.ui import Button, draw_text


class GameOverScene:
    def __init__(self, game, fish_count, elapsed, mean_survival):
        self.game = game
        self.fish_count = fish_count
        self.elapsed = elapsed
        self.mean_survival = mean_survival
        cx = config.WINDOW_WIDTH // 2
        self.again = Button("Play again", (cx, 410), size=(240, 50))
        self.menu = Button("Menu", (cx, 475), size=(240, 50))

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
        draw_text(surface, "The pod is gone!", 80, config.COLOR_ACCENT, center=(cx, 190))
        draw_text(
            surface,
            f"You ate all {self.fish_count} fish in {format_time(self.elapsed)}",
            38, config.COLOR_TEXT, center=(cx, 265),
        )
        draw_text(
            surface, f"Average fish survived {self.mean_survival:.1f} s",
            30, config.COLOR_TEXT_DIM, center=(cx, 305),
        )
        draw_text(surface, self.game.learner.status_text(), 26, config.COLOR_TEXT_DIM, center=(cx, 340))
        self.again.draw(surface)
        self.menu.draw(surface)
