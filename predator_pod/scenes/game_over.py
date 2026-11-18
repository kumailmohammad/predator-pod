import numpy as np
import pygame

from predator_pod import config
from predator_pod.brains.learner import ROLLING_WINDOW, training_running
from predator_pod.scenes.play import format_time
from predator_pod.scenes.ui import Button, draw_text


class GameOverScene:
    def __init__(self, game, fish_count, elapsed, mean_survival, learned, stats=None):
        self.game = game
        self.fish_count = fish_count
        self.elapsed = elapsed
        self.mean_survival = mean_survival
        self.learned = learned
        self.stats = stats or {}
        cx = config.WINDOW_WIDTH // 2
        self.again = Button("Play again", (cx - 130, 580), size=(220, 46), font_size=30)
        self.menu = Button("Menu", (cx + 130, 580), size=(220, 46), font_size=30)

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
        L = self.game.learner
        surface.fill(config.COLOR_BG)
        cx = config.WINDOW_WIDTH // 2
        draw_text(surface, "The pod is gone!", 64, config.COLOR_ACCENT, center=(cx, 60))
        draw_text(surface, f"You ate all {self.fish_count} fish in {format_time(self.elapsed)}"
                  f"   -   average fish survived {self.mean_survival:.1f} s",
                  28, config.COLOR_TEXT, center=(cx, 110))

        if not self.learned and training_running():
            msg = "Training (train.py) is running, so this round wasn't saved to the brain."
        elif L.converged:
            msg = f"Training finished: the brain stopped improving and is now frozen ({L.rounds} rounds)."
        elif self.learned:
            msg = f"Brain updated from this round. Rounds learned from: {L.rounds}"
        else:
            msg = "This round was too short to learn from."
        draw_text(surface, msg, 26, config.COLOR_TEXT, center=(cx, 142))
        if L.upgraded_from:
            msg = "Brain upgraded with new senses and escape behaviors - earlier training kept."
            draw_text(surface, msg, 20, config.COLOR_TEXT_DIM, center=(cx, 164))

        st = self.stats
        if st:
            line = (f"In a group {st['school_pct']:.0f}% of the time   -   "
                    f"eaten at walls: {st['wall_deaths']}   -   fast-starts: {st['bursts']}")
            draw_text(surface, line, 22, config.COLOR_TEXT, center=(cx, 186))
            if st.get("top_actions"):
                top = ", ".join(f"{n} {p:.0f}%" for n, p in st["top_actions"])
                draw_text(surface, f"Top behaviors: {top}", 22, config.COLOR_TEXT_DIM,
                          center=(cx, 208))

        chart = pygame.Rect(80, 226, config.WINDOW_WIDTH - 160, 280)
        draw_chart(surface, chart, L.history)
        draw_text(surface, "Average fish survival per round (thin) and rolling average (thick)",
                  22, config.COLOR_TEXT_DIM, center=(cx, chart.bottom + 18))
        draw_text(surface, L.status_text(), 22, config.COLOR_TEXT_DIM, center=(cx, chart.bottom + 42))
        self.again.draw(surface)
        self.menu.draw(surface)


def draw_chart(surface, rect, history):
    pygame.draw.rect(surface, config.COLOR_ARENA, rect)
    pygame.draw.rect(surface, config.COLOR_BORDER, rect, 1)
    if len(history) < 2:
        draw_text(surface, "Play a few rounds to see the learning curve", 24, config.COLOR_TEXT_DIM,
                  center=rect.center)
        return
    h = np.array(history)
    top = max(h.max(), 1.0) * 1.15

    def pt(i, v):
        return (rect.left + 10 + (rect.width - 20) * i / (len(h) - 1),
                rect.bottom - 10 - (rect.height - 20) * v / top)

    pygame.draw.lines(surface, config.COLOR_TEXT_DIM, False, [pt(i, v) for i, v in enumerate(h)], 1)
    for i, v in enumerate(h):
        pygame.draw.circle(surface, config.COLOR_TEXT_DIM, pt(i, v), 3)
    if len(h) >= ROLLING_WINDOW:
        roll = np.convolve(h, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
        pts = [pt(i + ROLLING_WINDOW - 1, v) for i, v in enumerate(roll)]
        if len(pts) >= 2:
            pygame.draw.lines(surface, config.COLOR_TEXT, False, pts, 3)
    draw_text(surface, f"{top:.0f}s", 20, config.COLOR_TEXT_DIM, topleft=(rect.left + 6, rect.top + 4))
    draw_text(surface, f"round {len(h)}", 20, config.COLOR_TEXT_DIM, topleft=(rect.right - 70, rect.bottom - 20))
