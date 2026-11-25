import pygame

from predator_pod import config
from predator_pod.brains.learner import training_running
from predator_pod.brains.shark_brain import AISharkController, SharkBrain
from predator_pod.scenes.ui import Button, draw_text, font
from predator_pod.world.scripted_sharks import ScriptedShark
from predator_pod.world.simulation import World

WATCH_ROUND_CAP = 60.0        # s, same cap as training rounds


class PlayScene:
    """A round of the game. watch=True: the AI shark hunts and you watch."""

    def __init__(self, game, fish_count, watch=False):
        self.game = game
        self.fish_count = fish_count
        self.watch = watch
        self.shark_label = None
        controller = None
        if watch:
            brain = SharkBrain.load_or_none()
            if brain is not None:
                controller = AISharkController(brain, greedy=True, record=False)
                self.shark_label = f"AI shark ({brain.rounds:,} training rounds)"
            else:
                controller = ScriptedShark("intercept")
                self.shark_label = "scripted shark (no AI shark trained yet - run train.py)"
        self.world = World(game.learner, fish_count, controller)
        self.finished = False
        self.training = training_running()
        right = self.world.arena.rect.right
        self.leave = Button("Leave game", (right - 60, 52), size=(120, 28), font_size=22)

    def _finish_round(self):
        """Learn from this round (once). Returns True if the brain was updated.

        Only called when every fish has been eaten, so only complete rounds teach the brain.
        """
        if self.finished or self.watch:
            return False
        self.finished = True
        return self.world.learn()

    def leave_round(self):
        """Leave early. The round is discarded and never teaches the brain."""
        self.finished = True
        self.game.show_landing(self.fish_count)

    def handle_event(self, event):
        if self.leave.clicked(event):
            self.leave_round()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.leave_round()

    def update(self, dt):
        if self.watch:
            self.world.step(dt)
            if not self.world.fishes or self.world.elapsed >= WATCH_ROUND_CAP:
                self.game.start_watch(self.fish_count)       # next round
            return

        keys = pygame.key.get_pressed()
        turn = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            turn -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            turn += 1.0

        self.world.step(dt, pygame.mouse.get_pos(), turn)
        if not self.world.fishes:
            times = self.world.survival_times()
            learned = self._finish_round()
            self.game.show_game_over(self.fish_count, self.world.elapsed, sum(times) / len(times),
                                     learned, self.world.stats())

    def draw(self, surface):
        w = self.world
        surface.fill(config.COLOR_BG)
        w.arena.draw(surface)
        for fish in w.fishes:
            fish.draw(surface)
        w.shark.draw(surface)

        left, right, y = w.arena.rect.left, w.arena.rect.right, 18
        draw_text(surface, f"Fish left: {len(w.fishes)}/{w.total}", 30, config.COLOR_TEXT, topleft=(left, y))
        draw_text(
            surface, f"Time: {format_time(w.elapsed)}", 30, config.COLOR_TEXT,
            center=(config.WINDOW_WIDTH // 2, y + 9),
        )
        fps_text = f"FPS: {self.game.clock.get_fps():.0f}"
        draw_text(surface, fps_text, 26, config.COLOR_TEXT_DIM, topleft=(right - font(26).size(fps_text)[0], y + 2))
        if self.watch:
            status = f"Watching: {self.shark_label} vs the trained fish  -  nothing is learned here"
        elif self.training:
            status = "Training (train.py) is running - this round won't be saved to the brain"
        else:
            status = self.game.learner.status_text()
        draw_text(surface, status, 22, config.COLOR_TEXT_DIM, center=(config.WINDOW_WIDTH // 2, y + 36))
        self.leave.draw(surface)


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
