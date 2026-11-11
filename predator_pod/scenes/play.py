import pygame

from predator_pod import config
from predator_pod.brains.neural import NeuralBrain
from predator_pod.scenes.ui import draw_text, font
from predator_pod.world.simulation import World

MIN_ROUND_SECONDS = 5.0   # shorter rounds (quick Esc) don't count for learning


class PlayScene:
    def __init__(self, game, fish_count):
        self.game = game
        self.fish_count = fish_count
        brains = [NeuralBrain(p) for p in game.learner.sample(fish_count)]
        self.world = World(brains)
        self.finished = False

    def _finish_round(self):
        """Feed this round's survival times to the learner (once)."""
        if self.finished:
            return
        self.finished = True
        if self.world.elapsed >= MIN_ROUND_SECONDS:
            self.game.learner.update(self.world.fitness())

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._finish_round()
            self.game.show_landing(self.fish_count)

    def update(self, dt):
        keys = pygame.key.get_pressed()
        turn = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            turn -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            turn += 1.0

        self.world.step(dt, pygame.mouse.get_pos(), turn)
        if not self.world.fishes:
            fitness = self.world.fitness()
            self._finish_round()
            self.game.show_game_over(self.fish_count, self.world.elapsed, sum(fitness) / len(fitness))

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

        status = self.game.learner.status_text()
        draw_text(surface, status, 22, config.COLOR_TEXT_DIM, center=(config.WINDOW_WIDTH // 2, y + 36))


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
