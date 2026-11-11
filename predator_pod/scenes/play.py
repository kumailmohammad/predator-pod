import pygame

from predator_pod import config
from predator_pod.brains.random_brain import RandomBrain
from predator_pod.entities.fish import Fish
from predator_pod.entities.shark import Shark
from predator_pod.scenes.ui import draw_text, font
from predator_pod.world import physics
from predator_pod.world.arena import Arena
from predator_pod.world.sensors import observe_all


class PlayScene:
    def __init__(self, game, fish_count):
        self.game = game
        self.fish_count = fish_count
        self.arena = Arena()
        self.elapsed = 0.0

        # Shark starts at the left, fish spawn away from it.
        start = pygame.Vector2(self.arena.rect.left + 80, self.arena.rect.centery)
        self.shark = Shark(start)
        spots = physics.spawn_positions(
            fish_count, config.FISH_RADIUS, self.arena,
            avoid_pos=start, avoid_radius=config.SHARK_RADIUS * 5,
        )
        self.fishes = [Fish(p, RandomBrain()) for p in spots]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.game.show_landing(self.fish_count)

    def update(self, dt):
        self.elapsed += dt
        self.shark.update(pygame.mouse.get_pos(), dt, self.arena)

        for fish, obs in zip(self.fishes, observe_all(self.fishes, self.shark, self.arena)):
            fish.update(obs, dt, self.arena)

        physics.resolve_overlaps(self.fishes, self.arena, config.OVERLAP_PASSES)

        eaten = physics.eaten_fish(self.shark, self.fishes)
        if eaten:
            self.fishes = [f for f in self.fishes if f not in eaten]
            if not self.fishes:
                self.game.show_game_over(self.fish_count, self.elapsed)

    def draw(self, surface):
        surface.fill(config.COLOR_BG)
        self.arena.draw(surface)
        for fish in self.fishes:
            fish.draw(surface)
        self.shark.draw(surface)

        left = self.arena.rect.left
        y = 24
        draw_text(
            surface,
            f"Fish left: {len(self.fishes)}/{self.fish_count}",
            32, config.COLOR_TEXT, topleft=(left, y),
        )
        draw_text(
            surface, f"Time: {format_time(self.elapsed)}", 32, config.COLOR_TEXT,
            center=(config.WINDOW_WIDTH // 2, y + 10),
        )
        fps_text = f"FPS: {self.game.clock.get_fps():.0f}"
        width = font(28).size(fps_text)[0]
        draw_text(
            surface, fps_text, 28, config.COLOR_TEXT_DIM,
            topleft=(self.arena.rect.right - width, y + 2),
        )


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
