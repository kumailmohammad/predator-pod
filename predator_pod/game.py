import os

import pygame

from predator_pod import config
from predator_pod.brains.learner import Learner
from predator_pod.scenes.game_over import GameOverScene
from predator_pod.scenes.landing import LandingScene
from predator_pod.scenes.play import PlayScene


class Game:
    def __init__(self):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.display.set_caption(config.TITLE)
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.learner = Learner.load_or_new()
        self.brain_mtime = self._mtime()
        self.scene = LandingScene(self)

    def _mtime(self):
        try:
            return self.learner.path.stat().st_mtime
        except OSError:
            return None

    def refresh_brain(self):
        """Pick up a better brain that train.py saved while the game was open."""
        m = self._mtime()
        if m != self.brain_mtime:
            self.learner = Learner.load_or_new()
            self.brain_mtime = self._mtime()

    def show_landing(self, fish_count=config.FISH_DEFAULT_COUNT):
        self.refresh_brain()
        self.scene = LandingScene(self, fish_count)

    def start_play(self, fish_count):
        self.refresh_brain()
        self.scene = PlayScene(self, fish_count)

    def start_watch(self, fish_count):
        self.refresh_brain()
        self.scene = PlayScene(self, fish_count, watch=True)

    def show_game_over(self, fish_count, elapsed, mean_survival, learned, stats=None):
        self.scene = GameOverScene(self, fish_count, elapsed, mean_survival, learned, stats)

    def run(self):
        while self.running:
            dt = min(self.clock.tick(config.FPS) / 1000.0, config.MAX_DT)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Closing the window mid-round is leaving early: nothing is learned.
                    self.running = False
                else:
                    self.scene.handle_event(event)
            self.scene.update(dt)
            self.scene.draw(self.screen)
            pygame.display.flip()
        pygame.quit()
