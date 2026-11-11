import pygame

from predator_pod import config
from predator_pod.scenes.ui import Button, draw_text


class LandingScene:
    def __init__(self, game, fish_count=config.FISH_DEFAULT_COUNT):
        self.game = game
        self.fish_count = fish_count
        cx = config.WINDOW_WIDTH // 2
        self.minus = Button("-", (cx - 170, 330), size=(50, 50), font_size=44)
        self.plus = Button("+", (cx + 170, 330), size=(50, 50), font_size=44)
        self.slider = pygame.Rect(0, 0, 260, 8)
        self.slider.center = (cx, 330)
        self.dragging = False
        self.play = Button("Play", (cx, 430))
        self.quit = Button("Quit", (cx, 495))

    def _set_count(self, n):
        self.fish_count = max(config.FISH_MIN_COUNT, min(config.FISH_MAX_COUNT, n))

    def _count_from_x(self, x):
        t = (x - self.slider.left) / self.slider.width
        t = max(0.0, min(1.0, t))
        span = config.FISH_MAX_COUNT - config.FISH_MIN_COUNT
        self._set_count(round(config.FISH_MIN_COUNT + t * span))

    def _knob_x(self):
        span = config.FISH_MAX_COUNT - config.FISH_MIN_COUNT
        t = (self.fish_count - config.FISH_MIN_COUNT) / span
        return self.slider.left + t * self.slider.width

    def handle_event(self, event):
        if self.play.clicked(event):
            self.game.start_play(self.fish_count)
        elif self.quit.clicked(event):
            self.game.running = False
        elif self.minus.clicked(event):
            self._set_count(self.fish_count - 1)
        elif self.plus.clicked(event):
            self._set_count(self.fish_count + 1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.slider.inflate(20, 30).collidepoint(event.pos):
                self.dragging = True
                self._count_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._count_from_x(event.pos[0])
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_DOWN):
                self._set_count(self.fish_count - 1)
            elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                self._set_count(self.fish_count + 1)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.game.start_play(self.fish_count)
            elif event.key == pygame.K_ESCAPE:
                self.game.running = False

    def update(self, dt):
        pass

    def draw(self, surface):
        surface.fill(config.COLOR_BG)
        cx = config.WINDOW_WIDTH // 2
        draw_text(surface, "PREDATOR-POD", 96, config.COLOR_ACCENT, center=(cx, 130))
        draw_text(
            surface,
            "You are the shark. Hunt down the whole pod.",
            30,
            config.COLOR_TEXT_DIM,
            center=(cx, 195),
        )
        draw_text(
            surface,
            "Move with the mouse or touchpad  -  Esc returns to menu",
            24,
            config.COLOR_TEXT_DIM,
            center=(cx, 225),
        )

        draw_text(surface, f"Fish in the pod: {self.fish_count}", 36, config.COLOR_TEXT, center=(cx, 285))
        pygame.draw.rect(surface, config.COLOR_BUTTON, self.slider, border_radius=4)
        filled = self.slider.copy()
        filled.width = int(self._knob_x() - self.slider.left)
        pygame.draw.rect(surface, config.COLOR_ACCENT, filled, border_radius=4)
        pygame.draw.circle(surface, config.COLOR_TEXT, (int(self._knob_x()), self.slider.centery), 12)
        draw_text(surface, str(config.FISH_MIN_COUNT), 22, config.COLOR_TEXT_DIM, center=(self.slider.left, 360))
        draw_text(surface, str(config.FISH_MAX_COUNT), 22, config.COLOR_TEXT_DIM, center=(self.slider.right, 360))

        for b in (self.minus, self.plus, self.play, self.quit):
            b.draw(surface)
