import pygame

from predator_pod import config
from predator_pod.entities.drawing import draw_shark

_AXES = [pygame.Vector2(1, 0), pygame.Vector2(-1, 0), pygame.Vector2(0, 1), pygame.Vector2(0, -1)]


class Shark:
    """Player-controlled shark.

    It always swims forward. Moving the cursor steers it smoothly toward the
    cursor (it slows down in sharp turns, like a real animal, so it can turn
    tightly); once it reaches the cursor it keeps going in its last direction.
    At a border it turns to swim along the wall in whichever direction it was
    already facing.
    """

    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.radius = config.SHARK_RADIUS
        self.speed = config.SHARK_SPEED
        self.heading = pygame.Vector2(1, 0)
        self.tail_phase = 0.0
        self.chasing = False
        self.last_mouse = None
        self.turn_speed = 0.0      # deg/s, eased
        self.velocity = pygame.Vector2(0, 0)   # px/s, last frame's actual motion

    def update(self, mouse, dt, arena, turn=0.0):
        """mouse: cursor/target point, or None (AI steering with turn only).
        turn: -1 (left) .. 1 (right), like the keyboard; overrides the cursor."""
        if mouse is None:
            self.chasing = False
            to_target = pygame.Vector2(0, 0)
        else:
            mouse = pygame.Vector2(mouse)
            if self.last_mouse is None or mouse.distance_to(self.last_mouse) > config.SHARK_MOUSE_MOVE_EPS:
                self.chasing = True          # cursor moved: new command
                self.last_mouse = mouse
            target = pygame.Vector2(mouse)
            arena.clamp(target, self.radius)
            to_target = target - self.pos

        if turn:
            self.chasing = False
            want_turn = turn * config.SHARK_TURN_RATE
        elif self.chasing:
            if to_target.length() < self.radius:
                self.chasing = False     # reached it: keep last direction
                want_turn = 0.0
            else:
                diff = (self.heading.angle_to(to_target) + 180) % 360 - 180
                # Turn faster the further off-target, eased so it never jerks.
                want_turn = max(-config.SHARK_TURN_RATE,
                                min(config.SHARK_TURN_RATE, diff * config.SHARK_STEER_GAIN))
        else:
            want_turn = 0.0

        # Ease turn speed and swim speed toward their targets.
        ease = min(1.0, config.SHARK_STEER_SMOOTHING * dt)
        self.turn_speed += (want_turn - self.turn_speed) * ease
        sharpness = min(abs(self.turn_speed) / config.SHARK_TURN_RATE, 1.0)
        want_speed = config.SHARK_SPEED * (1 - (1 - config.SHARK_MIN_SPEED_FACTOR) * sharpness)
        self.speed += (want_speed - self.speed) * ease

        self.heading = self.heading.rotate(self.turn_speed * dt)
        before = pygame.Vector2(self.pos)
        self.pos += self.heading * self.speed * dt
        hit_x, hit_y = arena.clamp(self.pos, self.radius)
        if hit_x or hit_y:
            self._slide_along_wall(hit_x, hit_y, to_target, arena)
        if dt > 0:
            self.velocity = (self.pos - before) / dt

        self.tail_phase += dt * 8

    def _slide_along_wall(self, hit_x, hit_y, to_target, arena):
        """Drop the part of the heading that points into the wall."""
        h = pygame.Vector2(self.heading)
        if hit_x:
            h.x = 0
        if hit_y:
            h.y = 0
        if h.length_squared() > 1e-4:
            self.heading = h.normalize()
            return
        # Facing straight into a wall or a corner: pick an open direction
        # along the wall, preferring the cursor's side.
        prefer = to_target if to_target.length_squared() > 1e-6 else (
            pygame.Vector2(arena.rect.center) - self.pos
        )
        open_dirs = []
        for d in _AXES:
            probe = self.pos + d
            if arena.clamp(pygame.Vector2(probe), self.radius) == (False, False):
                open_dirs.append(d)
        if open_dirs:
            self.heading = pygame.Vector2(max(open_dirs, key=lambda d: d.dot(prefer)))

    def draw(self, surface):
        draw_shark(
            surface,
            self.pos,
            self.heading.as_polar()[1],
            self.radius,
            config.COLOR_SHARK,
            config.COLOR_SHARK_BELLY,
            config.COLOR_EYE,
            self.tail_phase,
        )
