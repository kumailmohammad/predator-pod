"""What each fish can sense. Brains decide using only this."""

from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class Observation:
    # World-frame values (handy for hand-written brains)
    wall_left: float
    wall_right: float
    wall_top: float
    wall_bottom: float
    to_shark: pygame.Vector2      # vector from fish to shark
    to_pod: pygame.Vector2        # vector from fish to pod center
    pod_heading: pygame.Vector2   # average heading of the pod
    heading: pygame.Vector2       # fish's own unit heading
    velocity: pygame.Vector2

    def to_array(self, arena_w, arena_h, max_speed):
        """Normalized vector in the fish's local frame, for a neural network.

        Layout (11 values):
          4 wall distances, shark (fwd, side, dist), pod (fwd, side, dist),
          own speed.
        """
        scale = max(arena_w, arena_h)
        angle = self.heading.as_polar()[1]

        def local(v):
            r = v.rotate(-angle)
            return r.x / scale, r.y / scale

        sx, sy = local(self.to_shark)
        px, py = local(self.to_pod)
        return np.array(
            [
                self.wall_left / arena_w,
                self.wall_right / arena_w,
                self.wall_top / arena_h,
                self.wall_bottom / arena_h,
                sx, sy, self.to_shark.length() / scale,
                px, py, self.to_pod.length() / scale,
                self.velocity.length() / max_speed,
            ],
            dtype=np.float32,
        )


def observe_all(fishes, shark, arena):
    """Build an Observation for every fish."""
    if not fishes:
        return []
    center = sum((f.pos for f in fishes), pygame.Vector2()) / len(fishes)
    avg_heading = sum((f.heading for f in fishes), pygame.Vector2())
    if avg_heading.length_squared() > 1e-9:
        avg_heading = avg_heading.normalize()

    obs = []
    for f in fishes:
        left, right, top, bottom = arena.wall_distances(f.pos)
        obs.append(
            Observation(
                wall_left=left,
                wall_right=right,
                wall_top=top,
                wall_bottom=bottom,
                to_shark=shark.pos - f.pos,
                to_pod=center - f.pos,
                pod_heading=pygame.Vector2(avg_heading),
                heading=pygame.Vector2(f.heading),
                velocity=f.heading * f.speed,
            )
        )
    return obs
