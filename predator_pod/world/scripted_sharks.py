"""Hand-written sharks that hunt the way people usually do.

They steer with a target point, exactly like a human moving the cursor, so
the fish also learn to escape human-style chasing. They are never trained;
they are a fixed yardstick for the fish benchmark.
"""

import pygame

from predator_pod import config

CROWD_RANGE = 60.0
RETARGET_EVERY = 1.0          # s, the "isolated" hunter commits to a target this long
MAX_LEAD = 1.0                # s, how far ahead the interceptor aims


class ScriptedShark:
    KINDS = ("nearest", "isolated", "intercept")

    def __init__(self, kind):
        assert kind in self.KINDS, kind
        self.kind = kind
        self.target_index = None
        self.timer = 0.0

    def command(self, world, dt):
        fishes, shark = world.fishes, world.shark
        if not fishes:
            return None, 0.0
        nearest = min(fishes, key=lambda f: (f.pos - shark.pos).length_squared())
        if self.kind == "nearest":
            return pygame.Vector2(nearest.pos), 0.0
        if self.kind == "intercept":
            lead = min((nearest.pos - shark.pos).length() / config.SHARK_SPEED, MAX_LEAD)
            return nearest.pos + nearest.heading * nearest.speed * lead, 0.0
        # isolated: pick the fish with the fewest neighbors (closest breaks ties)
        self.timer -= dt
        by_index = {f.index: f for f in fishes}
        if self.timer <= 0 or self.target_index not in by_index:
            self.timer = RETARGET_EVERY
            r2 = CROWD_RANGE * CROWD_RANGE

            def score(f):
                crowd = sum(1 for g in fishes if g is not f and (g.pos - f.pos).length_squared() < r2)
                return crowd, (f.pos - shark.pos).length_squared()

            self.target_index = min(fishes, key=score).index
        return pygame.Vector2(by_index[self.target_index].pos), 0.0

    def on_eat(self, count):
        pass
