"""One round of the game world."""

import numpy as np
import pygame

from predator_pod import config
from predator_pod.brains import learner as L
from predator_pod.entities.fish import Fish
from predator_pod.entities.shark import Shark
from predator_pod.world import physics
from predator_pod.world.arena import Arena
from predator_pod.world.sensors import sense

WALL_DEATH_RANGE = 40.0       # px, a fish eaten this close to a wall counts as a wall death
SCHOOL_MIN_NEIGHBORS = 2      # "in a group" for the school % stat
ZIGZAG_ANGLE = 60.0           # deg off the straight escape line


class World:
    def __init__(self, learner, fish_count):
        self.learner = learner
        self.arena = Arena()
        start = pygame.Vector2(self.arena.rect.left + 80, self.arena.rect.centery)
        self.shark = Shark(start)
        spots = physics.spawn_positions(
            fish_count, config.FISH_RADIUS, self.arena,
            avoid_pos=start, avoid_radius=config.SHARK_RADIUS * 6,
        )
        self.fishes = [Fish(p, i) for i, p in enumerate(spots)]
        self.total = len(self.fishes)
        self.death_time = [None] * self.total
        self.episode = L.Episode(self.total)
        self.elapsed = 0.0
        self.decision_timer = 0.0
        # round stats
        self.action_counts = np.zeros(L.N_ACTIONS, dtype=int)
        self.decisions = 0
        self.grouped = 0
        self.bursts = 0
        self.wall_deaths = 0

    def _direction(self, fish, k, action, s):
        """Turn a chosen behavior into the direction the fish should swim."""
        if action < L.N_DIRECTION_ACTIONS:
            return pygame.Vector2(*L.ACTION_DIRS[action])
        away = pygame.Vector2(*s.away_dir[k])
        if action == L.CRUISE:
            return pygame.Vector2(fish.heading)
        if action == L.FLEE:
            return away
        if action == L.ALIGN:
            v = pygame.Vector2(*s.school_dir[k])
            return v if v.length_squared() > 1e-9 else pygame.Vector2(fish.heading)
        if action == L.HERD:
            v = pygame.Vector2(*s.herd_dir[k])
            return v if v.length_squared() > 1e-9 else pygame.Vector2(fish.heading)
        if action == L.FOUNTAIN:
            # Sideways to the shark's line of attack, on the side the fish is on.
            side = pygame.Vector2(-s.shark_heading[1], s.shark_heading[0])
            return side if side.dot(away) >= 0 else -side
        # ZIGZAG: away from the shark, alternating left/right of the escape line
        fish.zig = -fish.zig
        return away.rotate(ZIGZAG_ANGLE * fish.zig)

    def _decide(self):
        """Score each fish's last decision, then let the brain pick new behaviors."""
        s = sense(self.fishes, self.shark, self.arena)
        rewards = L.reward(s.shark_dist, s.pod_dist, s.wall_dist, s.crowd_count, s.corner)
        ready = np.array([f.burst_ready for f in self.fishes])
        actions, bursts, probs = self.learner.act(s.features, ready)
        for k, fish in enumerate(self.fishes):
            a, b = int(actions[k]), bool(bursts[k])
            self.episode.reward(fish.index, float(rewards[k]))
            self.episode.decide(fish.index, s.features[k], a, int(b), bool(ready[k]),
                                float(probs[k]))
            d = self._direction(fish, k, a, s)
            if d.length_squared() > 1e-9:
                fish.desired = d.normalize()
            if b and fish.burst():
                self.bursts += 1
            self.action_counts[a] += 1
        self.decisions += len(self.fishes)
        self.grouped += int((s.crowd_count >= SCHOOL_MIN_NEIGHBORS).sum())

    def step(self, dt, shark_target, shark_turn=0.0):
        self.elapsed += dt
        self.shark.update(shark_target, dt, self.arena, shark_turn)

        self.decision_timer -= dt
        if self.decision_timer <= 0 and self.fishes:
            self.decision_timer += L.DECISION_INTERVAL
            self._decide()

        for fish in self.fishes:
            fish.update(dt, self.arena)
        physics.resolve_overlaps(self.fishes, self.arena, config.OVERLAP_PASSES)

        eaten = physics.eaten_fish(self.shark, self.fishes)
        if eaten:
            gone = {f.index for f in eaten}
            for f in eaten:
                if min(self.arena.wall_distances(f.pos)) < WALL_DEATH_RANGE:
                    self.wall_deaths += 1
            for i in gone:
                self.death_time[i] = self.elapsed
                self.episode.reward(i, L.R_EATEN)
            self.fishes = [f for f in self.fishes if f.index not in gone]
        return eaten

    def survival_times(self):
        """Seconds each fish survived; survivors get the full round time."""
        return [t if t is not None else self.elapsed for t in self.death_time]

    def stats(self):
        school = 100.0 * self.grouped / self.decisions if self.decisions else 0.0
        total = self.action_counts.sum()
        top = []
        if total:
            for a in self.action_counts.argsort()[::-1][:3]:
                if self.action_counts[a]:
                    top.append((L.ACTION_NAMES[a], 100.0 * self.action_counts[a] / total))
        return {"school_pct": school, "wall_deaths": self.wall_deaths,
                "bursts": self.bursts, "top_actions": top}

    def learn(self):
        """Update the brain from this round. Returns True if it learned."""
        return self.learner.learn(self.episode, self.survival_times(), self.elapsed, self.stats())
