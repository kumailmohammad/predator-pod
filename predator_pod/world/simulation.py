"""One round of the game world."""

import pygame

from predator_pod import config
from predator_pod.brains.learner import ACTION_DIRS, DECISION_INTERVAL, R_EATEN, Episode, reward
from predator_pod.entities.fish import Fish
from predator_pod.entities.shark import Shark
from predator_pod.world import physics
from predator_pod.world.arena import Arena
from predator_pod.world.sensors import sense


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
        self.episode = Episode(self.total)
        self.elapsed = 0.0
        self.decision_timer = 0.0

    def _decide(self):
        """Score each fish's last decision, then let the brain pick new directions."""
        s = sense(self.fishes, self.shark, self.arena)
        rewards = reward(s.shark_dist, s.pod_dist, s.wall_dist)
        actions, probs = self.learner.act(s.features)
        for k, fish in enumerate(self.fishes):
            self.episode.reward(fish.index, float(rewards[k]))
            self.episode.decide(fish.index, s.features[k], int(actions[k]), float(probs[k]))
            dx, dy = ACTION_DIRS[actions[k]]
            fish.desired = pygame.Vector2(dx, dy)

    def step(self, dt, shark_target, shark_turn=0.0):
        self.elapsed += dt
        self.shark.update(shark_target, dt, self.arena, shark_turn)

        self.decision_timer -= dt
        if self.decision_timer <= 0 and self.fishes:
            self.decision_timer += DECISION_INTERVAL
            self._decide()

        for fish in self.fishes:
            fish.update(dt, self.arena)
        physics.resolve_overlaps(self.fishes, self.arena, config.OVERLAP_PASSES)

        eaten = physics.eaten_fish(self.shark, self.fishes)
        if eaten:
            gone = {f.index for f in eaten}
            for i in gone:
                self.death_time[i] = self.elapsed
                self.episode.reward(i, R_EATEN)
            self.fishes = [f for f in self.fishes if f.index not in gone]
        return eaten

    def survival_times(self):
        """Seconds each fish survived; survivors get the full round time."""
        return [t if t is not None else self.elapsed for t in self.death_time]

    def learn(self):
        """Update the brain from this round. Returns True if it learned."""
        return self.learner.learn(self.episode, self.survival_times(), self.elapsed)
