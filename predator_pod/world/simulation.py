"""One round of the game world, shared by the playable game and the trainer."""

import random

import pygame

from predator_pod import config
from predator_pod.entities.fish import Fish
from predator_pod.entities.shark import Shark
from predator_pod.world import physics
from predator_pod.world.arena import Arena
from predator_pod.world.sensors import observe_all


class World:
    def __init__(self, brains, random_shark_start=False):
        self.arena = Arena()
        r = self.arena.rect
        if random_shark_start:
            start = pygame.Vector2(
                random.uniform(r.left + 60, r.right - 60), random.uniform(r.top + 60, r.bottom - 60)
            )
        else:
            start = pygame.Vector2(r.left + 80, r.centery)
        self.shark = Shark(start)
        if random_shark_start:
            self.shark.heading = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))

        spots = physics.spawn_positions(
            len(brains), config.FISH_RADIUS, self.arena,
            avoid_pos=start, avoid_radius=config.SHARK_RADIUS * 6,
        )
        self.fishes = []
        for i, (p, brain) in enumerate(zip(spots, brains)):
            fish = Fish(p, brain)
            fish.index = i
            self.fishes.append(fish)
        self.total = len(self.fishes)
        self.death_time = [None] * self.total
        self.elapsed = 0.0

    def step(self, dt, shark_target, shark_turn=0.0):
        self.elapsed += dt
        self.shark.update(shark_target, dt, self.arena, shark_turn)
        for fish, obs in zip(self.fishes, observe_all(self.fishes, self.shark, self.arena)):
            fish.update(obs, dt, self.arena)
        physics.resolve_overlaps(self.fishes, self.arena, config.OVERLAP_PASSES)

        eaten = physics.eaten_fish(self.shark, self.fishes)
        if eaten:
            for f in eaten:
                self.death_time[f.index] = self.elapsed
            self.fishes = [f for f in self.fishes if f.index not in {e.index for e in eaten}]
        return eaten

    def fitness(self):
        """Seconds each fish survived; survivors get the full round time."""
        return [t if t is not None else self.elapsed for t in self.death_time]
