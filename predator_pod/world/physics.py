"""Body collisions: fish can't overlap each other, and the shark eats fish."""

import random

import pygame


def resolve_overlaps(bodies, arena, passes):
    """Push apart any two bodies whose circles overlap.

    Each body needs `.pos` (Vector2) and `.radius`. Each overlapping pair is
    separated by half the overlap each, along the line between centers.
    Several passes settle crowded clusters.
    """
    n = len(bodies)
    for _ in range(passes):
        moved = False
        for i in range(n):
            a = bodies[i]
            for j in range(i + 1, n):
                b = bodies[j]
                min_dist = a.radius + b.radius
                delta = b.pos - a.pos
                dist_sq = delta.length_squared()
                if dist_sq >= min_dist * min_dist:
                    continue
                if dist_sq < 1e-9:
                    # Exactly on top of each other: pick a random direction.
                    delta = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
                    dist = 0.0
                else:
                    dist = dist_sq ** 0.5
                    delta /= dist
                push = (min_dist - dist) / 2
                a.pos -= delta * push
                b.pos += delta * push
                moved = True
        for body in bodies:
            arena.clamp(body.pos, body.radius)
        if not moved:
            break


def eaten_fish(shark, fishes):
    """Fish whose center is inside the shark's body."""
    r_sq = shark.radius * shark.radius
    return [f for f in fishes if (f.pos - shark.pos).length_squared() < r_sq]


def spawn_positions(count, radius, arena, avoid_pos=None, avoid_radius=0.0):
    """Random non-overlapping spawn points (rejection sampling)."""
    rect = arena.rect
    points = []
    min_gap = radius * 2 + 2
    attempts = 0
    while len(points) < count and attempts < count * 500:
        attempts += 1
        p = pygame.Vector2(
            random.uniform(rect.left + radius, rect.right - radius),
            random.uniform(rect.top + radius, rect.bottom - radius),
        )
        if avoid_pos is not None and p.distance_to(avoid_pos) < avoid_radius:
            continue
        if all(p.distance_to(q) >= min_gap for q in points):
            points.append(p)
    return points
