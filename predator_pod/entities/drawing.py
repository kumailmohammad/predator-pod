"""Simple 2D shapes for fish and shark, rotated to face their heading."""

import math

import pygame


def _transform(points, pos, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [(pos.x + x * c - y * s, pos.y + x * s + y * c) for x, y in points]


def _ellipse_points(length, width, steps=16):
    return [
        (math.cos(t) * length / 2, math.sin(t) * width / 2)
        for t in (2 * math.pi * i / steps for i in range(steps))
    ]


def draw_fish(surface, pos, angle, radius, body_color, tail_color, eye_color, tail_phase):
    length, width = radius * 2.4, radius * 1.5
    wag = math.sin(tail_phase) * radius * 0.35
    tail = [
        (-length / 2 + 1, 0),
        (-length / 2 - radius * 0.9, -radius * 0.75 + wag),
        (-length / 2 - radius * 0.9, radius * 0.75 + wag),
    ]
    pygame.draw.polygon(surface, tail_color, _transform(tail, pos, angle))
    pygame.draw.polygon(surface, body_color, _transform(_ellipse_points(length, width), pos, angle))
    eye = _transform([(length * 0.25, -width * 0.15)], pos, angle)[0]
    pygame.draw.circle(surface, eye_color, eye, max(1, radius // 5))


def draw_shark(surface, pos, angle, radius, body_color, belly_color, eye_color, tail_phase):
    length, width = radius * 2.6, radius * 1.15
    wag = math.sin(tail_phase) * radius * 0.25
    tail = [
        (-length / 2 + 4, 0),
        (-length / 2 - radius * 0.7, -radius * 0.7 + wag),
        (-length / 2 - radius * 0.45, wag * 0.5),
        (-length / 2 - radius * 0.7, radius * 0.55 + wag),
    ]
    fins = [
        [(radius * 0.1, -width * 0.4), (-radius * 0.5, -width * 1.0), (-radius * 0.45, -width * 0.35)],
        [(radius * 0.1, width * 0.4), (-radius * 0.5, width * 1.0), (-radius * 0.45, width * 0.35)],
    ]
    pygame.draw.polygon(surface, body_color, _transform(tail, pos, angle))
    for fin in fins:
        pygame.draw.polygon(surface, body_color, _transform(fin, pos, angle))
    pygame.draw.polygon(surface, body_color, _transform(_ellipse_points(length, width, 24), pos, angle))
    pygame.draw.polygon(
        surface, belly_color, _transform(_ellipse_points(length * 0.7, width * 0.45, 20), pos, angle)
    )
    for side in (-1, 1):
        eye = _transform([(length * 0.32, side * width * 0.25)], pos, angle)[0]
        pygame.draw.circle(surface, eye_color, eye, max(2, radius // 9))
