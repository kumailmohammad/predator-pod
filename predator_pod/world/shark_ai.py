"""A computer shark used to train the fish without a human player.

It steers with exactly the same rules as the player's shark (constant speed,
turn rate, wall sliding); only the target point is chosen by code.
"""

SCORE_FRONT_BONUS = 0.6    # prefer fish in front of the shark
MAX_LEAD_TIME = 0.8        # s, how far ahead it predicts fish movement


def ai_target(shark, fishes):
    best, best_score, best_d = None, float("inf"), 0.0
    for f in fishes:
        to = f.pos - shark.pos
        d = to.length()
        if d < 1e-6:
            return f.pos
        cos = shark.heading.dot(to) / d
        score = d * (1 + SCORE_FRONT_BONUS * (1 - cos))
        if score < best_score:
            best, best_score, best_d = f, score, d
    lead = min(best_d / shark.speed, MAX_LEAD_TIME)
    return best.pos + best.heading * best.speed * lead
