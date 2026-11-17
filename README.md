# Predator-Pod

A 2D learning project: you play a shark hunting a pod of fish. Every fish shares one neural-network brain. **It learns only from your games.** Each time a round ends, the brain updates its weights and biases from what happened, and the next round the fish are a little better at staying alive. When it stops improving, it freezes and saves itself.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

## Controls
| Input | Action |
|---|---|
| Mouse / touchpad | Steer the shark toward the cursor |
| Left / Right or A / D | Turn the shark |
| Esc | Back to menu (a round of 5 s or more still counts for learning) |
| Landing page | Pick 10–30 fish, Play, Reset brain |

The shark never stops swimming. It keeps its last direction, slows down in sharp turns so it can turn tightly, and slides along walls.

## The game world
| | Fish | Shark |
|---|---|---|
| Body radius | 7 px | 22 px |
| Speed | 120 px/s (constant) | 200 px/s cruising, 90 px/s in hard turns |
| Turn rate | 200°/s | up to 300°/s |

## The fish brain
```
18 inputs ──► 32 hidden neurons (tanh) ──► 4 outputs (softmax)
                                            up / down / left / right
```
About 7 times a second (every 0.15 s), each fish feeds its senses into the network and swims toward the direction it picks.

### Inputs (what a fish senses)
| # | Input | Why it matters |
|---|---|---|
| 1–2 | Direction to the shark (x, y) | Which way the danger is |
| 3 | Distance to the shark | How much danger |
| 4 | Closeness to the shark (spikes when it's near) | When to panic |
| 5–6 | Which way the shark is heading | Get out of its path |
| 7–10 | Distance to the left, right, top and bottom walls | Corners are traps |
| 11–13 | Direction and distance to the school's center | Stay with the school |
| 14–16 | Direction and distance to the nearest fish | Spacing |
| 17–18 | Its own heading | Where it's going now |

### How it learns from your play
It uses reinforcement learning (policy gradient with PPO-style updates), in [predator_pod/brains/learner.py](predator_pod/brains/learner.py).

After every decision, a fish gets a **reward**:

| Reward | Amount | Purpose |
|---|---|---|
| Far from the shark | up to +1.0 (full at 300 px) | **The main goal: stay as far from the shark as possible** |
| Near the school center | up to +0.25 | Stick with the school, while still free to wander off and rejoin |
| Hugging a wall | −0.3 | Don't get cornered |
| Getting eaten | −10 | The worst outcome |

When your round ends, every decision is scored by the rewards that followed it. Decisions that turned out better than average become more likely, and worse ones less likely. The round's data is reused for a few passes of small, clipped updates, so one round teaches a lot without the network changing wildly. The brain is then saved to `models/pod_brain.npz`.

**Learning stops automatically** once the rolling 5-round average survival time hasn't improved by 3% for 10 rounds, with at least 20 rounds played. The brain is then frozen, and from then on every fish always picks its best direction.

**Where to see training:** the top of the play screen shows the brain's status and how many rounds it has learned from. After every round, the game-over screen shows the learning curve: average fish survival per round, plus a rolling average. If the line is going up, the fish are learning.

**Reset brain** on the landing page starts over from an untrained brain. An untrained brain swims randomly, so the first few rounds look chaotic.

## Code layout
```
main.py              start the game
predator_pod/
  config.py          sizes, speeds, colors, FPS
  game.py            main loop and scenes
  scenes/            landing, play, game over (with learning chart)
  entities/          Fish, Shark, drawing
  world/arena.py     border
  world/physics.py   no-overlap rule, eating, spawning
  world/sensors.py   the 18 inputs
  world/simulation.py one round: decisions, rewards, collisions
  brains/learner.py  the network, rewards, learning, convergence, save/load
models/pod_brain.npz saved brain (created after your first round)
```
