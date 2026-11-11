# Predator-Pod

A 2D learning project: you play a shark hunting a pod of fish. The fish have a small neural network brain that learns, round after round, how to survive longer. It stops learning by itself once it can't get any better.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py                # play (the fish keep learning while you play)
python train.py               # train headless against a computer shark
python train.py --reset       # throw away the saved brain and train from scratch
```

## Controls
| Input | Action |
|---|---|
| Mouse / touchpad | Steer the shark toward the cursor |
| Left / Right or A / D | Turn the shark |
| Esc | Back to menu / quit |
| Landing page | Pick 10–30 fish, Play, Reset brain |

The shark never stops swimming. It keeps its last direction and slides along walls when it hits them.

## The game world
| | Fish | Shark |
|---|---|---|
| Body radius | 7 px | 22 px |
| Speed | 120 px/s (constant) | 200 px/s (constant) |
| Turn rate | 240°/s | 220°/s |

A fish can't outrun the shark, but it can turn tighter. That is the whole game for the fish.

## What a fish can sense (18 inputs)
Everything is measured in the fish's own frame ("ahead" and "to my side"), so one brain works for every direction.

| # | Input | Why it matters |
|---|---|---|
| 1–4 | Distance to the wall ahead, left, right, behind | Corners are death traps |
| 5–6 | Direction to the shark | Which way to flee |
| 7–8 | Shark distance, and a "closeness" value that spikes when it's near | When to panic |
| 9–10 | Which way the shark is facing | Dodge out of its path, not straight ahead of it |
| 11 | Is the shark aiming at me? | Am I the target? |
| 12–14 | Direction and distance to the pod center | Safety in numbers |
| 15–17 | Direction and distance to the nearest fish | Spacing and following the pod |
| 18 | How aligned I am with the pod's swim direction | Moving together |

**Actions:** forward, backward (turn around), left, right. The fish only chooses a direction; its speed never changes.

**Network:** 18 inputs → 16 hidden (tanh) → 4 outputs. The highest output is the chosen action, for 372 weights and biases in total.

## How it learns
It uses an evolution strategy (ES), in [predator_pod/brains/evolution.py](predator_pod/brains/evolution.py). There is one shared brain, θ. Each step, it tries small random changes to θ in +change/−change pairs. Then θ moves toward the changes that survived longest and away from the ones that did worst. The size of the random change slowly shrinks, so learning gets finer over time.

- **Headless training (`train.py`):** the main way the brain learns. Each generation tests 10 candidate brains, and in each round the whole pod uses one candidate. The score is the pod's average survival time. Both candidates in a +/− pair get the exact same spawn positions and shark start, so the comparison between them is fair.
- **While you play:** every fish in the round gets its own variation, and fish are ranked by how long they survived. This signal is noisier, since who gets eaten first is partly luck. So playing nudges the brain toward your play style rather than training it from scratch.

**When does it stop?** The trainer keeps a 20-generation rolling average of survival time. If that average hasn't beaten its best by 1% for 60 generations (with at least 150 generations done), the brain is marked converged. From then on it is frozen, and every fish uses the exact learned weights.

While you play, every round that lasts at least 5 seconds counts as one generation. The computer shark used for training ([shark_ai.py](predator_pod/world/shark_ai.py)) moves by the same rules as yours and chases the most catchable fish.

The brain is saved after every generation to `models/pod_brain.npz`. **Reset brain** on the landing page, or `train.py --reset`, starts over.

## Code layout
```
main.py / train.py   play / headless training
predator_pod/
  config.py          sizes, speeds, colors, FPS
  game.py            main loop and scenes
  scenes/            landing, play, game over
  entities/          Fish, Shark, drawing
  world/arena.py     border
  world/physics.py   no-overlap rule, eating, spawning
  world/sensors.py   the 18 inputs
  world/simulation.py one round of the world (shared by game and trainer)
  world/shark_ai.py  computer shark for training
  brains/neural.py   the network
  brains/evolution.py the learner + convergence + save/load
models/pod_brain.npz saved brain
```
