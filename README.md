# Predator-Pod

A 2D learning project: you play a shark hunting a pod of fish. Every fish shares one neural-network brain, and the goal is to train that brain until the pod is as hard as possible for a human to wipe out.

The fish learn two ways:
- **Automatically** (`python train.py`): an AI shark with its own neural network hunts the fish in thousands of headless rounds. Both sides learn at the same time, so each keeps forcing the other to improve.
- **From your games:** every round you play also updates the fish brain.

## Quick start: play in 3 steps
1. **Install** Python 3.12 or newer, then run the commands below.
2. **Start the game** with `python main.py`.
3. **Pick a pod size** (10–30 fish) and press **Play**. Steer the shark with the mouse, and eat the whole pod.

Want to see the trained AI shark instead? Press **Watch AI** on the landing page.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py          # play (or "Watch AI" to see the AI shark hunt the trained fish)
python train.py         # automatic training, 30 min (Ctrl+C stops; run again to resume)
python progress.py      # graphs of the prey and shark training (--live to auto-refresh)
```

## Important things to know
- **The game ships with a trained fish brain** (`models/pod_brain.npz`), so the fish are already hard to catch. You don't need to train before playing.
- **Your games train the fish too.** Each round you finish updates the brain, unless `train.py` is running at the same time. In that case the game says so and doesn't save the round, so it can't overwrite the trainer's work.
- **Training is safe to stop.** Ctrl+C stops `train.py` cleanly and keeps the best brain. Run `python train.py` again to resume.
- **Reset brain** on the landing page starts the fish from scratch, but first it copies the old brain to `models/backups/`.
- **Watch the progress** with `python progress.py` (add `--live` to refresh while training runs).
- **Large files are tracked in git.** The brains (`models/pod_brain.npz` and `models/shark_brain.npz`) are committed. Checkpoints, logs and backups are not.

## Controls
| Input | Action |
|---|---|
| Mouse / touchpad | Steer the shark toward the cursor |
| Left / Right or A / D | Turn the shark |
| Esc | Back to menu (a round of 5 s or more still counts for learning) |
| Landing page | Pick 10–30 fish, Play, Watch AI, Reset brain (keeps a backup in `models/backups/`) |

The shark never stops swimming. It keeps its last direction, slows down in sharp turns so it can turn tightly, and slides along walls.

## The game world
| | Fish | Shark |
|---|---|---|
| Body radius | 7 px | 22 px |
| Speed | 120 px/s (constant) | 200 px/s cruising, 90 px/s in hard turns |
| Turn rate | 200°/s | up to 300°/s |

## The fish brain
```
31 inputs ──► 32 hidden neurons (tanh) ──┬─► 14 escape behaviors (softmax)
                                         └─► fast-start yes/no (sigmoid)
```
About 7 times a second (every 0.15 s), each fish feeds its senses into the network. It picks one behavior, and it may also fire a fast-start.

### Inputs (what a fish senses)
| # | Input | Why it matters |
|---|---|---|
| 1–2 | Direction to the shark (x, y) | Which way the danger is |
| 3 | Distance to the shark | How much danger |
| 4 | Closeness to the shark (spikes when it's near) | When to panic |
| 5–6 | Which way the shark is heading | Get out of its path |
| 7–10 | Distance to the left, right, top and bottom walls | Walls are traps |
| 11–13 | Direction and distance to the school's center | Stay with the school |
| 14–16 | Direction and distance to the nearest fish | Spacing |
| 17–18 | Its own heading | Where it's going now |
| 19 | Am I the target? (1.0 = closest fish to the shark) | The target must flee; the others stay grouped |
| 20 | Crowd: fish within 60 px | Alone or inside the school |
| 21–24 | Free water up, down, left and right (to a wall or fish) | Don't flee into a dead end |
| 25 | In a corner (two walls within 60 px) | Corners are death traps |
| 26 | In the shark's path (shark aimed at me) | Real danger versus a passing shark |
| 27 | Closing speed (is the gap shrinking?) | Being chased or not |
| 28–29 | Average heading of my group | Swim with the school |
| 30 | Shark's distance to a wall | A shark sliding along a wall turns badly |
| 31 | Fast-start ready | Whether the burst is available |

### Outputs: real prey-fish escape behaviors
| # | Behavior | Based on | What the fish does |
|---|---|---|---|
| 0–7 | Up, down, left, right and the 4 diagonals | Plain swimming | Swim that way |
| 8 | Cruise | Holding course | Keep its current heading |
| 9 | Flee | Direct escape | Straight away from the shark |
| 10 | Align | Schooling (polarized swimming) | Match the heading of nearby fish |
| 11 | Selfish herd | Hamilton's selfish herd | Push into the middle of nearby fish so others are on the outside |
| 12 | Fountain | Fountain effect | Swim sideways to the shark's line of attack, so the school splits around it |
| 13 | Zigzag | Protean (erratic) escape | Away from the shark at ±60°, switching sides each decision |
| + | Fast-start | C-start (Mauthner-cell reflex) | Instant turn, then 2.5× speed for 0.35 s, then a 3 s cooldown |

Flash expansion, bait-balling and the vacuole aren't separate actions. They emerge when many fish combine flee, selfish herd and fountain at once.

### How the fish learn
They use reinforcement learning (policy gradient with PPO-style updates). The network and rewards are in [predator_pod/brains/learner.py](predator_pod/brains/learner.py), and the PPO code is in [predator_pod/brains/ppo.py](predator_pod/brains/ppo.py).

After every decision, a fish gets a **reward**. Here *threat* is 1 when the shark is on top of the fish and 0 when it's 250 px or more away.

| Reward | Amount | Purpose |
|---|---|---|
| Far from the shark | up to +1.0 (full at 300 px) | **The main goal: stay as far from the shark as possible** |
| Near the school center | up to +0.25 | Stick with the school, while still free to wander off and rejoin |
| In a group | up to +0.4 × (0.5 + threat) | Safety in numbers, most of all when the shark is close |
| Alone under threat | −0.5 × threat | Lone fish get picked off |
| Near a wall (within 60 px) | up to −0.6 × (1 + threat) | Don't get pinned against a wall |
| In a corner | −0.4 × (1 + threat) | Corners are death traps |
| Getting eaten | −10 | The worst outcome |

When a round ends, every decision is scored by the rewards that followed it. Decisions that turned out better than average become more likely, and worse ones less likely. The data is reused for a few passes of small, clipped updates, so the network learns a lot without changing wildly.

**Learning from your games** stops automatically once the rolling 5-round average survival time of your rounds hasn't improved by 3% for 10 rounds, with at least 20 rounds played. From then on every fish always picks its best behavior. While `train.py` is running, your rounds aren't saved, so they can't overwrite the trainer's brain. The game tells you when this happens.

## The AI shark (sparring partner)
```
29 inputs ──► 32 hidden neurons (tanh) ──► 5 turn commands (softmax)
                                            hard left / left / straight / right / hard right
```
It decides 10 times a second and steers through **exactly the same physics as you do with the keyboard**: same speed, turn rate, slow-down in sharp turns and wall sliding. So the fish never learn to beat a shark that's stronger than a human could be. Its code is in [predator_pod/brains/shark_brain.py](predator_pod/brains/shark_brain.py) and [predator_pod/world/shark_sensors.py](predator_pod/world/shark_sensors.py).

| Inputs (all in the shark's own frame: "ahead" and "to my right") | Count |
|---|---|
| Own speed and turn speed | 2 |
| Distance to a wall along 5 rays (ahead, ahead-left, ahead-right, left, right) | 5 |
| 3 nearest fish: direction and distance | 9 |
| Nearest fish: its heading, how crowded it is, its distance to a wall, its fast-start ready | 5 |
| Most isolated fish within 300 px: direction and distance | 3 |
| Pod center: direction and distance, plus how spread out the pod is | 4 |
| Fraction of fish left | 1 |

| Shark reward | Amount |
|---|---|
| Eating a fish | +10 |
| Getting closer to the nearest fish | potential-based shaping (can't be farmed by circling) |
| Every decision | −0.02 (hunt fast) |

## Automatic training: `python train.py`
Self-play training is in [train.py](train.py) and [predator_pod/selfplay.py](predator_pod/selfplay.py).
- **Each iteration** plays a batch of headless 60-second rounds in parallel (one process per CPU core, minus 2), with 10–30 fish. The opponents are:
  - **60%** the current learning AI shark
  - **20%** older AI sharks from a snapshot pool, so the fish don't forget how to beat old tactics
  - **20%** scripted human-style sharks: *chase the nearest fish*, *chase the most isolated fish*, *intercept*. These steer with a target point, just like your cursor
- Both brains are then updated with PPO.
- **Every 10 iterations, a benchmark** runs. The fish play fixed-seed rounds against the 3 scripted sharks, and against the AI shark for reference. The brain the game uses, `models/pod_brain.npz`, is replaced **only when the scripted-shark score improves**, so the game always has the best fish so far.
- **It stops** after `--minutes` (30 by default), or once the benchmark hasn't improved for `--patience` evaluations (15 by default). **Ctrl+C** stops cleanly. Running it again resumes from `models/checkpoints/`. If you played the game since the last run, it starts from your newer brain instead, so your games are never thrown away.

Options: `--minutes N`, `--workers N`, `--patience N`, `--fresh-shark`.

## Training graphs: `python progress.py`
Opens a window and saves `models/logs/training_progress.png`. Use `--live` to refresh every 30 s while training runs.
- **Prey:**
  - benchmark survival against each test shark, with stars where a new best was saved
  - survival during training
  - schooling % and wall deaths
- **Shark:**
  - catches per minute
  - reward per decision
- **Your games:** fish survival in each round you played.

**Where to see training:** the top of the play screen shows the brain's status and how many rounds it has learned from. After every round, the game-over screen shows the learning curve: average fish survival per round, plus a rolling average. If the line is going up, the fish are learning. It also shows how much of the time fish spent in a group, how many were eaten at walls, how many fast-starts they fired, and their top 3 behaviors.

**Upgrades keep training.** A brain saved by an older version of the game (fewer senses or behaviors) is widened automatically when it loads. New inputs start with zero weight, and new behaviors start at about 10% of choices. What the fish already learned is kept.

**Reset brain** on the landing page starts over from an untrained brain, after copying the old one to `models/backups/pod_brain_before_reset.npz`.

## Code layout
```
main.py                   start the game
train.py                  automatic self-play training
progress.py               training graphs
predator_pod/
  config.py               sizes, speeds, colors, FPS
  game.py                 main loop and scenes
  selfplay.py             one headless training round (runs in worker processes)
  scenes/                 landing, play / watch, game over (with learning chart)
  entities/               Fish, Shark, drawing
  world/arena.py          border
  world/physics.py        no-overlap rule, eating, spawning
  world/sensors.py        the fish's 31 inputs
  world/shark_sensors.py  the AI shark's 29 inputs
  world/scripted_sharks.py human-style hunters (training opponents and benchmark)
  world/simulation.py     one round: decisions, rewards, collisions
  brains/ppo.py           policy network, PPO update, Adam
  brains/learner.py       fish brain: behaviors, rewards, human-play learning, save/load
  brains/shark_brain.py   AI shark brain, rewards, controller
models/
  pod_brain.npz           the fish brain the game uses (best so far)
  shark_brain.npz         the AI shark (used by Watch AI)
  checkpoints/ logs/ backups/   training state, CSV logs and graphs, backups (not in git)
```
