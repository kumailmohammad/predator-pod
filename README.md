# Predator-Pod

A 2D learning project: you play a shark hunting a pod of fish. The long-term goal is to train the fish with a neural network (neuroevolution) so they learn the best way to survive: staying together, keeping away from the shark, and avoiding getting cornered at the walls.

## Current status: milestone 1
- A landing page where you pick the pod size (10–30 fish).
- A bordered arena inside a 960×640 window, running at 60 FPS.
- Fish that wander loosely as a pod at a constant speed, with solid bodies that never overlap.
- A shark (about 3× the fish size) that never stops swimming. It turns toward your mouse or touchpad cursor, keeps going in its last direction once it reaches the cursor, and swims along the border when it hits a wall. It eats any fish it touches.
- A game-over screen showing how long it took to eat the whole pod.

The fish ignore the shark for now. Brains get added in milestone 2.

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
| Mouse / touchpad | Move the shark |
| Left / Right arrows (menu) | Change the fish count |
| Enter | Start |
| Esc | Back to menu / quit |

## How the code is organized
```
predator_pod/
  config.py          every tunable value (sizes, speeds, colors, FPS)
  game.py            main loop and scene switching
  scenes/            landing, play, game over
  entities/          Fish, Shark and how they're drawn
  world/arena.py     border and wall distances
  world/physics.py   no-overlap rule, eating, spawning
  world/sensors.py   what each fish can sense (input to its brain)
  brains/            Brain interface + the current random-wander brain
```

### What a fish senses
`world/sensors.py` gives every fish:
- its distance to each of the 4 walls
- the direction and distance to the shark
- the direction and distance to the center of the pod
- its own velocity

`Observation.to_array()` packs these into 11 normalized numbers in the fish's own frame of reference. That array is the input layer for the neural network in the next milestone. The network picks one of four actions (forward, backward, left, right). Speed never changes, only direction.

## Roadmap
1. ~~Arena, landing page, wandering pod, player shark~~
2. Neural-network fish brain + headless neuroevolution trainer
3. Watch mode: trained pod vs. your shark
