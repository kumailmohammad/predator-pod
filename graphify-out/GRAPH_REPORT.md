# Graph Report - predator-pod  (2026-11-25)

## Corpus Check
- Corpus is ~12,530 words - fits in a single context window. You may not need a graph.

## Summary
- 266 nodes · 562 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Training Scripts & Imports
- Fish/Shark Drawing
- Learner (RL Training)
- Config & Game Loop
- README: Brains & Controls
- Game Class & Entry Point
- PPO Trajectories & Returns
- Simulation Rounds & Physics
- Fish Sensors

## God Nodes (most connected - your core abstractions)
1. `main()` - 19 edges
2. `Learner` - 18 edges
3. `PPOPolicy` - 18 edges
4. `PlayScene` - 17 edges
5. `World` - 17 edges
6. `Game` - 14 edges
7. `LandingScene` - 13 edges
8. `Button` - 13 edges
9. `SharkBrain` - 12 edges
10. `AISharkController` - 12 edges

## Surprising Connections (you probably didn't know these)
- `benchmark()` --indirect_call--> `run_round()`  [INFERRED]
  train.py → predator_pod/selfplay.py
- `main()` --indirect_call--> `run_round()`  [INFERRED]
  train.py → predator_pod/selfplay.py
- `load_pool()` --calls--> `SharkBrain`  [EXTRACTED]
  train.py → predator_pod/brains/shark_brain.py
- `load_shark()` --calls--> `SharkBrain`  [EXTRACTED]
  train.py → predator_pod/brains/shark_brain.py
- `main()` --indirect_call--> `worker_init()`  [INFERRED]
  train.py → predator_pod/selfplay.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Self-play training loop** — readme_self_play_training, readme_fish_brain, readme_ai_shark, readme_ppo [EXTRACTED 0.95]

## Communities (11 total, 2 thin omitted)

### Community 0 - "Training Scripts & Imports"
Cohesion: 0.09
Nodes (19): reward(), threat_level(), merge_batches(), worker_init(), ScriptedShark, draw(), human_history(), main() (+11 more)

### Community 1 - "Fish/Shark Drawing"
Cohesion: 0.08
Nodes (9): draw_fish(), draw_shark(), _ellipse_points(), _transform(), Fish, Shark, Arena, resolve_overlaps() (+1 more)

### Community 2 - "Learner (RL Training)"
Cohesion: 0.08
Nodes (4): Learner, PPOPolicy, _sigmoid(), SharkBrain

### Community 3 - "Config & Game Loop"
Cohesion: 0.14
Nodes (7): training_running(), draw_chart(), GameOverScene, format_time(), Button, draw_text(), font()

### Community 4 - "README: Brains & Controls"
Cohesion: 0.09
Nodes (26): AI shark (29-32-5 network), Benchmark evaluation, Brain widening on upgrade, Game controls, Escape behaviors (14 + fast-start), Fast-start (C-start), Fish brain (31-32-14+1 network), Fish inputs (31 senses) (+18 more)

### Community 5 - "Game Class & Entry Point"
Cohesion: 0.14
Nodes (3): Game, LandingScene, PlayScene

### Community 6 - "PPO Trajectories & Returns"
Cohesion: 0.12
Nodes (5): discounted_returns(), Trajectories, AISharkController, _ray_to_wall(), shark_sense()

### Community 7 - "Simulation Rounds & Physics"
Cohesion: 0.15
Nodes (3): run_round(), eaten_fish(), World

### Community 8 - "Fish Sensors"
Cohesion: 0.47
Nodes (3): sense(), Senses, _unit()

## Knowledge Gaps
- **17 isolated node(s):** `Fish inputs (31 senses)`, `Fast-start (C-start)`, `Selfish herd behavior`, `Fish reward design`, `AI shark inputs (29)` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 92 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Learner` connect `Learner (RL Training)` to `Training Scripts & Imports`, `Config & Game Loop`, `Game Class & Entry Point`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `main()` (e.g. with `run_round()` and `worker_init()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Fish inputs (31 senses)`, `Fast-start (C-start)`, `Selfish herd behavior` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Training Scripts & Imports` be split into smaller, more focused modules?**
  _Cohesion score 0.08953900709219859 - nodes in this community are weakly interconnected._
- **Why does `World` connect `Simulation Rounds & Physics` to `Training Scripts & Imports`, `Fish/Shark Drawing`, `Config & Game Loop`, `Game Class & Entry Point`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PlayScene` (e.g. with `Game` and `AISharkController`) actually correct?**
  _`PlayScene` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `Fish/Shark Drawing` be split into smaller, more focused modules?**
  _Cohesion score 0.07822410147991543 - nodes in this community are weakly interconnected._