# INF FORTRESS + FISHING FOR ISLANDS — Full Project Sitrep

## Project Summary

**INF FORTRESS** is a JAX-native turn-based ASCII game engine with RL-TDD (Reinforcement Learning Test-Driven Design). A PPO agent on a 96GB Blackwell GPU playtests game mechanics at ~1M frames/sec, acting as a behavioral compiler. 20 games across 2 seasons, each composing more ECS systems:

- **Season 1 (Games 01-14): INF FORTRESS** — violent mechanics (combat, magma, siege, stress)
- **Season 2 (Games 15-20): FISHING FOR ISLANDS** — cozy ecological survival (fire, farming, husbandry, emergence)

**Key insight:** Season 1's "violent" primitives become Season 2's "cozy" mechanics. Magma CA = fertility spread. Stress system = stamina/thirst meters. Siege timers = growth cycles.

**Hardware:** RTX PRO 6000 Blackwell (96GB VRAM, GPU:0) + RTX 5070 Ti (16GB, GPU:1). Local.

**Branch:** `inf-fortress` (ahead of `main`). All JAX engine work lives here.

---

## What Exists Today

### Engine Architecture (`jaxswarm/`)

```
jaxswarm/
  core/          # Foundation (all JIT-compatible, no Python loops)
    state.py     — EnvState (chex.dataclass pytree), EnvConfig, init_state
    grid_ops.py  — Entity CRUD, spatial index, grid queries
    obs.py       — Channel-first grid tensor + scalar vector observation builder
    movement.py  — Greedy Manhattan one-step pathfinding
    wave.py      — Cellular automata distance field via min-pooling

  systems/       # Composable ECS systems (each is EnvState -> EnvState)
    movement.py  — Player movement with solid-entity blocking
    behavior.py  — NPC dispatch (wanderer, chaser, patroller) via lax.switch
    collision.py — Exit/hazard detection
    interact.py  — Adjacent entity interaction (key pickup, door unlock)
    combat.py    — Bump-attack damage exchange
    hunger.py    — Food depletion per turn, starvation
    push.py      — Sokoban-style block pushing
    farming.py   — Growth ticks (seed -> sprout -> mature crop)
    crafting.py  — Workbench resource combination
    slide.py     — Ice sliding physics
    magma.py     — NEW: Cellular automata spread via grid convolution
    time_cycle.py — NEW: Day/night with multi-meter depletion
    fire.py      — NEW: Fuel decay + warmth radius via distance field
    emergence.py — NEW: Submerged masking (flip water to terrain)
    processing.py — NEW: NPC busy timer → output product
    husbandry.py — NEW: Feed animal → produce outputs
    fertility.py — NEW: Manure CA propagation → faster crop growth
    island_effects.py — NEW: Global multipliers from unlocked chunks

  games/         # 20 games (10 complete + 10 new)
  network.py     — ActorCritic (games 01-10) + ScaledActorCritic (games 11+)
  train.py       — PureJaxRL PPO with chunked progress output + auto_train_config
  curriculum.py  — NEW: Multi-phase weight transfer across games
```

### Games Status

| # | Game | Grid | Entities | Season | Status |
|---|------|------|----------|--------|--------|
| 01 | Empty Exit | 8x8 | 8 | S1 | ✓ Trained |
| 02 | Dodge | 10x10 | 8 | S1 | ✓ Trained |
| 03 | Lock & Key | 12x12 | 64 | S1 | ✓ Trained |
| 04 | Dungeon Crawl | 16x16 | 128 | S1 | ✓ Trained |
| 05 | Pac-Man Collect | 12x12 | 160 | S1 | ✓ Trained |
| 06 | Ice Sliding | 10x10 | 16 | S1 | ✓ Trained |
| 07 | Hunger Clock | 14x14 | 27 | S1 | ✓ Trained |
| 08 | Block Push | 8x8 | 10 | S1 | ✓ Trained |
| 09 | Inventory Crafting | 16x16 | 28 | S1 | ✓ Trained |
| 10 | Farming Growth | 14x14 | 24 | S1 | ✓ Trained |
| 11 | Digging Deep | 16x16 | 256 | S1 | ✓ Built (needs GPU eval) |
| 12 | Tantrum Spiral | 12x12 | 16 | S1 | ✓ Built (needs GPU eval) |
| 13 | Siege Architecture | 14x14 | 128 | S1 | ✓ Built (needs GPU eval) |
| 14 | INF FORTRESS | 32x32 | 512 | S1 | ✓ Built (needs curriculum) |
| 15 | The Campfire | 16x16 | 32 | S2 | ✓ Built (needs GPU eval) |
| 16 | The Golden Field | 24x24 | 128 | S2 | ✓ Built (needs GPU eval) |
| 17 | The Shepherd | 24x24 | 192 | S2 | ✓ Built (needs GPU eval) |
| 18 | The Oven | 32x32 | 256 | S2 | In progress |
| 19 | The Deep | 32x32 | 384 | S2 | In progress |
| 20 | Fishing for Islands | 32x32 | 512 | S2 | In progress |

### Training Infrastructure: COMPLETE

| Component | File | Status |
|-----------|------|--------|
| ScaledActorCritic | `network.py` | ✓ Auto-selects by game size |
| Chunked training loop | `train.py` | ✓ Progress output, block_until_ready |
| Auto train config | `train.py` | ✓ VRAM-aware num_envs/updates |
| Multi-GPU training | `train_parallel.py` | ✓ Built |
| Curriculum learning | `curriculum.py` | ✓ Built + weight transfer tested |
| 5-layer evaluator | `evaluate_game.py` | ✓ Complete |

### System Reuse Map

| Existing System | S1 Games | S2 Games |
|-----------------|----------|----------|
| `farming.py` | 10, 14 | 16-20 |
| `hunger.py` | 7, 14 | 15-20 |
| `wave.py` | 9, 11, 14 | 15-20 |
| `behavior.py` | 4-5, 13-14 | 16-20 |
| `crafting.py` | 9, 14 | 16-20 |
| `push.py` | 8, 12 | — |
| `combat.py` | 4, 13-14 | — |
| `magma.py` | 11, 14 | 17, 19 |
| `emergence.py` | — | 16-20 |
| `time_cycle.py` | — | 15-20 |
| `fire.py` | — | 15-20 |
| `processing.py` | — | 18-20 |
| `husbandry.py` | — | 17-20 |
| `island_effects.py` | — | 19-20 |
| `fertility.py` | — | 17-20 |

---

## Next Steps

### Immediate: GPU Evaluation
Run `evaluate_game.py` on Games 11-17 to get trained weights. Priority order:
1. Game 11 (magma CA — foundation for Game 14)
2. Game 12 (stress — small, fast)
3. Game 13 (siege — depends on 11's mining pattern)
4. Game 15 (campfire — Season 2 foundation)
5. Game 16 (emergence — the JAX-hard problem)
6. Game 17 (husbandry + fertility)

### Phase 2: Curriculum Training
1. Season 1 curriculum: 01-03 → 07+10 → 11+13 → 14
2. Season 2 curriculum: 15 → 16 → 17 → 18 → 19 → 20

### Phase 3: Capstone Verification
- Game 14: curriculum PPO >40%, weaponized fluids observed
- Game 20: curriculum PPO >30%, all 4 chunks unlocked in winning episodes

---

## Key Design Decisions

- **No event system** — explicit phase-based turn loop
- **Fixed-shape state** — all arrays static for JIT. `alive` mask selects active entities.
- **struct-of-arrays** — columnar entity storage (x[], y[], tags[][], props[][])
- **Submerged masking** — Season 2's emergence system: full grid pre-allocated, water tiles flipped to terrain
- **No gymnasium/SB3/torch** — pure JAX + equinox + optax + chex
- **PureJaxRL** — entire training loop compiled to single XLA program

---

## File Map

```
bootstrap/
  DEVELOPMENT_PLAN.md    — Master plan
  TRAINING_INFRA_PLAN.md — This file
  game-specs/            — 20 game specifications (01-20)

jaxswarm/
  core/                  — Engine foundation (5 modules)
  systems/               — 18 composable ECS systems
  games/                 — 20 game implementations
  network.py             — ActorCritic + ScaledActorCritic
  train.py               — PPO loop + auto_train_config
  curriculum.py          — Multi-phase curriculum training

evaluate_game.py         — 5-layer RL-TDD evaluator
train_parallel.py        — Multi-GPU training
play_games.py            — Human play (curses)
play_pygame.py           — Human play (pygame)
replay_games.py          — Agent replay to MP4

tests/core/              — 7 test files
weights/                 — 10 trained weight files (13MB total, games 01-10)
```
