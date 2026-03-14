# Game Spec 04: Dungeon Crawl

## Overview
A multi-room dungeon with combat, health management, and three enemy types. The player must fight or avoid enemies, collect health potions, and reach the exit. First game intended for AGENT SWARM to build autonomously.

## Grid
- Dimensions: 16×16

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=128,      # walls + enemies + potions + player + exit
    max_stack=3,
    num_entity_types=7,    # 0=unused, 1=player, 2=exit, 3=wall, 4=wanderer, 5=chaser, 6=sentinel, 7=potion
    num_tags=6,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc
    num_props=3,           # 0=health, 1=attack, 2=direction (for wanderer)
    num_actions=6,         # standard 6
    max_turns=500,
    step_penalty=-0.005,
    game_state_size=2,     # 0=enemies_killed, 1=potions_used
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | wall | `#` |
| 4 | wanderer | `w` |
| 5 | chaser | `c` |
| 6 | sentinel | `s` |
| 7 | potion | `!` |

## Tag Index Mapping

Standard 6-tag layout (same as specs 01–03).

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | health | player (init 10, max 10), wanderer (1), chaser (2), sentinel (3) |
| 1 | attack | player (init 2), wanderer (1), chaser (2), sentinel (1) |
| 2 | direction | wanderer: random walk direction |

## Player Properties (for scalar observation)

| Key | Max | Description |
|-----|-----|-------------|
| health (prop 0) | 10 | Current HP |
| attack (prop 1) | 5 | Damage dealt |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center of first room |
| wall (3) | solid (1) | Room boundaries and corridors |
| wanderer (4) | hazard (2) | 1–2 per room, random empty cell |
| chaser (5) | hazard (2) | 1 per room (rooms 3+) |
| sentinel (6) | hazard (2) | 1 per room (rooms 4+) |
| potion (7) | pickup (3) | 1–2 per room |
| exit (2) | exit (4) | Center of last room |

## Room Generation

Generate 3–5 rectangular rooms (4×4 to 6×6), non-overlapping. Connect with 1-tile corridors. All rooms must be connected (verify via Python-side BFS before JIT). Use `jax.random` for placement.

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (wanderer) | Random cardinal direction (via rng_key). Attempt move. If blocked, stay. |
| 5 (chaser) | If player within Manhattan distance 5: move one step toward player (prefer axis with greater distance, break ties via rng_key). Otherwise: random walk like wanderer. |
| 6 (sentinel) | Does not move. Stationary. |

## Turn Phases

### Phase 1: Process Input
- Move player in chosen direction. Before completing move, check target cell:
  - If target has solid (1) entity (wall): cancel move.
  - If target has hazard (2) entity: cancel move, execute combat (see below).
  - If target has pickup (3) entity: move succeeds, pick up potion.
  - If target has exit (4) entity: move succeeds, `status = 1`.

**Combat**: Reduce player health by enemy attack. Reduce enemy health by player attack. If enemy health ≤ 0, destroy enemy. If player health ≤ 0, `status = -1`.

**Potion pickup**: Increase player health by 3 (cap at 10). Destroy potion. Add 0.1 to `reward_acc`.

### Phase 2: Run Behaviors
- `jax.lax.fori_loop` over entity slots. For each alive entity, `jax.lax.switch(entity_type, [noop, noop, noop, noop, wanderer_fn, chaser_fn, sentinel_fn, noop])`.
- After each enemy moves, check if enemy shares cell with player → combat (enemy attacks player).

### Phase 3: Turn End
Nothing extra.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | enemies_killed |
| 1 | potions_used |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 1–10% |
| PPO learning delta (500k vs 50k) | >0 |

## Invariant Tests

1. All rooms connected (BFS from player reaches exit).
2. Every room has at least one potion.
3. Total enemies between 5 and 20.
4. Player starts with health > 0.
5. No enemy spawns on player's cell.

## Notes
- Combat is walk-into: player attempts to move into enemy cell, move is cancelled, damage exchanged.
- Enemy health defaults: wanderer=1, chaser=2, sentinel=3.
- Enemy attack defaults: wanderer=1, chaser=2, sentinel=1.
- Room generation happens in `reset()` — this can be Python-side before the state is JIT'd, or use `jax.lax.while_loop` for procedural generation within JAX.
