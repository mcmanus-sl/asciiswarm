# Game Spec 05: Pac-Man Collect

## Overview
A collection game where the player must pick up all dots while avoiding two ghosts. One ghost chases (Manhattan distance), the other patrols a fixed path. Win by collecting every dot. Lose by colliding with a ghost.

## Grid
- Dimensions: 12×12

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=12, grid_h=12,
    max_entities=128,      # dots fill most cells (~80), walls, 2 ghosts, player
    max_stack=2,
    num_entity_types=6,    # 0=unused, 1=player, 2=dot, 3=chaser, 4=patroller, 5=wall
    num_tags=6,            # standard 6
    num_props=3,           # 0=patrol_direction, 1=patrol_steps, 2=unused
    num_actions=6,         # standard 6
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=2,     # 0=dots_remaining, 1=dots_collected
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | dot | `.` |
| 3 | chaser | `C` |
| 4 | patroller | `P` |
| 5 | wall | `#` |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | patrol_direction | patroller: 0=east, 1=south, 2=west, 3=north |
| 1 | patrol_steps | patroller: steps taken in current direction |
| 2 | (unused) | — |

## Grid Layout

Border walls on all edges. Interior walls form a cross pattern:
- Horizontal wall: y=5, x=3 to x=8 (gap at x=5 and x=6)
- Vertical wall: x=5, y=3 to y=8 (gap at y=5 and y=6)

Dots fill all remaining empty cells (not occupied by walls, player, or ghosts).

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center of grid (6, 6) |
| dot (2) | pickup (3) | Every empty cell not occupied by player, ghost, or wall |
| chaser (3) | hazard (2) | Top-left corner (1, 1) |
| patroller (4) | hazard (2) | Top-right corner (10, 1) |
| wall (5) | solid (1) | Border + cross pattern |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 3 (chaser) | Move one step toward player (Manhattan). Prefer axis with greater |dx| vs |dy|. Break ties: prefer horizontal. If blocked by solid, try other axis. If both blocked, stay. |
| 4 (patroller) | Rectangular patrol: east along y=1 → south along x=10 → west along y=10 → north along x=1. 9 steps per leg. If blocked, advance to next direction. |

## Turn Phases

### Phase 1: Process Input
- Move player. Check target for solid → cancel. Check target for pickup (dot) → move succeeds, destroy dot, add 0.05 to `reward_acc`, decrement `game_state[0]` (dots_remaining), increment `game_state[1]` (dots_collected). If `game_state[0] == 0`, set `status = 1`.
- Check target for hazard → `status = -1`.

### Phase 2: Run Behaviors
- Run chaser and patroller behaviors via `jax.lax.switch`.
- After each ghost moves, check if ghost shares cell with player → `status = -1`.

### Phase 3: Turn End
Nothing extra.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | dots_remaining |
| 1 | dots_collected |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–2% |
| PPO win rate at 100k steps | >10% |
| PPO learning delta (100k - 10k) | >5% |

## Invariant Tests

1. Exactly one chaser and one patroller at game start.
2. At least 20 dots at game start.
3. Player starts at center (6, 6).
4. No dot on a ghost or wall cell.
5. Player not on a ghost cell.
6. All non-wall cells reachable from player (BFS over non-solid).

## Notes
- Collection win condition (ALL dots) is harder than single exit.
- Per-dot +0.05 reward provides frequent positive signal.
- Dot count tracked in `game_state[0]` to avoid scanning all entities each step.
- The cross pattern creates corridors that make ghost avoidance strategic.
