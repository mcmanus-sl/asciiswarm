# Game Spec 07: Hunger Clock

## Overview
The player must reach an exit before starving. Food decreases by 1 every turn. Food pickups restore food. Player must balance exploring toward exit with detours to eat. Tests ticking resource depletion.

## Grid
- Dimensions: 14×14

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=14, grid_h=14,
    max_entities=48,       # player + exit + 10-15 food + 10-20 walls
    max_stack=2,
    num_entity_types=5,    # 0=unused, 1=player, 2=exit, 3=food, 4=wall
    num_tags=6,            # standard 6
    num_props=2,           # 0=food (player), 1=unused
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=2,     # 0=food_eaten_count, 1=unused
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | food | `f` |
| 4 | wall | `#` |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | food | player: starts at 20, max 20 |
| 1 | (unused) | — |

## Player Properties (for scalar observation)

| Key | Max | Description |
|-----|-----|-------------|
| food (prop 0) | 20 | Current food level |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Bottom-left corner (0, 13) |
| exit (2) | exit (4) | Top-right corner (13, 0) |
| food (3) | pickup (3) | 10–15 scattered randomly |
| wall (4) | solid (1) | 3–5 clusters of 2–4 walls each |

## Behaviors

None.

## Turn Phases

### Phase 1: Process Input
- Move player. Check target for solid → cancel. Check target for pickup (food) → move succeeds, increase food by 5 (cap at 20), destroy food entity, add 0.05 to `reward_acc`.
- Check target for exit → move succeeds, `status = 1`.

### Phase 2: Run Behaviors
None.

### Phase 3: Turn End
- Decrease `properties[player_idx, 0]` (food) by 1.
- If food ≤ 0, set `status = -1` (lost/starved).

## Wall Placement (in `reset`)

3–5 wall clusters placed randomly. Each cluster is 2–4 adjacent tiles. Walls cannot be on player start, exit, or food locations. After placement, verify exit is reachable from player (BFS, Python-side). Regenerate if unreachable (max 100 attempts).

## game_state Slots

| Index | Name |
|-------|------|
| 0 | food_eaten_count |
| 1 | (unused) |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–3% |
| PPO win rate at 100k steps | >15% |
| PPO learning delta (100k - 10k) | >8% |

## Invariant Tests

1. Player starts with food == 20.
2. Between 10 and 15 food entities at game start.
3. Player starts at (0, 13).
4. Exit at (13, 0).
5. Exit reachable from player (BFS over non-solid).
6. No food on player cell, exit cell, or wall cell.

## Notes
- Manhattan distance (0,13)→(13,0) = 26, exceeding starting food of 20. Player MUST eat at least once. This is the core design tension.
- Hunger clock creates time pressure distinguishing this from simple pathfinding.
- Food decrease happens in Turn End, after player movement, so eating food on the same turn food would tick still works.
