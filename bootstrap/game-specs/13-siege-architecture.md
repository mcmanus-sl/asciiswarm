# Game Spec 13: Siege Architecture (Chokepoints)

## Overview
A goblin invasion at turn 100. The player's attack power is too low to fight them directly. The player must mine corridors in rock and build traps to funnel goblins into destruction. Tests strategic base-building over brute force.

## Grid
- Dimensions: 14×14

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=14, grid_h=14,
    max_entities=128,      # rock fills most of grid + goblins + traps
    max_stack=3,           # trap under goblin
    num_entity_types=6,    # 0=unused, 1=player, 2=rock, 3=gears, 4=trap, 5=goblin
    num_tags=7,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=mineable, 6=trap
    num_props=3,           # 0=mechanisms (player), 1=health (goblin), 2=unused
    num_actions=6,
    max_turns=300,
    step_penalty=-0.005,
    game_state_size=4,     # 0=goblins_alive, 1=traps_built, 2=trap_kills, 3=turn_goblins_spawn
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | rock | `#` |
| 3 | gears | `*` |
| 4 | trap | `^` |
| 5 | goblin | `g` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | mineable |
| 6 | trap |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | mechanisms | player: collected gear mechanisms (0–5) |
| 1 | health | goblin: HP (default 3) |
| 2 | (unused) | — |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| mechanisms (prop 0) | 5 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center (7, 7) |
| rock (2) | solid (1), mineable (5) | Fills x=0 to x=10. Open plains x>10. |
| gears (3) | pickup (3) | 3 placed inside rock mass |
| trap (4) | trap (6) | Built dynamically via interact |
| goblin (5) | hazard (2) | Spawns at turn 100: 3 goblins at (13,1), (13,7), (13,12) |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 5 (goblin) | Move one step toward player. Use greedy Manhattan (not A* — too complex for JIT). Avoid solid blocks. If direct path blocked, try alternate axis. |

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move into mineable): Cancel move, destroy rock. If rock type, no reward. If gears (type 3): increment mechanisms by 1, add 0.1 to `reward_acc`.
- Actions 0–3 (move into empty): Normal move.
- Action 4 (interact — build trap): If mechanisms ≥ 1: decrement mechanisms, create trap at player's current cell. Increment `game_state[1]`.
- Action 5: No-op.

### Phase 2: Run Behaviors
- **Goblin spawn check**: If turn_number == 100 (stored in `game_state[3]`), spawn 3 goblins at designated positions.
- **Goblin movement**: Each alive goblin moves one step toward player via greedy Manhattan.
- **Goblin-trap collision**: After each goblin moves, check if goblin cell has a trap. If yes: destroy goblin, destroy trap (single-use). Add 1.0 to `reward_acc`. Increment `game_state[2]`.
- **Goblin-player collision**: If goblin shares cell with player: `status = -1`.

### Phase 3: Turn End
- If all 3 goblins are dead (no alive entities of type 5): `status = 1`.

## Win Condition
All goblins killed (by traps).

## Lose Condition
Any goblin reaches the player.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | goblins_alive |
| 1 | traps_built |
| 2 | trap_kills |
| 3 | turn_goblins_spawn (constant: 100) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Survival past turn 150 | >50% | Agent hides in rock |
| Trap kills | >2 per winning episode | Agent uses traps instead of fleeing |
| Chokepoint behavior | Observed | Agent mines a narrow corridor and places traps in it, forcing goblins to queue through a kill zone |

## Invariant Tests

1. 3 gears inside rock at game start.
2. No goblins at game start (they spawn at turn 100).
3. Rock fills x=0 to x=10.
4. Open plains at x>10.
5. Player starts at center.

## Notes
- The emergent insight: placing 3 traps in the open is inefficient. Mining a 1-tile-wide tunnel and placing a single trap at the choke forces all 3 goblins through it. The agent discovers architecture.
- Goblin pathfinding uses greedy Manhattan, not A*. BFS needs dynamic queues incompatible with JIT. Greedy is sufficient — goblins will navigate around obstacles but may not find optimal paths.
- The 100-turn grace period gives the agent time to prepare. The agent must learn that turns 1–100 are for building, not wandering.
