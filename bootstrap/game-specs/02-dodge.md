# Game Spec 02: Dodge

## Overview
A player must reach an exit while avoiding a patrolling enemy. The enemy bounces horizontally across the grid. Contact with the enemy kills the player. First game with autonomous entity behavior and hazard avoidance.

## Grid
- Dimensions: 10×10

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=10, grid_h=10,
    max_entities=8,
    max_stack=2,
    num_entity_types=4,    # 0=unused, 1=player, 2=exit, 3=wanderer
    num_tags=6,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc
    num_props=2,           # 0=direction (for wanderer), 1=unused
    num_actions=6,         # 0=move_n, 1=move_s, 2=move_e, 3=move_w, 4=interact, 5=wait
    max_turns=200,
    step_penalty=-0.01,
    game_state_size=1,
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused slot) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | wanderer | `w` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | npc |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | direction | wanderer: 1=east, -1=west |
| 1 | (unused) | — |

## Entities

| Type | Tags | Glyph | Spawning |
|------|------|-------|----------|
| player (1) | player (0) | `@` | Bottom-left quadrant (x < 5, y >= 5), random empty cell |
| exit (2) | exit (4) | `>` | Top-right quadrant (x >= 5, y < 5), random empty cell |
| wanderer (3) | hazard (2) | `w` | Center row (y=4 or y=5), random x. Starts with direction=1 (east). |

## Behaviors

### wanderer (type 3)
Each turn, move one step in the entity's current direction (property 0: 1=east, -1=west). After moving, check if the next step in the same direction would be out of bounds (x+direction < 0 or x+direction >= grid_w). If so, reverse direction (multiply by -1).

Use `move_entity(state, slot, x + direction, y)`.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move_n/s/e/w): Move player.
- Actions 4–5: No-op.

### Phase 2: Run Behaviors
- Iterate all alive entities. For type 3 (wanderer): execute wanderer behavior.
- Use `jax.lax.switch` on entity_type to dispatch to the correct behavior function.

### Phase 3: Turn End
- Check if player shares a cell with any entity tagged hazard (2). If yes, `status = -1` (lost).
- Check if player shares a cell with exit entity. If yes, `status = 1` (won).

## Win Condition
Player walks onto the exit tile.

## Lose Condition
Player collides with the wanderer (either player walks into wanderer, or wanderer walks into player).

## game_state Slots
None used.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 1–15% |
| PPO win rate at 100k steps | >50% |
| PPO learning delta (100k - 10k) | >15% |

## Invariant Tests

1. Exactly one wanderer exists at game start.
2. Wanderer starts in center rows (y=4 or y=5).
3. Player starts in bottom-left quadrant (x < 5, y >= 5).
4. Exit is in top-right quadrant (x >= 5, y < 5).
5. Player and wanderer do not start on the same cell.

## Notes
- The wanderer's bounce pattern is deterministic — no randomness in its movement, only in initial x position.
- Spawning quadrants ensure the player must cross the wanderer's path to reach the exit.
- Collision detection is done in Turn End by checking if any hazard-tagged entity shares the player's cell.
