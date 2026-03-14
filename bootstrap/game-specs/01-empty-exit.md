# Game Spec 01: Empty Exit

## Overview
The simplest possible game. A player on a grid must walk to an exit tile. No enemies, no obstacles. This exists purely to validate the engine pipeline, JIT compilation, and vmap correctness.

## Grid
- Dimensions: 8×8

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=8, grid_h=8,
    max_entities=8,
    max_stack=2,
    num_entity_types=3,    # 0=unused, 1=player, 2=exit
    num_tags=6,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc
    num_props=1,           # unused, but minimum 1 for array shape
    num_actions=6,         # 0=move_n, 1=move_s, 2=move_e, 3=move_w, 4=interact, 5=wait
    max_turns=200,
    step_penalty=-0.01,
    game_state_size=1,     # unused
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused slot) | — |
| 1 | player | `@` |
| 2 | exit | `>` |

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

No properties used. Array size 1 (minimum), all zeros.

## Entities

| Type | Tags | Glyph | Spawning |
|------|------|-------|----------|
| player (1) | player (0) | `@` | Random empty cell |
| exit (2) | exit (4) | `>` | Random empty cell (not player's cell) |

## Behaviors

None. No entity has autonomous behavior.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move_n/s/e/w): Compute target position from direction. Call `move_entity(state, player_idx, target_x, target_y)`.
- Action 4 (interact): No-op.
- Action 5 (wait): No-op.

### Phase 2: Run Behaviors
No behaviors to run.

### Phase 3: Turn End
- Check if player shares a cell with the exit entity. If yes, set `status = 1` (won).

## Win Condition
Player walks onto the exit tile.

## Lose Condition
None. This game cannot be lost. Truncates at `max_turns` (200).

## game_state Slots
None used.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 5–40% |
| PPO win rate at 100k steps | >90% |
| PPO learning delta (100k - 10k) | >30% |

## Notes
- Spawning uses `jax.random` with key splitting for deterministic placement.
- The simplest possible collision check: compare `(player_x, player_y)` to `(exit_x, exit_y)` after movement.
- `max_entities=8` is generous — only 2 entities used. The extra slots are headroom.
