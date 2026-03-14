# Game Spec 08: Block Push

## Overview
Simplified Sokoban: push 2 blocks onto 2 target tiles. Walking into a block pushes it one tile — unless the block is blocked, in which case the player's move is also cancelled. Tests push mechanics and spatial puzzle solving.

## Grid
- Dimensions: 8×8

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=8, grid_h=8,
    max_entities=32,       # border walls (28) + player + 2 blocks + 2 targets
    max_stack=3,           # block can be on top of target
    num_entity_types=5,    # 0=unused, 1=player, 2=block, 3=target, 4=wall
    num_tags=8,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=pushable, 7=target
    num_props=1,
    num_actions=6,
    max_turns=300,
    step_penalty=-0.005,
    game_state_size=2,     # 0=blocks_on_target, 1=total_pushes
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | block | `B` |
| 3 | target | `X` |
| 4 | wall | `#` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | npc |
| 6 | pushable |
| 7 | target |

## Property Index Mapping

No meaningful properties. Array size 1.

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Random empty cell, bottom half (y >= 4) |
| block (2) | solid (1), pushable (6) | 2 blocks, center area (2 ≤ x ≤ 5, 2 ≤ y ≤ 5) |
| target (3) | target (7) | 2 targets, top half (y ≤ 3), not on walls |
| wall (4) | solid (1) | Border walls on all 4 edges |

## Behaviors

None.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Compute target cell and direction vector (dx, dy).
  - If target has wall (solid, not pushable): cancel move.
  - If target has block (solid AND pushable): attempt push.
    - Compute block's push destination: `(block_x + dx, block_y + dy)`.
    - Check if push destination has any solid entity or is out of bounds → cancel both block push and player move.
    - If push destination clear: move block, then move player into block's old cell. Increment `game_state[1]` (total_pushes).
    - After push: count how many targets have a block on the same cell. If count == 2, set `status = 1`.
  - Otherwise: move player normally.
- Actions 4–5: No-op.

### Phase 2: Run Behaviors
None.

### Phase 3: Turn End
Nothing.

## Push Detection in JAX

The push mechanic requires checking:
1. Is target cell occupied by a pushable entity? → `find_by_tag(state, 6)` (pushable) mask, check if any alive pushable is at target.
2. Is push destination blocked? → Check for solid entities at `(block_x + dx, block_y + dy)`.
3. All via `jnp.where`, no Python branching.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | blocks_on_target (win at 2) |
| 1 | total_pushes |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0.5–5% |
| PPO win rate at 100k steps | >10% |
| PPO learning delta (100k - 10k) | >5% |

## Invariant Tests

1. Exactly 2 blocks at game start.
2. Exactly 2 targets at game start.
3. Player starts in bottom half (y >= 4).
4. Blocks in center area (2 ≤ x ≤ 5, 2 ≤ y ≤ 5).
5. Targets in top half (y ≤ 3).
6. No block starts on a target (not pre-solved).
7. No two blocks on same cell.
8. No two targets on same cell.
9. Puzzle is solvable (verified during generation).

## Notes
- Simplified from full Sokoban: only 2 blocks, 2 targets, open 8×8 interior.
- Push chain: player → block → wall. If wall blocks block, everything stays.
- Solvability BFS over `(player_pos, block1_pos, block2_pos)` state space. Python-side in `reset`.
- Distance-based reward shaping recommended: `0.02 * (prev_distance_sum - curr_distance_sum)`.
- `pushable` and `target` are custom tags (indices 6, 7) declared in this game's config.
