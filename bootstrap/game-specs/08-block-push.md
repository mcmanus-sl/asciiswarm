# Game Spec 08: Block Push

## Overview
A simplified Sokoban: the player pushes 2 blocks onto 2 target tiles. Walking into a block pushes it one tile in the same direction — unless the block is itself blocked, in which case the player's move is also cancelled. This tests push mechanics via collision chains and spatial puzzle solving.

## Grid
- Dimensions: 8×8

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc', 'pushable', 'target'],
    'grid': (8, 8),
    'max_turns': 300,
    'step_penalty': -0.005,
    'player_properties': [],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Random empty cell in bottom half (y >= 4) |
| `block` | `solid`, `pushable` | `B` | 5 | 2 blocks, random empty cells in the center area (2 <= x <= 5, 2 <= y <= 5) |
| `target` | `target` | `X` | 3 | 2 targets, random empty cells in the top half (y <= 3), not on walls |
| `wall` | `solid` | `#` | 1 | Border walls on all 4 edges |

## Player Properties
None. No properties needed for this game.

## Layout

The grid has border walls on all edges (x=0, x=7, y=0, y=7). The interior (x=1–6, y=1–6) is open except for the blocks and targets. The small grid and open interior keep the puzzle tractable for PPO.

All placement uses `env.random()`. After placement, verify solvability: BFS over (player_pos, block1_pos, block2_pos) state space. If no solution exists within 50 moves, regenerate (max 100 attempts, then use a known-good hardcoded layout).

## Behaviors
None. No entity has autonomous behavior.

## Event Handlers

- **`input` (Player Movement)**: The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` does nothing. `interact` does nothing.

- **`collision` (player pushes block)**: If mover is `player` and any occupant is tagged `pushable`:
  - Compute the push direction from the player's movement (dx = occupant.x - mover_source.x, dy = occupant.y - mover_source.y).
  - Attempt to move the block: `env.move_entity(block.id, block.x + dx, block.y + dy)`.
  - If the block move succeeds: allow the player's move (do NOT cancel). Then check the win condition.
  - If the block move fails (blocked by `solid` or out of bounds): cancel the player's move (player stays in place).

- **`collision` (win condition check)**: After a successful block push, check if ALL entities tagged `target` have an entity tagged `pushable` on the same cell. If so, call `env.end_game('won')`.

- **`before_move` (solid blocks movement)**: If target cell contains any entity tagged `solid`, cancel the move. Note: blocks are tagged `solid`, so this fires when the player walks into a block — the `collision` handler then overrides this by attempting the push.

### Collision/Before-Move Interaction

The push mechanic requires careful handler ordering:
1. `before_move` fires first and would cancel movement into a `solid` block.
2. The `collision` handler must fire INSTEAD of (or override) the `before_move` cancellation for `pushable` entities specifically.
3. **Implementation approach**: The `before_move` handler should check: if the target cell contains a `pushable` entity, do NOT cancel (let the collision handler handle it). Only cancel for non-pushable `solid` entities (walls, other blocked blocks after a failed push).

Alternatively, skip `before_move` for the player entirely and handle all player-vs-solid interactions in the `collision` handler:
- If occupant is `solid` but NOT `pushable` → cancel move.
- If occupant is `pushable` → attempt push (as described above).

Either approach works. The game implementer should choose whichever is cleaner.

## Interact Mapping
`interact` does nothing in this game.

## Win Condition
Both blocks are on target tiles simultaneously. Checked after every successful block push.

## Lose Condition
None. This game cannot be lost. The engine truncates at `max_turns` (300). (Deadlocked states — where blocks are pushed into unmovable positions — effectively make the game unwinnable, but the engine handles this via truncation, not an explicit loss.)

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0.5–5% |
| PPO win rate at 100k steps | >10% |
| PPO learning delta (100k - 10k) | >5% |

## Invariant Tests (game-specific)

1. Exactly 2 blocks exist at game start.
2. Exactly 2 targets exist at game start.
3. Player starts in the bottom half (y >= 4).
4. Blocks start in the center area (2 <= x <= 5, 2 <= y <= 5).
5. Targets are in the top half (y <= 3).
6. No block starts on a target (puzzle is not pre-solved).
7. No two blocks start on the same cell.
8. No two targets are on the same cell.
9. The puzzle is solvable (verified during generation).

## Notes
- This is deliberately simplified from full Sokoban: only 2 blocks, only 2 targets, small 8×8 grid with open interior. Full Sokoban with many blocks is notoriously hard for PPO.
- The push mechanic creates a collision chain: player→block→wall. If the wall blocks the block, both the block and player stay in place.
- Reward shaping: consider adding distance-based intermediate rewards. For each block, compute Manhattan distance to the nearest unoccupied target. Sum of distances decreasing gives a positive reward signal. Example: `reward = 0.02 * (prev_total_distance - curr_total_distance)`. This is optional but strongly recommended for PPO convergence.
- Deadlocked states (block in corner, no way to push it out) are common in Sokoban. The small grid and open layout minimize these. The solvability BFS during generation helps, but pushes during play can still create deadlocks.
- The `target` and `pushable` tags are custom to this game (declared in `GAME_CONFIG['tags']`).
