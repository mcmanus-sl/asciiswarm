# Game Spec 06: Ice Sliding

## Overview
The player slides in the chosen direction until hitting a solid obstacle or grid edge. Rocks create a routing puzzle where the player must choose directions carefully. Tests momentum-style physics via repeated `move_entity` calls.

## Grid
- Dimensions: 10×10

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=10, grid_h=10,
    max_entities=16,       # player + exit + 8-12 rocks
    max_stack=2,
    num_entity_types=4,    # 0=unused, 1=player, 2=exit, 3=rock
    num_tags=6,            # standard 6
    num_props=1,
    num_actions=6,         # standard 6
    max_turns=200,
    step_penalty=-0.01,
    game_state_size=1,
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | rock | `O` |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Bottom-left area: random cell in (x=0–2, y=7–9) |
| exit (2) | exit (4) | Top-right area: random cell in (x=7–9, y=0–2) |
| rock (3) | solid (1) | 8–12 rocks, random positions. Not on player or exit cell. |

## Behaviors

None.

## Turn Phases

### Phase 1: Process Input (Ice Sliding)
- Actions 0–3: Compute direction vector (dx, dy). Then **slide loop**: repeatedly call `move_entity(state, player_idx, x+dx, y+dy)`.
  - If move succeeds: update position, continue sliding.
  - If move fails (out of bounds or target cell has solid entity): stop. Player stays at last valid position.
  - After each successful move, check if player is on exit cell. If yes, set `status = 1` and stop sliding.
- The slide loop has a bounded iteration count (max `grid_w + grid_h` steps) implemented via `jax.lax.while_loop` or `jax.lax.fori_loop`.
- Actions 4–5: No-op.

### Phase 2: Run Behaviors
None.

### Phase 3: Turn End
Nothing.

## Rock Placement (in `reset`)

Rocks placed procedurally:
1. Place 8–12 rocks at random positions avoiding player and exit cells.
2. Verify solvability: BFS over `(x, y)` states where each transition simulates a full slide in each direction. If exit reachable from player start, accept.
3. This solvability check can be Python-side (before JIT), since `reset` only needs to produce a valid initial state.
4. Max 100 regeneration attempts, then fall back to a known-good layout.

## game_state Slots
None used.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 1–10% |
| PPO win rate at 100k steps | >30% |
| PPO learning delta (100k - 10k) | >10% |

## Invariant Tests

1. Between 8 and 12 rocks at game start.
2. Player starts in bottom-left area (x <= 2, y >= 7).
3. Exit in top-right area (x >= 7, y <= 2).
4. No rock on player's cell or exit cell.
5. Exit reachable from player via ice-sliding BFS.

## Notes
- The sliding loop is the key JIT challenge — must be a bounded loop (`jax.lax.fori_loop` with max iterations = grid dimension).
- Indirect movement (choose direction, can't choose distance) is the core learning challenge.
- `status = 1` during slide stops further movement because subsequent moves check `status != 0`.
