# Game Spec 06: Ice Sliding

## Overview
The player slides in the chosen direction until hitting a solid obstacle or the grid edge. Rocks are placed to create a routing puzzle where the player must reach the exit by choosing directions carefully, knowing they will overshoot. This tests `before_move` chaining and momentum-style physics.

## Grid
- Dimensions: 10×10

## GAME_CONFIG

```python
GAME_CONFIG = {
    'grid': (10, 10),
    'max_turns': 200,
    'step_penalty': -0.01,
    'player_properties': [],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Bottom-left corner area: random cell in (0–2, 7–9) |
| `exit` | `exit` | `>` | 5 | Top-right corner area: random cell in (7–9, 0–2) |
| `rock` | `solid` | `O` | 5 | 8–12 rocks placed to create a solvable routing puzzle (see layout below) |

## Player Properties
None. No properties needed for this game.

## Rock Placement

Rocks are placed procedurally to ensure the puzzle is solvable. The placement algorithm:
1. Place 8–12 rocks at random positions (via `env.random()`) avoiding the player's cell, the exit cell, and each other.
2. Verify solvability: BFS over (x, y, direction) states where each state transition simulates a full slide. If the exit is reachable from the player's start, accept. Otherwise, regenerate.
3. Maximum 100 regeneration attempts before falling back to a known-good hardcoded layout.

## Behaviors
None. No entity has autonomous behavior.

## Event Handlers

- **`input` (Player Movement — Ice Sliding)**: The game module MUST register an `input` event handler that implements the sliding mechanic. When the player chooses a direction (`move_n`, `move_s`, `move_e`, `move_w`):
  1. Compute the direction vector (dx, dy) from the action.
  2. Repeatedly attempt `env.move_entity(player.id, player.x + dx, player.y + dy)`.
  3. If the move succeeds, continue sliding (repeat step 2 from the new position).
  4. If the move fails (out of bounds, or blocked by `solid` via `before_move`), stop. The player stays at the last valid position.
  5. `wait` does nothing. `interact` does nothing.

- **`collision` (player slides onto exit)**: If mover is `player` and any occupant is tagged `exit`, call `env.end_game('won')`. Note: this fires during the slide loop, so the player stops on the exit tile.

- **`before_move` (solid blocks movement)**: If target cell contains any entity tagged `solid`, cancel the move. This is what stops the player's slide — hitting a rock.

## Interact Mapping
`interact` does nothing in this game.

## Win Condition
Player slides onto the exit tile. Triggered by collision handler during the slide loop.

## Lose Condition
None. This game cannot be lost. The engine truncates at `max_turns` (200).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 1–10% |
| PPO win rate at 100k steps | >30% |
| PPO learning delta (100k - 10k) | >10% |

## Invariant Tests (game-specific)

1. Between 8 and 12 rocks exist at game start.
2. Player starts in the bottom-left area (x <= 2, y >= 7).
3. Exit is in the top-right area (x >= 7, y <= 2).
4. No rock occupies the player's starting cell or the exit cell.
5. The exit is reachable from the player's start via ice-sliding BFS.

## Notes
- The sliding mechanic is implemented entirely in the `input` handler using repeated `env.move_entity()` calls. Each call triggers `before_move` and `collision` normally.
- `end_game('won')` during the slide loop stops further movement because the engine ignores actions after game end.
- The indirect movement (you choose direction but can't choose distance) is the core learning challenge. PPO must learn that moving east might slide past the target.
- Random agent win rate is nonzero because random slides can land on the exit by chance, especially with rocks creating stop points near it.
- Solvability verification ensures the puzzle always has a solution, but finding it requires planning multiple slides ahead.
