# Game Spec 05: Pac-Man Collect

## Overview
A collection game where the player must pick up all dots on the grid while avoiding two ghosts. One ghost chases the player (Manhattan distance), the other patrols a fixed path. Win by collecting every dot. Lose by colliding with a ghost. This is the first AGENT SWARM game that combines collection-based win conditions with deterministic enemy AI patterns.

## Grid
- Dimensions: 12×12

## GAME_CONFIG

```python
GAME_CONFIG = {
    'grid': (12, 12),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Center of grid (6, 6) |
| `dot` | `pickup` | `.` | 3 | Every empty cell not occupied by player, ghost, or wall (grid-filling) |
| `chaser` | `hazard` | `C` | 5 | Top-left corner (1, 1) |
| `patroller` | `hazard` | `P` | 5 | Top-right corner (10, 1) |
| `wall` | `solid` | `#` | 1 | Border walls and a few interior walls to create corridors (see layout below) |

## Player Properties
None. No properties needed for this game.

## Grid Layout

The grid has border walls on all edges. Interior walls form a simple cross pattern creating 4 quadrant corridors:
- Horizontal wall segment: y=5, x=3 to x=8 (with a gap at x=5 and x=6)
- Vertical wall segment: x=5, y=3 to y=8 (with a gap at y=5 and y=6)

This creates open corridors while giving ghosts and the player limited pathways. Dots fill all remaining empty cells.

## Behaviors

### `chaser`
Each turn, move one step toward the player using Manhattan distance. Choose the axis with the greater distance (e.g., if |dx| > |dy|, move horizontally). Break ties by preferring horizontal movement. If the chosen move is blocked (by `solid` or out of bounds), try the other axis. If both are blocked, stay put. Uses `env.move_entity()`.

### `patroller`
Follows a fixed rectangular patrol path around the grid interior: east along y=1, south along x=10, west along y=10, north along x=1, then repeat. Stores current `patrol_direction` as a property (0=east, 1=south, 2=west, 3=north) and `patrol_steps` counting steps in current direction. After 9 steps in a direction, advance to the next direction. If movement is blocked (by `solid`), advance to the next direction immediately. Uses `env.move_entity()`.

## Event Handlers

- **`input` (Player Movement)**: The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` does nothing. `interact` does nothing.

- **`collision` (player walks into pickup)**: If mover is `player` and any occupant is tagged `pickup`:
  - Allow the move (do NOT cancel).
  - Destroy the dot entity.
  - Emit `reward` event with `{ 'amount': 0.05 }`.
  - After destruction, check if any entities tagged `pickup` remain. If none remain, call `env.end_game('won')`.

- **`collision` (player walks into hazard)**: If mover is `player` and any occupant is tagged `hazard`, call `env.end_game('lost')`.

- **`collision` (hazard walks into player)**: If mover is tagged `hazard` and any occupant is tagged `player`, call `env.end_game('lost')`.

- **`before_move` (solid blocks movement)**: If target cell contains any entity tagged `solid`, cancel the move.

## Interact Mapping
`interact` does nothing in this game.

## Win Condition
All entities tagged `pickup` have been collected (destroyed). Checked after each dot pickup in the collision handler.

## Lose Condition
Player collides with any ghost (chaser or patroller).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–2% |
| PPO win rate at 100k steps | >10% |
| PPO learning delta (100k - 10k) | >5% |

## Invariant Tests (game-specific)

1. Exactly one chaser and one patroller exist at game start.
2. At least 20 dots exist at game start.
3. Player starts at center of grid.
4. No dot occupies the same cell as a ghost or wall.
5. Player does not start on a ghost's cell.
6. All non-wall, non-entity cells are reachable from the player (BFS over non-`solid` tiles).

## Notes
- The collection-based win condition (ALL dots) is harder than a single exit — the player must cover the entire grid while staying alive.
- Dot count is high (filling empty cells), so reward shaping via per-dot +0.05 provides frequent positive signal.
- The chaser creates pressure to keep moving; the patroller creates predictable danger zones the agent must learn to avoid.
- The interior wall pattern is simple and deterministic — no procedural generation needed.
