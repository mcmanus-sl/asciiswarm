# Game Spec 02: Dodge

## Overview
A player must reach an exit while avoiding a patrolling enemy. The enemy bounces horizontally across the grid. Contact with the enemy kills the player. This is the first game with autonomous entity behavior and introduces hazard avoidance.

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
| `player` | `player` | `@` | 10 | Bottom-left quadrant, random empty cell |
| `exit` | `exit` | `>` | 5 | Top-right quadrant, random empty cell |
| `wanderer` | `hazard` | `w` | 5 | Center row (y=4 or y=5), random x. Starts moving east. |

## Player Properties
None. No properties needed for this game.

## Behaviors

### `wanderer`
Each turn, move one step in the entity's current direction (stored as a `direction` property: `1` for east, `-1` for west). After moving, check if the next step in the same direction would be out of bounds (x+direction < 0 or x+direction >= grid width). If so, reverse direction (`direction *= -1`). Uses `env.move_entity()` for movement.

## Event Handlers

- **`input` (Player Movement)**: The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` does nothing. `interact` does nothing. Out-of-bounds moves are handled safely by `env.move_entity()` returning False.

- **`collision` (player walks into hazard)**: If mover is `player` and any occupant is tagged `hazard`, call `env.end_game('lost')`.

- **`collision` (hazard walks into player)**: If mover is tagged `hazard` and any occupant is tagged `player`, call `env.end_game('lost')`.

- **`collision` (player walks into exit)**: If mover is `player` and any occupant is tagged `exit`, call `env.end_game('won')`.

## Interact Mapping
`interact` does nothing in this game.

## Win Condition
Player walks onto the exit tile. Triggered by collision handler above.

## Lose Condition
Player collides with the wanderer (either player walks into wanderer, or wanderer walks into player).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 1–15% |
| PPO win rate at 100k steps | >50% |
| PPO learning delta (100k - 10k) | >15% |

## Invariant Tests (game-specific)

1. Exactly one wanderer exists at game start.
2. Wanderer starts in the center rows (y=4 or y=5).
3. Player starts in the bottom-left quadrant (x < 5, y >= 5).
4. Exit is in the top-right quadrant (x >= 5, y < 5).
5. Player and wanderer do not start on the same cell.

## Notes
- The wanderer's bounce pattern is deterministic — no randomness in its movement, only in initial x position.
- The player must learn to time movement past the wanderer's patrol path.
- Spawning quadrants ensure the player must cross the wanderer's path to reach the exit.
