# Game Spec 07: Hunger Clock

## Overview
The player must reach an exit on the far side of the grid before starving. Food decreases by 1 every turn. Food pickups are scattered across the grid and restore food when collected. The player must balance exploring toward the exit with detours to eat. This tests ticking resource depletion as a core mechanic.

## Grid
- Dimensions: 14×14

## GAME_CONFIG

```python
GAME_CONFIG = {
    'grid': (14, 14),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'food', 'max': 20},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Bottom-left corner: (0, 13) |
| `exit` | `exit` | `>` | 5 | Top-right corner: (13, 0) |
| `food` | `pickup` | `f` | 5 | 10–15 food entities scattered randomly across the grid (via `env.random()`) |
| `wall` | `solid` | `#` | 1 | A few wall clusters (3–5 clusters of 2–4 walls each) to create routing obstacles |

## Player Properties

| Key | Initial Value | Max (for normalization) | Description |
|-----|--------------|------------------------|-------------|
| `food` | 20 | 20 | Current food level. Decreases by 1 each turn. |

## Wall Placement

Place 3–5 wall clusters randomly (via `env.random()`). Each cluster is 2–4 adjacent wall tiles (horizontal or vertical). Walls cannot be placed on the player start, exit, or food locations. After placement, verify the exit is reachable from the player (BFS over non-`solid` tiles). Regenerate if unreachable (max 100 attempts, then use a known-good layout).

## Behaviors
None. No entity has autonomous behavior.

## Event Handlers

- **`input` (Player Movement)**: The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` does nothing. `interact` does nothing.

- **`turn_end` (Hunger Tick)**: At the end of every turn:
  - Decrease `player.properties['food']` by 1.
  - If `player.properties['food'] <= 0`, call `env.end_game('lost')`.

- **`collision` (player walks into pickup)**: If mover is `player` and any occupant is tagged `pickup`:
  - Allow the move (do NOT cancel).
  - Increase `player.properties['food']` by 5, capped at 20.
  - Destroy the food entity.
  - Emit `reward` event with `{ 'amount': 0.05 }`.

- **`collision` (player walks into exit)**: If mover is `player` and any occupant is tagged `exit`, call `env.end_game('won')`.

- **`before_move` (solid blocks movement)**: If target cell contains any entity tagged `solid`, cancel the move.

## Interact Mapping
`interact` does nothing in this game.

## Win Condition
Player walks onto the exit tile before starving.

## Lose Condition
Player's food reaches 0 (checked at end of each turn via `turn_end` handler).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–3% |
| PPO win rate at 100k steps | >15% |
| PPO learning delta (100k - 10k) | >8% |

## Invariant Tests (game-specific)

1. Player starts with `food == 20`.
2. Between 10 and 15 food entities exist at game start.
3. Player starts at (0, 13).
4. Exit is at (13, 0).
5. Exit is reachable from the player (BFS over non-`solid` tiles).
6. No food entity spawns on the player's cell, exit cell, or a wall cell.

## Notes
- The Manhattan distance from (0,13) to (13,0) is 26, which is greater than the starting food of 20. The player MUST eat at least one food pickup to survive the journey. This is the core design tension.
- Food placement via `env.random()` means some runs are easier than others, but the 10–15 food count ensures enough exists.
- The hunger clock creates time pressure that distinguishes this from a simple pathfinding game — the player can't wander aimlessly.
- Reward shaping (+0.05 per food pickup) encourages the agent to eat, preventing starvation deaths early in training.
- Wall clusters add routing complexity without making the map feel maze-like.
