# Game Spec 01: Empty Exit

## Overview
The simplest possible game. A player on a grid must walk to an exit tile. No enemies, no obstacles. This exists purely to validate the engine pipeline and Gym interface.

## Grid
- Dimensions: 8×8

## GAME_CONFIG

```python
GAME_CONFIG = {
    'grid': (8, 8),
    'max_turns': 200,
    'player_properties': [],
    # uses all defaults — 6 standard actions, 6 standard tags
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Random empty cell |
| `exit` | `exit` | `>` | 5 | Random empty cell (not player's cell) |

## Player Properties
None. No properties needed for this game.

## Behaviors
None. No entity has autonomous behavior.

## Event Handlers

- **`input` (Player Movement)**: The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` does nothing. Out-of-bounds moves are handled safely by `env.move_entity()` returning False.

- **`collision`**: If mover is `player` and any occupant is tagged `exit`, call `env.end_game('won')`.

## Interact Mapping
`interact` does nothing in this game.

## Win Condition
Player walks onto the exit tile. Triggered by collision handler above.

## Lose Condition
None. This game cannot be lost. The engine truncates at `max_turns` (200) — the game itself never calls `end_game('lost')`.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 5–40% (small grid, no obstacles, random walk will find exit eventually within time limit) |
| PPO win rate at 100k steps | >90% |
| PPO learning delta (100k - 10k) | >30% |

## Notes
- The engine natively truncates at `max_turns=200` — no external `TimeLimit` wrapper needed.
- Spawning uses `env.random()` for deterministic placement.
