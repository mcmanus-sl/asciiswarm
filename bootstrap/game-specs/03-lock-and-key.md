# Game Spec 03: Lock & Key

## Overview
A multi-room grid where the player must find a key, use it to unlock a door, and reach the exit. This is the first game that requires the `interact` action to do something meaningful, testing entity state mutation and multi-step dependency chains.

## Grid
- Dimensions: 12×12

## GAME_CONFIG

```python
GAME_CONFIG = {
    'grid': (12, 12),
    'max_turns': 300,
    'step_penalty': -0.01,
    'player_properties': [
        {'key': 'has_key', 'max': 1},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Random empty cell in room 1 |
| `wall` | `solid` | `#` | 1 | Room boundaries and dividers |
| `door` | `solid` | `+` | 5 | In a wall divider between rooms, blocking the corridor |
| `key` | `pickup` | `k` | 5 | Random empty cell in room 2 (not the room with the player or exit) |
| `exit` | `exit` | `>` | 5 | Random empty cell in room 3 (behind the door) |

## Player Properties

| Key | Initial Value | Max (for normalization) | Description |
|-----|--------------|------------------------|-------------|
| `has_key` | 0 | 1 | Whether the player is holding the key (0 = no, 1 = yes) |

## Room Layout

The grid is divided into 3 rooms by vertical wall dividers:
- **Room 1** (left): x = 0–3. Player starts here.
- **Room 2** (center): x = 5–7. Key is here.
- **Room 3** (right): x = 9–11. Exit is here.

Wall dividers at x=4 and x=8, floor-to-ceiling. Each divider has one corridor opening (1 tile wide, random y position via `env.random()`). The corridor between rooms 2 and 3 is blocked by a `door` entity. The corridor between rooms 1 and 2 is open.

All room generation uses `env.random()` for any random placement decisions.

## Behaviors
None. No entity has autonomous behavior.

## Event Handlers

- **`input` (Player Movement)**: The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` does nothing. Out-of-bounds moves are handled safely by `env.move_entity()` returning False.

- **`input` (Interact)**: If action is `interact`:
  - Check all 4 cardinal neighbors of the player for an entity tagged `solid` with type `door`.
  - If a door is found AND `player.properties['has_key'] == 1`:
    - Destroy the door entity (`env.destroy_entity(door.id)`).
    - Set `player.properties['has_key'] = 0` (key is consumed).
    - Emit `reward` event with `{ 'amount': 0.2 }`.
  - Otherwise, do nothing.

- **`collision` (player walks into pickup)**: If mover is `player` and any occupant is tagged `pickup`:
  - Allow the move (do NOT cancel).
  - Set `player.properties['has_key'] = 1`.
  - Destroy the key entity.
  - Emit `reward` event with `{ 'amount': 0.2 }`.

- **`collision` (player walks into exit)**: If mover is `player` and any occupant is tagged `exit`, call `env.end_game('won')`.

- **`before_move` (solid blocks movement)**: If target cell contains any entity tagged `solid`, cancel the move.

## Interact Mapping
`interact` adjacent to a `door` while holding the key: destroys the door and consumes the key.

## Win Condition
Player walks onto the exit tile. Triggered by collision handler above.

## Lose Condition
None. This game cannot be lost. The engine truncates at `max_turns` (300).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0.1–3% |
| PPO win rate at 100k steps | >20% |
| PPO learning delta (100k - 10k) | >10% |

## Invariant Tests (game-specific)

1. Exactly one key exists at game start.
2. Exactly one door exists at game start.
3. Player starts in room 1 (x <= 3).
4. Key is in room 2 (5 <= x <= 7).
5. Exit is in room 3 (x >= 9).
6. Door blocks the corridor between rooms 2 and 3.
7. The corridor between rooms 1 and 2 is not blocked.
8. Player starts with `has_key == 0`.

## Notes
- The multi-step dependency (find key → go to door → interact → reach exit) is the core learning challenge.
- The `interact` action must be adjacent to the door, not on the same tile (since the door is `solid` and blocks movement).
- Reward shaping (+0.2 for key pickup, +0.2 for door unlock) guides PPO toward the correct sequence.
- The room layout is simple enough that spatial navigation is not the bottleneck — learning the dependency chain is.
