# Game Spec 04: Dungeon Crawl

## Overview
A multi-room dungeon with combat, health management, and three enemy types. The player must fight or avoid enemies, collect health potions, and reach the exit on the far side of the map. This is the first game intended for AGENT SWARM to build autonomously.

## Grid
- Dimensions: 16×16

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit'],
    'grid': (16, 16),
    'max_turns': 500,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'health', 'max': 10},
        {'key': 'attack', 'max': 5},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Center of the first (leftmost) room |
| `wall` | `solid` | `#` | 1 | Room boundaries and corridor walls |
| `floor` | *(inert, no tags required — use a special `inert` tag)* | `.` | 0 | Interior of rooms and corridors. NOTE: you may also simply leave these cells empty and rely on the renderer's default `'.'` character. Either approach is acceptable. |
| `wanderer` | `hazard` | `w` | 5 | 1–2 per room, random empty cell in room |
| `chaser` | `hazard` | `c` | 5 | 1 per room (rooms 3+), random empty cell |
| `sentinel` | `hazard` | `s` | 5 | 1 per room (rooms 4+), random empty cell |
| `potion` | `pickup` | `!` | 3 | 1–2 per room, random empty cell |
| `exit` | `exit` | `>` | 5 | Center of the last (rightmost) room |

## Player Properties

| Key | Initial Value | Max (for normalization) | Description |
|-----|--------------|------------------------|-------------|
| `health` | 10 | 10 | Current HP |
| `attack` | 2 | 5 | Damage dealt per hit |

`player_properties` config for observation: `[{'key': 'health', 'max': 10}, {'key': 'attack', 'max': 5}]`.

## Room Generation

Generate 3–5 rectangular rooms (random sizes within 4×4 to 6×6) placed non-overlapping on the grid. Connect adjacent rooms with corridors (1 tile wide). All rooms must be connected — verify with BFS. Use `env.random()` for all random placement.

## Behaviors

### `wanderer`
Each turn, pick a random cardinal direction (via `env.random()`). Attempt to move. If blocked (by `solid` or out of bounds), stay put.

### `chaser`
If the player is within Manhattan distance 5, move one step toward the player (prefer axis with greater distance, break ties via `env.random()`). Otherwise, behave like `wanderer`.

### `sentinel`
Does not move. On each turn, if the player is within Manhattan distance 2, emit a custom `sentinel_alert` event with the sentinel's position. Other chasers within the same room could optionally respond to this (stretch goal — not required for pass).

## Event Handlers

### `input` (Player Movement)
The game module MUST register an `input` event handler that moves the player. If action is `move_n`, attempt `env.move_entity(player.id, player.x, player.y - 1)`. Map `move_s` to +y, `move_e` to +x, `move_w` to -x. `wait` and `interact` do nothing. Out-of-bounds or blocked moves are handled safely by `env.move_entity()` returning False or being cancelled by `before_move` handlers.

### `collision` (player walks into hazard)
If mover is `player` and any occupant is tagged `hazard`:
- Cancel the move (player stays in place).
- Reduce player `health` by occupant's `attack` property (default 1).
- Reduce occupant `health` by player `attack` property.
- If occupant `health` ≤ 0, destroy the occupant.
- If player `health` ≤ 0, call `env.end_game('lost')`.

### `collision` (hazard walks into player)
If mover is tagged `hazard` and any occupant is tagged `player`:
- Cancel the move.
- Same damage exchange as above.

### `collision` (player walks into pickup)
If mover is `player` and any occupant is tagged `pickup`:
- Allow the move (do NOT cancel).
- Increase player `health` by potion's `heal_amount` property (default 3), capped at max health.
- Destroy the potion.
- Emit `reward` event with `{ 'amount': 0.1 }`.

### `collision` (player walks into exit)
If mover is `player` and any occupant is tagged `exit`:
- Call `env.end_game('won')`.
- Emit `reward` event with `{ 'amount': 1.0 }`.

### `before_move` (solid blocks all movement)
If target cell contains any entity tagged `solid`, cancel the move.

## Interact Mapping
`interact` does nothing in this game. All interaction is via walk-into collision.

## Win Condition
Player walks onto the exit tile.

## Lose Condition
Player health reaches 0.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 1–10% |
| PPO learning delta (500k vs 50k) | >0 |

## Invariant Tests (game-specific)

These are registered by the game module using the invariant test framework:
1. All rooms are connected (BFS from player start reaches exit).
2. Every room contains at least one potion.
3. Total enemy count is between 5 and 20.
4. Player starts with health > 0.
5. No enemy spawns in the same cell as the player.
6. No enemy spawns in a corridor (only in rooms).

## Notes
- Wall collision uses `before_move`, not `collision`, because walls don't need entity-to-entity interaction logic — they just block movement unconditionally.
- The damage exchange on collision means the player can "attack" by walking into enemies. This is the simplest possible combat system.
- Enemy `health` defaults: wanderer=1, chaser=2, sentinel=3.
- Enemy `attack` defaults: wanderer=1, chaser=2, sentinel=1.
