# Game Spec 13: Ecology & Spawning

## Overview
A survival game on a living map. Prey animals (rabbits) reproduce over time, predators (wolves) hunt them, and the player must hunt rabbits for food while avoiding wolves. If rabbits go extinct, the food supply collapses. If wolves overpopulate, the map becomes lethal. The player must reach the exit before starving. This introduces population dynamics and carrying capacity — Dwarf Fortress's wildlife simulation in miniature.

## Grid
- Dimensions: 20×20

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (20, 20),
    'max_turns': 500,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'food', 'max': 20},
        {'key': 'health', 'max': 10},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Bottom-left corner area |
| `wall` | `solid` | `#` | 1 | Outer boundary, scattered rock clusters |
| `rabbit` | `npc` | `r` | 5 | 8–10 scattered randomly. Properties: `{age: 0}` |
| `wolf` | `hazard` | `W` | 5 | 2–3 scattered, away from player (distance ≥ 8) |
| `bush` | `npc` | `*` | 2 | 5–7 scattered. Rabbits reproduce near bushes. |
| `exit` | `exit` | `>` | 5 | Top-right corner area |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `food` | 15 | 20 | Decreases by 1 every 3 turns. Hunting rabbits adds 3. |
| `health` | 10 | 10 | Wolf attacks deal 2 damage. |

## Behaviors

### `rabbit`
1. Increment `age` by 1 each turn.
2. Movement: pick random cardinal direction (via `env.random()`), move if not blocked.
3. Flee: if a wolf is within Manhattan distance 3, move away from nearest wolf instead.
4. Reproduction: if `age >= 20` AND a `bush` entity is within Manhattan distance 2 AND total rabbit count < 15 (carrying capacity): create a new `rabbit` at an adjacent empty cell (via `env.random()`). Reset `age` to 0.

### `wolf`
1. If a rabbit is within Manhattan distance 5, move toward nearest rabbit.
2. If no rabbit nearby, random movement.
3. Wolf reproduction: every 40 turns (tracked via `age` property), if total wolf count < 5, spawn a new wolf at adjacent empty cell. Reset age.

## Event Handlers

### `input` (Movement)
Standard 4-direction movement.

### `input` (Interact)
If standing on a cell with a dead rabbit (there are no dead rabbits as entities — see collision below) — `interact` does nothing in this game.

### `turn_end` (Hunger clock)
Every 3 turns (check `env.turn_number % 3 == 0`), decrement player `food` by 1. If `food <= 0`: `env.end_game('lost')`.

### `collision` (player hunts rabbit)
If mover is `player` and occupant is `rabbit`:
- Destroy the rabbit.
- Increment `food` by 3 (cap at 20).
- Emit `reward` `{ 'amount': 0.1 }`.

### `collision` (wolf attacks player)
If mover has tag `hazard` and occupant has tag `player`, OR mover has tag `player` and occupant has tag `hazard`:
- Cancel the move.
- Player takes 2 damage.
- Wolf takes 1 damage (wolf HP=3).
- If wolf HP ≤ 0, destroy wolf. Emit `reward` `{ 'amount': 0.2 }`.
- If player health ≤ 0: `env.end_game('lost')`.

### `collision` (wolf hunts rabbit)
If mover has tag `hazard` and occupant type is `rabbit`:
- Destroy the rabbit.
- Wolf `age` resets to 0 (fed).

### `collision` (exit)
Player reaches exit: `env.end_game('won')`.

### `before_move`
Standard solid blocking.

## Win Condition
Player reaches the exit before starving or dying to wolves.

## Lose Condition
- Food reaches 0 (starvation).
- Health reaches 0 (wolf attack).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–2% |
| PPO win rate at 500k steps | >5% |
| PPO learning delta (500k - 100k) | >3% |

## Invariant Tests

1. 8–10 rabbits at start.
2. 2–3 wolves at start.
3. No wolf within Manhattan distance 8 of player at start.
4. 5–7 bushes exist.
5. Player starts with food=15, health=10.
6. Rabbit carrying capacity is 15 (code-level check).

## Notes
- The ecological balance creates emergent gameplay: if the player kills too many rabbits early, food becomes scarce later. If the player ignores wolves, they eat all the rabbits.
- The carrying capacity caps (15 rabbits, 5 wolves) prevent population explosions from making the game unplayable.
- Rabbit flee behavior creates "herding" patterns that the RL agent can learn to exploit.
- This is the first game where the environment's dynamics matter more than the player's direct actions — a key DF concept.
