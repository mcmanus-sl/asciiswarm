# Game Spec 14: Fluid & Pressure

## Overview
A flooding mine. Water pours from a source and spreads one tile per turn into adjacent empty cells. The player must navigate the mine, activate pumps to drain flooded sections, and reach the exit before drowning. This introduces cellular automata-style fluid simulation — a defining feature of Dwarf Fortress.

## Grid
- Dimensions: 20×16

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (20, 16),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'air', 'max': 10},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Left side, dry area |
| `wall` | `solid` | `#` | 1 | Mine tunnels and chamber walls |
| `water_source` | `npc` | `S` | 5 | 1–2 sources in the upper area. Continuously spawns water. |
| `water` | `hazard` | `~` | 3 | Spawned by sources and spreads. Properties: `{depth: 1}` |
| `pump` | `npc` | `P` | 5 | 2–3 pumps at strategic chokepoints |
| `drain` | `npc` | `D` | 2 | 2–3 drains, paired with pumps. Water reaching a drain is destroyed. |
| `valve` | `npc` | `V` | 5 | 1 valve. Interact to shut off a water_source permanently. |
| `exit` | `exit` | `>` | 5 | Far right, behind a flooded section |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `air` | 10 | 10 | Decreases by 1 each turn while standing on water. Resets to 10 on dry land. |

## Behaviors

### `water_source`
Each turn, check all 4 cardinal neighbors. For each neighbor that is empty (no `solid`, no `water`, and within bounds), create a `water` entity there with `depth=1`. Maximum 1 new water tile per source per turn (pick randomly via `env.random()` if multiple open neighbors).

### `water` (spreading)
Each turn, if `depth >= 1`, check all 4 cardinal neighbors. For each empty neighbor (no `solid`, no `water`), create a new `water` entity with `depth=1`. Each water tile spreads to at most 1 neighbor per turn (random choice).

### `pump`
Each turn, if pump's `active` property is 1, destroy all `water` entities within Manhattan distance 2.

### `drain`
Passive. Water entities that spread onto a drain tile are destroyed at end of turn.

## Event Handlers

### `input` (Movement)
Standard 4-direction movement. Player CAN walk onto water tiles (water is `hazard`, not `solid`).

### `input` (Interact)
If action is `interact`:
1. Check 4 cardinal neighbors for a `pump`: toggle pump's `active` property (0→1 or 1→0). Emit `reward` `{ 'amount': 0.1 }`.
2. Check 4 cardinal neighbors for a `valve`: destroy the nearest `water_source`. Emit `reward` `{ 'amount': 0.3 }`.

### `turn_end` (Drowning)
If player is on a `water` tile: decrement `air` by 1. If `air <= 0`: `env.end_game('lost')`.
If player is NOT on water: reset `air` to 10.

### `collision` (exit)
Player reaches exit: `env.end_game('won')`.

### `before_move`
Standard solid blocking. Water does NOT block movement.

## Win Condition
Reach the exit.

## Lose Condition
Air reaches 0 (drowning).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–1% |
| PPO win rate at 500k steps | >5% |
| PPO learning delta (500k - 100k) | >3% |

## Invariant Tests

1. 1–2 water sources exist at start.
2. 2–3 pumps exist, all starting with `active=0`.
3. 2–3 drains exist.
4. 1 valve exists.
5. Player starts on a dry tile (no water).
6. Player starts with air=10.
7. Exit exists and is initially behind at least one water-source-adjacent zone.

## Notes
- Water spreading is capped: each water tile spreads to only 1 neighbor per turn, and water_source spawns 1 per turn. This prevents exponential flooding while still creating time pressure.
- The pump+drain system creates a controllable counter to flooding. Strategic pump activation is key.
- The valve is a permanent solution but requires navigating through partially flooded areas to reach it.
- The air mechanic means the player can survive short dashes through water but not prolonged submersion.
- This is a simplified version of DF's fluid pressure system — no pressure levels, no vertical flow, just horizontal spread.
