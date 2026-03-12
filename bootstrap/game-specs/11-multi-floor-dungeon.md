# Game Spec 11: Multi-Floor Dungeon

## Overview
A 3-floor dungeon represented as side-by-side zones on a wide grid. The player descends via staircase entities, fighting enemies and collecting keys to unlock the exit on floor 3. This introduces Z-level navigation — a core Dwarf Fortress concept — within the single-grid engine by placing floors side by side.

## Grid
- Dimensions: 36×12 (three 12×12 zones side by side)

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (36, 12),
    'max_turns': 600,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'health', 'max': 10},
        {'key': 'keys_held', 'max': 3},
        {'key': 'floor', 'max': 3},
    ],
}
```

## Floor Layout

- **Floor 1** (x=0–11): Starting area. 1–2 wanderers, 1 potion, staircase down at far end.
- **Floor 2** (x=12–23): Mid dungeon. 2–3 chasers, 1–2 potions, 1 key, staircase down.
- **Floor 3** (x=24–35): Final floor. 1 sentinel + 2 chasers, 1 potion, locked exit.

Each floor has walls at its left and right boundaries (x=0, x=11, x=12, x=23, x=24, x=35) and top/bottom (y=0, y=11). Floors are visually separated by double-thick walls.

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Random empty cell in floor 1 |
| `wall` | `solid` | `#` | 1 | Floor boundaries, internal room walls |
| `stairs_down` | `npc` | `<` | 5 | One per floor (floors 1, 2). Right side of floor. |
| `stairs_up` | `npc` | `>` | 5 | One per floor (floors 2, 3). Left side of floor. Paired with stairs_down from previous floor. |
| `wanderer` | `hazard` | `w` | 5 | Floor 1 |
| `chaser` | `hazard` | `c` | 5 | Floors 2 and 3 |
| `sentinel` | `hazard` | `s` | 5 | Floor 3 |
| `potion` | `pickup` | `!` | 3 | 1 per floor |
| `floor_key` | `pickup` | `k` | 3 | 1 on floor 2 |
| `locked_exit` | `solid` | `+` | 5 | Floor 3, blocking the exit |
| `exit` | `exit` | `E` | 5 | Floor 3, behind the locked exit |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `health` | 10 | 10 | Current HP |
| `keys_held` | 0 | 3 | Keys in inventory |
| `floor` | 1 | 3 | Current floor (for observation) |

## Behaviors

### `wanderer`
Random cardinal movement each turn (same as spec 04).

### `chaser`
Move toward player if on the same floor zone AND within Manhattan distance 6. Otherwise random movement. Floor zone check: chaser at x in [0,11] = floor 1, [12,23] = floor 2, [24,35] = floor 3. Only chase if player's `floor` property matches.

### `sentinel`
Does not move. If player is within Manhattan distance 3 (and same floor), emit `sentinel_alert`.

## Event Handlers

### `input` (Player Movement)
Standard 4-direction movement.

### `input` (Interact)
If action is `interact`:
1. Check if player is standing on `stairs_down`: teleport player to the paired `stairs_up` on the next floor. Update `floor` property. Emit `reward` `{ 'amount': 0.2 }`.
2. Check if player is standing on `stairs_up`: teleport to paired `stairs_down` on previous floor. Update `floor` property.
3. Check 4 cardinal neighbors for `locked_exit` and `keys_held >= 1`: destroy locked_exit, decrement `keys_held`. Emit `reward` `{ 'amount': 0.3 }`.

### `collision` (combat)
Same damage exchange as spec 04. Player attack=2. Wanderer: 1 HP / 1 ATK. Chaser: 2 HP / 2 ATK. Sentinel: 3 HP / 1 ATK. Cancel move on combat.

### `collision` (pickup)
Potion: +3 health (capped). Floor_key: +1 `keys_held`. Destroy pickup. Emit reward.

### `collision` (exit)
`env.end_game('won')`.

### `before_move` (solid blocks)
Standard solid blocking.

## Win Condition
Reach the exit on floor 3 (after unlocking it).

## Lose Condition
Health reaches 0.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–0.5% |
| PPO win rate at 300k steps | >5% |
| PPO learning delta (300k - 100k) | >3% |

## Invariant Tests

1. Grid is exactly 36×12.
2. Exactly one player, starts in floor 1 zone (x < 12).
3. Each floor has its own wall boundaries.
4. At least one stair connection between each adjacent floor pair.
5. Exactly one floor_key exists (on floor 2).
6. Exactly one locked_exit and one exit on floor 3.
7. Exit is not reachable from floor 3 start without destroying locked_exit.

## Notes
- The side-by-side floor encoding means the RL agent sees all 3 floors simultaneously in the grid observation. This is intentional — it lets the CNN learn spatial relationships between floors.
- The `floor` player property helps the scalar observation track which zone the player is in.
- Stair teleportation is handled in the `interact` handler by directly setting `entity.x` and updating the grid — use `env.move_entity()` would trigger collisions, so instead: remove from old grid cell, set new coordinates, add to new grid cell.
