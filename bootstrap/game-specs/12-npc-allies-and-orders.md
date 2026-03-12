# Game Spec 12: NPC Allies & Orders

## Overview
The player commands a team of 3 NPC allies to accomplish tasks cooperatively. The player must direct NPCs to harvest resources and defend a base while personally reaching the exit. This introduces delegation and multi-agent coordination — the seed of Dwarf Fortress's labor system.

## Grid
- Dimensions: 20×20

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'order_follow', 'order_guard', 'order_harvest'],
    'grid': (20, 20),
    'max_turns': 500,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'wood_stockpile', 'max': 10},
        {'key': 'allies_alive', 'max': 3},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Center-left (x=2, y=10) |
| `wall` | `solid` | `#` | 1 | Outer boundary + base walls (small room at left side) |
| `ally` | `npc` | `A` | 8 | 3 allies, spawned inside the base room |
| `raider` | `hazard` | `r` | 5 | 2–3 enemies, right side of map (x ≥ 15). Chase behavior. |
| `tree` | `pickup` | `T` | 3 | 6–8 trees scattered in the middle zone (5 ≤ x ≤ 14) |
| `stockpile` | `npc` | `S` | 5 | Single tile inside the base room. NPCs deliver wood here. |
| `barricade_slot` | `npc` | `_` | 2 | 3 tiles in a line at x=14, y=9–11. Barricades can be built here. |
| `barricade` | `solid` | `=` | 5 | Created when an ally builds at a barricade_slot (costs 1 wood from stockpile). |
| `exit` | `exit` | `>` | 5 | Far right (x=19, y=10), behind the raider zone |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `wood_stockpile` | 0 | 10 | Wood stored at stockpile. Allies auto-deliver. |
| `allies_alive` | 3 | 3 | Number of living allies |

## NPC Ally Modes

Each ally has a `mode` property:
- **`follow`** (default): Follow the player (move toward player each turn, stay within 2 tiles).
- **`guard`**: Move toward the nearest `barricade_slot` or `barricade`. If adjacent to a raider, attack it (1 damage). Stay near barricade zone.
- **`harvest`**: Move toward nearest `tree`. If adjacent, destroy tree and carry wood. Move toward `stockpile`. If adjacent, deposit wood (+1 to `wood_stockpile`). Repeat.

## Behaviors

### `ally`
Execute behavior based on `mode` property. Movement uses `env.move_entity()`. Each mode is a simple state machine:
- `follow`: move one step toward player (Manhattan).
- `guard`: if raider within 3 tiles, move toward it; else move toward barricade zone center.
- `harvest`: if not carrying wood, move toward nearest tree; if adjacent to tree, destroy it, set `carrying=1`; if carrying, move toward stockpile; if adjacent to stockpile, set `carrying=0`, increment `wood_stockpile`.

### `raider`
Chase the nearest entity tagged `player` or `npc` (prefer closer). Move one step toward target per turn. On collision with ally: both take 1 damage. Raider HP=2, Ally HP=3.

## Event Handlers

### `input` (Movement)
Standard 4-direction movement.

### `input` (Orders)
- `order_follow`: set the nearest ally (within 5 tiles) to mode `follow`.
- `order_guard`: set the nearest ally (within 5 tiles) to mode `guard`.
- `order_harvest`: set the nearest ally (within 5 tiles) to mode `harvest`.

### `input` (Interact)
If adjacent to `barricade_slot` and `wood_stockpile >= 2` and no barricade already there:
- Decrement `wood_stockpile` by 2.
- Create `barricade` entity at the slot position.
- Emit `reward` `{ 'amount': 0.2 }`.

### `collision` (combat)
Player/ally vs raider: mutual damage, cancel move. If entity health ≤ 0, destroy it. If player health ≤ 0: `env.end_game('lost')`. Track `allies_alive`.

### `collision` (exit)
Player reaches exit: `env.end_game('won')`.

### `before_move`
Standard solid blocking.

## Win Condition
Player reaches the exit.

## Lose Condition
Player health reaches 0. (Player has 5 HP, raider does 1 damage per hit.)

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–0.5% |
| PPO win rate at 500k steps | >5% |
| PPO learning delta (500k - 100k) | >3% |

## Invariant Tests

1. Exactly 3 allies exist at start, all in `follow` mode.
2. All allies start inside the base room.
3. At least 2 raiders exist.
4. At least 6 trees exist.
5. Exactly one stockpile and 3 barricade_slots exist.
6. Exit is at x=19.

## Notes
- The custom actions (`order_follow`, `order_guard`, `order_harvest`) expand the action space to 9. This is the first game with non-standard actions.
- The delegation challenge: the optimal strategy is to set 1-2 allies to harvest, 1 to guard, gather enough wood for barricades, then sprint through the raider zone.
- Ally AI is intentionally simple (greedy movement) — the learning challenge is choosing which orders to give, not micro-managing ally pathfinding.
