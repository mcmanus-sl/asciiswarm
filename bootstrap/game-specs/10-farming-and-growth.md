# Game Spec 10: Farming & Growth

## Overview
A turn-based farming game. The player must plant seeds on tilled soil, wait for crops to grow over multiple turns, harvest them, and deliver a quota to a collection bin before time runs out. This introduces temporal planning — the agent must learn that actions have delayed payoffs.

## Grid
- Dimensions: 14×14

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'pickup', 'exit', 'npc'],
    'grid': (14, 14),
    'max_turns': 300,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'seeds', 'max': 10},
        {'key': 'crops', 'max': 10},
        {'key': 'delivered', 'max': 5},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Random empty cell in farmhouse area (x < 4, y < 4) |
| `wall` | `solid` | `#` | 1 | Outer boundary + farmhouse walls |
| `soil` | `npc` | `~` | 2 | 3×3 patch of tilled soil (center of map, x=5–7, y=5–7). 9 tiles total. |
| `seedbag` | `pickup` | `s` | 3 | Single entity near farmhouse (x=2, y=5). Respawns NOT — player gets 6 seeds from walking over it. |
| `sprout` | `npc` | `,` | 3 | Created when player plants. Grows into `mature` after 15 turns. |
| `mature` | `pickup` | `*` | 3 | Replaces sprout after growth. Harvestable by walking onto it. |
| `bin` | `npc` | `B` | 5 | Collection bin at (12, 1). Deliver crops here via `interact`. |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `seeds` | 0 | 10 | Seeds in hand |
| `crops` | 0 | 10 | Harvested crops in hand |
| `delivered` | 0 | 5 | Crops delivered to bin. Win at 5. |

## Behaviors

### `sprout`
Each turn, increment the sprout's `age` property by 1. When `age >= 15`, replace the sprout with a `mature` entity at the same position (destroy sprout, create mature).

## Event Handlers

### `input` (Player Movement)
Standard 4-direction movement.

### `input` (Interact)
If action is `interact`:
1. Check if player is standing on a `soil` tile AND `seeds >= 1` AND no `sprout` or `mature` already at this position:
   - Create a `sprout` entity at player's position with `properties={'age': 0}`.
   - Decrement `seeds` by 1.
   - Emit `reward` `{ 'amount': 0.05 }`.
2. Else check 4 cardinal neighbors for a `bin`. If found and `crops >= 1`:
   - Decrement `crops` by 1. Increment `delivered` by 1.
   - Emit `reward` `{ 'amount': 0.3 }`.
   - If `delivered >= 5`: `env.end_game('won')`.

### `collision` (player walks into pickup)
If mover is `player`:
- If occupant type is `seedbag`: set `seeds = 6`. Destroy seedbag. Emit `reward` `{ 'amount': 0.1 }`.
- If occupant type is `mature`: increment `crops` by 1 (cap at 10). Destroy mature plant. Emit `reward` `{ 'amount': 0.15 }`.

### `before_move` (solid blocks movement)
Standard solid blocking.

## Win Condition
Deliver 5 crops to the bin (`delivered >= 5`).

## Lose Condition
None. Truncates at `max_turns`. (The implicit lose is running out of time before crops grow.)

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–0.5% |
| PPO win rate at 300k steps | >8% |
| PPO learning delta (300k - 100k) | >4% |

## Invariant Tests

1. Exactly 9 soil tiles exist (3×3 patch).
2. Exactly one seedbag exists at start.
3. Exactly one bin exists.
4. Player starts with seeds=0, crops=0, delivered=0.
5. No sprouts or mature plants exist at start.

## Notes
- The temporal delay (15 turns for growth) is the core learning challenge. The agent must learn to plant early, do something else, come back to harvest.
- 6 seeds with a quota of 5 gives one margin of error.
- The soil patch is small (9 tiles) so the agent must return to it — spatial memory is tested.
- Sprout behavior ticks every turn regardless of player position, creating a real-time pressure element within the turn-based system.
