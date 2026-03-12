# Game Spec 16: Mood & Needs

## Overview
A caretaker game. The player manages 4 NPCs ("dwarves") who have needs: hunger, rest, and social. Needs decay over time. If any dwarf's mood drops to 0, they throw a tantrum — destroying nearby objects and potentially triggering a cascade of tantrums in other dwarves. The player must keep all dwarves alive and happy while completing a construction project (building 5 wall segments). This is a direct analog to Dwarf Fortress's mood/needs/tantrum spiral system.

## Grid
- Dimensions: 16×16

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'order_eat', 'order_sleep', 'order_socialize', 'order_build'],
    'grid': (16, 16),
    'max_turns': 500,
    'step_penalty': -0.003,
    'player_properties': [
        {'key': 'walls_built', 'max': 5},
        {'key': 'dwarves_alive', 'max': 4},
        {'key': 'avg_mood', 'max': 10},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Center of the fort |
| `wall` | `solid` | `#` | 1 | Fort boundaries |
| `dwarf` | `npc` | `D` | 8 | 4 dwarves inside the fort. Properties: `{hunger: 8, rest: 8, social: 8, mood: 10, task: 'idle'}` |
| `food_store` | `npc` | `F` | 5 | 2 food storage areas inside the fort |
| `bed` | `npc` | `b` | 5 | 4 beds inside the fort |
| `tavern` | `npc` | `T` | 5 | 1 tavern area inside the fort |
| `build_site` | `npc` | `_` | 2 | 5 sites along the fort's outer edge where walls can be constructed |
| `stone` | `pickup` | `o` | 3 | 8–10 stone piles scattered outside the fort |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `walls_built` | 0 | 5 | Constructed wall segments |
| `dwarves_alive` | 4 | 4 | Living dwarves |
| `avg_mood` | 10 | 10 | Average dwarf mood (for observation signal) |

## Dwarf Needs System

Each dwarf has three need values (0–10, start at 8):
- **hunger**: decreases by 1 every 5 turns. Restored to 10 when dwarf is at a `food_store`.
- **rest**: decreases by 1 every 6 turns. Restored to 10 when dwarf is at a `bed`.
- **social**: decreases by 1 every 8 turns. Restored to 10 when dwarf is at the `tavern`.

**Mood** = min(hunger, rest, social). Updated each turn.

### Tantrum Cascade
If any dwarf's mood reaches 0:
1. The dwarf enters `tantrum` state.
2. For 3 turns, the tantrum dwarf moves randomly and destroys any non-wall, non-player entity it collides with (food stores, beds, build sites, stone).
3. Any other dwarf within Manhattan distance 3 of a tantruming dwarf loses 2 mood immediately.
4. If that mood drop causes another dwarf to hit 0 → cascade tantrum.
5. After 3 turns, the tantruming dwarf calms down with all needs reset to 3.

## Behaviors

### `dwarf`
Execute current `task`:
- **`idle`**: stand still.
- **`eat`**: move toward nearest `food_store`. If adjacent or on it, set `hunger = 10`, then set task to `idle`.
- **`sleep`**: move toward nearest `bed`. If adjacent or on it, set `rest = 10`, then set task to `idle`.
- **`socialize`**: move toward `tavern`. If adjacent or on it, set `social = 10`, then set task to `idle`.
- **`build`**: move toward nearest `stone`. If adjacent, pick it up (destroy stone, set `carrying = 1`). Then move toward nearest `build_site`. If adjacent and carrying, create a `wall` at the build_site position, destroy the build_site, set `carrying = 0`. Increment player's `walls_built`.
- **`tantrum`**: random destructive movement (see above).

## Event Handlers

### `input` (Movement)
Standard 4-direction movement.

### `input` (Orders)
- `order_eat`: set nearest dwarf (within 5 tiles) to task `eat`.
- `order_sleep`: set nearest dwarf (within 5 tiles) to task `sleep`.
- `order_socialize`: set nearest dwarf (within 5 tiles) to task `socialize`.
- `order_build`: set nearest dwarf (within 5 tiles) to task `build`.

### `turn_end` (Needs decay + mood check)
1. Decay needs for all dwarves based on turn number modulo.
2. Recalculate mood for all dwarves.
3. Trigger tantrum cascade for any dwarf hitting mood 0.
4. Update `avg_mood` player property.
5. If `walls_built >= 5`: `env.end_game('won')`.

### `collision` (pickup)
Player walks onto stone: nothing (player doesn't carry stone — dwarves do).

### `before_move`
Standard solid blocking.

## Win Condition
Build 5 wall segments at the build sites before time runs out or all dwarves die.

## Lose Condition
All dwarves die (tantrum cascade destroys all beds/food → unrecoverable spiral). Tracked via `dwarves_alive`. If `dwarves_alive == 0`: `env.end_game('lost')`.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0% |
| PPO win rate at 1M steps | >3% |
| PPO learning delta (1M - 200k) | >2% |

## Invariant Tests

1. Exactly 4 dwarves at start, all with mood=10.
2. At least 2 food stores, 4 beds, 1 tavern.
3. Exactly 5 build sites.
4. At least 8 stone piles.
5. All dwarves start in `idle` task.
6. Player starts with walls_built=0.

## Notes
- The tantrum cascade is the signature DF mechanic. A single neglected dwarf can trigger a chain reaction that destroys infrastructure, which makes other dwarves unhappy, which triggers more tantrums.
- The player must balance short-term productivity (ordering builds) with long-term stability (ordering rest/food/social before needs get critical).
- The `avg_mood` scalar gives the RL agent a summary signal of colony health without needing to observe individual dwarf states.
- 10 custom actions (including 4 orders) make this the largest action space so far.
