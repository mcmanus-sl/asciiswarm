# Game Spec 18: Fortress Mode

## Overview
The capstone game. A full fortress management simulation integrating systems from specs 09–17: NPCs with needs, farming, crafting, resource economy, ecology, fluid hazards, and siege defense. The player manages a colony of 6 dwarves through an open-ended survival scenario. There is no single win condition — the game uses a continuous score based on colony survival time, population, and wealth. This is AsciiSwarm's Dwarf Fortress.

## Grid
- Dimensions: 40×30

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'order_eat', 'order_sleep', 'order_mine', 'order_farm',
                'order_craft', 'order_build', 'order_guard', 'order_haul'],
    'grid': (40, 30),
    'max_turns': 2000,
    'step_penalty': 0.0,
    'player_properties': [
        {'key': 'population', 'max': 12},
        {'key': 'food_stock', 'max': 50},
        {'key': 'stone_stock', 'max': 50},
        {'key': 'wood_stock', 'max': 50},
        {'key': 'wealth', 'max': 200},
        {'key': 'avg_mood', 'max': 10},
        {'key': 'wave', 'max': 10},
        {'key': 'score', 'max': 1000},
    ],
}
```

## Layout

The map has four zones:
- **Fortress** (x=0–12): Enclosed base with rooms for beds, food stores, workshops, tavern. Pre-built outer walls with one entrance.
- **Farm zone** (x=13–19): Open soil tiles for agriculture. Water source nearby.
- **Wilderness** (x=20–32): Trees, ore deposits, rabbits, wolves. Resource gathering zone.
- **Siege approach** (x=33–39): Where enemy waves spawn from the east.

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Center of fortress |
| `wall` | `solid` | `#` | 1 | Fortress walls, cave walls |
| `dwarf` | `npc` | `D` | 8 | 6 dwarves inside fortress. Properties: `{hunger: 10, rest: 10, social: 10, mood: 10, task: 'idle', hp: 5}` |
| `bed` | `npc` | `b` | 5 | 6 beds in dormitory |
| `food_store` | `npc` | `F` | 5 | 2 food storage areas |
| `workshop` | `npc` | `W` | 5 | 1 crafting workshop |
| `tavern` | `npc` | `T` | 5 | 1 tavern |
| `soil` | `npc` | `~` | 2 | 4×4 patch in farm zone (16 tiles) |
| `sprout` | `npc` | `,` | 3 | Created when dwarf farms. Grows in 12 turns. |
| `mature` | `pickup` | `*` | 3 | Grown crop. Harvestable. |
| `tree` | `pickup` | `t` | 3 | 8–12 in wilderness. Yield wood. |
| `ore` | `pickup` | `o` | 3 | 6–8 in wilderness. Yield stone/metal. |
| `rabbit` | `npc` | `r` | 5 | 6–8 in wilderness. Huntable for food. Reproduce near bushes. |
| `wolf` | `hazard` | `w` | 5 | 2–3 in wilderness. Hostile. |
| `bush` | `npc` | `*` | 2 | 4–5 in wilderness. Rabbit reproduction sites. |
| `water_source` | `npc` | `S` | 5 | 1 near farm zone. Floods if not managed. |
| `water` | `hazard` | `~` | 3 | Spreads from source if unchecked. |
| `pump` | `npc` | `P` | 5 | 1 near farm (controls irrigation/flooding). |
| `built_wall` | `solid` | `=` | 5 | Player/dwarf constructed. HP=3. |
| `trap` | `npc` | `^` | 3 | Constructed defense. |
| `build_site` | `npc` | `_` | 2 | 8 sites along fortress eastern wall |
| `grunt` | `hazard` | `g` | 5 | Wave enemies |
| `brute` | `hazard` | `B` | 5 | Wave enemies (later waves) |
| `merchant` | `npc` | `M` | 5 | Arrives every 200 turns at fortress entrance. Trades goods for wealth. |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `population` | 6 | 12 | Living dwarves. New dwarves arrive if mood avg > 7 and food > 20 (every 150 turns, up to 12). |
| `food_stock` | 10 | 50 | Stored food units |
| `stone_stock` | 5 | 50 | Stored stone |
| `wood_stock` | 5 | 50 | Stored wood |
| `wealth` | 0 | 200 | Accumulated from crafting and trade |
| `avg_mood` | 10 | 10 | Average colony mood |
| `wave` | 0 | 10 | Current siege wave |
| `score` | 0 | 1000 | Continuous score (see scoring) |

## Scoring (Reward Function)

No binary win/loss. Instead, a continuous score updated each turn:
- **+0.01** per living dwarf per turn (survival time × population)
- **+0.5** per successful harvest delivered to food_store
- **+0.5** per wall built
- **+1.0** per siege wave survived
- **+0.3** per trade completed with merchant
- **-2.0** per dwarf death
- **-5.0** if population reaches 0 (game ends: `env.end_game('lost')`)

The score is emitted as `reward` events. The game "wins" (`env.end_game('won')`) if the colony survives all 2000 turns with population > 0.

## Dwarf Needs System (from Spec 16)

Identical to spec 16: hunger/rest/social decay, mood = min(needs), tantrum cascade at mood 0.
- Hunger: -1 every 5 turns. Restored at food_store (consumes 1 food_stock).
- Rest: -1 every 6 turns. Restored at bed.
- Social: -1 every 8 turns. Restored at tavern.

## Dwarf Tasks

| Task | Behavior |
|------|----------|
| `idle` | Stand still |
| `eat` | Move to food_store, consume 1 food_stock, restore hunger |
| `sleep` | Move to bed, restore rest |
| `mine` | Move to nearest ore, mine it (+2 stone_stock), return to workshop |
| `farm` | Move to empty soil, plant (if food_stock > 0, costs 1), tend sprouts, harvest mature→food_store (+2 food_stock) |
| `craft` | Move to workshop. If stone_stock ≥ 2 and wood_stock ≥ 1: produce 1 wealth (+3). |
| `build` | Move to stone_stock, pick up material, move to build_site, construct wall |
| `guard` | Move to fortress entrance, attack nearby enemies |
| `haul` | Move to nearest pickup in wilderness, bring to fortress stockpile |

## Ecology (from Spec 13, simplified)

- Rabbits reproduce near bushes every 25 turns (cap 10).
- Wolves hunt rabbits, reproduce every 50 turns (cap 4).
- Dwarves on `guard` or `haul` task can be attacked by wolves in wilderness.

## Water (from Spec 14, simplified)

- Water source spawns 1 water tile per 10 turns if pump is off.
- Pump `interact` toggles: ON drains all water within 3 tiles, OFF lets water spread.
- Water on farm soil speeds crop growth (8 turns instead of 12).
- Water flooding fortress rooms destroys food_store contents.

## Siege (from Spec 17, simplified)

Waves spawn every 300 turns (turns 300, 600, 900, 1200, 1500):

| Wave | Enemies |
|------|---------|
| 1 | 3 grunts |
| 2 | 5 grunts |
| 3 | 4 grunts + 2 brutes |
| 4 | 3 grunts + 3 brutes |
| 5 | 5 grunts + 4 brutes |

Enemies move toward fortress entrance. Built walls slow them. Traps destroy them. Guard-task dwarves fight them. Enemies that reach the fortress interior attack dwarves and destroy furniture.

## Trade (from Spec 15, simplified)

Merchant arrives at fortress entrance every 200 turns, stays for 20 turns. Player `interact` adjacent to merchant: trade 5 food_stock for +5 wealth, or 5 stone_stock for +5 wealth, or 5 wood_stock for +5 wealth.

## Event Handlers

### `input` (Movement)
Standard 4-direction movement for the player.

### `input` (Orders)
`order_eat`, `order_sleep`, `order_mine`, `order_farm`, `order_craft`, `order_build`, `order_guard`, `order_haul`: set nearest dwarf (within 5 tiles) to the specified task.

### `input` (Interact)
- Adjacent to pump: toggle pump.
- Adjacent to merchant: execute trade (if resources available).
- Adjacent to workshop: player crafts directly (+3 wealth if materials available).

### `turn_end` (Tick all systems)
1. Dwarf needs decay.
2. Mood check → tantrum cascade.
3. Crop growth tick.
4. Water spread tick (every 10 turns).
5. Ecology tick (rabbit/wolf reproduction).
6. Wave spawn check.
7. Merchant arrival/departure check.
8. Population growth check (every 150 turns).
9. Score update: emit `reward` for per-turn survival bonus.
10. Update all player properties from global state.

### `collision`, `before_move`
Composite handlers covering all interaction types from component specs.

## Win Condition
Survive 2000 turns with population > 0.

## Lose Condition
Population reaches 0.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent avg score | 5–15 |
| PPO avg score at 2M steps | >50 |
| PPO score delta (2M - 500k) | >20 |

## Invariant Tests

1. 6 dwarves at start, all mood=10.
2. 6 beds, 2 food stores, 1 workshop, 1 tavern.
3. 16 soil tiles in farm zone.
4. Resource entities exist in wilderness.
5. No enemies at start (wave 0).
6. Fortress entrance exists (gap in wall).
7. Water source and pump exist near farm.
8. 8 build sites along eastern fortress wall.

## Notes
- This is intentionally the most complex game in the progression. It integrates 6 subsystems (needs, farming, crafting, ecology, fluid, siege) into a single episode.
- The continuous scoring replaces binary win/loss, which is more appropriate for complex management games and provides denser RL signal.
- The 14-action space (6 movement + 8 orders) is the largest in the suite.
- The 40×30 grid with 2000 max turns creates long episodes that test RL sample efficiency.
- An agent that learns to prioritize food security → mood management → defense preparation → trade wealth will substantially outperform random. The challenge is learning this priority ordering from reward signal alone.
- This is the "step toward 100" — future specs can add deeper systems: individual dwarf skills/preferences, material quality tiers, noble mandates, megabeasts, cavern layers, magma forges, and multi-fortress trade networks.
