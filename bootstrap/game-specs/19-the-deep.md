# Game Spec 19: The Deep (Island Effects + Deep Fishing)

## Overview
A Pier chunk emerges south of the Shell, jutting into deep water. The Pier emits an Island Effect — a global multiplier that boosts crop growth, sheep production, and fishing yields across all chunks. But the Pier also unlocks Deep Fishing: dangerous fishing spots that can catch Atlantean relics (Deep Fish) but drain stamina rapidly. The agent must dynamically alter its established routine to exploit the Pier's benefits while managing the increased risk.

This game introduces the `island_effects` system: the Pier acts as a global buff emitter that changes the economics of the entire island.

## Grid
- Dimensions: 32x32
- Turtle Shell: 6x6 (x=13..18, y=13..18)
- Farm Chunk: 6x6 (x=7..12, y=13..18)
- Pasture Chunk: 6x6 (x=7..12, y=7..12)
- Mill Chunk: 6x6 (x=19..24, y=13..18)
- Pier Chunk: 3x8 (x=14..16, y=19..26) — narrow pier extending south into water
- Bridge Shell↔Farm: (x=12, y=15)
- Bridge Farm↔Pasture: (x=9, y=12)
- Bridge Shell↔Mill: (x=19, y=15)
- Pier connects directly to Shell south edge: (x=15, y=19)
- All other cells are water

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=384,
    max_stack=2,
    num_entity_types=26,   # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush,
                           # 7=bait_trader, 8=fishing_spot, 9=bridge, 10=soil, 11=seed, 12=sprout,
                           # 13=mature_wheat, 14=david_npc, 15=sheep, 16=manure, 17=fence,
                           # 18=pierre_npc, 19=oven, 20=bread, 21=mouse, 22=pier_plank,
                           # 23=deep_fishing_spot, 24=deep_fish, 25=pier_beacon
    num_tags=12,           # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth,
                           # 7=water_source, 8=farmable, 9=harvestable, 10=processor, 11=secret
    num_props=10,          # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held,
                           # 6=wool_held, 7=wheat_held, 8=bread_held, 9=processing_timer
    num_actions=8,
    max_turns=1200,
    step_penalty=-0.002,
    game_state_size=16,    # 0=day_night_phase, 1=phase_timer, 2=deep_fish_caught, 3=pier_active,
                           # 4=island_effect_multiplier, 5=bread_baked, 6=golden_seed_found,
                           # 7=wool_collected, 8=manure_held, 9=seeds_held, 10=pierre_busy,
                           # 11=pierre_output_ready, 12=pier_fuel, 13=total_wheat_harvested,
                           # 14=deep_fishing_attempts, 15=pier_activated_turn
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | water | `~` |
| 3 | stick | `/` |
| 4 | campfire | `*` |
| 5 | well | `O` |
| 6 | berry_bush | `%` |
| 7 | bait_trader | `T` |
| 8 | fishing_spot | `o` |
| 9 | bridge | `=` |
| 10 | soil | `.` |
| 11 | seed | `,` |
| 12 | sprout | `'` |
| 13 | mature_wheat | `W` |
| 14 | david_npc | `D` |
| 15 | sheep | `S` |
| 16 | manure | `m` |
| 17 | fence | `#` |
| 18 | pierre_npc | `P` |
| 19 | oven | `n` |
| 20 | bread | `B` |
| 21 | mouse | `r` |
| 22 | pier_plank | `-` |
| 23 | deep_fishing_spot | `O` |
| 24 | deep_fish | `F` |
| 25 | pier_beacon | `!` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | npc |
| 6 | warmth |
| 7 | water_source |
| 8 | farmable |
| 9 | harvestable |
| 10 | processor |
| 11 | secret |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | food | player: hunger meter (0–10) |
| 1 | stamina | player: energy meter (0–20) |
| 2 | thirst | player: hydration meter (0–10) |
| 3 | fire_fuel | campfire: fuel (0–5) / sheep: hunger (0–5) / soil: fertility (0–3) |
| 4 | sticks_held | player: sticks (0–5) |
| 5 | water_held | player: water (0–3) |
| 6 | wool_held | player: wool (0–10) |
| 7 | wheat_held | player: wheat (0–10) |
| 8 | bread_held | player: bread (0–5) |
| 9 | processing_timer | pierre_npc: countdown (0–10) |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| food (prop 0) | 10.0 |
| stamina (prop 1) | 20.0 |
| thirst (prop 2) | 10.0 |
| sticks_held (prop 4) | 5.0 |
| water_held (prop 5) | 3.0 |
| wool_held (prop 6) | 10.0 |
| wheat_held (prop 7) | 10.0 |
| bread_held (prop 8) | 5.0 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | (x=15, y=15) on shell |
| water (2) | hazard (2) | All cells outside land chunks and pier |
| All Game 18 entities | — | Same positions as Game 18, adjusted for 5-chunk layout |
| pier_plank (22) | — | Fills Pier chunk cells (x=14..16, y=19..26), walkable |
| deep_fishing_spot (23) | — | 3 spots on water cells adjacent to Pier tip (x=13..17, y=26) |
| deep_fish (24) | pickup (3) | Spawned on successful deep fishing catch |
| pier_beacon (25) | solid (1) | (x=15, y=26) at Pier tip, activates Island Effect when lit |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| All Game 18 behaviors | Same as Game 18. |
| 23 (deep_fishing_spot) | **Deep fishing**: When player interacts adjacent and bait >= 2, consume 2 bait. 40% chance to catch Deep Fish (Atlantean relic). On catch: spawn deep_fish entity, add 1.0 to reward_acc, increment game_state[2]. On fail: stamina -= 3 (exhausting attempt). Always: stamina -= 2 (base deep fishing cost). |
| 25 (pier_beacon) | **Activation**: Player interacts with 3 sticks → beacon lit. Sets game_state[3] = 1 (pier_active). Beacon requires 1 stick per 100 turns to stay lit (pier_fuel tracked in game_state[12]). When lit, emits Island Effect. |

### Island Effects System

When pier_active == 1 (beacon lit):
- **Crop growth speed**: 1.5x (growth timers tick 1.5x faster — implemented as 2 ticks every 3 turns via modular arithmetic)
- **Sheep production**: feeding sheep produces 2 wool instead of 1
- **Fishing yield**: regular fishing spots give food += 5 instead of 3
- **Deep fishing success**: 40% → 50% catch rate
- `game_state[4]` stores the active multiplier (1.0 when off, 1.5 when on)

The beacon consumes fuel: pier_fuel decrements by 1 every 100 turns. When pier_fuel hits 0, beacon goes dark, Island Effect deactivates. Player must return to refuel.

## Turn Phases

### Phase 1: Process Input
- Actions 0–7: Same as Game 18, plus:
  - Adjacent to deep_fishing_spot (bait >= 2): deep fish attempt.
  - Adjacent to pier_beacon (sticks_held >= 3, beacon off): light beacon, consume 3 sticks.
  - Adjacent to pier_beacon (sticks_held >= 1, beacon on, fuel < max): refuel, consume 1 stick, pier_fuel += 1.
  - Pick up deep_fish on walk-over: deep_fish added to inventory (tracked in game_state[2]).

### Phase 2: Run Behaviors
- All Game 18 behaviors.
- Island Effect application (modify growth ticks, sheep output, fishing yields).
- Pier beacon fuel countdown.
- Deep fishing spot availability.

### Phase 3: Turn End
- Death checks: stamina/food/thirst <= 0 → `status = -1`.
- Win check: game_state[2] (deep_fish_caught) >= 3 → `status = 1`. Add 3.0 to reward_acc.
- Auto-drink water.
- Auto-eat bread if food < 3 and bread_held > 0.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | day_night_phase |
| 1 | phase_timer |
| 2 | deep_fish_caught (lifetime, win condition) |
| 3 | pier_active (0 or 1) |
| 4 | island_effect_multiplier (1.0 or 1.5) |
| 5 | bread_baked (lifetime) |
| 6 | golden_seed_found (0 or 1) |
| 7 | wool_collected (lifetime) |
| 8 | manure_held (player inventory) |
| 9 | seeds_held (player inventory) |
| 10 | pierre_busy (0 or 1) |
| 11 | pierre_output_ready (0 or 1) |
| 12 | pier_fuel (0–5) |
| 13 | total_wheat_harvested (lifetime) |
| 14 | deep_fishing_attempts (lifetime) |
| 15 | pier_activated_turn (turn number when pier was first lit) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Cannot sustain survival while deep fishing |
| PPO win rate at 800k steps | >15% | Agent context-switches routine after Pier unlock |
| PPO "activate beacon" rate | >60% | Agent learns Pier activation is critical |
| PPO "refuel beacon" rate | >40% | Agent maintains Island Effect over time |
| PPO deep fishing efficiency | avg < 8 attempts to catch 3 | Agent manages bait economy for deep fishing |

## Invariant Tests

1. Five land chunks at specified positions.
2. Pier is 3x8 at (x=14..16, y=19..26).
3. Pier connects to Shell at (x=15, y=19).
4. 3 deep fishing spots adjacent to Pier tip.
5. Beacon at (x=15, y=26).
6. All Game 18 entities present.
7. Player starts at (x=15, y=15).
8. Pier beacon starts unlit (pier_active = 0).

## Notes
- The Pier fundamentally changes the game's economy. Before activation, the agent plays Game 18's routine. After activation, everything is 1.5x more efficient — but the agent must maintain the beacon and route to the Pier tip for deep fishing.
- Deep fishing is expensive (2 bait, stamina cost) but required for victory. The agent must build a surplus economy before attempting deep fishing.
- The Island Effect creates a phase transition in gameplay. The agent's pre-Pier and post-Pier strategies should be qualitatively different — this is the "context-switching" the RL evaluation tests.
- The Pier's narrow 3-wide layout creates a chokepoint. Night trips to the Pier are risky (far from campfire warmth).
- `max_entities=384` handles the expanded map with 5 chunks + pier + dynamic entities.
- Auto-eat bread when food is low helps the agent survive while focused on logistics.
