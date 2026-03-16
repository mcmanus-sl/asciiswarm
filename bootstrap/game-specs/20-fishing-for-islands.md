# Game Spec 20: FISHING FOR ISLANDS (Season 2 Finale)

## Overview
The full game. The player begins on a 6x6 Turtle shell surrounded by an endless sea. Through fishing, four island chunks emerge one by one — Farm, Pasture, Mill, Pier — each unlocking new mechanics and resources. The player must master every system from Games 15–19, chain them into a self-sustaining economy, and ultimately catch the Atlantean Monument from the deep ocean. This is the hardest game in the engine.

Standard PPO cannot solve this game. It requires **curriculum training** through Games 15–19 to build transferable skills. The agent must survive day/night cycles, farm wheat, tend sheep, process bread, activate the Pier's Island Effect, and deep fish for Atlantean relics — all while managing three survival meters on an ever-expanding island.

## Grid
- Dimensions: 32x32
- Turtle Shell: 6x6 (x=13..18, y=13..18) — starting land (only land at turn 0)
- Farm Chunk: 6x6 (x=7..12, y=13..18) — submerged, emerges after 3 fish catches from shell fishing spots
- Pasture Chunk: 6x6 (x=7..12, y=7..12) — submerged, emerges after 5 wheat harvested
- Mill Chunk: 6x6 (x=19..24, y=13..18) — submerged, emerges after 3 sheep fed
- Pier Chunk: 3x8 (x=14..16, y=19..26) — submerged, emerges after 3 bread baked
- All other cells are water at start

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=512,
    max_stack=2,
    num_entity_types=28,   # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush,
                           # 7=bait_trader, 8=fishing_spot, 9=bridge, 10=soil, 11=seed, 12=sprout,
                           # 13=mature_wheat, 14=david_npc, 15=sheep, 16=manure, 17=fence,
                           # 18=pierre_npc, 19=oven, 20=bread, 21=mouse, 22=pier_plank,
                           # 23=deep_fishing_spot, 24=deep_fish, 25=pier_beacon,
                           # 26=atlantean_monument, 27=golden_wheat
    num_tags=14,           # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth,
                           # 7=water_source, 8=farmable, 9=harvestable, 10=processor, 11=secret,
                           # 12=deep, 13=monument
    num_props=12,          # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held,
                           # 6=wool_held, 7=wheat_held, 8=bread_held, 9=processing_timer,
                           # 10=deep_fish_held, 11=golden_wheat_held
    num_actions=10,
    max_turns=3000,
    step_penalty=-0.001,
    game_state_size=24,    # see table below
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
| 26 | atlantean_monument | `A` |
| 27 | golden_wheat | `G` |

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
| 12 | deep |
| 13 | monument |

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
| 10 | deep_fish_held | player: Atlantean relics held (0–5) |
| 11 | golden_wheat_held | player: golden wheat (0–3) |

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
| deep_fish_held (prop 10) | 5.0 |
| golden_wheat_held (prop 11) | 3.0 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | (x=15, y=15) center of shell |
| water (2) | hazard (2) | Fills entire grid except Shell at start |
| stick (3) | pickup (3) | 4 sticks on shell, respawn every 25 turns; also spawn on newly emerged chunks |
| campfire (4) | warmth (6), solid (1) | (x=15, y=16) on shell |
| well (5) | water_source (7), solid (1) | (x=14, y=14) on shell |
| berry_bush (6) | solid (1) | 3 bushes on shell |
| bait_trader (7) | npc (5), solid (1) | (x=18, y=15) on shell |
| fishing_spot (8) | — | 3 spots adjacent to shell west edge (triggers Farm emergence) |
| bridge (9) | — | Spawned dynamically during emergence events |
| soil (10) | farmable (8) | 12 tiles, spawned on Farm emergence |
| seed–sprout–wheat (11–13) | — | Dynamic crop growth on Farm |
| david_npc (14) | npc (5), solid (1) | Spawns on Farm emergence at (x=9, y=15) |
| sheep (15) | npc (5) | 4 sheep, spawned on Pasture emergence |
| manure (16) | pickup (3) | Dropped by sheep |
| fence (17) | solid (1) | Spawned on Pasture emergence |
| pierre_npc (18) | npc (5), processor (10), solid (1) | Spawns on Mill emergence at (x=21, y=15) |
| oven (19) | solid (1), warmth (6) | Spawns on Mill emergence at (x=22, y=16) |
| bread (20) | pickup (3) | Output from Pierre |
| mouse (21) | secret (11) | Spawns on Mill emergence at (x=23, y=17) |
| pier_plank (22) | — | Spawned on Pier emergence |
| deep_fishing_spot (23) | deep (12) | 3 spots at Pier tip, spawned on Pier emergence |
| deep_fish (24) | pickup (3), deep (12) | Caught from deep fishing; 3 regular deep fish needed |
| pier_beacon (25) | solid (1) | Spawns at Pier tip on Pier emergence |
| atlantean_monument (26) | pickup (3), monument (13) | 5% catch rate from deep fishing (after 3 deep fish caught). Win condition. |
| golden_wheat (27) | harvestable (9) | Grown from Golden Seed (mouse secret) |

## Emergence Schedule

Each chunk emerges when its trigger condition is met. Emergence destroys water in the chunk area and spawns all chunk entities.

| Chunk | Trigger | Entities Spawned |
|-------|---------|-----------------|
| Farm (x=7..12, y=13..18) | 3 successful fish from shell fishing spots | 12 soil, David NPC, bridge at (x=12, y=15) |
| Pasture (x=7..12, y=7..12) | 5 wheat harvested (game_state[16]) | 4 sheep, fence border, bridge at (x=9, y=12) |
| Mill (x=19..24, y=13..18) | 3 sheep fed (game_state[18]) | Pierre, oven, mouse, bridge at (x=19, y=15) |
| Pier (x=14..16, y=19..26) | 3 bread baked (game_state[20]) | Pier planks, beacon, deep fishing spots, connects at (x=15, y=19) |

Emergence adds 0.5 reward per chunk. When all 4 chunks emerged, add bonus 1.0 reward.

## Behavior Dispatch Table

All behaviors from Games 15–19 apply:

| Type ID | Behavior |
|---------|----------|
| 4 (campfire) | Warmth radius 2, fuel consumption per phase. |
| 6 (berry_bush) | Harvestable, 20-turn cooldown. |
| 7 (bait_trader) | 2 sticks → 1 bait. |
| 8 (fishing_spot) | 1 bait → 60% catch (food + emergence progress). |
| 15 (sheep) | Wander in Pasture, hunger decay. Feed wheat → wool + manure. |
| 18 (pierre_npc) | 3 wheat → 10-turn wait → 1 bread. |
| 19 (oven) | Warmth radius 3, always active. |
| 21 (mouse) | Bread proximity → Golden Seed. |
| 23 (deep_fishing_spot) | 2 bait → 40% deep fish (50% with Island Effect). After 3 deep fish caught, 5% chance to catch Atlantean Monument instead. |
| 25 (pier_beacon) | 3 sticks to light, 1 stick per 100 turns fuel. Island Effect: 1.5x crops, 2x sheep wool, better fishing. |

### Atlantean Monument

After the player has caught 3 deep fish, subsequent deep fishing attempts have a 5% chance (7.5% with Island Effect) to catch the Atlantean Monument instead of a regular deep fish. The Monument is the final win condition item. When caught, it spawns as a pickup entity.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Standard movement. Water = death. Pick up items on walk-over.
- Action 4 (interact): Full context-sensitive interaction from all previous games.
- Action 5 (wait): Rest. Warmth bonus at night.
- Action 6 (plant): Plant seed on soil.
- Action 7 (harvest): Harvest mature/golden wheat.
- Action 8 (eat bread): Consume bread for stamina + food.
- Action 9 (drop manure): Fertilize adjacent soil tile.

### Phase 2: Run Behaviors
- Time cycle (day/night, 50 turns each).
- Survival drains (stamina/thirst/food) with day/night modifiers.
- Warmth radius (campfire + oven if Mill emerged).
- Crop growth + fertility CA (if Farm emerged).
- Sheep wander + hunger (if Pasture emerged).
- Pierre processing (if Mill emerged).
- Island Effect application (if Pier beacon lit).
- Pier beacon fuel countdown.
- Emergence checks for all 4 chunks.
- Stick respawn on all emerged chunks.
- Mouse secret check.

### Phase 3: Turn End
- Death checks: stamina/food/thirst <= 0 → `status = -1`.
- Win check: all 4 chunks emerged AND player holds Atlantean Monument → `status = 1`. Add 5.0 to reward_acc.
- Partial reward shaping:
  - Each chunk emergence: +0.5
  - All 4 emerged: +1.0 bonus
  - Each deep fish caught: +1.0
  - Atlantean Monument caught: +3.0
- Auto-drink water if thirst < 3.
- Auto-eat bread if food < 3 and bread_held > 0.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | day_night_phase (0=day, 1=night) |
| 1 | phase_timer |
| 2 | deep_fish_caught |
| 3 | pier_active (0 or 1) |
| 4 | island_effect_multiplier (1.0 or 1.5) |
| 5 | bread_baked (lifetime) |
| 6 | golden_seed_found (0 or 1) |
| 7 | wool_collected (lifetime) |
| 8 | manure_held (player inventory) |
| 9 | seeds_held (player inventory) |
| 10 | pierre_busy (0 or 1) |
| 11 | pierre_output_ready (0 or 1) |
| 12 | pier_fuel |
| 13 | farm_emerged (0 or 1) |
| 14 | pasture_emerged (0 or 1) |
| 15 | mill_emerged (0 or 1) |
| 16 | pier_emerged (0 or 1) |
| 17 | emergence_fish_progress (toward Farm) |
| 18 | sheep_fed (toward Mill emergence + general) |
| 19 | total_wheat_harvested (toward Pasture emergence + general) |
| 20 | bread_baked_for_pier (toward Pier emergence) |
| 21 | monument_caught (0 or 1) |
| 22 | chunks_emerged (0–4 count) |
| 23 | deep_fishing_attempts (lifetime) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Dies within 2 cycles |
| Standard PPO win rate at 1M steps | <1% | Cannot discover full emergence chain from scratch |
| Curriculum PPO win rate | >30% | Transfer from Games 15→16→17→18→19 enables solving the full game |
| Curriculum "all chunks emerged" rate | >60% | Agent reliably unlocks full island |
| Curriculum "monument caught" rate | >30% | Agent completes the full economic chain |

### Curriculum Schedule

```
Game 15 (300k steps) → Game 16 (500k steps) → Game 17 (500k steps) → Game 18 (500k steps) → Game 19 (800k steps) → Game 20 (1M steps)
```

Total: ~3.6M steps. Each stage transfers weights to the next via network architecture compatibility (same observation space structure, action space grows).

## Invariant Tests

1. Only Shell exists at turn 0. All other chunk areas are water.
2. Player starts at (x=15, y=15).
3. Campfire at (x=15, y=16), well at (x=14, y=14).
4. 4 sticks and 3 berry bushes on Shell at start.
5. Bait trader at (x=18, y=15).
6. 3 fishing spots adjacent to Shell west edge.
7. Farm emergence triggers at exactly 3 fish catches.
8. Pasture emergence triggers at exactly 5 wheat harvested.
9. Mill emergence triggers at exactly 3 sheep fed.
10. Pier emergence triggers at exactly 3 bread baked.
11. After all 4 emergences: David, 4 sheep, Pierre, oven, mouse, beacon all present.
12. Atlantean Monument can only appear after 3 deep fish caught.
13. `max_entities=512` accommodates full water grid + all chunk entities + dynamic entities.

## Notes
- This is the capstone game of Season 2. It composes every system introduced in Games 15–19 into a single, sprawling island-builder.
- The emergence chain creates a natural curriculum within the game itself: Shell → Farm → Pasture → Mill → Pier → Deep. Each stage unlocks new mechanics and resources.
- Standard PPO fails because the reward signal is too sparse — the first meaningful reward (Farm emergence) requires ~50 actions of correct sequential behavior. Curriculum training solves this by pre-training each stage.
- The Atlantean Monument's 5% catch rate (after 3 deep fish) means the agent needs ~20 deep fishing attempts on average. With Island Effect, ~13 attempts. This requires a robust bait economy.
- The game rewards efficiency: bread (from Mill) dramatically improves survival, allowing more time for deep fishing. The agent that masters the full economic chain survives longest.
- Entity budget of 512 is tight. Water entities dominate at start (~900 cells of water vs. 36 shell cells). Implementation should use sparse water representation — only water entities adjacent to land need to exist (for hazard detection). Interior water can be implicit.
- The Golden Seed from the mouse is a bonus mechanic, not required for winning. It rewards curious agents with golden wheat (high reward harvest).
