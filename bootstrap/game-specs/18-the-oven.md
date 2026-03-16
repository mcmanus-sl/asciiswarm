# Game Spec 18: The Oven (Processing + Logistics)

## Overview
The island economy matures. A Mill chunk has emerged east of the Shell, containing Pierre the Miller and an oven. The player harvests wheat, delivers it to Pierre for milling, waits for processing, and collects bread — a super-food worth 5x stamina. A hidden mechanic rewards curiosity: placing bread near a Mouse entity spawns a Golden Seed that grows into a golden wheat crop worth massive points.

This game introduces the `processing` system: drop-off resources, wait N turns, collect output. It tests spatial logistics — the agent must efficiently route between Farm, Mill, and Pasture.

## Grid
- Dimensions: 32x32
- Turtle Shell: 6x6 (x=13..18, y=13..18)
- Farm Chunk: 6x6 (x=7..12, y=13..18)
- Pasture Chunk: 6x6 (x=7..12, y=7..12)
- Mill Chunk: 6x6 (x=19..24, y=13..18)
- Bridge Shell↔Farm: (x=12, y=15)
- Bridge Farm↔Pasture: (x=9, y=12)
- Bridge Shell↔Mill: (x=19, y=15)
- All other cells are water

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=256,
    max_stack=2,
    num_entity_types=22,   # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush,
                           # 7=bait_trader, 8=fishing_spot, 9=bridge, 10=soil, 11=seed, 12=sprout,
                           # 13=mature_wheat, 14=david_npc, 15=sheep, 16=manure, 17=fence,
                           # 18=pierre_npc, 19=oven, 20=bread, 21=mouse
    num_tags=12,           # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth,
                           # 7=water_source, 8=farmable, 9=harvestable, 10=processor, 11=secret
    num_props=10,          # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held,
                           # 6=wool_held, 7=wheat_held, 8=bread_held, 9=processing_timer
    num_actions=8,
    max_turns=1000,
    step_penalty=-0.002,
    game_state_size=12,    # 0=day_night_phase, 1=phase_timer, 2=bread_baked, 3=golden_seed_found,
                           # 4=wheat_delivered, 5=bread_collected, 6=mouse_fed, 7=golden_wheat_harvested,
                           # 8=manure_held, 9=seeds_held, 10=pierre_busy, 11=pierre_output_ready
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
| 9 | processing_timer | pierre_npc: turns until output ready (0–10) |

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
| water (2) | hazard (2) | All cells outside land chunks |
| stick (3) | pickup (3) | 4 sticks on shell, respawn every 25 turns |
| campfire (4) | warmth (6), solid (1) | (x=15, y=16) on shell |
| well (5) | water_source (7), solid (1) | (x=14, y=14) on shell |
| berry_bush (6) | solid (1) | 3 bushes on shell |
| bait_trader (7) | npc (5), solid (1) | (x=18, y=15) on shell |
| fishing_spot (8) | — | 3 spots on water adjacent to shell |
| bridge (9) | — | 3 bridges connecting chunks |
| soil (10) | farmable (8) | 12 tiles on Farm chunk |
| seed–sprout–wheat (11–13) | — | Dynamic crop entities on Farm |
| david_npc (14) | npc (5), solid (1) | (x=9, y=15) on Farm |
| sheep (15) | npc (5) | 4 sheep on Pasture |
| manure (16) | pickup (3) | Dropped by sheep |
| fence (17) | solid (1) | Borders Pasture |
| pierre_npc (18) | npc (5), processor (10), solid (1) | (x=21, y=15) in Mill |
| oven (19) | solid (1), warmth (6) | (x=22, y=16) in Mill, cosmetic/warmth source |
| bread (20) | pickup (3) | Spawned by Pierre when processing completes |
| mouse (21) | secret (11) | (x=23, y=17) hidden in Mill corner, 1 entity |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (campfire) | Warmth emission, same as Game 15. |
| 6 (berry_bush) | Harvestable with cooldown, same as Game 15. |
| 15 (sheep) | Wander + hunger, same as Game 17. |
| 18 (pierre_npc) | **Processing**: When player interacts and wheat_held >= 3 and pierre_busy == 0: consume 3 wheat, set pierre_busy = 1, processing_timer = 10. Each turn, decrement timer. When timer hits 0, set pierre_output_ready = 1, pierre_busy = 0. Player interacts again to collect 1 bread. |
| 19 (oven) | **Warmth**: Emits warmth radius 3 in Mill chunk (computed via wave.py). Always active (no fuel cost). |
| 21 (mouse) | **Secret interaction**: If bread entity exists within Manhattan distance 1 of mouse, destroy the bread, spawn Golden Seed on mouse's cell. Mouse disappears. Set game_state[6] = 1. Add 0.5 to reward_acc. Golden Seed grows into golden_wheat in 20 turns (uses mature_wheat type with a special flag in prop — golden wheat harvests for 0.5 reward instead of normal). |

### Processing System

Pierre operates as a state machine tracked in game_state:
- `game_state[10]` (pierre_busy): 0 = idle, 1 = processing
- `game_state[11]` (pierre_output_ready): 0 = no output, 1 = bread ready for pickup
- Pierre's prop 9 (processing_timer): countdown from 10 to 0

The player must:
1. Deliver 3 wheat to Pierre (interact when adjacent, wheat_held >= 3).
2. Wait 10 turns (do other tasks — farm, tend sheep, survive).
3. Return to Pierre and interact to collect 1 bread.

Bread effect: consuming bread restores stamina += 10 (capped at 20) and food += 5 (capped at 10). Massively more efficient than berries.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Same movement rules. Walk over bread/manure to pick up.
- Action 4 (interact): Context-sensitive:
  - All previous interactions from Games 15–17.
  - Adjacent to pierre_npc (wheat_held >= 3, not busy): deliver wheat, start processing.
  - Adjacent to pierre_npc (output_ready == 1): collect bread. bread_held += 1. Add 0.3 to reward_acc.
  - Adjacent to mouse (bread_held > 0): drop bread near mouse → triggers Golden Seed.
- Action 5 (wait): Same.
- Action 6 (plant): Same.
- Action 7 (eat bread): If bread_held > 0, consume 1 bread. stamina += 10, food += 5 (capped). Add 0.1 to reward_acc.

### Phase 2: Run Behaviors
- Time cycle (day/night).
- Survival drains.
- Warmth radius (campfire + oven).
- Crop growth + fertility CA.
- Sheep wander + hunger.
- Pierre processing timer countdown.
- Mouse secret check (bread proximity).
- Stick respawn.

### Phase 3: Turn End
- Death checks: stamina/food/thirst <= 0 → `status = -1`.
- Win check: game_state[2] (bread_baked) >= 5 AND game_state[3] (golden_seed_found) == 1 → `status = 1`. Add 2.0 to reward_acc.
- Partial win: bread_baked >= 5 without Golden Seed → `status = 1` with 1.0 reward (lesser win).
- Auto-drink water.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | day_night_phase |
| 1 | phase_timer |
| 2 | bread_baked (lifetime total) |
| 3 | golden_seed_found (0 or 1) |
| 4 | wheat_delivered (to Pierre, lifetime) |
| 5 | bread_collected (from Pierre, lifetime) |
| 6 | mouse_fed (0 or 1) |
| 7 | golden_wheat_harvested (0 or 1) |
| 8 | manure_held (player inventory) |
| 9 | seeds_held (player inventory) |
| 10 | pierre_busy (0 or 1) |
| 11 | pierre_output_ready (0 or 1) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Cannot chain harvest→deliver→wait→collect |
| PPO win rate (bread only) at 500k steps | >30% | Agent masters spatial logistics: Farm↔Mill routing |
| PPO Golden Seed discovery | >10% | Agent discovers hidden mouse mechanic through exploration |
| PPO "efficient routing" metric | avg < 50 turns between bread collections | Agent optimizes travel between chunks |

## Invariant Tests

1. Four land chunks at specified positions.
2. Three bridges connecting chunks.
3. Pierre at (x=21, y=15), oven at (x=22, y=16).
4. Mouse at (x=23, y=17).
5. Pierre starts idle (not busy, no output ready).
6. Player starts at (x=15, y=15).
7. All Game 17 entities present (sheep, David, soil, etc.).

## Notes
- The processing system introduces **temporal delays** — the agent must plan around 10-turn wait periods. During the wait, the agent should tend other tasks (farming, sheep, survival).
- The 4-chunk layout creates a spatial logistics problem. The agent must route efficiently between Shell (survival), Farm (wheat), Pasture (wool/manure), and Mill (bread).
- The mouse secret is deliberately hidden — it rewards exploration. PPO may or may not discover it. If it does, that validates curiosity-driven behavior.
- Bread is a game-changer for survival: 1 bread = 10 stamina + 5 food. This incentivizes the processing pipeline heavily.
- The oven provides warmth in the Mill chunk, creating a second safe zone at night. The agent can split night time between campfire and oven.
- `max_entities=256` handles 4 chunks of water + terrain + dynamic entities.
