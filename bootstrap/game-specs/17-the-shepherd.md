# Game Spec 17: The Shepherd (Husbandry + Fertility)

## Overview
The island expands again. A Pasture chunk now sits north of the Farm, home to sheep. The player must grow wheat on the Farm, carry it to the Pasture to feed sheep, collect wool and manure, and use manure to accelerate crop growth via a fertility cellular automaton. The agent discovers an ecological feedback loop: wheat feeds sheep, manure from sheep fertilizes soil, fertilized soil grows wheat faster.

This game introduces the `husbandry` system (feed animals → produce resources) and the `fertility` CA (manure spreads fertility to adjacent soil).

## Grid
- Dimensions: 24x24
- Turtle Shell: 6x6 (x=9..14, y=9..14) — starting land
- Farm Chunk: 6x6 (x=3..8, y=9..14) — starts emerged (pre-unlocked from Game 16 progression)
- Pasture Chunk: 6x6 (x=3..8, y=3..8) — starts emerged
- Bridge 1: (x=8, y=11) connecting Shell to Farm
- Bridge 2: (x=5, y=8) connecting Farm to Pasture
- All other cells are water

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=24, grid_h=24,
    max_entities=192,
    max_stack=2,
    num_entity_types=18,   # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush,
                           # 7=bait_trader, 8=fishing_spot, 9=bridge, 10=soil, 11=seed, 12=sprout,
                           # 13=mature_wheat, 14=david_npc, 15=sheep, 16=manure, 17=fence
    num_tags=10,           # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth,
                           # 7=water_source, 8=farmable, 9=harvestable
    num_props=8,           # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held,
                           # 6=wool_held, 7=wheat_held
    num_actions=8,
    max_turns=800,
    step_penalty=-0.002,
    game_state_size=10,    # 0=day_night_phase, 1=phase_timer, 2=wool_collected, 3=wheat_surplus,
                           # 4=manure_placed, 5=fertile_tiles, 6=sheep_fed, 7=total_wheat_harvested,
                           # 8=manure_held, 9=seeds_held
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

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | food | player: hunger meter (0–10) |
| 1 | stamina | player: energy meter (0–20) |
| 2 | thirst | player: hydration meter (0–10) |
| 3 | fire_fuel | campfire: fuel (0–5) / sheep: hunger (0–5) / soil: fertility (0–3) |
| 4 | sticks_held | player: carried sticks (0–5) |
| 5 | water_held | player: carried water (0–3) |
| 6 | wool_held | player: carried wool (0–10) |
| 7 | wheat_held | player: carried wheat (0–10) |

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

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center of shell (x=11, y=11) |
| water (2) | hazard (2) | All cells outside land chunks |
| stick (3) | pickup (3) | 4 sticks on shell, respawn every 25 turns |
| campfire (4) | warmth (6), solid (1) | (x=11, y=12) on shell |
| well (5) | water_source (7), solid (1) | (x=10, y=10) on shell |
| berry_bush (6) | solid (1) | 3 bushes on shell |
| bait_trader (7) | npc (5), solid (1) | (x=14, y=11) on shell |
| fishing_spot (8) | — | 3 spots on water adjacent to shell west edge |
| bridge (9) | — | (x=8, y=11) shell↔farm, (x=5, y=8) farm↔pasture |
| soil (10) | farmable (8) | 12 tiles on Farm chunk |
| seed (11) | — | Planted by player |
| sprout (12) | — | Grows from seed |
| mature_wheat (13) | harvestable (9) | Grows from sprout |
| david_npc (14) | npc (5), solid (1) | (x=5, y=11) on Farm |
| sheep (15) | npc (5) | 4 sheep on Pasture chunk, wander randomly within Pasture bounds |
| manure (16) | pickup (3) | Dropped by fed sheep, 1 per feeding |
| fence (17) | solid (1) | Borders Pasture chunk edges (prevents sheep escape) |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (campfire) | Warmth emission, same as Game 15. |
| 6 (berry_bush) | Harvestable with 20-turn cooldown, same as Game 15. |
| 15 (sheep) | **Wander**: Each turn, 30% chance to move 1 cell in a random cardinal direction within Pasture bounds. **Hunger**: sheep have prop 3 (hunger) that decreases by 0.1 per turn. When hunger <= 0, sheep stops producing wool/manure when fed. Feeding resets hunger to 5. |
| 16 (manure) | **Pickup**: Player walks over manure to collect it (game_state[8] += 1). Manure entity destroyed. |

### Husbandry System

When player interacts adjacent to a sheep and wheat_held > 0:
1. Decrement wheat_held by 1.
2. Reset sheep hunger (prop 3) to 5.
3. Sheep produces: 1 wool (player wool_held += 1, capped at 10) and 1 manure (spawned on an adjacent empty cell).
4. Increment game_state[6] (sheep_fed).
5. Add 0.15 to reward_acc.

### Fertility CA System

When player drops manure on a soil tile (interact on soil while manure_held > 0):
1. Soil's prop 3 (fertility) set to 3.
2. **Spread**: Each turn, fertile soil (prop 3 > 0) has a 30% chance to spread fertility to each adjacent soil tile (set adjacent soil's prop 3 to min(current + 1, 3)).
3. **Effect**: Fertile soil (prop 3 > 0) halves crop growth time — seeds grow in 8 turns instead of 15, sprouts mature in 8 turns instead of 15.
4. Fertility decrements by 0.1 per turn (fades over time).

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Same as Game 15/16. Walk over manure to pick up.
- Action 4 (interact): Context-sensitive:
  - Adjacent to campfire/well/berry_bush/bait_trader/fishing_spot: same as Game 16.
  - Adjacent to sheep (with wheat_held > 0): feed sheep → wool + manure.
  - Adjacent to soil (with seed): plant seed.
  - Adjacent to soil (with manure_held > 0): drop manure, fertilize tile.
  - Adjacent to mature_wheat: harvest → wheat_held += 1.
  - Adjacent to david_npc: receive 2 seeds.
- Action 5 (wait): Same as Game 15.
- Action 6 (plant): Plant seed on current soil tile.
- Action 7 (harvest): Harvest wheat on current tile.

### Phase 2: Run Behaviors
- Time cycle (day/night 50-turn phases).
- Survival drains (stamina/thirst/food).
- Warmth radius.
- Stick respawn.
- Sheep wander + hunger decay.
- Crop growth (affected by fertility).
- Fertility CA spread + decay.

### Phase 3: Turn End
- Death checks: stamina/food/thirst <= 0 → `status = -1`.
- Win check: wool_held + game_state[2] >= 5 AND game_state[7] >= 20 → `status = 1`. Add 1.5 to reward_acc.
- Auto-drink water.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | day_night_phase (0=day, 1=night) |
| 1 | phase_timer |
| 2 | wool_collected (lifetime banked — deposited wool, not currently held) |
| 3 | wheat_surplus (wheat harvested minus wheat fed to sheep) |
| 4 | manure_placed (lifetime total) |
| 5 | fertile_tiles (current count of soil with fertility > 0) |
| 6 | sheep_fed (lifetime total) |
| 7 | total_wheat_harvested (lifetime total) |
| 8 | manure_held (player inventory, not a prop — tracked in game_state) |
| 9 | seeds_held (player inventory, tracked in game_state) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Cannot chain wheat→sheep→wool loop |
| PPO win rate at 500k steps | >20% | Agent discovers ecological feedback loop |
| PPO "use manure" rate | >40% of wins | Agent fertilizes soil — validates fertility CA design |
| PPO wheat→sheep conversion | >10 feedings per win | Agent actively feeds sheep rather than hoarding wheat |

## Invariant Tests

1. Shell at (x=9..14, y=9..14), Farm at (x=3..8, y=9..14), Pasture at (x=3..8, y=3..8).
2. Bridges at (x=8, y=11) and (x=5, y=8).
3. 4 sheep on Pasture at start.
4. Fence borders Pasture edges.
5. 12 soil tiles on Farm.
6. Player starts at (x=11, y=11) with same initial props as Game 15.
7. All sheep start with hunger = 5.

## Notes
- The ecological loop is the emergent behavior: wheat → sheep → manure → fertile soil → faster wheat → more sheep feedings. The agent that discovers this loop wins faster.
- Sheep wander adds spatial unpredictability — the agent must chase sheep within the Pasture.
- Fertility CA creates a spreading benefit zone, rewarding strategic manure placement (center of Farm spreads to more tiles).
- The win condition requires both wool (5) and wheat surplus (20 harvested total), forcing the agent to balance farming and husbandry.
- `manure_held` and `seeds_held` are in game_state rather than props due to prop budget limits. They are included in the scalar observation vector.
