# Game Spec 16: The Golden Field (Emergence + Farming)

## Overview
The island grows. The player starts on the familiar Turtle shell but can now trade sticks for bait, fish at fishing spots, and — through fishing — cause a submerged Farm chunk to emerge from the water. Once the Farm is revealed, the player plants wheat seeds and tends crops alongside David, a friendly NPC. The agent must chain a multi-step economic loop: survive → gather sticks → trade for bait → fish → unlock Farm → plant → harvest wheat.

This game introduces the `emergence` system: fishing at specific spots unmasks submerged terrain, expanding the playable area.

## Grid
- Dimensions: 24x24
- Turtle Shell: center 6x6 (x=9..14, y=9..14) — starting land
- Farm Chunk: 6x6 (x=3..8, y=9..14) — submerged (water) at start, emerges after 3 successful fishing actions
- All other cells are water

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=24, grid_h=24,
    max_entities=128,
    max_stack=2,
    num_entity_types=15,   # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush,
                           # 7=bait_trader, 8=fishing_spot, 9=bridge, 10=soil, 11=seed, 12=sprout,
                           # 13=mature_wheat, 14=david_npc
    num_tags=10,           # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth,
                           # 7=water_source, 8=farmable, 9=harvestable
    num_props=8,           # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held,
                           # 6=bait, 7=wheat_harvested
    num_actions=8,
    max_turns=600,
    step_penalty=-0.002,
    game_state_size=8,     # 0=day_night_phase, 1=phase_timer, 2=fish_caught, 3=farm_emerged,
                           # 4=seeds_planted, 5=wheat_harvested, 6=bait_traded, 7=emergence_progress
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
| 3 | fire_fuel | campfire: remaining fuel (0–5) |
| 4 | sticks_held | player: carried sticks (0–5) |
| 5 | water_held | player: carried water units (0–3) |
| 6 | bait | player: bait for fishing (0–3) |
| 7 | wheat_harvested | player: harvested wheat count (0–15) |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| food (prop 0) | 10.0 |
| stamina (prop 1) | 20.0 |
| thirst (prop 2) | 10.0 |
| sticks_held (prop 4) | 5.0 |
| water_held (prop 5) | 3.0 |
| bait (prop 6) | 3.0 |
| wheat_harvested (prop 7) | 15.0 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center of shell (x=11, y=11) |
| water (2) | hazard (2) | All cells outside shell (and Farm chunk before emergence) |
| stick (3) | pickup (3) | 4 sticks on shell, respawn every 25 turns |
| campfire (4) | warmth (6), solid (1) | (x=11, y=12) on shell |
| well (5) | water_source (7), solid (1) | (x=10, y=10) on shell |
| berry_bush (6) | solid (1) | 3 bushes on shell |
| bait_trader (7) | npc (5), solid (1) | (x=14, y=11) on shell east edge |
| fishing_spot (8) | — | 3 spots on water cells adjacent to shell's west edge (x=8, y=10..12) |
| bridge (9) | — | Spawned during emergence connecting shell to Farm chunk (x=8, y=11) |
| soil (10) | farmable (8) | Spawned on Farm chunk cells after emergence (up to 12 soil tiles) |
| seed (11) | — | Planted by player on soil |
| sprout (12) | — | Grows from seed after 15 turns |
| mature_wheat (13) | harvestable (9) | Grows from sprout after 15 more turns |
| david_npc (14) | npc (5), solid (1) | Spawns on Farm chunk after emergence at (x=5, y=11) |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (campfire) | Same as Game 15: warmth emission radius 2, fuel decrement on phase transition. |
| 6 (berry_bush) | Same as Game 15: harvestable with 20-turn cooldown. |
| 7 (bait_trader) | **Trade**: When player interacts adjacent, if sticks_held >= 2, consume 2 sticks, grant 1 bait. |
| 8 (fishing_spot) | **Fish**: When player interacts adjacent and bait > 0, consume 1 bait, 60% chance to catch fish (food += 3, add 0.2 to reward_acc). Each successful catch increments game_state[7] (emergence_progress). |
| 11 (seed) | **Growth timer**: After 15 turns, transform to sprout (type 12). |
| 12 (sprout) | **Growth timer**: After 15 more turns, transform to mature_wheat (type 13), add harvestable tag. |
| 14 (david_npc) | **David's guidance**: If player is adjacent and interacts, David gives 1 seed (if player has none). David is cosmetic otherwise — represents the NPC from the story. |

### Emergence System

- `game_state[7]` tracks emergence_progress (successful fish catches).
- When emergence_progress reaches 3, trigger Farm emergence:
  1. Destroy all water entities in the Farm chunk area (x=3..8, y=9..14).
  2. Spawn bridge entity at (x=8, y=11) connecting shell to Farm.
  3. Spawn 12 soil entities on Farm chunk.
  4. Spawn David NPC at (x=5, y=11).
  5. Set game_state[3] = 1 (farm_emerged).
  6. Add 0.5 to reward_acc.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Same as Game 15. Moving into water → `status = -1`.
  - Picking up sticks on contact.
  - Night movement stamina penalty applies.
- Action 4 (interact): Context-sensitive:
  - Adjacent to campfire: feed stick (same as Game 15).
  - Adjacent to well: draw water (same as Game 15).
  - Adjacent to berry_bush: harvest food (same as Game 15).
  - Adjacent to bait_trader: trade 2 sticks → 1 bait. Add 0.1 to reward_acc.
  - Adjacent to fishing_spot: if bait > 0, fish (see behavior table).
  - Adjacent to soil: if player has seed, plant seed (replace soil with seed entity). Add 0.1 to reward_acc.
  - Adjacent to mature_wheat: harvest wheat. Increment wheat_harvested. Destroy wheat, restore soil. Add 0.15 to reward_acc.
  - Adjacent to david_npc: receive 1 seed if player holds 0 seeds. (Seeds tracked via game_state, not a prop — implicit in planting action.)
- Action 5 (wait): Same as Game 15.
- Action 6 (plant): Plant seed on current tile if it is soil. Alternative to interact-based planting.
- Action 7 (harvest): Harvest mature_wheat on current tile. Alternative to interact-based harvesting.

### Phase 2: Run Behaviors
- Time cycle system (same as Game 15): day/night alternation, 50 turns each.
- Stamina/thirst/food drains (same as Game 15).
- Warmth radius computation (same as Game 15).
- Stick respawn on shell (same as Game 15).
- Crop growth: advance seed → sprout → mature_wheat timers.
- Emergence check: if emergence_progress >= 3 and farm_emerged == 0, trigger emergence.

### Phase 3: Turn End
- Death checks: stamina/food/thirst <= 0 → `status = -1`.
- Win check: wheat_harvested >= 10 → `status = 1`. Add 1.0 to reward_acc.
- Auto-drink water if thirst < 3 and water_held > 0.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | day_night_phase (0=day, 1=night) |
| 1 | phase_timer (counts down from 50) |
| 2 | fish_caught (lifetime total) |
| 3 | farm_emerged (0 or 1) |
| 4 | seeds_planted (lifetime total) |
| 5 | wheat_harvested (lifetime total) |
| 6 | bait_traded (lifetime total) |
| 7 | emergence_progress (fish catches toward Farm unlock) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Cannot chain multi-step economic loop |
| PPO win rate at 500k steps | >40% | Agent solves temporal credit assignment across gather→trade→fish→farm→harvest |
| PPO emergence unlock rate | >70% | Agent reliably fishes 3 times to unlock Farm |
| PPO "feed fire before night" rate | >50% | Survival rhythm from Game 15 persists |

## Invariant Tests

1. Shell is 6x6 at (x=9..14, y=9..14).
2. Farm chunk area is (x=3..8, y=9..14), starts as water.
3. Fishing spots at (x=8, y=10..12) — adjacent to shell's west edge.
4. Bait trader at (x=14, y=11).
5. Campfire at (x=11, y=12), well at (x=10, y=10).
6. Player starts at (x=11, y=11).
7. Emergence triggers at exactly 3 fish catches.
8. After emergence: 12 soil tiles, 1 bridge, 1 David NPC present.
9. Player starts with same initial props as Game 15.

## Notes
- The emergence mechanic is the key innovation: the player literally fishes new land into existence. This is implemented by destroying water entities and spawning terrain entities, all within fixed entity budget.
- The multi-step economic chain (sticks → bait → fish → emergence → seeds → wheat) creates deep temporal credit assignment. Random agents cannot stumble into this.
- David NPC is a seed dispenser — keeps the loop flowing without requiring seed crafting.
- Crop growth takes 30 total turns (15 seed + 15 sprout), so the agent must plant early and manage time.
- `max_entities=128` accommodates water tiles (most of the 24x24 grid) plus dynamic farm entities.
- The fishing 60% success rate adds stochasticity that the agent must account for (may need 5 bait to get 3 catches).
