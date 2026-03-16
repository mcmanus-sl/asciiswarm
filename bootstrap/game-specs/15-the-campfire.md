# Game Spec 15: The Campfire (Survival Rhythm)

## Overview
The Turtle's shell floats in a vast sea. The player must survive 4 full day/night cycles on a tiny island by mastering a survival rhythm: gather sticks and berries during the day, feed the campfire and huddle near it at night. Night drains stamina brutally unless the player stays within the fire's warmth radius. Water must be drawn from the well. The agent must learn a rhythmic loop — hoard resources by day, conserve by night.

This is the first game of **Season 2: Fishing for Islands**. It introduces the `time_cycle` and `fire` systems.

## Grid
- Dimensions: 16x16
- Center 6x6 (x=5..10, y=5..10) is the Turtle shell (land). All other cells are water.

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=32,
    max_stack=2,
    num_entity_types=7,    # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush
    num_tags=8,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth, 7=water_source
    num_props=6,           # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held
    num_actions=6,
    max_turns=400,
    step_penalty=-0.002,
    game_state_size=6,     # 0=day_night_phase, 1=phase_timer, 2=sticks_gathered, 3=fires_fed, 4=berries_eaten, 5=water_drawn
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

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | food | player: hunger meter (0–10) |
| 1 | stamina | player: energy meter (0–20) |
| 2 | thirst | player: hydration meter (0–10) |
| 3 | fire_fuel | campfire: remaining fuel (0–5) |
| 4 | sticks_held | player: carried sticks (0–5) |
| 5 | water_held | player: carried water units (0–3) |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| food (prop 0) | 10.0 |
| stamina (prop 1) | 20.0 |
| thirst (prop 2) | 10.0 |
| sticks_held (prop 4) | 5.0 |
| water_held (prop 5) | 3.0 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center of shell (x=7, y=7) |
| water (2) | hazard (2) | All cells outside the 6x6 shell |
| stick (3) | pickup (3) | 4 sticks scattered on shell edge cells, respawn every 25 turns on random empty shell cells |
| campfire (4) | warmth (6), solid (1) | Center of shell (x=7, y=8), permanent |
| well (5) | water_source (7), solid (1) | (x=6, y=6), permanent |
| berry_bush (6) | solid (1) | 3 bushes on shell cells; interact to harvest 1 food (cooldown: 20 turns per bush, tracked via prop 3 overloaded as cooldown) |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (campfire) | **Warmth emission**: When fire_fuel > 0, emits warmth in a radius of 2 cells (Manhattan distance). Warmth radius computed via wave.py flood-fill from campfire position, blocked by nothing (warmth passes through all entities). fire_fuel decrements by 1 every 50 turns (once per phase transition). When fire_fuel = 0, no warmth emitted. |
| 6 (berry_bush) | **Regrowth**: Each bush has a cooldown counter. After harvest, cooldown = 20 turns. Decrements each turn. When cooldown = 0, bush is harvestable again. |

### Time Cycle System

The game alternates between day (50 turns) and night (50 turns):
- `game_state[0]`: 0 = day, 1 = night
- `game_state[1]`: phase timer (counts down from 50 to 0, then toggles phase)

**Day effects**:
- Stamina drain: -0.1 per turn (light activity)
- Thirst drain: -0.1 per turn
- Food drain: -0.05 per turn
- Sticks can be picked up

**Night effects**:
- Stamina drain: -0.5 per turn if OUTSIDE warmth radius, -0.1 per turn if INSIDE warmth radius
- Thirst drain: -0.15 per turn
- Food drain: -0.1 per turn
- Sticks still visible but moving costs double stamina (additional -0.2 per move)

### Warmth Radius via wave.py

The campfire's warmth radius is computed each turn using a Manhattan distance flood-fill from the campfire's position, with a max radius of 2. All cells within this radius are considered "warm". The player's position is checked against this set to determine night stamina drain rate.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Move player on shell. Moving into water → `status = -1`. Moving into solid → cancel move.
  - If target cell has a stick (pickup): pick up stick, increment sticks_held (capped at 5). Destroy stick entity.
  - Night movement: additional -0.2 stamina cost per move.
- Action 4 (interact): Context-sensitive based on adjacent entity:
  - Adjacent to campfire: If sticks_held > 0, decrement sticks_held by 1, increment fire_fuel by 1 (capped at 5). Add 0.1 to `reward_acc`.
  - Adjacent to well: If water_held < 3, increment water_held by 1. Thirst += 3 (capped at 10). Add 0.05 to `reward_acc`.
  - Adjacent to berry_bush (cooldown = 0): Harvest 1 food (capped at 10). Set bush cooldown = 20. Add 0.05 to `reward_acc`.
- Action 5 (wait): No-op. If inside warmth radius at night, add 0.01 to `reward_acc` (encourages huddling).

### Phase 2: Run Behaviors
- Decrement phase timer. On phase transition (timer hits 0):
  - Toggle day/night phase.
  - Reset timer to 50.
  - Campfire: decrement fire_fuel by 1 (phase transition fuel cost).
- Apply time-of-day drains to stamina, thirst, food.
- Compute warmth radius from campfire.
- Apply warmth-dependent stamina drain modifier.
- Berry bush cooldown decrement.
- Stick respawn: every 25 turns, if fewer than 4 sticks alive on shell, spawn 1 stick on a random empty shell cell.

### Phase 3: Turn End
- If stamina <= 0 → `status = -1` (exhaustion death).
- If food <= 0 → `status = -1` (starvation death).
- If thirst <= 0 → `status = -1` (dehydration death).
- If turn >= 200 → `status = 1` (survived 4 full day/night cycles). Add 1.0 to `reward_acc`.
- Drink water: if water_held > 0 and thirst < 3, auto-consume 1 water_held, thirst += 3.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | day_night_phase (0=day, 1=night) |
| 1 | phase_timer (counts down from 50) |
| 2 | sticks_gathered (lifetime total) |
| 3 | fires_fed (lifetime total) |
| 4 | berries_eaten (lifetime total) |
| 5 | water_drawn (lifetime total) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Dies from meter depletion within 2 cycles |
| PPO win rate at 300k steps | >25% | Agent learns basic survival loop |
| PPO "huddle at night" rate | >60% of nights | Agent stays within warmth radius during night phases — validates rhythm learning |
| PPO "feed fire before night" rate | >50% | Agent feeds campfire during day in anticipation of night |

## Invariant Tests

1. Shell is exactly 6x6 at center (x=5..10, y=5..10).
2. All cells outside shell are water (type 2).
3. Campfire at (7, 8), well at (6, 6).
4. Player starts at (7, 7).
5. Exactly 4 sticks at start.
6. Exactly 3 berry bushes at start.
7. Player starts with: food=8, stamina=20, thirst=8, sticks_held=0, water_held=0.
8. fire_fuel starts at 3.

## Notes
- The tiny island creates a tight resource loop. The agent cannot wander — it must optimize pathing within 36 cells.
- The warmth radius mechanic forces spatial awareness: the agent must be near the fire at night.
- The rhythmic day/night cycle is the core learning signal. Random agents die because they don't adapt behavior to the phase.
- `prop_maxes = (10.0, 20.0, 10.0, 5.0, 5.0, 3.0)` — used for observation normalization.
- Stick respawn prevents resource exhaustion but requires active gathering.
- Win condition at turn 200 (not 400) means max_turns=400 is a hard ceiling — the agent should win well before timeout.
