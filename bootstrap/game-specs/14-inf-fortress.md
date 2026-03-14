# Game Spec 14: INF FORTRESS (The Sandbox)

## Overview
The capstone environment. A large-scale sandbox where hunger, farming, mining, fluid dynamics, psychology, and sieges run concurrently. The player must establish a sustainable loop: mine ore → craft tools → dig moats → farm food → build defenses → hoard wealth. All mechanics from specs 04, 07, 09, 10, 11, 12, and 13 overlap simultaneously.

This spec tests whether the RL agent can learn a multi-system survival loop — the "Dwarf Fortress" experience.

## Grid
- Dimensions: 32×32

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=512,      # large grid, many systems running simultaneously
    max_stack=4,           # multiple entities per cell (trap under goblin, sprout on soil, etc.)
    num_entity_types=16,   # see enum below
    num_tags=10,           # expanded tag set
    num_props=8,           # many properties per entity
    num_actions=8,         # expanded action set
    max_turns=1000,
    step_penalty=-0.002,
    game_state_size=16,    # many counters
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | rock | `%` |
| 3 | adamantine | `$` |
| 4 | magma | `~` |
| 5 | wall | `#` |
| 6 | soil | `=` |
| 7 | sprout | `,` |
| 8 | mature | `*` |
| 9 | food | `f` |
| 10 | workbench | `W` |
| 11 | keg | `&` |
| 12 | goblin | `g` |
| 13 | trap | `^` |
| 14 | barricade | `X` |
| 15 | vault | `V` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | npc |
| 6 | mineable |
| 7 | pushable |
| 8 | trap |
| 9 | defense |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | health/food | player: food level (0–30), goblin: HP |
| 1 | stress | player: 0–20 |
| 2 | ore | player: carried ore (0–10) |
| 3 | stone | player: carried stone (0–20) |
| 4 | has_pickaxe | player: 0 or 1 |
| 5 | wealth | player: deposited in vault |
| 6 | age/charges | sprout: growth age; keg: drink charges |
| 7 | direction | goblin: movement direction |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| health/food (prop 0) | 30 |
| stress (prop 1) | 20 |
| ore (prop 2) | 10 |
| stone (prop 3) | 20 |
| has_pickaxe (prop 4) | 1 |
| wealth (prop 5) | 100 |

## Action Mapping

| Action | Name |
|--------|------|
| 0 | move_n |
| 1 | move_s |
| 2 | move_e |
| 3 | move_w |
| 4 | interact |
| 5 | wait |
| 6 | build_barricade |
| 7 | build_trap |

## Map Zones

The 32×32 grid is divided into zones:

| Zone | Location | Contents |
|------|----------|----------|
| Safe zone | Top-left (x < 8, y < 8) | Player start, workbench, keg |
| Farm zone | Top-center (x=10–16, y=2–8) | Soil patches |
| Mine zone | Center-right (x=16–28, y=8–24) | Rock, adamantine, magma pocket |
| Vault zone | Bottom-left (x=2–6, y=26–30) | Vault tile |
| Invasion edge | Right edge (x=31) | Goblin spawn points |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (magma) | CA spread: 20% chance to adjacent empty cells. Cannot enter solid. Destroys pickups. |
| 7 (sprout) | Age +1 each turn. At age 20, transform to mature. |
| 11 (keg) | Regenerate 1 charge every 30 turns. Max 3. |
| 12 (goblin) | Move toward player (greedy Manhattan). Destroy barricades/traps on contact. |

## Turn Phases

### Phase 1: Process Input
- Standard movement, mining, pickup collection.
- Interact: context-dependent (workbench → craft pickaxe, vault → deposit ore as wealth, soil → plant).
- Build barricade (action 6): costs 3 stone.
- Build trap (action 7): costs 2 ore.

### Phase 2: Run Behaviors
- Magma CA spread.
- Sprout growth.
- Keg regeneration.
- Goblin movement + combat.
- **Goblin spawning**: 3 goblins spawn at right edge every 100 turns (turns 100, 200, 300, ...).

### Phase 3: Turn End
- Hunger tick: food -= 1 each turn. If food ≤ 0, `status = -1`.
- Stress tick: stress += 1 every 10 turns.
- Tantrum check: if stress ≥ 20, tantrum mode for 10 turns.
- Win check: if wealth ≥ 100, `status = 1`.

## Win Condition
Deposit 100 wealth in the vault before turn 1000.

## Lose Condition
- Starve (food ≤ 0).
- Killed by goblin.
- Killed by magma.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | wealth_deposited |
| 1 | goblins_killed |
| 2 | food_eaten |
| 3 | barricades_built |
| 4 | traps_built |
| 5 | tantrum_count |
| 6 | magma_cells |
| 7 | adamantine_mined |
| 8 | crops_harvested |
| 9 | turn_next_invasion |
| 10–15 | (reserved) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Standard PPO win rate | <1% | Fails to link the multi-step resource loop |
| Curriculum PPO win rate | >40% | Agent transfers policies from specs 01–13 |
| Weaponized fluids | >1.0 per run | Agent mines a path to release magma onto goblins instead of fighting |

## Notes

### Curriculum Learning

Standard PPO will fail on this spec — the 12-step resource loop is too long for random exploration to discover. The curriculum pipeline:
1. Pre-train on specs 01–03 (basic navigation).
2. Fine-tune on specs 07 + 10 (hunger + farming).
3. Fine-tune on specs 11 + 13 (mining + siege).
4. Fine-tune on spec 14.

### The "Dwarf Fortress" Metric

The CI pipeline passes ONLY if the agent demonstrates at least one instance of "weaponized fluids" — intentionally mining a path to release magma onto approaching goblins instead of fighting. This validates that the overlapping systems create emergent strategic options.

### Entity Budget

`max_entities=512` is large but necessary: the 32×32 grid with rock fill, multiple magma cells, farm sprouts, and periodic goblin waves all need slots. The alive mask ensures only active entities consume computation.

### Complexity Management

This spec intentionally combines everything. If the agent can't converge, the spec should be simplified (reduce grid size, fewer concurrent systems, longer turn limit) before declaring the engine broken. The spec is the hardest test — its purpose is to stress-test emergence, not to be easily solvable.
