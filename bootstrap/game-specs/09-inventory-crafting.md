# Game Spec 09: Inventory & Crafting

## Overview
The player collects raw materials (wood, ore), brings them to a workbench, crafts a pickaxe, and mines through a rubble wall to reach the exit. Tests multi-slot inventory, crafting recipes, and workbench interaction.

## Grid
- Dimensions: 16×16

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=96,       # outer walls + rubble wall + wood + ore + workbench + player + exit
    max_stack=2,
    num_entity_types=8,    # 0=unused, 1=player, 2=exit, 3=wall, 4=wood, 5=ore, 6=workbench, 7=rubble
    num_tags=6,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc
    num_props=4,           # 0=wood, 1=ore, 2=has_pickaxe, 3=unused
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=4,     # 0=wood_collected, 1=ore_collected, 2=pickaxe_crafted, 3=rubble_mined
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | wall | `#` |
| 4 | wood | `t` |
| 5 | ore | `o` |
| 6 | workbench | `W` |
| 7 | rubble | `%` |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | wood | player: carried wood (0–5) |
| 1 | ore | player: carried ore (0–5) |
| 2 | has_pickaxe | player: 0 or 1 |
| 3 | (unused) | — |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| wood (prop 0) | 5 |
| ore (prop 1) | 5 |
| has_pickaxe (prop 2) | 1 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Random empty cell, left third (x < 5) |
| wall (3) | solid (1) | Outer boundary |
| wood (4) | pickup (3) | 4–6 scattered randomly |
| ore (5) | pickup (3) | 3–5 scattered randomly |
| workbench (6) | npc (5) | Single, center area (6 ≤ x ≤ 9, 6 ≤ y ≤ 9) |
| rubble (7) | solid (1) | Vertical wall at x=12, one gap blocked by rubble (1 tile) |
| exit (2) | exit (4) | Random empty cell, right section (x ≥ 13) |

## Behaviors

None.

## Crafting Recipes

| Recipe | Ingredients | Product | Where |
|--------|-------------|---------|-------|
| Pickaxe | 2 wood + 2 ore | has_pickaxe = 1 | Adjacent to workbench, via interact |

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Check target cell:
  - Solid → cancel.
  - Pickup (wood/ore) → move succeeds, increment relevant property (cap at 5), destroy pickup, add 0.05 to `reward_acc`.
  - Exit → move succeeds, `status = 1`.
- Action 4 (interact):
  1. Check 4 cardinal neighbors for workbench (type 6). If found AND wood ≥ 2 AND ore ≥ 2: wood -= 2, ore -= 2, has_pickaxe = 1. Add 0.3 to `reward_acc`. Update `game_state[2] = 1`.
  2. Else check 4 cardinal neighbors for rubble (type 7). If found AND has_pickaxe == 1: destroy rubble, has_pickaxe = 0 (breaks). Add 0.3 to `reward_acc`. Update `game_state[3] = 1`.
- Action 5 (wait): No-op.

### Phase 2–3
Nothing.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | wood_collected |
| 1 | ore_collected |
| 2 | pickaxe_crafted |
| 3 | rubble_mined |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–1% |
| PPO win rate at 200k steps | >10% |
| PPO learning delta (200k - 50k) | >5% |

## Invariant Tests

1. Exactly one workbench.
2. Exactly one rubble entity.
3. At least 2 wood and 2 ore (enough to craft).
4. Player starts with all inventory at 0.
5. Exit behind rubble wall (x ≥ 13).
6. Rubble at x=12.

## Notes
- Dependency chain: gather wood → gather ore → interact workbench → go to rubble → interact rubble → walk to exit. Six steps minimum.
- Pickaxe breaks after one use.
- Workbench is tagged `npc` (not solid) so player can stand adjacent easily.
- Neighbor checking for interact: check `(x±1, y)` and `(x, y±1)` for target entity type.
