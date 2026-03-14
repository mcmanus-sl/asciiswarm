# Game Spec 10: Farming & Growth

## Overview
Turn-based farming. Plant seeds on tilled soil, wait for crops to grow over 15 turns, harvest, deliver to a bin. Tests temporal planning — the agent must learn that actions have delayed payoffs.

## Grid
- Dimensions: 14×14

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=14, grid_h=14,
    max_entities=48,       # walls + soil(9) + seedbag + bin + sprouts + mature + player
    max_stack=3,           # sprout/mature on top of soil
    num_entity_types=8,    # 0=unused, 1=player, 2=wall, 3=soil, 4=seedbag, 5=sprout, 6=mature, 7=bin
    num_tags=6,            # standard 6
    num_props=4,           # 0=seeds, 1=crops, 2=delivered, 3=age (sprout)
    num_actions=6,
    max_turns=300,
    step_penalty=-0.005,
    game_state_size=4,     # 0=seeds_planted, 1=crops_harvested, 2=crops_delivered, 3=unused
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | wall | `#` |
| 3 | soil | `~` |
| 4 | seedbag | `s` |
| 5 | sprout | `,` |
| 6 | mature | `*` |
| 7 | bin | `B` |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | seeds | player: seeds in hand (0–10) |
| 1 | crops | player: harvested crops (0–10) |
| 2 | delivered | player: crops delivered to bin (win at 5) |
| 3 | age | sprout: turns since planting (matures at 15) |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| seeds (prop 0) | 10 |
| crops (prop 1) | 10 |
| delivered (prop 2) | 5 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Random cell in farmhouse area (x < 4, y < 4) |
| wall (2) | solid (1) | Outer boundary + farmhouse walls |
| soil (3) | npc (5) | 3×3 patch, center (x=5–7, y=5–7). 9 tiles. |
| seedbag (4) | pickup (3) | Single, near farmhouse (x=2, y=5) |
| bin (7) | npc (5) | Collection bin at (12, 1) |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 5 (sprout) | Increment `properties[slot, 3]` (age) by 1. When age ≥ 15: destroy sprout, create mature entity at same position. |

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move): Check target:
  - Solid → cancel.
  - Pickup: if seedbag → set seeds = 6, destroy seedbag, add 0.1 to `reward_acc`. If mature → increment crops (cap 10), destroy mature, add 0.15 to `reward_acc`.
- Action 4 (interact):
  1. If player standing on soil (type 3) AND seeds ≥ 1 AND no sprout/mature at this cell: create sprout (type 5) at player position with age=0. Decrement seeds. Add 0.05 to `reward_acc`.
  2. Else if adjacent to bin (type 7) AND crops ≥ 1: decrement crops, increment delivered. Add 0.3 to `reward_acc`. If delivered ≥ 5: `status = 1`.
- Action 5: No-op.

### Phase 2: Run Behaviors
- Iterate entities. For type 5 (sprout): increment age. If age ≥ 15, transform to mature.
- **JAX note**: Sprout → mature transformation means destroying one entity and creating another. Use `destroy_entity` + `create_entity` within the behavior loop. The entity budget must accommodate having sprouts AND mature plants simultaneously.

### Phase 3: Turn End
Nothing extra.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | seeds_planted |
| 1 | crops_harvested |
| 2 | crops_delivered |
| 3 | (unused) |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–0.5% |
| PPO win rate at 300k steps | >8% |
| PPO learning delta (300k - 100k) | >4% |

## Invariant Tests

1. Exactly 9 soil tiles (3×3 patch).
2. Exactly one seedbag at start.
3. Exactly one bin.
4. Player starts with seeds=0, crops=0, delivered=0.
5. No sprouts or mature plants at start.

## Notes
- 15-turn growth delay is the core learning challenge. Agent must plant early, do something else, come back.
- 6 seeds with quota of 5 gives one margin of error.
- Soil patch is small (9 tiles) → spatial memory tested.
- Sprout ticks every turn regardless of player position.
