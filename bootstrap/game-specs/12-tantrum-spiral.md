# Game Spec 12: Tantrum Spiral (Psychology)

## Overview
Simulates Dwarf psychology and cascading failures. The player hauls boulders to a stockpile — each haul increases stress. Drinking ale reduces stress. If stress maxes out, the player "tantrums" (loses control, moves randomly, smashes items). The agent must learn work-life balance.

## Grid
- Dimensions: 12×12

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=12, grid_h=12,
    max_entities=16,       # player + 5 boulders + 9 stockpile + keg
    max_stack=3,           # boulder on stockpile
    num_entity_types=5,    # 0=unused, 1=player, 2=boulder, 3=stockpile, 4=keg
    num_tags=8,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=pushable, 7=target
    num_props=4,           # 0=stress, 1=boulders_hauled, 2=charges (keg), 3=tantrum_turns
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=4,     # 0=boulders_hauled, 1=drinks_taken, 2=tantrum_count, 3=items_smashed
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | boulder | `O` |
| 3 | stockpile | `_` |
| 4 | keg | `&` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | npc |
| 6 | pushable |
| 7 | target |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | stress | player: 0–20, tantrum at 20 |
| 1 | boulders_hauled | player: win at 5 |
| 2 | charges | keg: starts at 3, regains 1 every 20 turns |
| 3 | tantrum_turns | player: remaining tantrum turns (0 = normal) |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| stress (prop 0) | 20 |
| boulders_hauled (prop 1) | 5 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Center (6, 6) |
| boulder (2) | solid (1), pushable (6) | 5 scattered around map |
| stockpile (3) | target (7) | 3×3 zone, bottom-left corner |
| keg (4) | npc (5) | Top-right area (tavern zone) |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (keg) | Every 20 turns: if charges < 3, increment charges by 1. |

## Turn Phases

### Phase 1: Process Input
- **Tantrum check**: If `properties[player_idx, 3]` (tantrum_turns) > 0:
  - **Override input**: Ignore the agent's action. Instead, pick a random direction (via rng_key) and move player. If player bumps into a keg or boulder, destroy it. Increment `game_state[3]`.
  - Decrement tantrum_turns by 1.
  - Skip normal input processing.
- **Normal input**:
  - Actions 0–3 (move): Push mechanics (same as Block Push). If boulder pushed onto a stockpile tile: increment boulders_hauled, add stress += 4, add 0.1 to `reward_acc`. If boulders_hauled == 5, `status = 1`.
  - Action 4 (interact): Check adjacency to keg. If keg found AND charges > 0: reduce stress by 10 (clamp at 0), decrement charges. Add 0.05 to `reward_acc`.
  - Action 5: No-op.

### Phase 2: Run Behaviors
- Keg charge regeneration (every 20 turns).

### Phase 3: Turn End
- Every 5 turns: stress += 1.
- If stress ≥ 20: enter tantrum mode. Set tantrum_turns = 10. Increment `game_state[2]`.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | boulders_hauled |
| 1 | drinks_taken |
| 2 | tantrum_count |
| 3 | items_smashed |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Tantrums destroy necessary items |
| PPO win rate at 300k steps | >25% | Agent discovers "pacing" — haul some, drink, haul more |

## Invariant Tests

1. Exactly 5 boulders at start.
2. Exactly 9 stockpile tiles (3×3).
3. Exactly 1 keg.
4. Player starts with stress=0, boulders_hauled=0.
5. Keg starts with charges=3.

## Notes
- The core emergent behavior: the agent learns that working without breaks triggers tantrums that destroy items needed to win. It naturally discovers "work-life balance."
- Tantrum mode overrides agent input — the `jax.lax.cond` branch selects random movement when tantrum_turns > 0.
- Stress ticks passively (+1 every 5 turns) plus actively (+4 per boulder hauled). This creates time pressure even when idle.
- The keg's limited charges (3, regenerating slowly) prevent spamming drink actions.
