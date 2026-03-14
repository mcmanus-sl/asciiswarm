# Game Spec 03: Lock & Key

## Overview
A multi-room grid where the player must find a key, use it to unlock a door, and reach the exit. First game requiring the `interact` action, entity state mutation, and multi-step dependency chains.

## Grid
- Dimensions: 12×12

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=12, grid_h=12,
    max_entities=64,       # walls + player + key + door + exit
    max_stack=2,
    num_entity_types=6,    # 0=unused, 1=player, 2=exit, 3=wall, 4=door, 5=key
    num_tags=6,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc
    num_props=2,           # 0=has_key (player), 1=unused
    num_actions=6,         # 0=move_n, 1=move_s, 2=move_e, 3=move_w, 4=interact, 5=wait
    max_turns=300,
    step_penalty=-0.01,
    game_state_size=2,     # 0=key_picked_up, 1=door_unlocked
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused slot) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | wall | `#` |
| 4 | door | `+` |
| 5 | key | `k` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | npc |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | has_key | player: 0=no, 1=yes |
| 1 | (unused) | — |

## Entities

| Type | Tags | Glyph | Spawning |
|------|------|-------|----------|
| player (1) | player (0) | `@` | Random empty cell in room 1 |
| wall (3) | solid (1) | `#` | Room boundaries and dividers |
| door (4) | solid (1) | `+` | In wall divider between rooms 2–3, blocking corridor |
| key (5) | pickup (3) | `k` | Random empty cell in room 2 |
| exit (2) | exit (4) | `>` | Random empty cell in room 3 |

## Room Layout

Grid divided into 3 rooms by vertical wall dividers:
- **Room 1** (left): x=0–3. Player starts here.
- **Room 2** (center): x=5–7. Key is here.
- **Room 3** (right): x=9–11. Exit is here.

Wall dividers at x=4 and x=8, floor-to-ceiling. Each divider has one corridor opening (1 tile wide, random y via `jax.random`). Corridor between rooms 2–3 blocked by a `door`. Corridor between rooms 1–2 is open.

## Behaviors

None. No entity has autonomous behavior.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move_n/s/e/w): Compute target. Check target cell for solid-tagged entities. If any solid entity exists at target AND it's not the door with key in hand, cancel move. Otherwise, move player.
  - When player moves onto a cell with the key: set `properties[player_idx, 0] = 1` (has_key), destroy key entity (set `alive[key_slot] = False`), add `0.2` to `reward_acc`, update `game_state[0] = 1`.
- Action 4 (interact): Check 4 cardinal neighbors for the door entity. If found AND `properties[player_idx, 0] == 1` (has_key):
  - Destroy door entity. Set `properties[player_idx, 0] = 0` (key consumed). Add `0.2` to `reward_acc`. Update `game_state[1] = 1`.
- Action 5 (wait): No-op.

### Phase 2: Run Behaviors
No behaviors.

### Phase 3: Turn End
- Check if player shares a cell with exit entity. If yes, `status = 1` (won).

## Win Condition
Player walks onto the exit tile.

## Lose Condition
None. Truncates at `max_turns` (300).

## game_state Slots

| Index | Name | Description |
|-------|------|-------------|
| 0 | key_picked_up | 1.0 if key has been picked up |
| 1 | door_unlocked | 1.0 if door has been unlocked |

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0.1–3% |
| PPO win rate at 100k steps | >20% |
| PPO learning delta (100k - 10k) | >10% |

## Invariant Tests

1. Exactly one key exists at game start.
2. Exactly one door exists at game start.
3. Player starts in room 1 (x <= 3).
4. Key is in room 2 (5 <= x <= 7).
5. Exit is in room 3 (x >= 9).
6. Door blocks corridor between rooms 2–3.
7. Corridor between rooms 1–2 is not blocked.
8. Player starts with has_key == 0.

## Notes
- The multi-step dependency (find key → go to door → interact → reach exit) is the core learning challenge.
- `interact` must be adjacent to the door, not on the same tile (door is solid).
- Reward shaping (+0.2 for key, +0.2 for door) guides PPO toward correct sequence.
- Wall entities consume most of the `max_entities=64` budget.
- Room layout is deterministic given the rng_key.
