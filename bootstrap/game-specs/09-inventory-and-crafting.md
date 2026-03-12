# Game Spec 09: Inventory & Crafting

## Overview
A workshop survival game. The player collects raw materials (wood, ore) scattered across the map, brings them to a workbench, and crafts a pickaxe to mine through a rubble wall blocking the exit. This introduces multi-slot inventory, crafting recipes, and workbench interaction — the foundation of every production chain in Dwarf Fortress.

## Grid
- Dimensions: 16×16

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'pickup', 'exit', 'npc'],
    'grid': (16, 16),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'wood', 'max': 5},
        {'key': 'ore', 'max': 5},
        {'key': 'has_pickaxe', 'max': 1},
    ],
}
```

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Random empty cell in left third (x < 5) |
| `wall` | `solid` | `#` | 1 | Outer boundary |
| `wood` | `pickup` | `t` | 3 | 4–6 scattered, random empty cells |
| `ore` | `pickup` | `o` | 3 | 3–5 scattered, random empty cells |
| `workbench` | `npc` | `W` | 5 | Single, center area (6 ≤ x ≤ 9, 6 ≤ y ≤ 9) |
| `rubble` | `solid` | `%` | 5 | Vertical wall at x=12, floor-to-ceiling, one gap blocked by rubble entity (1 tile). Rubble is mineable, regular walls are not. |
| `exit` | `exit` | `>` | 5 | Random empty cell in right section (x ≥ 13) |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `wood` | 0 | 5 | Wood carried |
| `ore` | 0 | 5 | Ore carried |
| `has_pickaxe` | 0 | 1 | Whether player has crafted a pickaxe |

## Behaviors
None. No entity has autonomous behavior.

## Crafting Recipes

| Recipe | Ingredients | Product | Where |
|--------|------------|---------|-------|
| Pickaxe | 2 wood + 2 ore | `has_pickaxe = 1` | Adjacent to workbench, via `interact` |

## Event Handlers

### `input` (Player Movement)
Standard 4-direction movement. `wait` does nothing.

### `input` (Interact)
If action is `interact`:
1. Check 4 cardinal neighbors for a `workbench`. If found and player has `wood >= 2` AND `ore >= 2`:
   - Set `wood -= 2`, `ore -= 2`, `has_pickaxe = 1`.
   - Emit `reward` with `{ 'amount': 0.3 }`.
2. Else check 4 cardinal neighbors for a `rubble` entity. If found and `has_pickaxe == 1`:
   - Destroy the rubble.
   - Set `has_pickaxe = 0` (pickaxe breaks after one use).
   - Emit `reward` with `{ 'amount': 0.3 }`.

### `collision` (player walks into pickup)
If mover is `player` and any occupant is tagged `pickup`:
- If occupant type is `wood`: increment `player.properties['wood']` by 1 (cap at 5). Destroy occupant. Emit `reward` `{ 'amount': 0.05 }`.
- If occupant type is `ore`: increment `player.properties['ore']` by 1 (cap at 5). Destroy occupant. Emit `reward` `{ 'amount': 0.05 }`.

### `collision` (player walks into exit)
If mover is `player` and occupant tagged `exit`: `env.end_game('won')`.

### `before_move` (solid blocks movement)
If target cell contains any entity tagged `solid`, cancel the move.

## Win Condition
Player reaches the exit (behind the rubble wall).

## Lose Condition
None. Truncates at `max_turns`.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0–1% |
| PPO win rate at 200k steps | >10% |
| PPO learning delta (200k - 50k) | >5% |

## Invariant Tests

1. Exactly one workbench exists.
2. Exactly one rubble entity exists.
3. At least 2 wood and 2 ore entities exist (enough to craft).
4. Player starts with all inventory properties at 0.
5. Exit is behind the rubble wall (x ≥ 13).
6. Rubble is at x=12.

## Notes
- The dependency chain is: gather wood → gather ore → interact at workbench → go to rubble → interact at rubble → walk to exit. Six steps minimum — the longest chain so far.
- Pickaxe breaks after one use to prevent trivializing future rubble-heavy maps.
- The workbench is tagged `npc` (not `solid`) so the player can stand adjacent to it easily.
