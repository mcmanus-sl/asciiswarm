# Game Spec 11: Digging Deep (Fluid Dynamics)

## Overview
Mining carries risks. The player mines valuable adamantine ore near a pressurized magma chamber. Magma spreads via cellular automata. The agent must mine carefully, extract adamantine, build barricades to contain magma, and escape. Tests spatial containment, dynamic hazard propagation, and risk assessment (greed vs. safety).

This is the first **macro-spec** — it tests emergent behavior from overlapping primitives.

## Grid
- Dimensions: 16×16

## EnvConfig

```python
CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=256,      # rock fills most of grid + magma can spread
    max_stack=2,
    num_entity_types=7,    # 0=unused, 1=player, 2=exit, 3=rock, 4=magma, 5=adamantine, 6=barricade
    num_tags=7,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=mineable, 6=defense
    num_props=3,           # 0=adamantine (player), 1=stone (player), 2=spread_chance (magma)
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=4,     # 0=adamantine_mined, 1=barricades_built, 2=magma_cells, 3=panic_walls
)
```

## Entity Type Enum

| Type ID | Name | Glyph |
|---------|------|-------|
| 0 | (unused) | — |
| 1 | player | `@` |
| 2 | exit | `>` |
| 3 | rock | `%` |
| 4 | magma | `~` |
| 5 | adamantine | `$` |
| 6 | barricade | `X` |

## Tag Index Mapping

| Tag Index | Name |
|-----------|------|
| 0 | player |
| 1 | solid |
| 2 | hazard |
| 3 | pickup |
| 4 | exit |
| 5 | mineable |
| 6 | defense |

## Property Index Mapping

| Prop Index | Name | Used By |
|-----------|------|---------|
| 0 | adamantine | player: collected adamantine (0–5) |
| 1 | stone | player: collected stone (0–20) |
| 2 | (unused) | — |

## Player Properties (for scalar observation)

| Key | Max |
|-----|-----|
| adamantine (prop 0) | 5 |
| stone (prop 1) | 20 |

## Entities

| Type | Tags | Spawning |
|------|------|----------|
| player (1) | player (0) | Safe zone (x < 4, y < 4) |
| exit (2) | exit (4) | Top right corner (15, 0) |
| rock (3) | solid (1), mineable (5) | Fills all non-safe-zone cells |
| magma (4) | hazard (2) | 3×3 pocket, center (x=7–9, y=7–9) |
| adamantine (5) | solid (1), mineable (5) | 5 blocks strictly adjacent to magma pocket |
| barricade (6) | solid (1), defense (6) | Built dynamically by player |

## Behavior Dispatch Table

| Type ID | Behavior |
|---------|----------|
| 4 (magma) | **Cellular automata spread**: For each of 4 cardinal neighbors, if neighbor cell is empty (no alive entity), 20% chance to spawn new magma there. Magma destroys any pickup entities it flows over. Cannot flow into solid blocks. |

### Magma CA in JAX

The magma spread is best implemented as a **grid convolution**, not per-entity iteration:
1. Build a binary magma grid: `magma_grid[y, x] = 1` where magma exists.
2. Build an occupancy grid: `occupied[y, x] = 1` where any solid entity exists.
3. For each empty cell adjacent to magma (convolution with cross kernel), roll `jax.random.uniform` < 0.2 → spawn magma.
4. This is vectorized and avoids per-entity loops.

## Turn Phases

### Phase 1: Process Input
- Actions 0–3 (move into mineable): If target cell has a mineable entity:
  - Cancel move (player stays — simulates swinging pickaxe). Destroy the mineable entity.
  - If it was rock (type 3): increment stone by 1.
  - If it was adamantine (type 5): increment adamantine by 1, add 0.3 to `reward_acc`.
- Actions 0–3 (move into empty/exit): Normal move. If exit cell, `status = 1`.
- Action 4 (interact — build barricade): If stone ≥ 2: create barricade at a designated cell (e.g., the cell behind the player, or a cell the player previously occupied). Subtract 2 stone.
  - **Panic wall bonus**: If any magma entity within Chebyshev distance 2 of the barricade, add 0.5 to `reward_acc`. Increment `game_state[3]`.
- Action 5 (wait): No-op.

### Phase 2: Run Behaviors
- Run magma CA spread for all magma entities.
- After spread, check if any magma shares cell with player → `status = -1`.

### Phase 3: Turn End
Nothing extra.

## game_state Slots

| Index | Name |
|-------|------|
| 0 | adamantine_mined |
| 1 | barricades_built |
| 2 | magma_cells (current count) |
| 3 | panic_walls (barricades near magma) |

## RL Evaluation Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Random agent win rate | 0% | Killed by magma |
| PPO "Panic Wall" execution | >50% of wins | Agent builds barricade near approaching magma — validates design tension |
| PPO win rate at 300k steps | >15% | Agent learns to mine carefully and escape |

## Invariant Tests

1. Exactly 5 adamantine blocks at start.
2. All adamantine blocks adjacent to magma pocket.
3. Player starts in safe zone (x < 4, y < 4).
4. Exit at (15, 0).
5. Magma pocket is 3×3 at center.
6. Safe zone is empty (no rock).

## Notes
- The core emergent behavior: the agent learns to mine AROUND the magma, not through it. It builds barricades to contain magma while extracting adamantine from the edges.
- `max_entities=256` is large because rock fills most of the grid AND magma can spread.
- Magma spread is probabilistic (20%), creating unpredictable danger. The agent must be conservative.
- Barricade placement is the "panic wall" — Dwarf Fortress's signature moment.
