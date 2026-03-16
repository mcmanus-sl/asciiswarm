"""Fertility system — manure CA propagation that halves crop growth time."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def fertility_system(
    state: EnvState, config: EnvConfig,
    manure_type: int,
    spread_chance: float = 0.3,
) -> tuple[EnvState, jnp.ndarray]:
    """
    Fertility CA spread from manure entities. Similar to magma.py but:
    - Spreads fertility (a float grid), not new entities
    - Uses zero-padded boundaries (no wrap-around)
    - Returns fertility_field: float32[H, W] — 0.0 to 1.0

    Crops growing on cells with fertility > 0.5 have their growth threshold halved.
    """
    H, W = config.grid_h, config.grid_w

    # Build manure grid
    is_manure = state.alive & (state.entity_type == manure_type)
    manure_grid = jnp.zeros(H * W, dtype=jnp.float32)
    indices = state.y * W + state.x
    manure_vals = is_manure.astype(jnp.float32)
    manure_grid = manure_grid.at[indices].add(manure_vals)
    manure_grid = jnp.clip(manure_grid.reshape(H, W), 0.0, 1.0)

    # Spread via cross kernel (average of neighbors + self, iterated)
    def spread_step(grid, _):
        up = jnp.pad(grid[:-1, :], ((1, 0), (0, 0)), constant_values=0.0)
        down = jnp.pad(grid[1:, :], ((0, 1), (0, 0)), constant_values=0.0)
        left = jnp.pad(grid[:, :-1], ((0, 0), (1, 0)), constant_values=0.0)
        right = jnp.pad(grid[:, 1:], ((0, 0), (0, 1)), constant_values=0.0)
        avg = (grid + up + down + left + right) / 5.0
        # Keep manure sources at 1.0
        avg = jnp.maximum(avg, manure_grid)
        return avg, None

    # Iterate spread (3 steps gives ~3-cell radius)
    fertility, _ = jax.lax.scan(spread_step, manure_grid, None, length=3)

    return state, fertility
