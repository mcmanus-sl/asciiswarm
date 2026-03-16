"""Magma system — cellular automata spread via grid convolution."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def magma_system(
    state: EnvState, config: EnvConfig,
    magma_type: int = 4,
    spread_chance: float = 0.2,
) -> EnvState:
    """
    Magma CA spread: for each empty cell adjacent to magma, 20% chance to spawn magma.
    Uses grid convolution (jnp.roll shifts), not per-entity iteration.

    Returns updated state with new magma entities spawned.
    Uses zero-padding (not wrap-around) for boundaries.
    """
    H, W = config.grid_h, config.grid_w

    # Build magma grid: bool[H, W]
    is_magma = state.alive & (state.entity_type == magma_type)
    magma_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    indices = state.y * W + state.x
    magma_grid = magma_grid.at[indices].set(magma_grid[indices] | is_magma)
    magma_grid = magma_grid.reshape(H, W)

    # Build occupied grid: bool[H, W] where any alive solid entity exists
    is_solid = state.alive & state.tags[:, 1]  # solid tag
    occupied = jnp.zeros(H * W, dtype=jnp.bool_)
    occupied = occupied.at[indices].set(occupied[indices] | is_solid)
    occupied = occupied.reshape(H, W)

    # Build any-alive grid: bool[H, W]
    any_alive = jnp.zeros(H * W, dtype=jnp.bool_)
    any_alive = any_alive.at[indices].set(any_alive[indices] | state.alive)
    any_alive = any_alive.reshape(H, W)

    # Cross-kernel neighbor check using pad+slice (zero-padded, no wrap-around)
    up = jnp.pad(magma_grid[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down = jnp.pad(magma_grid[1:, :], ((0, 1), (0, 0)), constant_values=False)
    left = jnp.pad(magma_grid[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = jnp.pad(magma_grid[:, 1:], ((0, 0), (0, 1)), constant_values=False)

    has_magma_neighbor = up | down | left | right

    # Candidate cells: adjacent to magma, not occupied by solid, not already having any entity
    can_spread = has_magma_neighbor & ~occupied & ~any_alive

    # Stochastic spread: roll per cell
    key, subkey = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)
    rolls = jax.random.uniform(subkey, (H, W))
    will_spread = can_spread & (rolls < spread_chance)

    # Count new magma cells to spawn
    spread_ys, spread_xs = jnp.where(will_spread, size=32, fill_value=-1)
    # size=32 is a static upper bound on new magma per turn; excess slots get (-1,-1)

    # Spawn new magma entities in free slots
    hazard_tags = jnp.zeros(config.num_tags, dtype=jnp.bool_).at[2].set(True)
    magma_props = jnp.zeros(config.num_props, dtype=jnp.float32)

    def spawn_one(state, idx):
        sy = spread_ys[idx]
        sx = spread_xs[idx]
        is_valid = (sy >= 0) & (sx >= 0)

        # Find first free slot
        free_slot = jnp.argmin(state.alive)
        has_free = ~state.alive.all()
        can_spawn = is_valid & has_free

        new_alive = state.alive.at[free_slot].set(True)
        new_etype = state.entity_type.at[free_slot].set(magma_type)
        new_x = state.x.at[free_slot].set(sx)
        new_y = state.y.at[free_slot].set(sy)
        new_tags = state.tags.at[free_slot].set(hazard_tags)
        new_props = state.properties.at[free_slot].set(magma_props)

        spawned = state.replace(
            alive=new_alive, entity_type=new_etype,
            x=new_x, y=new_y, tags=new_tags, properties=new_props,
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(can_spawn, n, o), spawned, state
        )
        return state, None

    state, _ = jax.lax.scan(spawn_one, state, jnp.arange(32))
    return state
