"""Fire system — fuel decay and warmth radius via distance field."""

import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.wave import compute_distance_field


def fire_system(
    state: EnvState, config: EnvConfig,
    fire_type: int = 4,
    fuel_prop: int = 3,
    fuel_decay: float = 0.2,
    warmth_radius: int = 3,
) -> tuple[EnvState, jnp.ndarray]:
    """
    Process fire entities: decay fuel, compute warmth field.

    Returns (updated_state, warmth_field) where:
    - warmth_field: float32[H, W] — 1.0 at fire, decays to 0 at warmth_radius.
    - Fire entity dies when fuel reaches 0.
    """
    H, W = config.grid_h, config.grid_w

    # Decay fuel for all fire entities
    is_fire = state.alive & (state.entity_type == fire_type)
    fuel = state.properties[:, fuel_prop]
    new_fuel = jnp.where(is_fire, fuel - fuel_decay, fuel)
    new_fuel = jnp.maximum(new_fuel, 0.0)

    # Kill fires with no fuel
    fire_dead = is_fire & (new_fuel <= 0)
    new_alive = jnp.where(fire_dead, False, state.alive)
    new_etype = jnp.where(fire_dead, 0, state.entity_type)

    state = state.replace(
        alive=new_alive,
        entity_type=new_etype,
        properties=state.properties.at[:, fuel_prop].set(new_fuel),
    )

    # Compute warmth field (distance from fire entities)
    # Build fire mask on grid
    still_fire = state.alive & (state.entity_type == fire_type)
    fire_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    indices = state.y * W + state.x
    fire_grid = fire_grid.at[indices].set(fire_grid[indices] | still_fire)
    fire_grid = fire_grid.reshape(H, W)

    # Wall mask (solid entities block warmth)
    is_solid = state.alive & state.tags[:, 1]
    wall_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    wall_grid = wall_grid.at[indices].set(wall_grid[indices] | is_solid)
    wall_grid = wall_grid.reshape(H, W)

    # Distance field from fire
    dist = compute_distance_field(wall_grid, fire_grid)

    # Convert to warmth: 1.0 at fire, linear decay to 0 at warmth_radius
    warmth = jnp.clip(1.0 - dist / warmth_radius, 0.0, 1.0)
    warmth = jnp.where(dist >= 999.0, 0.0, warmth)

    return state, warmth
