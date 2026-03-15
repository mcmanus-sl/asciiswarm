"""Farming system — growth ticks for planted seeds."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import create_entity, destroy_entity


def farming_system(
    state: EnvState, config: EnvConfig,
    sprout_type: int = 5,
    mature_type: int = 6,
    age_prop: int = 3,
    growth_threshold: int = 15,
) -> EnvState:
    """
    Iterate over entities. If a sprout, increment age.
    If age >= threshold, destroy sprout and create mature crop at same position.
    """
    mature_tags = jnp.zeros(config.num_tags, dtype=jnp.bool_).at[3].set(True)  # pickup
    mature_props = jnp.zeros(config.num_props, dtype=jnp.float32)

    def grow_entity(i, state):
        is_sprout = state.alive[i] & (state.entity_type[i] == sprout_type)
        age = state.properties[i, age_prop]
        new_age = age + 1.0

        # Update age
        aged_state = state.replace(
            properties=state.properties.at[i, age_prop].set(new_age),
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(is_sprout, n, o), aged_state, state
        )

        # Check if mature
        is_mature = is_sprout & (new_age >= growth_threshold)
        sx, sy = state.x[i], state.y[i]

        # Destroy sprout, create mature
        destroyed = destroy_entity(state, config, i)
        matured, _ = create_entity(
            destroyed, config, jnp.int32(mature_type), sx, sy, mature_tags, mature_props
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(is_mature, n, o), matured, state
        )
        return state

    return jax.lax.fori_loop(0, config.max_entities, grow_entity, state)
