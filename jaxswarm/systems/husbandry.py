"""Husbandry system — feed entity, produce outputs (wool, manure)."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def husbandry_system(
    state: EnvState, config: EnvConfig,
    animal_type: int,
    fed_prop: int = 6,
    production_interval: int = 10,
    output_type: int = 0,
    output_tags: jnp.ndarray = None,
    output_props: jnp.ndarray = None,
) -> EnvState:
    """
    Process all animals of animal_type: if fed (prop > 0), decrement.
    Every production_interval turns of being fed, spawn output product.

    The feeding is handled by the game's interact phase (sets fed_prop).
    This system handles production ticks.
    """
    is_animal = state.alive & (state.entity_type == animal_type)

    if output_tags is None:
        output_tags = jnp.zeros(config.num_tags, dtype=jnp.bool_).at[3].set(True)
    if output_props is None:
        output_props = jnp.zeros(config.num_props, dtype=jnp.float32)

    def process_animal(carry, eidx):
        state = carry
        is_this_animal = is_animal[eidx]
        fed = state.properties[eidx, fed_prop]
        is_fed = is_this_animal & (fed > 0)

        # Decrement fed counter
        new_fed = jnp.where(is_fed, fed - 1.0, fed)

        # Produce when counter hits 0
        just_produced = is_fed & (new_fed <= 0)

        # Reset fed counter for next cycle
        reset_fed = jnp.where(just_produced, 0.0, new_fed)
        state = state.replace(
            properties=state.properties.at[eidx, fed_prop].set(reset_fed),
        )

        # Spawn output
        free_slot = jnp.argmin(state.alive)
        has_free = ~state.alive.all()
        can_spawn = just_produced & has_free

        spawned_alive = state.alive.at[free_slot].set(True)
        spawned_etype = state.entity_type.at[free_slot].set(output_type)
        spawned_x = state.x.at[free_slot].set(state.x[eidx])
        spawned_y = state.y.at[free_slot].set(state.y[eidx])
        spawned_tags = state.tags.at[free_slot].set(output_tags)
        spawned_props = state.properties.at[free_slot].set(output_props)

        spawned_state = state.replace(
            alive=spawned_alive, entity_type=spawned_etype,
            x=spawned_x, y=spawned_y,
            tags=spawned_tags, properties=spawned_props,
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(can_spawn, n, o), spawned_state, state
        )
        return state, None

    state, _ = jax.lax.scan(process_animal, state, jnp.arange(config.max_entities))
    return state
