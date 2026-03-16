"""Processing system — NPC takes input item, busy for N turns, outputs product."""

import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def processing_system(
    state: EnvState, config: EnvConfig,
    npc_slot: int,
    busy_prop: int = 6,
    busy_duration: int = 10,
    output_type: int = 0,
    output_tags: jnp.ndarray = None,
    output_props: jnp.ndarray = None,
) -> EnvState:
    """
    Process a single NPC processor: if busy, decrement timer.
    When timer hits 0, spawn output product at NPC position.

    The NPC becomes "busy" when the game's interact phase sets busy_prop > 0.
    This system just handles the countdown and output.
    """
    is_alive = state.alive[npc_slot]
    busy_turns = state.properties[npc_slot, busy_prop]
    is_busy = is_alive & (busy_turns > 0)

    # Decrement timer
    new_busy = jnp.where(is_busy, busy_turns - 1.0, busy_turns)
    state = state.replace(
        properties=state.properties.at[npc_slot, busy_prop].set(new_busy),
    )

    # Output when timer reaches 0
    just_finished = is_busy & (new_busy <= 0)

    # Find free slot
    free_slot = jnp.argmin(state.alive)
    has_free = ~state.alive.all()
    can_output = just_finished & has_free

    if output_tags is None:
        output_tags = jnp.zeros(config.num_tags, dtype=jnp.bool_).at[3].set(True)  # pickup
    if output_props is None:
        output_props = jnp.zeros(config.num_props, dtype=jnp.float32)

    spawned_alive = state.alive.at[free_slot].set(True)
    spawned_etype = state.entity_type.at[free_slot].set(output_type)
    spawned_x = state.x.at[free_slot].set(state.x[npc_slot])
    spawned_y = state.y.at[free_slot].set(state.y[npc_slot])
    spawned_tags = state.tags.at[free_slot].set(output_tags)
    spawned_props = state.properties.at[free_slot].set(output_props)

    import jax
    spawned_state = state.replace(
        alive=spawned_alive, entity_type=spawned_etype,
        x=spawned_x, y=spawned_y,
        tags=spawned_tags, properties=spawned_props,
    )
    state = jax.tree.map(
        lambda n, o: jnp.where(can_output, n, o), spawned_state, state
    )
    return state
