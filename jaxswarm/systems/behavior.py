"""Behavior system — sequential NPC dispatch via fori_loop + lax.switch."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import move_entity


def _noop_behavior(state, slot, config):
    """Default: do nothing."""
    return state


def _wanderer_behavior(state, slot, config):
    """Wanderer: move in current direction, bounce at walls."""
    direction = state.properties[slot, 0].astype(jnp.int32)  # 1=east, -1=west
    cur_x = state.x[slot]
    cur_y = state.y[slot]

    new_x = cur_x + direction
    state, moved = move_entity(state, config, slot, new_x, cur_y)

    # Check if next step would be out of bounds — if so, reverse direction
    next_x = state.x[slot] + direction
    should_reverse = (next_x < 0) | (next_x >= config.grid_w)
    new_direction = jnp.where(should_reverse, -direction, direction).astype(jnp.float32)
    state = state.replace(
        properties=state.properties.at[slot, 0].set(new_direction)
    )
    return state


# Behavior dispatch table indexed by entity_type
# Index 0 = unused, 1 = player, 2 = exit, 3 = wanderer
# Extend this list as new NPC types are added
BEHAVIOR_TABLE = [
    _noop_behavior,      # 0: unused
    _noop_behavior,      # 1: player (handled by movement_system)
    _noop_behavior,      # 2: exit
    _wanderer_behavior,  # 3: wanderer
]


def behavior_system(state: EnvState, config: EnvConfig, behavior_table=None) -> EnvState:
    """
    Iterate all entities, dispatch NPC behavior by type.
    Uses fori_loop (NOT vmap) to avoid concurrent grid collisions.
    """
    if behavior_table is None:
        behavior_table = BEHAVIOR_TABLE

    # Pad table to num_entity_types if needed
    while len(behavior_table) < config.num_entity_types:
        behavior_table = list(behavior_table) + [_noop_behavior]

    def loop_body(i, state):
        is_npc = state.alive[i] & (state.entity_type[i] >= 3)  # types >= 3 are NPCs

        # Dispatch behavior based on entity type
        type_idx = state.entity_type[i]
        branches = [lambda s, sl=i, c=config: fn(s, sl, c) for fn in behavior_table]
        new_state = jax.lax.switch(type_idx, branches, state)

        # Only apply if entity is alive NPC
        state = jax.tree.map(
            lambda n, o: jnp.where(is_npc, n, o), new_state, state
        )
        return state

    return jax.lax.fori_loop(0, config.max_entities, loop_body, state)
