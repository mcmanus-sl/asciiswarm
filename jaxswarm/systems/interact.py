"""Interact system — adjacent entity interaction for action=4."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import get_entities_at, destroy_entity

# Cardinal direction offsets for adjacency check
ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

INTERACT_ACTION = 4


def interact_system(state: EnvState, action: jnp.int32, config: EnvConfig) -> EnvState:
    """
    If action is interact (4), check 4 cardinal neighbors for interactable entities.
    Game 03 pattern: destroy door if player has key, consume key, add reward.
    """
    is_interact = action == INTERACT_ACTION
    pidx = state.player_idx
    px = state.x[pidx]
    py = state.y[pidx]
    has_key = state.properties[pidx, 0] >= 1.0

    # Check all 4 adjacent cells for door entity (type 4)
    def check_direction(d, carry):
        state, found_door = carry
        nx = px + ADJ_DX[d]
        ny = py + ADJ_DY[d]

        in_bounds = (nx >= 0) & (nx < config.grid_w) & (ny >= 0) & (ny < config.grid_h)
        safe_nx = jnp.clip(nx, 0, config.grid_w - 1)
        safe_ny = jnp.clip(ny, 0, config.grid_h - 1)

        slots, count = get_entities_at(state, safe_nx, safe_ny)

        # Check each slot in the cell for a door (type 4)
        def check_slot(j, inner_carry):
            st, fd = inner_carry
            slot = slots[j]
            is_valid = (j < count) & in_bounds
            is_door = is_valid & st.alive[slot] & (st.entity_type[slot] == 4)
            can_unlock = is_door & has_key & is_interact & (~fd)

            # Destroy door and consume key
            new_st = destroy_entity(st, config, slot)
            new_st = new_st.replace(
                properties=new_st.properties.at[pidx, 0].set(0.0),
                reward_acc=new_st.reward_acc + 2.0,
                game_state=new_st.game_state.at[1].set(1.0),
            )

            st = jax.tree.map(
                lambda n, o: jnp.where(can_unlock, n, o), new_st, st
            )
            fd = fd | can_unlock
            return (st, fd)

        state, found_door = jax.lax.fori_loop(0, config.max_stack, check_slot, (state, found_door))
        return (state, found_door)

    state, _ = jax.lax.fori_loop(0, 4, check_direction, (state, jnp.bool_(False)))
    return state
