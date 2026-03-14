"""Collision system — checks player cell for exit/hazard entities."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import get_entities_at


def collision_system(state: EnvState, config: EnvConfig) -> EnvState:
    """Check if player shares cell with exit or hazard. Update status."""
    pidx = state.player_idx
    px = state.x[pidx]
    py = state.y[pidx]

    slots, count = get_entities_at(state, px, py)

    # Scan the stack for exit (tag 4) and hazard (tag 2)
    def check_cell(i, carry):
        found_exit, found_hazard = carry
        slot = slots[i]
        is_valid = (i < count) & (slot != pidx)  # skip player itself
        is_exit = is_valid & state.alive[slot] & state.tags[slot, 4]
        is_hazard = is_valid & state.alive[slot] & state.tags[slot, 2]
        return (found_exit | is_exit, found_hazard | is_hazard)

    found_exit, found_hazard = jax.lax.fori_loop(
        0, config.max_stack, check_cell, (jnp.bool_(False), jnp.bool_(False))
    )

    # Hazard takes priority (lose), then exit (win). Only update if still playing.
    still_playing = state.status == 0
    new_status = jnp.where(
        still_playing & found_hazard, jnp.int32(-1),
        jnp.where(still_playing & found_exit, jnp.int32(1), state.status)
    )
    return state.replace(status=new_status)
