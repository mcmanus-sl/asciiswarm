"""Movement system — maps actions to player movement with solid-entity blocking."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import move_entity, get_entities_at

# Direction deltas: N=0(-y), S=1(+y), E=2(+x), W=3(-x), interact=4(noop), wait=5(noop)
DX = jnp.array([0, 0, 1, -1, 0, 0], dtype=jnp.int32)
DY = jnp.array([-1, 1, 0, 0, 0, 0], dtype=jnp.int32)


def movement_system(state: EnvState, action: jnp.int32, config: EnvConfig) -> EnvState:
    """Move player based on action. Blocked by solid-tagged entities at target cell."""
    pidx = state.player_idx
    cur_x = state.x[pidx]
    cur_y = state.y[pidx]

    target_x = cur_x + DX[action]
    target_y = cur_y + DY[action]

    # Check for solid entities at target (only if in bounds)
    safe_tx = jnp.clip(target_x, 0, config.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, config.grid_h - 1)
    slots, count = get_entities_at(state, safe_tx, safe_ty)

    # Check if any entity in the target cell has the solid tag (tag index 1)
    def check_solid(i, has_solid):
        slot = slots[i]
        is_valid = i < count
        is_solid = state.alive[slot] & state.tags[slot, 1]
        return has_solid | (is_valid & is_solid)

    target_has_solid = jax.lax.fori_loop(0, config.max_stack, check_solid, jnp.bool_(False))

    # Only move if target is not solid
    new_state, moved = move_entity(state, config, pidx, target_x, target_y)
    state = jax.tree.map(
        lambda n, o: jnp.where(~target_has_solid, n, o), new_state, state
    )
    return state
