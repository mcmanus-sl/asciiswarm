"""Slide system — ice sliding physics. Entity slides until hitting solid or edge."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import move_entity, get_entities_at
from jaxswarm.systems.movement import DX, DY


def slide_player(state: EnvState, action: jnp.int32, config: EnvConfig) -> EnvState:
    """
    Slide player in direction until hitting solid entity or grid edge.
    After each step, check for exit collision.
    Uses fori_loop with max iterations = max(grid_w, grid_h).
    """
    is_move = action < 4
    dx = DX[action]
    dy = DY[action]
    pidx = state.player_idx
    max_steps = max(config.grid_w, config.grid_h)

    def slide_step(_, carry):
        state, still_sliding = carry

        cur_x = state.x[pidx]
        cur_y = state.y[pidx]
        next_x = cur_x + dx
        next_y = cur_y + dy

        # Check if next cell has solid entity
        safe_nx = jnp.clip(next_x, 0, config.grid_w - 1)
        safe_ny = jnp.clip(next_y, 0, config.grid_h - 1)
        in_bounds = (next_x >= 0) & (next_x < config.grid_w) & (next_y >= 0) & (next_y < config.grid_h)

        slots, count = get_entities_at(state, safe_nx, safe_ny)

        def check_solid(i, has_solid):
            slot = slots[i]
            is_valid = (i < count) & in_bounds
            is_solid = is_valid & state.alive[slot] & state.tags[slot, 1]
            return has_solid | is_solid

        next_solid = jax.lax.fori_loop(0, config.max_stack, check_solid, jnp.bool_(False))

        can_move = still_sliding & is_move & in_bounds & ~next_solid & (state.status == 0)
        new_state, moved = move_entity(state, config, pidx, next_x, next_y)
        state = jax.tree.map(
            lambda n, o: jnp.where(can_move, n, o), new_state, state
        )

        # Check for exit at new position
        new_x = state.x[pidx]
        new_y = state.y[pidx]
        cell_slots, cell_count = get_entities_at(state, new_x, new_y)

        def check_exit(i, found_exit):
            slot = cell_slots[i]
            is_valid = i < cell_count
            is_exit = is_valid & state.alive[slot] & state.tags[slot, 4]  # tag 4 = exit
            return found_exit | is_exit

        on_exit = jax.lax.fori_loop(0, config.max_stack, check_exit, jnp.bool_(False))
        on_exit = on_exit & can_move

        new_status = jnp.where(
            on_exit & (state.status == 0), jnp.int32(1), state.status
        )
        state = state.replace(status=new_status)

        # Stop sliding if we didn't move or hit exit
        still_sliding = can_move & ~on_exit

        return (state, still_sliding)

    (state, _), _ = jax.lax.scan(
        lambda carry, _: (slide_step(None, carry), None),
        (state, jnp.bool_(True)),
        None,
        length=max_steps,
    )

    return state
