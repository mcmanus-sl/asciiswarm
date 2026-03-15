"""move_toward — greedy Manhattan one-step movement helper."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import move_entity, get_entities_at


def _has_solid(state, config, x, y):
    """Check if cell (x, y) contains a solid entity (tag 1)."""
    safe_x = jnp.clip(x, 0, config.grid_w - 1)
    safe_y = jnp.clip(y, 0, config.grid_h - 1)
    in_bounds = (x >= 0) & (x < config.grid_w) & (y >= 0) & (y < config.grid_h)
    slots, count = get_entities_at(state, safe_x, safe_y)

    def check(i, has_solid):
        slot = slots[i]
        valid = (i < count) & in_bounds & state.alive[slot] & state.tags[slot, 1]
        return has_solid | valid

    return jax.lax.fori_loop(0, config.max_stack, check, jnp.bool_(False))


def move_toward(
    state: EnvState,
    config: EnvConfig,
    slot: jnp.int32,
    target_x: jnp.int32,
    target_y: jnp.int32,
    rng_key: jax.Array,
) -> tuple[EnvState, jnp.bool_]:
    """
    Move entity one step toward (target_x, target_y) using greedy Manhattan.
    Prefer axis with greater distance. Break ties with rng_key.
    Returns (new_state, moved).
    """
    cur_x = state.x[slot]
    cur_y = state.y[slot]

    dx = target_x - cur_x  # positive = move right
    dy = target_y - cur_y  # positive = move down

    abs_dx = jnp.abs(dx)
    abs_dy = jnp.abs(dy)

    at_target = (abs_dx == 0) & (abs_dy == 0)

    # Step directions
    step_x = jnp.sign(dx)  # -1, 0, or 1
    step_y = jnp.sign(dy)

    # Decide which axis to move on
    # prefer_x: move along x axis
    # When tied, use rng to break tie
    tie = abs_dx == abs_dy
    prefer_x_by_dist = abs_dx > abs_dy
    coin = jax.random.bernoulli(rng_key)
    prefer_x = jnp.where(tie, coin, prefer_x_by_dist)

    # Compute candidate position
    move_x = jnp.where(prefer_x, cur_x + step_x, cur_x)
    move_y = jnp.where(prefer_x, cur_y, cur_y + step_y)

    # If at target, don't move
    final_x = jnp.where(at_target, cur_x, move_x)
    final_y = jnp.where(at_target, cur_y, move_y)

    # Check for solid entities at target cells (walls block NPC movement)
    primary_solid = _has_solid(state, config, final_x, final_y)

    # Try primary move (skip if solid)
    state1, moved1_raw = move_entity(state, config, slot, final_x, final_y)
    moved1 = moved1_raw & ~primary_solid

    # If primary blocked, try the other axis
    alt_x = jnp.where(prefer_x, cur_x, cur_x + step_x)
    alt_y = jnp.where(prefer_x, cur_y + step_y, cur_y)
    alt_x = jnp.where(at_target, cur_x, alt_x)
    alt_y = jnp.where(at_target, cur_y, alt_y)

    alt_solid = _has_solid(state, config, alt_x, alt_y)
    state2, moved2_raw = move_entity(state, config, slot, alt_x, alt_y)
    moved2 = moved2_raw & ~alt_solid

    # Use primary if it worked, else try alternate
    final_state = jax.tree.map(
        lambda s1, s2, s0: jnp.where(moved1, s1, jnp.where(moved2, s2, s0)),
        state1, state2, state
    )
    final_moved = moved1 | moved2
    # If at target, no move
    final_moved = final_moved & ~at_target

    final_state = jax.tree.map(
        lambda f, o: jnp.where(at_target, o, f), final_state, state
    )

    return final_state, final_moved
