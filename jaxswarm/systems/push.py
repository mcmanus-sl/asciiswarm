"""Push system — Sokoban-style block pushing."""

import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import move_entity, get_entities_at
from jaxswarm.systems.movement import DX, DY


def push_system(
    state: EnvState, action: jnp.int32, config: EnvConfig,
    pushable_tag: int = 6,
) -> EnvState:
    """
    If player tries to move into a pushable block, push it.
    Must run BEFORE standard movement so the cell is clear for the player.

    Returns state with block moved (if valid) and player NOT yet moved.
    The caller should attempt player movement after this.
    """
    pidx = state.player_idx
    px, py = state.x[pidx], state.y[pidx]
    is_move = action < 4

    dx = DX[action]
    dy = DY[action]
    target_x = px + dx
    target_y = py + dy

    # Clamp to prevent OOB
    safe_tx = jnp.clip(target_x, 0, config.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, config.grid_h - 1)
    in_bounds = (target_x >= 0) & (target_x < config.grid_w) & (target_y >= 0) & (target_y < config.grid_h)

    # Check for pushable entity at target
    slots, count = get_entities_at(state, safe_tx, safe_ty)

    def find_pushable(j, carry):
        found_slot, found = carry
        slot = slots[j]
        is_pushable = (j < count) & in_bounds & is_move & state.alive[slot] & state.tags[slot, pushable_tag]
        found_slot = jnp.where(is_pushable & ~found, slot, found_slot)
        found = found | is_pushable
        return (found_slot, found)

    pushable_slot, has_pushable = jnp.int32(-1), jnp.bool_(False)
    pushable_slot, has_pushable = jax.lax.fori_loop(
        0, config.max_stack, find_pushable, (pushable_slot, has_pushable)
    )

    # Compute push destination (block moves in same direction as player)
    push_x = jnp.clip(target_x + dx, 0, config.grid_w - 1)
    push_y = jnp.clip(target_y + dy, 0, config.grid_h - 1)
    push_in_bounds = ((target_x + dx) >= 0) & ((target_x + dx) < config.grid_w) & \
                     ((target_y + dy) >= 0) & ((target_y + dy) < config.grid_h)

    # Check push destination is clear (no solid)
    dest_slots, dest_count = get_entities_at(state, push_x, push_y)

    def check_solid(j, has_solid):
        slot = dest_slots[j]
        is_solid = (j < dest_count) & state.alive[slot] & state.tags[slot, 1]
        return has_solid | is_solid

    dest_blocked = jax.lax.fori_loop(0, config.max_stack, check_solid, jnp.bool_(False))

    # Push succeeds if: has pushable, push dest in bounds, not blocked
    can_push = has_pushable & push_in_bounds & ~dest_blocked

    # Move the block
    new_state, _ = move_entity(state, config, pushable_slot, push_x, push_y)
    state = jax.tree.map(
        lambda n, o: jnp.where(can_push, n, o), new_state, state
    )

    return state


import jax
