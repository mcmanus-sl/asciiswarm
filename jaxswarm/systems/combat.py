"""Combat system — bump-attack: player moves into enemy, damage exchanged."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import get_entities_at, destroy_entity
from jaxswarm.systems.movement import DX, DY

# Property indices for combat
HP_IDX = 0
ATK_IDX = 1


def combat_system(state: EnvState, action: jnp.int32, config: EnvConfig) -> EnvState:
    """
    If player's target cell contains a hazard entity, exchange damage.
    Movement is NOT performed (movement_system handles solid/hazard blocking).
    This system only handles the combat damage exchange.
    """
    pidx = state.player_idx
    px = state.x[pidx]
    py = state.y[pidx]

    target_x = px + DX[action]
    target_y = py + DY[action]

    # Clamp for safe indexing
    safe_tx = jnp.clip(target_x, 0, config.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, config.grid_h - 1)
    in_bounds = (target_x >= 0) & (target_x < config.grid_w) & (target_y >= 0) & (target_y < config.grid_h)

    # Check if movement action (0-3)
    is_move = action < 4

    slots, count = get_entities_at(state, safe_tx, safe_ty)

    # Find first hazard entity at target cell
    def find_hazard(i, carry):
        found, target_slot = carry
        slot = slots[i]
        is_valid = (i < count) & in_bounds & is_move
        is_hazard = is_valid & state.alive[slot] & state.tags[slot, 2]  # tag 2 = hazard
        target_slot = jnp.where(is_hazard & ~found, slot, target_slot)
        found = found | is_hazard
        return (found, target_slot)

    found_hazard, enemy_slot = jax.lax.fori_loop(
        0, config.max_stack, find_hazard, (jnp.bool_(False), jnp.int32(0))
    )

    # Exchange damage
    player_atk = state.properties[pidx, ATK_IDX]
    enemy_atk = state.properties[enemy_slot, ATK_IDX]
    player_hp = state.properties[pidx, HP_IDX]
    enemy_hp = state.properties[enemy_slot, HP_IDX]

    new_enemy_hp = enemy_hp - player_atk
    new_player_hp = player_hp - enemy_atk

    # Apply damage
    new_props = state.properties.at[pidx, HP_IDX].set(
        jnp.where(found_hazard, new_player_hp, player_hp)
    )
    new_props = new_props.at[enemy_slot, HP_IDX].set(
        jnp.where(found_hazard, new_enemy_hp, enemy_hp)
    )
    state = state.replace(properties=new_props)

    # Kill enemy if HP <= 0
    enemy_dead = found_hazard & (new_enemy_hp <= 0)
    killed_state = destroy_entity(state, config, enemy_slot)
    killed_state = killed_state.replace(
        reward_acc=killed_state.reward_acc + 2.0,
        game_state=killed_state.game_state.at[0].set(killed_state.game_state[0] + 1),
    )
    state = jax.tree.map(
        lambda k, o: jnp.where(enemy_dead, k, o), killed_state, state
    )

    # Kill player if HP <= 0
    player_dead = found_hazard & (new_player_hp <= 0)
    new_status = jnp.where(
        player_dead & (state.status == 0), jnp.int32(-1), state.status
    )
    state = state.replace(status=new_status)

    return state


def enemy_attack_player(state: EnvState, enemy_slot: jnp.int32, config: EnvConfig) -> EnvState:
    """Enemy at enemy_slot attacks player if sharing cell. Called after enemy movement."""
    pidx = state.player_idx
    same_cell = (state.x[enemy_slot] == state.x[pidx]) & (state.y[enemy_slot] == state.y[pidx])
    is_alive = state.alive[enemy_slot]
    should_attack = same_cell & is_alive & (state.status == 0)

    enemy_atk = state.properties[enemy_slot, ATK_IDX]
    player_hp = state.properties[pidx, HP_IDX]
    new_hp = player_hp - enemy_atk

    new_props = state.properties.at[pidx, HP_IDX].set(
        jnp.where(should_attack, new_hp, player_hp)
    )
    state = state.replace(properties=new_props)

    player_dead = should_attack & (new_hp <= 0)
    new_status = jnp.where(
        player_dead & (state.status == 0), jnp.int32(-1), state.status
    )
    state = state.replace(status=new_status)
    return state
