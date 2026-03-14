"""Game 01: Empty Exit — player walks to exit tile. Pipeline validation only."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import create_entity
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import movement_system
from jaxswarm.systems.collision import collision_system


CONFIG = EnvConfig(
    grid_w=8, grid_h=8,
    max_entities=8,
    max_stack=2,
    num_entity_types=3,
    num_tags=6,
    num_props=1,
    num_actions=6,
    max_turns=200,
    step_penalty=-0.01,
    game_state_size=1,
    prop_maxes=(1.0,),
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Boustrophedon: go to (0,0), then snake the grid
_go_to_origin = [0] * 7 + [3] * 7
_snake = []
for row in range(8):
    if row % 2 == 0:
        _snake += [2] * 7
    else:
        _snake += [3] * 7
    if row < 7:
        _snake += [1]
DETERMINISTIC_TRACE = _go_to_origin + _snake


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    k1, k2, k3 = jax.random.split(rng_key, 3)

    player_cell = jax.random.randint(k1, (), 0, CONFIG.grid_w * CONFIG.grid_h)
    player_x = player_cell % CONFIG.grid_w
    player_y = player_cell // CONFIG.grid_w

    offset = jax.random.randint(k2, (), 1, CONFIG.grid_w * CONFIG.grid_h)
    exit_cell = (player_cell + offset) % (CONFIG.grid_w * CONFIG.grid_h)
    exit_x = exit_cell % CONFIG.grid_w
    exit_y = exit_cell // CONFIG.grid_w

    player_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
    player_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, player_slot = create_entity(
        state, CONFIG, jnp.int32(1), player_x, player_y, player_tags, player_props
    )
    state = state.replace(player_idx=player_slot)

    exit_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[4].set(True)
    exit_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(2), exit_x, exit_y, exit_tags, exit_props
    )

    state = state.replace(rng_key=k3)
    obs = get_obs(state, CONFIG)
    return state, obs


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    # Phase 1: Player movement
    state = movement_system(state, action, CONFIG)

    # Phase 2: No behaviors

    # Phase 3: Collision check (exit)
    state = collision_system(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    player_count = int((state.alive & (state.entity_type == 1)).sum())
    results.append(("exactly_one_player", player_count == 1))
    exit_count = int((state.alive & (state.entity_type == 2)).sum())
    results.append(("exactly_one_exit", exit_count == 1))
    pidx = int(state.player_idx)
    exit_mask = state.alive & (state.entity_type == 2)
    exit_slot = int(jnp.argmax(exit_mask))
    diff_pos = (int(state.x[pidx]) != int(state.x[exit_slot])) or (int(state.y[pidx]) != int(state.y[exit_slot]))
    results.append(("player_exit_different_position", diff_pos))
    alive_count = int(state.alive.sum())
    results.append(("entity_budget_ok", alive_count <= CONFIG.max_entities))
    px, py = int(state.x[pidx]), int(state.y[pidx])
    results.append(("player_in_bounds", 0 <= px < CONFIG.grid_w and 0 <= py < CONFIG.grid_h))
    ex, ey = int(state.x[exit_slot]), int(state.y[exit_slot])
    results.append(("exit_in_bounds", 0 <= ex < CONFIG.grid_w and 0 <= ey < CONFIG.grid_h))
    return results
