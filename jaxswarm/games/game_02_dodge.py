"""Game 02: Dodge — reach exit while avoiding a bouncing wanderer."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import create_entity
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import movement_system
from jaxswarm.systems.collision import collision_system
from jaxswarm.systems.behavior import behavior_system


CONFIG = EnvConfig(
    grid_w=10, grid_h=10,
    max_entities=8,
    max_stack=2,
    num_entity_types=4,    # 0=unused, 1=player, 2=exit, 3=wanderer
    num_tags=6,
    num_props=2,           # 0=direction (wanderer), 1=unused
    num_actions=6,
    max_turns=200,
    step_penalty=-0.01,
    game_state_size=1,
    prop_maxes=(1.0, 1.0),
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace: go north to avoid wanderer, then east to exit quadrant, then north
# Player starts in bottom-left (x<5, y>=5), exit in top-right (x>=5, y<5)
# Strategy: go north past wanderer row, then east, should reach exit area
_go_to_origin = [0] * 9 + [3] * 9  # north 9, west 9 to reach (0,0)
_snake = []
for row in range(10):
    if row % 2 == 0:
        _snake += [2] * 9  # east 9
    else:
        _snake += [3] * 9  # west 9
    if row < 9:
        _snake += [1]       # south 1
DETERMINISTIC_TRACE = _go_to_origin + _snake


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    # Player: bottom-left quadrant (x < 5, y >= 5)
    player_x = jax.random.randint(k1, (), 0, 5)
    player_y = jax.random.randint(k2, (), 5, 10)

    player_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
    player_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, player_slot = create_entity(
        state, CONFIG, jnp.int32(1), player_x, player_y, player_tags, player_props
    )
    state = state.replace(player_idx=player_slot)

    # Exit: top-right quadrant (x >= 5, y < 5)
    k_ex, k_ey, k3 = jax.random.split(k3, 3)
    exit_x = jax.random.randint(k_ex, (), 5, 10)
    exit_y = jax.random.randint(k_ey, (), 0, 5)

    exit_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[4].set(True)
    exit_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(2), exit_x, exit_y, exit_tags, exit_props
    )

    # Wanderer: center row (y=4 or y=5), random x, direction=1 (east)
    k_wy, k_wx, k3 = jax.random.split(k3, 3)
    wanderer_y = jax.random.randint(k_wy, (), 4, 6)  # 4 or 5
    wanderer_x = jax.random.randint(k_wx, (), 0, 10)

    wanderer_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[2].set(True)  # hazard
    wanderer_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32).at[0].set(1.0)  # direction=east
    state, _ = create_entity(
        state, CONFIG, jnp.int32(3), wanderer_x, wanderer_y, wanderer_tags, wanderer_props
    )

    state = state.replace(rng_key=k3)
    obs = get_obs(state, CONFIG)
    return state, obs


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    # Phase 1: Player movement
    state = movement_system(state, action, CONFIG)

    # Phase 1.5: Check if player stepped onto hazard
    state = collision_system(state, CONFIG)

    # Phase 2: NPC behaviors (wanderer bounces, moves every other turn)
    should_move = (state.turn_number % 2 == 0)
    new_state = behavior_system(state, CONFIG)
    state = jax.tree.map(
        lambda n, o: jnp.where(should_move, n, o), new_state, state
    )

    # Phase 3: Check if hazard stepped onto player, or player on exit
    state = collision_system(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    # Exactly one wanderer
    wanderer_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("exactly_one_wanderer", wanderer_count == 1))

    # Wanderer in center rows
    w_mask = state.alive & (state.entity_type == 3)
    w_slot = int(jnp.argmax(w_mask))
    wy = int(state.y[w_slot])
    results.append(("wanderer_center_row", wy == 4 or wy == 5))

    # Player in bottom-left quadrant
    pidx = int(state.player_idx)
    px, py = int(state.x[pidx]), int(state.y[pidx])
    results.append(("player_bottom_left", px < 5 and py >= 5))

    # Exit in top-right quadrant
    e_mask = state.alive & (state.entity_type == 2)
    e_slot = int(jnp.argmax(e_mask))
    ex, ey = int(state.x[e_slot]), int(state.y[e_slot])
    results.append(("exit_top_right", ex >= 5 and ey < 5))

    # Player and wanderer not on same cell
    diff = (px != int(state.x[w_slot])) or (py != int(state.y[w_slot]))
    results.append(("player_wanderer_different_cell", diff))

    return results
