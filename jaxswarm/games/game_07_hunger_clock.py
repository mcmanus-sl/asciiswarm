"""Game 07: Hunger Clock — survive 100 turns by managing food."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import DX, DY

# No border walls — move_entity rejects OOB. No create_entity in reset or step.
# Slot layout: 0=player, 1-6=walls, 7-26=food (no exit — survive 100 turns to win)
CONFIG = EnvConfig(
    grid_w=14, grid_h=14,
    max_entities=27,
    max_stack=2,
    num_entity_types=5,    # 0=unused, 1=player, 2=unused, 3=food, 4=wall
    num_tags=6,
    num_props=2,           # 0=food, 1=unused
    num_actions=6,
    max_turns=100,
    step_penalty=-0.005,
    game_state_size=2,     # 0=food_eaten_count, 1=unused
    prop_maxes=(30.0, 1.0),
    max_behaviors=2,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace: collect food to survive 100 turns. Player starts near bottom
# (x=1-3, y=10-12). Sweep bottom-up via boustrophedon to find food quickly.
# N=0,S=1,E=2,W=3.
_trace07 = []
for row in range(14):
    if row % 2 == 0:
        _trace07 += [2] * 13  # east across
    else:
        _trace07 += [3] * 13  # west across
    _trace07 += [0]           # north one row
DETERMINISTIC_TRACE = _trace07[:100]


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 6)

    # Vectorized bulk init
    k_px, k_py = jax.random.split(keys[0])
    player_x = jax.random.randint(k_px, (), 1, 4)
    player_y = jax.random.randint(k_py, (), 10, 13)

    # keys[1] consumed but unused (was exit position)

    k_wx, k_wy = jax.random.split(keys[2])
    wall_xs = jax.random.randint(k_wx, (6,), 4, 10)
    wall_ys = jax.random.randint(k_wy, (6,), 4, 10)
    wall_xs = wall_xs.at[1].set(jnp.clip(wall_xs[0] + 1, 0, 13))
    wall_ys = wall_ys.at[1].set(wall_ys[0])
    wall_xs = wall_xs.at[3].set(jnp.clip(wall_xs[2] + 1, 0, 13))
    wall_ys = wall_ys.at[3].set(wall_ys[2])
    wall_xs = wall_xs.at[5].set(jnp.clip(wall_xs[4] + 1, 0, 13))
    wall_ys = wall_ys.at[5].set(wall_ys[4])

    k_fx, k_fy = jax.random.split(keys[3])
    food_xs = jax.random.randint(k_fx, (20,), 0, 14)
    food_ys = jax.random.randint(k_fy, (20,), 0, 14)

    n = CONFIG.max_entities
    alive = jnp.zeros(n, dtype=jnp.bool_)
    entity_type = jnp.zeros(n, dtype=jnp.int32)
    x = jnp.zeros(n, dtype=jnp.int32)
    y = jnp.zeros(n, dtype=jnp.int32)
    tags = jnp.zeros((n, CONFIG.num_tags), dtype=jnp.bool_)
    properties = jnp.zeros((n, CONFIG.num_props), dtype=jnp.float32)

    # Slot 0: player
    alive = alive.at[0].set(True)
    entity_type = entity_type.at[0].set(1)
    x = x.at[0].set(player_x)
    y = y.at[0].set(player_y)
    tags = tags.at[0, 0].set(True)
    properties = properties.at[0, 0].set(20.0)

    # No exit — survive 100 turns to win

    # Slots 1-6: walls
    alive = alive.at[1:7].set(True)
    entity_type = entity_type.at[1:7].set(4)
    x = x.at[1:7].set(wall_xs)
    y = y.at[1:7].set(wall_ys)
    tags = tags.at[1:7, 1].set(True)

    # Slots 7-26: food
    alive = alive.at[7:27].set(True)
    entity_type = entity_type.at[7:27].set(3)
    x = x.at[7:27].set(food_xs)
    y = y.at[7:27].set(food_ys)
    tags = tags.at[7:27, 3].set(True)

    state = state.replace(
        alive=alive, entity_type=entity_type, x=x, y=y,
        tags=tags, properties=properties,
        player_idx=jnp.int32(0),
        rng_key=keys[4],
    )
    state = rebuild_grid(state, CONFIG)
    obs = get_obs(state, CONFIG)
    return state, obs


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    pidx = state.player_idx
    px, py = state.x[pidx], state.y[pidx]
    is_move = action < 4

    target_x = px + DX[action]
    target_y = py + DY[action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)

    # Vectorized: check if any alive solid entity is at target cell
    at_target = state.alive & (state.x == target_x) & (state.y == target_y)
    has_solid = (at_target & state.tags[:, 1]).any()

    # Move player directly via array update (no move_entity/grid mutation)
    can_move = is_move & in_bounds & ~has_solid
    new_px = jnp.where(can_move, target_x, px)
    new_py = jnp.where(can_move, target_y, py)
    state = state.replace(
        x=state.x.at[pidx].set(new_px),
        y=state.y.at[pidx].set(new_py),
    )

    # Vectorized food collection: tombstone all food at player's new position
    is_eaten = state.alive & (state.entity_type == 3) & (state.x == new_px) & (state.y == new_py)
    food_eaten_count = is_eaten.sum()
    new_food = jnp.minimum(state.properties[pidx, 0] + food_eaten_count * 10.0, 30.0)
    state = state.replace(
        alive=state.alive & ~is_eaten,
        properties=state.properties.at[pidx, 0].set(
            jnp.where(food_eaten_count > 0, new_food, state.properties[pidx, 0])
        ),
        reward_acc=state.reward_acc + food_eaten_count * 0.05,
        game_state=state.game_state.at[0].set(state.game_state[0] + food_eaten_count),
    )

    # Hunger tick
    food = state.properties[pidx, 0] - 1.0
    starved = (food <= 0) & (state.status == 0)
    state = state.replace(
        properties=state.properties.at[pidx, 0].set(food),
        status=jnp.where(starved, jnp.int32(-1), state.status),
    )

    # Autowin: survive 100 turns
    survived = (state.turn_number >= CONFIG.max_turns) & (state.status == 0)
    state = state.replace(
        status=jnp.where(survived, jnp.int32(1), state.status),
    )

    # No rebuild_grid needed — get_obs builds obs directly from entity arrays.
    # Grid tensor is stale but unused by vectorized step logic.
    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -5.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    pidx = int(state.player_idx)
    food = float(state.properties[pidx, 0])
    results.append(("player_food_20", food == 20.0))

    food_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("food_count_ok", 10 <= food_count <= 25))

    return results
