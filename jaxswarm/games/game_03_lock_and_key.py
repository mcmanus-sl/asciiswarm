"""Game 03: Lock & Key — find key, unlock door, reach exit. Multi-step dependency chain."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import create_entity, destroy_entity, get_entities_at
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import movement_system
from jaxswarm.systems.collision import collision_system
from jaxswarm.systems.interact import interact_system


CONFIG = EnvConfig(
    grid_w=12, grid_h=12,
    max_entities=64,
    max_stack=2,
    num_entity_types=6,    # 0=unused, 1=player, 2=exit, 3=wall, 4=door, 5=key
    num_tags=6,
    num_props=2,           # 0=has_key, 1=unused
    num_actions=6,
    max_turns=300,
    step_penalty=-0.01,
    game_state_size=2,     # 0=key_picked_up, 1=door_unlocked
    prop_maxes=(1.0, 1.0),
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace for seed 0. Layout: Player(3,3), Key(5,10), Door(8,3), Exit(10,3).
# Corridor 1 at (4,2), Corridor 2 at (8,3) blocked by door.
# Route: north to corridor row, east through corridor, south to key, north to door, interact, east to exit.
DETERMINISTIC_TRACE = (
    [0] * 1 +        # north 1: (3,3) -> (3,2)
    [2] * 2 +        # east 2: through corridor at (4,2) -> (5,2)
    [1] * 8 +        # south 8: -> (5,10), picks up key
    [0] * 7 +        # north 7: -> (5,3)
    [2] * 2 +        # east 2: -> (7,3), adjacent to door at (8,3)
    [4] +            # interact: unlocks door (has key)
    [2] * 3          # east 3: through (8,3) opened corridor -> (10,3) = exit, WIN
)


def _place_walls(state, config, wall_x, corridor_y):
    """Place a vertical wall at column wall_x with a 1-tile corridor at corridor_y."""
    wall_tags = jnp.zeros(config.num_tags, dtype=jnp.bool_).at[1].set(True)  # solid
    wall_props = jnp.zeros(config.num_props, dtype=jnp.float32)

    def place_wall_tile(carry, y):
        state = carry
        is_corridor = (y == corridor_y)
        new_state, _ = create_entity(
            state, config, jnp.int32(3), jnp.int32(wall_x), y, wall_tags, wall_props
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(~is_corridor, n, o), new_state, state
        )
        return state, None

    state, _ = jax.lax.scan(place_wall_tile, state, jnp.arange(config.grid_h, dtype=jnp.int32))
    return state


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng_key, 7)

    corridor1_y = jax.random.randint(k1, (), 1, CONFIG.grid_h - 1)
    corridor2_y = jax.random.randint(k2, (), 1, CONFIG.grid_h - 1)

    # Place wall 1 at x=4 (open corridor)
    state = _place_walls(state, CONFIG, 4, corridor1_y)
    # Place wall 2 at x=8 (door will block corridor)
    state = _place_walls(state, CONFIG, 8, corridor2_y)

    # Place door at corridor 2
    door_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[1].set(True)  # solid
    door_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(4), jnp.int32(8), corridor2_y, door_tags, door_props
    )

    # Player in room 1 (x 0-3)
    player_x = jax.random.randint(k3, (), 0, 4)
    player_y = jax.random.randint(k4, (), 0, CONFIG.grid_h)
    player_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
    player_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, player_slot = create_entity(
        state, CONFIG, jnp.int32(1), player_x, player_y, player_tags, player_props
    )
    state = state.replace(player_idx=player_slot)

    # Key in room 2 (x 5-7)
    key_x = jax.random.randint(k5, (), 5, 8)
    key_y = jax.random.randint(k6, (), 0, CONFIG.grid_h)
    key_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)  # pickup
    key_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(5), key_x, key_y, key_tags, key_props
    )

    # Exit in room 3 (x 9-11)
    exit_x = jax.random.randint(k7, (), 9, 12)
    exit_y = jax.random.randint(jax.random.split(k7)[0], (), 0, CONFIG.grid_h)
    exit_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[4].set(True)
    exit_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(2), exit_x, exit_y, exit_tags, exit_props
    )

    state = state.replace(rng_key=jax.random.split(k7)[1])
    obs = get_obs(state, CONFIG)
    return state, obs


def _pickup_system(state: EnvState) -> EnvState:
    """Check if player is standing on a key. If so, pick it up."""
    pidx = state.player_idx
    px = state.x[pidx]
    py = state.y[pidx]

    slots, count = get_entities_at(state, px, py)

    def check_slot(j, state):
        slot = slots[j]
        is_valid = j < count
        is_key = is_valid & state.alive[slot] & (state.entity_type[slot] == 5)

        new_state = destroy_entity(state, CONFIG, slot)
        new_state = new_state.replace(
            properties=new_state.properties.at[pidx, 0].set(1.0),
            reward_acc=new_state.reward_acc + 2.0,
            game_state=new_state.game_state.at[0].set(1.0),
        )

        state = jax.tree.map(
            lambda n, o: jnp.where(is_key, n, o), new_state, state
        )
        return state

    return jax.lax.fori_loop(0, CONFIG.max_stack, check_slot, state)


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    # Phase 1: Player movement (blocked by solid entities)
    state = movement_system(state, action, CONFIG)

    # Phase 1.5: Check for key pickup after movement
    state = _pickup_system(state)

    # Phase 2: Interact system (unlock door if adjacent and has key)
    state = interact_system(state, action, CONFIG)

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

    key_count = int((state.alive & (state.entity_type == 5)).sum())
    results.append(("exactly_one_key", key_count == 1))

    door_count = int((state.alive & (state.entity_type == 4)).sum())
    results.append(("exactly_one_door", door_count == 1))

    pidx = int(state.player_idx)
    px = int(state.x[pidx])
    results.append(("player_in_room_1", px <= 3))

    k_mask = state.alive & (state.entity_type == 5)
    k_slot = int(jnp.argmax(k_mask))
    kx = int(state.x[k_slot])
    results.append(("key_in_room_2", 5 <= kx <= 7))

    e_mask = state.alive & (state.entity_type == 2)
    e_slot = int(jnp.argmax(e_mask))
    ex = int(state.x[e_slot])
    results.append(("exit_in_room_3", ex >= 9))

    has_key = float(state.properties[pidx, 0])
    results.append(("player_no_key", has_key == 0.0))

    d_mask = state.alive & (state.entity_type == 4)
    d_slot = int(jnp.argmax(d_mask))
    dx = int(state.x[d_slot])
    results.append(("door_at_wall_2", dx == 8))

    return results
