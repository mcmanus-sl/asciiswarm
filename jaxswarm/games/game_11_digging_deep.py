"""Game 11: Digging Deep — mine adamantine near magma, build barricades, escape."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import DX, DY
from jaxswarm.systems.magma import magma_system

CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=256,
    max_stack=2,
    num_entity_types=7,    # 0=unused, 1=player, 2=exit, 3=rock, 4=magma, 5=adamantine, 6=barricade
    num_tags=7,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=mineable, 6=defense
    num_props=3,           # 0=adamantine, 1=stone, 2=unused
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=4,     # 0=adamantine_mined, 1=barricades_built, 2=magma_cells, 3=panic_walls
    prop_maxes=(5.0, 20.0, 1.0),
    max_behaviors=2,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace for seed 0:
# Player starts in safe zone (x<4, y<4). Exit at (15,0).
# Adamantine is adjacent to magma pocket at (7-9, 7-9).
# Strategy: mine east through rock, collect adamantine, build barricades, escape to exit.
# N=0, S=1, E=2, W=3, I=4, WAIT=5
_trace11 = (
    [2] * 5 +              # mine east from safe zone toward center (x~4 to x~9)
    [1] * 4 +              # mine south toward magma pocket
    [2] * 2 +              # mine east to reach adamantine
    [1] +                  # mine adamantine (south of pocket at y=10, x~9)
    [0] +                  # mine north (adamantine)
    [2] +                  # mine east (adamantine)
    [3] +                  # mine west (adamantine)
    [1] +                  # mine south (adamantine — 5th)
    # Now have 5 adamantine, head to exit at (15, 0)
    [0] * 12 +             # mine north to top
    [2] * 6                 # mine east to x=15 — should hit exit
)
DETERMINISTIC_TRACE = _trace11[:395]


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 10)

    n = CONFIG.max_entities
    alive = jnp.zeros(n, dtype=jnp.bool_)
    entity_type = jnp.zeros(n, dtype=jnp.int32)
    x = jnp.zeros(n, dtype=jnp.int32)
    y = jnp.zeros(n, dtype=jnp.int32)
    tags = jnp.zeros((n, CONFIG.num_tags), dtype=jnp.bool_)
    properties = jnp.zeros((n, CONFIG.num_props), dtype=jnp.float32)

    # Slot 0: Player in safe zone
    k_px, k_py = jax.random.split(keys[0])
    player_x = jax.random.randint(k_px, (), 1, 3)
    player_y = jax.random.randint(k_py, (), 1, 3)
    alive = alive.at[0].set(True)
    entity_type = entity_type.at[0].set(1)
    x = x.at[0].set(player_x)
    y = y.at[0].set(player_y)
    tags = tags.at[0, 0].set(True)  # player tag

    # Slot 1: Exit at (15, 0)
    alive = alive.at[1].set(True)
    entity_type = entity_type.at[1].set(2)
    x = x.at[1].set(15)
    y = y.at[1].set(0)
    tags = tags.at[1, 4].set(True)  # exit tag

    # Slots 2-10: Magma 3x3 pocket at (7-9, 7-9) = 9 cells
    magma_idx = jnp.arange(9, dtype=jnp.int32)
    magma_xs = 7 + (magma_idx % 3)
    magma_ys = 7 + (magma_idx // 3)
    slot_start = 2
    alive = alive.at[slot_start:slot_start + 9].set(True)
    entity_type = entity_type.at[slot_start:slot_start + 9].set(4)  # magma
    x = x.at[slot_start:slot_start + 9].set(magma_xs)
    y = y.at[slot_start:slot_start + 9].set(magma_ys)
    tags = tags.at[slot_start:slot_start + 9, 2].set(True)  # hazard

    # Slots 11-15: Adamantine — 5 blocks adjacent to magma pocket
    # Place them at: (6,7), (6,8), (6,9), (10,7), (10,8)
    adam_positions = jnp.array([
        [6, 7], [6, 8], [6, 9], [10, 7], [10, 8]
    ], dtype=jnp.int32)
    slot_start = 11
    alive = alive.at[slot_start:slot_start + 5].set(True)
    entity_type = entity_type.at[slot_start:slot_start + 5].set(5)  # adamantine
    x = x.at[slot_start:slot_start + 5].set(adam_positions[:, 0])
    y = y.at[slot_start:slot_start + 5].set(adam_positions[:, 1])
    tags = tags.at[slot_start:slot_start + 5, 1].set(True)  # solid
    tags = tags.at[slot_start:slot_start + 5, 5].set(True)  # mineable

    # Slots 16+: Rock — fill all cells that are not:
    #   - safe zone (x < 4, y < 4)
    #   - magma pocket (7-9, 7-9)
    #   - adamantine positions
    #   - exit (15, 0)
    # Build rock positions programmatically
    all_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(16, dtype=jnp.int32),
        jnp.arange(16, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)  # [256, 2]

    ax = all_coords[:, 0]
    ay = all_coords[:, 1]

    # Masks for excluded regions
    is_safe = (ax < 4) & (ay < 4)
    is_magma = (ax >= 7) & (ax <= 9) & (ay >= 7) & (ay <= 9)
    is_exit = (ax == 15) & (ay == 0)

    # Check adamantine positions
    is_adam = jnp.zeros(256, dtype=jnp.bool_)
    for i in range(5):
        is_adam = is_adam | ((ax == adam_positions[i, 0]) & (ay == adam_positions[i, 1]))

    is_rock = ~is_safe & ~is_magma & ~is_exit & ~is_adam

    # Assign rock entities to slots 16+
    # Use cumsum to get slot indices for rocks
    rock_count = is_rock.sum()
    rock_slot_offsets = jnp.cumsum(is_rock.astype(jnp.int32)) - 1  # 0-indexed
    rock_slots = 16 + rock_slot_offsets  # slot indices

    # For each position, conditionally set entity data
    alive = alive.at[rock_slots].set(jnp.where(is_rock, True, alive[rock_slots]))
    entity_type = entity_type.at[rock_slots].set(jnp.where(is_rock, 3, entity_type[rock_slots]))
    x = x.at[rock_slots].set(jnp.where(is_rock, ax, x[rock_slots]))
    y = y.at[rock_slots].set(jnp.where(is_rock, ay, y[rock_slots]))
    tags = tags.at[rock_slots, 1].set(jnp.where(is_rock, True, tags[rock_slots, 1]))  # solid
    tags = tags.at[rock_slots, 5].set(jnp.where(is_rock, True, tags[rock_slots, 5]))  # mineable

    state = state.replace(
        alive=alive, entity_type=entity_type, x=x, y=y,
        tags=tags, properties=properties,
        player_idx=jnp.int32(0),
        rng_key=keys[1],
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
    is_interact = (action == 4)

    # --- Phase 1: Process Input ---

    # Compute target cell for movement
    target_x = px + DX[action]
    target_y = py + DY[action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)

    # Check what's at target cell (vectorized)
    at_target = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_solid = (at_target & state.tags[:, 1]).any()
    has_mineable = (at_target & state.tags[:, 5]).any()
    has_exit = (at_target & state.tags[:, 4]).any()
    has_hazard = (at_target & state.tags[:, 2]).any()

    # --- Mining: move into mineable = destroy it + collect resource ---
    # Find the first mineable entity at target
    mineable_mask = at_target & state.tags[:, 5]
    mineable_slot = jnp.argmax(mineable_mask)  # first True
    mineable_etype = state.entity_type[mineable_slot]

    can_mine = is_move & in_bounds & has_mineable

    # Destroy mined entity (tombstone)
    mined_alive = state.alive.at[mineable_slot].set(False)
    mined_etype = state.entity_type.at[mineable_slot].set(0)
    mined_tags = state.tags.at[mineable_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))
    mined_props = state.properties.at[mineable_slot].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))

    # Determine what was mined
    is_rock_mined = can_mine & (mineable_etype == 3)
    is_adam_mined = can_mine & (mineable_etype == 5)

    # Update player inventory
    new_stone = jnp.minimum(state.properties[pidx, 1] + 1.0, 20.0)
    new_adam = jnp.minimum(state.properties[pidx, 0] + 1.0, 5.0)

    mine_props = mined_props
    mine_props = mine_props.at[pidx, 1].set(
        jnp.where(is_rock_mined, new_stone, state.properties[pidx, 1])
    )
    mine_props = mine_props.at[pidx, 0].set(
        jnp.where(is_adam_mined, new_adam, state.properties[pidx, 0])
    )

    # Apply mining (player stays in place — "swinging pickaxe")
    mine_reward = jnp.where(is_adam_mined, 0.3, 0.0)
    mine_gs = state.game_state.at[0].set(
        jnp.where(is_adam_mined, state.game_state[0] + 1, state.game_state[0])
    )

    state = state.replace(
        alive=jnp.where(can_mine, mined_alive, state.alive),
        entity_type=jnp.where(can_mine, mined_etype, state.entity_type),
        tags=jnp.where(can_mine, mined_tags, state.tags),
        properties=jnp.where(can_mine, mine_props, state.properties),
        reward_acc=state.reward_acc + mine_reward,
        game_state=jnp.where(can_mine, mine_gs, state.game_state),
    )

    # --- Normal movement (only if not mining and not blocked) ---
    can_move = is_move & in_bounds & ~has_solid & ~can_mine
    new_x = state.x.at[pidx].set(jnp.where(can_move, target_x, state.x[pidx]))
    new_y = state.y.at[pidx].set(jnp.where(can_move, target_y, state.y[pidx]))
    state = state.replace(x=new_x, y=new_y)

    # Win on exit (need 5 adamantine)
    new_px, new_py = state.x[pidx], state.y[pidx]
    at_exit = (new_px == 15) & (new_py == 0)
    has_enough = state.properties[pidx, 0] >= 5
    win = can_move & at_exit & has_enough & (state.status == 0)
    state = state.replace(
        status=jnp.where(win, jnp.int32(1), state.status),
        reward_acc=state.reward_acc + jnp.where(win, 10.0, 0.0),
    )

    # --- Interact: build barricade ---
    # Place barricade at last position (where player was before current step)
    has_stone = state.properties[pidx, 1] >= 2
    can_build = is_interact & has_stone

    # Find free slot for barricade
    free_slot = jnp.argmin(state.alive)
    has_free = ~state.alive.all()
    do_build = can_build & has_free

    barricade_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[1].set(True).at[6].set(True)
    barricade_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

    built_alive = state.alive.at[free_slot].set(True)
    built_etype = state.entity_type.at[free_slot].set(6)
    # Place barricade at player's current position
    built_x = state.x.at[free_slot].set(new_px)
    built_y = state.y.at[free_slot].set(new_py)
    built_tags = state.tags.at[free_slot].set(barricade_tags)
    built_props_arr = state.properties.at[free_slot].set(barricade_props)
    # Deduct stone
    built_props_arr = built_props_arr.at[pidx, 1].set(state.properties[pidx, 1] - 2.0)

    # Panic wall bonus: check if any magma within Chebyshev distance 2
    is_magma = state.alive & (state.entity_type == 4)
    magma_dist_x = jnp.abs(state.x - new_px)
    magma_dist_y = jnp.abs(state.y - new_py)
    magma_chebyshev = jnp.maximum(magma_dist_x, magma_dist_y)
    magma_near = (is_magma & (magma_chebyshev <= 2)).any()
    panic_bonus = jnp.where(do_build & magma_near, 0.5, 0.0)

    state = state.replace(
        alive=jnp.where(do_build, built_alive, state.alive),
        entity_type=jnp.where(do_build, built_etype, state.entity_type),
        x=jnp.where(do_build, built_x, state.x),
        y=jnp.where(do_build, built_y, state.y),
        tags=jnp.where(do_build, built_tags, state.tags),
        properties=jnp.where(do_build, built_props_arr, state.properties),
        reward_acc=state.reward_acc + jnp.where(do_build, 0.1, 0.0) + panic_bonus,
        game_state=state.game_state
            .at[1].set(jnp.where(do_build, state.game_state[1] + 1, state.game_state[1]))
            .at[3].set(jnp.where(do_build & magma_near, state.game_state[3] + 1, state.game_state[3])),
    )

    # --- Phase 2: Magma CA spread ---
    state = magma_system(state, CONFIG, magma_type=4, spread_chance=0.2)

    # Check if magma killed player
    magma_at_player = state.alive & (state.entity_type == 4) & \
                      (state.x == state.x[pidx]) & (state.y == state.y[pidx])
    magma_death = magma_at_player.any() & (state.status == 0)
    state = state.replace(
        status=jnp.where(magma_death, jnp.int32(-1), state.status),
    )

    # Update magma cell count in game_state
    magma_count = (state.alive & (state.entity_type == 4)).sum().astype(jnp.float32)
    state = state.replace(
        game_state=state.game_state.at[2].set(magma_count),
    )

    # Rebuild grid after all entity changes
    state = rebuild_grid(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []

    # 1. Exactly 5 adamantine blocks at start
    adam_count = int((state.alive & (state.entity_type == 5)).sum())
    results.append(("five_adamantine", adam_count == 5))

    # 2. All adamantine adjacent to magma pocket (7-9, 7-9)
    adam_mask = state.alive & (state.entity_type == 5)
    magma_mask = state.alive & (state.entity_type == 4)
    # For each adamantine, check if any magma is within Manhattan distance 1
    adam_xs = jnp.where(adam_mask, state.x, -100)
    adam_ys = jnp.where(adam_mask, state.y, -100)
    magma_xs = jnp.where(magma_mask, state.x, -100)
    magma_ys = jnp.where(magma_mask, state.y, -100)
    # Check: for each adam, is there a magma within Chebyshev dist 1?
    all_adjacent = True
    # Simplified: check adamantine x range [6,10] and y range [6,10] (adjacent to 7-9)
    adam_in_range = adam_mask & (state.x >= 6) & (state.x <= 10) & (state.y >= 6) & (state.y <= 10)
    results.append(("adamantine_near_magma", int(adam_in_range.sum()) == adam_count))

    # 3. Player starts in safe zone
    pidx = int(state.player_idx)
    px, py = int(state.x[pidx]), int(state.y[pidx])
    results.append(("player_in_safe_zone", px < 4 and py < 4))

    # 4. Exit at (15, 0)
    exit_mask = state.alive & (state.entity_type == 2)
    exit_count = int(exit_mask.sum())
    results.append(("one_exit", exit_count == 1))
    if exit_count == 1:
        eidx = int(jnp.argmax(exit_mask))
        results.append(("exit_at_15_0", int(state.x[eidx]) == 15 and int(state.y[eidx]) == 0))
    else:
        results.append(("exit_at_15_0", False))

    # 5. Magma pocket is 3x3 at center
    magma_count = int(magma_mask.sum())
    results.append(("nine_magma_cells", magma_count == 9))

    # 6. Safe zone empty of rock
    rock_mask = state.alive & (state.entity_type == 3)
    rock_in_safe = rock_mask & (state.x < 4) & (state.y < 4)
    results.append(("safe_zone_clear", int(rock_in_safe.sum()) == 0))

    return results
