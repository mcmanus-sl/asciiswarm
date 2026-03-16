"""Game 13: Siege Architecture — mine corridors, build traps, funnel goblins."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.movement import move_toward
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=14, grid_h=14,
    max_entities=128,
    max_stack=3,           # trap under goblin
    num_entity_types=6,    # 0=unused, 1=player, 2=rock, 3=gears, 4=trap, 5=goblin
    num_tags=7,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=mineable, 6=trap
    num_props=3,           # 0=mechanisms (player), 1=health (goblin), 2=unused
    num_actions=6,
    max_turns=300,
    step_penalty=-0.005,
    game_state_size=4,     # 0=goblins_alive, 1=traps_built, 2=trap_kills, 3=turn_goblins_spawn
    prop_maxes=(5.0, 3.0, 1.0),
    max_behaviors=8,       # up to 3 goblins + player
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace: mine east, collect gears, build traps in corridor, wait for goblins
# N=0, S=1, E=2, W=3, I=4, WAIT=5
_trace13 = (
    # Phase 1: Mine east from center (7,7) to collect gears and reach open area
    [2] * 4 +              # mine east (hit gears on the way)
    [0] * 3 +              # mine north
    [2] * 2 +              # mine east — should hit a gear
    [1] * 3 +              # mine south — should hit another gear
    [2] * 2 +              # mine east to x=10 border
    # Phase 2: Build traps in the open area (x>10)
    [4] +                  # build trap at current position (costs 1 mechanism)
    [1] + [4] +            # move south, build another trap
    [1] + [4] +            # move south, build 3rd trap
    # Phase 3: Wait for goblins (turn 100)
    [5] * 85 +             # wait until goblin spawn
    # Goblins spawn and walk into traps
    [5] * 30               # wait for goblins to reach traps
)
DETERMINISTIC_TRACE = _trace13[:295]

# Goblin spawn positions (right edge at x=13)
GOBLIN_SPAWNS = jnp.array([
    [13, 1], [13, 7], [13, 12]
], dtype=jnp.int32)

GOBLIN_SPAWN_TURN = 100


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

    # Slot 0: Player at center (7, 7)
    alive = alive.at[0].set(True)
    entity_type = entity_type.at[0].set(1)
    x = x.at[0].set(7)
    y = y.at[0].set(7)
    tags = tags.at[0, 0].set(True)  # player tag

    # Slots 1-3: Gears inside rock mass (pre-allocated, alive=False until found)
    # Place gears at specific positions within rock
    gear_positions = jnp.array([
        [5, 4], [8, 6], [4, 9]
    ], dtype=jnp.int32)
    for i in range(3):
        slot = 1 + i
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(3)  # gears
        x = x.at[slot].set(gear_positions[i, 0])
        y = y.at[slot].set(gear_positions[i, 1])
        tags = tags.at[slot, 3].set(True)   # pickup
        tags = tags.at[slot, 1].set(True)   # solid (embedded in rock, mined to access)

    # Slots 4-6: Pre-allocated goblin slots (alive=False, spawn at turn 100)
    for i in range(3):
        slot = 4 + i
        # alive=False — will be flipped at turn 100
        entity_type = entity_type.at[slot].set(5)  # goblin
        x = x.at[slot].set(GOBLIN_SPAWNS[i, 0])
        y = y.at[slot].set(GOBLIN_SPAWNS[i, 1])
        tags = tags.at[slot, 2].set(True)   # hazard
        properties = properties.at[slot, 1].set(3.0)  # 3 HP

    # Slots 7+: Rock fill (x=0 to x=10)
    all_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(14, dtype=jnp.int32),
        jnp.arange(14, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)  # [196, 2]

    ax = all_coords[:, 0]
    ay = all_coords[:, 1]

    # Rock fills x=0 to x=10, excluding player position and gear positions
    is_rock_zone = ax <= 10
    is_player = (ax == 7) & (ay == 7)
    is_gear = jnp.zeros(196, dtype=jnp.bool_)
    for i in range(3):
        is_gear = is_gear | ((ax == gear_positions[i, 0]) & (ay == gear_positions[i, 1]))

    is_rock = is_rock_zone & ~is_player & ~is_gear

    rock_slot_offsets = jnp.cumsum(is_rock.astype(jnp.int32)) - 1
    rock_slots = 7 + rock_slot_offsets

    alive = alive.at[rock_slots].set(jnp.where(is_rock, True, alive[rock_slots]))
    entity_type = entity_type.at[rock_slots].set(jnp.where(is_rock, 2, entity_type[rock_slots]))
    x = x.at[rock_slots].set(jnp.where(is_rock, ax, x[rock_slots]))
    y = y.at[rock_slots].set(jnp.where(is_rock, ay, y[rock_slots]))
    tags = tags.at[rock_slots, 1].set(jnp.where(is_rock, True, tags[rock_slots, 1]))  # solid
    tags = tags.at[rock_slots, 5].set(jnp.where(is_rock, True, tags[rock_slots, 5]))  # mineable

    # Set game_state[3] = spawn turn
    game_state = jnp.zeros(CONFIG.game_state_size, dtype=jnp.float32)
    game_state = game_state.at[3].set(float(GOBLIN_SPAWN_TURN))

    state = state.replace(
        alive=alive, entity_type=entity_type, x=x, y=y,
        tags=tags, properties=properties,
        player_idx=jnp.int32(0),
        game_state=game_state,
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
    target_x = px + DX[action]
    target_y = py + DY[action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)

    at_target = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_mineable = (at_target & state.tags[:, 5]).any()
    has_solid = (at_target & state.tags[:, 1]).any()

    # --- Mining ---
    mineable_mask = at_target & state.tags[:, 5]
    mineable_slot = jnp.argmax(mineable_mask)
    mineable_etype = state.entity_type[mineable_slot]

    can_mine = is_move & in_bounds & has_mineable

    # Destroy mined entity
    mined_alive = state.alive.at[mineable_slot].set(False)
    mined_etype_arr = state.entity_type.at[mineable_slot].set(0)
    mined_tags = state.tags.at[mineable_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))
    mined_props = state.properties.at[mineable_slot].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))

    # If gears (type 3): collect mechanism
    is_gear_mined = can_mine & (mineable_etype == 3)
    new_mechanisms = jnp.minimum(state.properties[pidx, 0] + 1.0, 5.0)
    gear_props = mined_props.at[pidx, 0].set(
        jnp.where(is_gear_mined, new_mechanisms, state.properties[pidx, 0])
    )

    state = state.replace(
        alive=jnp.where(can_mine, mined_alive, state.alive),
        entity_type=jnp.where(can_mine, mined_etype_arr, state.entity_type),
        tags=jnp.where(can_mine, mined_tags, state.tags),
        properties=jnp.where(can_mine, gear_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(is_gear_mined, 0.1, 0.0),
    )

    # --- Normal movement ---
    at_target2 = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_solid2 = (at_target2 & state.tags[:, 1]).any()
    can_move = is_move & in_bounds & ~has_solid2 & ~can_mine
    new_x = state.x.at[pidx].set(jnp.where(can_move, target_x, state.x[pidx]))
    new_y = state.y.at[pidx].set(jnp.where(can_move, target_y, state.y[pidx]))
    state = state.replace(x=new_x, y=new_y)

    # --- Pickup gears at new position ---
    new_px, new_py = state.x[pidx], state.y[pidx]
    at_player = state.alive & (state.x == new_px) & (state.y == new_py)
    gear_at_player = at_player & (state.entity_type == 3) & state.tags[:, 3]  # pickup gears
    has_gear = gear_at_player.any()
    pickup_mechs = jnp.minimum(state.properties[pidx, 0] + gear_at_player.sum().astype(jnp.float32), 5.0)
    state = state.replace(
        alive=jnp.where(has_gear, state.alive & ~gear_at_player, state.alive),
        properties=jnp.where(has_gear,
                             state.properties.at[pidx, 0].set(pickup_mechs),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_gear, 0.1, 0.0),
    )

    # --- Interact: build trap ---
    has_mechanisms = state.properties[pidx, 0] >= 1
    free_slot = jnp.argmin(state.alive)
    has_free = ~state.alive.all()
    can_build = is_interact & has_mechanisms & has_free

    trap_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[6].set(True)
    trap_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

    built_alive = state.alive.at[free_slot].set(True)
    built_etype = state.entity_type.at[free_slot].set(4)  # trap
    built_x = state.x.at[free_slot].set(new_px)
    built_y = state.y.at[free_slot].set(new_py)
    built_tags = state.tags.at[free_slot].set(trap_tags)
    built_props = state.properties.at[free_slot].set(trap_props)
    built_props = built_props.at[pidx, 0].set(state.properties[pidx, 0] - 1.0)

    state = state.replace(
        alive=jnp.where(can_build, built_alive, state.alive),
        entity_type=jnp.where(can_build, built_etype, state.entity_type),
        x=jnp.where(can_build, built_x, state.x),
        y=jnp.where(can_build, built_y, state.y),
        tags=jnp.where(can_build, built_tags, state.tags),
        properties=jnp.where(can_build, built_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_build, 0.05, 0.0),
        game_state=state.game_state.at[1].set(
            jnp.where(can_build, state.game_state[1] + 1, state.game_state[1])
        ),
    )

    # --- Phase 2: Goblin spawn at turn 100 ---
    should_spawn = state.turn_number == GOBLIN_SPAWN_TURN
    # Flip alive=True for pre-allocated goblin slots (4, 5, 6)
    spawn_alive = state.alive.at[4].set(True).at[5].set(True).at[6].set(True)
    state = state.replace(
        alive=jnp.where(should_spawn, spawn_alive, state.alive),
        game_state=state.game_state.at[0].set(
            jnp.where(should_spawn, 3.0, state.game_state[0])
        ),
    )

    # --- Goblin movement: greedy Manhattan toward player ---
    # Only move on even turns after spawn (give player breathing room)
    goblins_active = (state.turn_number > GOBLIN_SPAWN_TURN) & (state.turn_number % 2 == 0)

    def move_goblin(state, goblin_slot):
        is_alive = state.alive[goblin_slot] & (state.entity_type[goblin_slot] == 5)
        should_move = goblins_active & is_alive

        key, subkey = jax.random.split(state.rng_key)
        state = state.replace(rng_key=key)

        moved_state, _ = move_toward(state, CONFIG, goblin_slot, state.x[pidx], state.y[pidx], subkey)
        state = jax.tree.map(
            lambda n, o: jnp.where(should_move, n, o), moved_state, state
        )
        return state

    # Move each goblin (slots 4, 5, 6)
    state = move_goblin(state, jnp.int32(4))
    state = move_goblin(state, jnp.int32(5))
    state = move_goblin(state, jnp.int32(6))

    # --- Goblin-trap collision ---
    def check_trap_collision(state, goblin_slot):
        gx, gy = state.x[goblin_slot], state.y[goblin_slot]
        is_alive_goblin = state.alive[goblin_slot] & (state.entity_type[goblin_slot] == 5)

        # Find trap at goblin position
        trap_mask = state.alive & (state.entity_type == 4) & (state.x == gx) & (state.y == gy)
        has_trap = trap_mask.any() & is_alive_goblin
        trap_slot = jnp.argmax(trap_mask)

        # Destroy both goblin and trap
        killed_alive = state.alive.at[goblin_slot].set(False).at[trap_slot].set(False)
        killed_etype = state.entity_type.at[goblin_slot].set(0).at[trap_slot].set(0)
        killed_tags = state.tags.at[goblin_slot].set(
            jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_)
        ).at[trap_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))

        killed_state = state.replace(
            alive=killed_alive, entity_type=killed_etype, tags=killed_tags,
            reward_acc=state.reward_acc + 1.0,
            game_state=state.game_state
                .at[0].set(state.game_state[0] - 1)
                .at[2].set(state.game_state[2] + 1),
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(has_trap, n, o), killed_state, state
        )
        return state

    state = check_trap_collision(state, jnp.int32(4))
    state = check_trap_collision(state, jnp.int32(5))
    state = check_trap_collision(state, jnp.int32(6))

    # --- Goblin-player collision ---
    def check_player_collision(state, goblin_slot):
        gx, gy = state.x[goblin_slot], state.y[goblin_slot]
        is_alive_goblin = state.alive[goblin_slot] & (state.entity_type[goblin_slot] == 5)
        same_cell = (gx == state.x[pidx]) & (gy == state.y[pidx])
        goblin_kills = is_alive_goblin & same_cell & (state.status == 0)
        state_new = state.replace(
            status=jnp.where(goblin_kills, jnp.int32(-1), state.status),
        )
        return state_new

    state = check_player_collision(state, jnp.int32(4))
    state = check_player_collision(state, jnp.int32(5))
    state = check_player_collision(state, jnp.int32(6))

    # --- Phase 3: Win check — all goblins dead ---
    goblins_spawned = state.turn_number >= GOBLIN_SPAWN_TURN
    goblin_alive = state.alive & (state.entity_type == 5)
    all_dead = goblins_spawned & (goblin_alive.sum() == 0)
    win = all_dead & (state.status == 0)
    state = state.replace(
        status=jnp.where(win, jnp.int32(1), state.status),
    )

    # Rebuild grid
    state = rebuild_grid(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -5.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []

    # 1. 3 gears inside rock at start
    gear_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("three_gears", gear_count == 3))

    # 2. No goblins at start
    goblin_count = int((state.alive & (state.entity_type == 5)).sum())
    results.append(("no_goblins_at_start", goblin_count == 0))

    # 3. Rock fills x=0 to x=10
    rock_mask = state.alive & (state.entity_type == 2)
    rock_in_zone = rock_mask & (state.x <= 10)
    results.append(("rock_in_zone", int(rock_mask.sum()) == int(rock_in_zone.sum())))

    # 4. Open plains at x>10 (no rock)
    rock_in_open = rock_mask & (state.x > 10)
    results.append(("open_plains", int(rock_in_open.sum()) == 0))

    # 5. Player starts at center
    pidx = int(state.player_idx)
    results.append(("player_at_center",
                    int(state.x[pidx]) == 7 and int(state.y[pidx]) == 7))

    return results
