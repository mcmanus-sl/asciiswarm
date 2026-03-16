"""Game 09: Inventory & Crafting — gather resources, craft pickaxe, mine rubble, reach exit."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.wave import compute_distance_field
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=28,
    max_stack=2,
    num_entity_types=8,    # 0=unused, 1=player, 2=exit, 3=wall, 4=wood, 5=ore, 6=workbench, 7=rubble
    num_tags=7,            # standard 6 + channel 6 = scent (wave distance field)
    num_props=4,           # 0=wood, 1=ore, 2=has_pickaxe, 3=unused
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=5,     # 0=wood_collected, 1=ore_collected, 2=pickaxe_crafted, 3=rubble_mined, 4=prev_wave_dist
    prop_maxes=(5.0, 5.0, 1.0, 1.0),
    max_behaviors=2,       # player only
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Adjacent directions for interact checks
ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Deterministic trace for seed 0: Player(2,9), Wood(2,3)(4,10), Ore(3,2)(5,5),
# Workbench(9,6), Rubble(12,10), Exit(14,5).
# N=0,S=1,E=2,W=3,I=4
_trace09 = (
    [0]*6 + [2, 0] +       # wood at (2,3), ore at (3,2)
    [2] + [1]*8 +           # wood at (4,10)
    [2] + [0]*5 +           # ore at (5,5)
    [2]*4 + [4] +           # craft at workbench (9,6) from (9,5)
    [2]*2 + [1]*5 + [4] +   # mine rubble at (12,10) from (11,10)
    [2]*3 + [0]*5            # through gap, to exit at (14,5)
)
DETERMINISTIC_TRACE = _trace09[:395]


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 20)

    # --- Generate random positions (same RNG sequence as before) ---

    # Player position
    k_px, k_py = jax.random.split(keys[0])
    player_x = jax.random.randint(k_px, (), 1, 5)
    player_y = jax.random.randint(k_py, (), 1, 15)

    # Gap y for rubble wall
    gap_y = jax.random.randint(keys[1], (), 4, 12)

    # Exit position
    k_ex, k_ey = jax.random.split(keys[2])
    exit_x = jax.random.randint(k_ex, (), 13, 15)
    exit_y = jax.random.randint(k_ey, (), 1, 15)

    # Workbench position
    k_wx, k_wy = jax.random.split(keys[3])
    wb_x = jax.random.randint(k_wx, (), 6, 10)
    wb_y = jax.random.randint(k_wy, (), 6, 10)

    # Wood positions (5 candidates, skip any on player)
    wood_keys = jax.random.split(keys[4], 5)
    wood_xs = jnp.zeros(5, dtype=jnp.int32)
    wood_ys = jnp.zeros(5, dtype=jnp.int32)
    wood_safe = jnp.zeros(5, dtype=jnp.bool_)
    for i in range(5):
        k1, k2 = jax.random.split(wood_keys[i])
        wx = jax.random.randint(k1, (), 1, 12)
        wy = jax.random.randint(k2, (), 1, 15)
        wood_xs = wood_xs.at[i].set(wx)
        wood_ys = wood_ys.at[i].set(wy)
        wood_safe = wood_safe.at[i].set(~((wx == player_x) & (wy == player_y)))

    # Ore positions (4 candidates, skip any on player)
    ore_keys = jax.random.split(keys[5], 4)
    ore_xs = jnp.zeros(4, dtype=jnp.int32)
    ore_ys = jnp.zeros(4, dtype=jnp.int32)
    ore_safe = jnp.zeros(4, dtype=jnp.bool_)
    for i in range(4):
        k1, k2 = jax.random.split(ore_keys[i])
        ox = jax.random.randint(k1, (), 1, 12)
        oy = jax.random.randint(k2, (), 1, 15)
        ore_xs = ore_xs.at[i].set(ox)
        ore_ys = ore_ys.at[i].set(oy)
        ore_safe = ore_safe.at[i].set(~((ox == player_x) & (oy == player_y)))

    # --- Slot layout ---
    # 0: player
    # 1: exit
    # 2: workbench
    # 3: rubble (gap at x=12, y=gap_y) — solid
    # 4-18: walls at x=12 for y=0..15 excluding gap_y (15 walls)
    # 19-23: wood (5 candidates, gated by safe mask)
    # 24-27: ore (4 candidates, gated by safe mask)

    # Entity type array
    entity_type = state.entity_type
    entity_type = entity_type.at[0].set(1)   # player
    entity_type = entity_type.at[1].set(2)   # exit
    entity_type = entity_type.at[2].set(6)   # workbench
    entity_type = entity_type.at[3].set(7)   # rubble
    for i in range(15):
        entity_type = entity_type.at[4 + i].set(3)   # wall
    for i in range(5):
        entity_type = entity_type.at[19 + i].set(4)  # wood
    for i in range(4):
        entity_type = entity_type.at[24 + i].set(5)  # ore

    # X positions
    x = state.x
    x = x.at[0].set(player_x)
    x = x.at[1].set(exit_x)
    x = x.at[2].set(wb_x)
    x = x.at[3].set(jnp.int32(12))          # rubble
    for i in range(15):
        x = x.at[4 + i].set(jnp.int32(12))  # wall column
    for i in range(5):
        x = x.at[19 + i].set(wood_xs[i])
    for i in range(4):
        x = x.at[24 + i].set(ore_xs[i])

    # Y positions — wall column at x=12: y=0..15 skipping gap_y
    # Build wall y-values: 0..15 with gap_y removed, giving 15 values
    all_ys = jnp.arange(16, dtype=jnp.int32)
    # Shift values at and above gap_y up by one to fill the gap
    wall_ys_col = jnp.where(all_ys[:15] >= gap_y, all_ys[:15] + 1, all_ys[:15])

    y = state.y
    y = y.at[0].set(player_y)
    y = y.at[1].set(exit_y)
    y = y.at[2].set(wb_y)
    y = y.at[3].set(gap_y)                   # rubble at the gap
    for i in range(15):
        y = y.at[4 + i].set(wall_ys_col[i])
    for i in range(5):
        y = y.at[19 + i].set(wood_ys[i])
    for i in range(4):
        y = y.at[24 + i].set(ore_ys[i])

    # Alive mask — all fixed entities alive, wood/ore gated by safe
    alive = state.alive
    alive = alive.at[0].set(True)    # player
    alive = alive.at[1].set(True)    # exit
    alive = alive.at[2].set(True)    # workbench
    alive = alive.at[3].set(True)    # rubble
    for i in range(15):
        alive = alive.at[4 + i].set(True)    # wall
    for i in range(5):
        alive = alive.at[19 + i].set(wood_safe[i])
    for i in range(4):
        alive = alive.at[24 + i].set(ore_safe[i])

    # Tags
    tags = state.tags
    tags = tags.at[0, 0].set(True)   # player: tag 0 (player)
    tags = tags.at[1, 4].set(True)   # exit: tag 4 (exit)
    tags = tags.at[2, 5].set(True)   # workbench: tag 5 (npc)
    tags = tags.at[3, 1].set(True)   # rubble: tag 1 (solid)
    for i in range(15):
        tags = tags.at[4 + i, 1].set(True)   # wall: tag 1 (solid)
    for i in range(5):
        tags = tags.at[19 + i, 3].set(True)  # wood: tag 3 (pickup)
    for i in range(4):
        tags = tags.at[24 + i, 3].set(True)  # ore: tag 3 (pickup)

    state = state.replace(
        alive=alive,
        entity_type=entity_type,
        x=x,
        y=y,
        tags=tags,
        player_idx=jnp.int32(0),
        rng_key=keys[6],
    )

    state = rebuild_grid(state, CONFIG)
    obs = get_obs(state, CONFIG)
    return state, obs


def _build_wall_mask(state, config):
    """Build bool[H,W] of solid cells — vectorized scatter."""
    H, W = config.grid_h, config.grid_w
    solid_mask = state.alive & state.tags[:, 1]  # [max_entities]
    solid = jnp.zeros(H * W, dtype=jnp.bool_)
    indices = state.y * W + state.x  # [max_entities]
    solid = solid.at[indices].set(solid[indices] | solid_mask)
    return solid.reshape(H, W)


def _build_type_mask(state, config, etype):
    """Build bool[H,W] for a given entity type — vectorized scatter."""
    H, W = config.grid_h, config.grid_w
    type_mask = state.alive & (state.entity_type == etype)
    grid = jnp.zeros(H * W, dtype=jnp.bool_)
    indices = state.y * W + state.x
    grid = grid.at[indices].set(grid[indices] | type_mask)
    return grid.reshape(H, W)


def _compute_quest_wave(state, config):
    """Compute distance field to current quest target — fully vectorized."""
    pidx = state.player_idx
    wood = state.properties[pidx, 0]
    ore = state.properties[pidx, 1]
    has_pickaxe = state.properties[pidx, 2]
    rubble_mined = state.game_state[3]

    wall_mask = _build_wall_mask(state, config)

    wood_target = _build_type_mask(state, config, 4)
    ore_target = _build_type_mask(state, config, 5)
    wb_target = _build_type_mask(state, config, 6)
    rubble_target = _build_type_mask(state, config, 7)
    exit_target = _build_type_mask(state, config, 2)

    need_wood = wood < 2
    need_ore = (wood >= 2) & (ore < 2)
    need_craft = (wood >= 2) & (ore >= 2) & (has_pickaxe < 1)
    need_mine = (has_pickaxe >= 1) & (rubble_mined < 1)

    target = jnp.where(need_wood, wood_target,
             jnp.where(need_ore, ore_target,
             jnp.where(need_craft, wb_target,
             jnp.where(need_mine, rubble_target,
             exit_target))))

    wave = compute_distance_field(wall_mask, target)
    return wave


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    pidx = state.player_idx
    px, py = state.x[pidx], state.y[pidx]
    is_move = action < 4
    is_interact = (action == 4)

    # --- Movement (vectorized) ---
    target_x = px + DX[action]
    target_y = py + DY[action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)

    at_target = state.alive & (state.x == target_x) & (state.y == target_y)
    has_solid = (at_target & state.tags[:, 1]).any()
    has_exit = (at_target & state.tags[:, 4]).any()

    can_move = is_move & in_bounds & ~has_solid
    new_px = jnp.where(can_move, target_x, px)
    new_py = jnp.where(can_move, target_y, py)
    state = state.replace(x=state.x.at[pidx].set(new_px), y=state.y.at[pidx].set(new_py))

    # Win on exit
    state = state.replace(
        status=jnp.where(can_move & has_exit & (state.status == 0), jnp.int32(1), state.status)
    )

    # --- Pickup resources (vectorized tombstone) ---
    at_player = state.alive & (state.x == new_px) & (state.y == new_py)
    is_wood_here = at_player & (state.entity_type == 4)
    is_ore_here = at_player & (state.entity_type == 5)
    wood_count = is_wood_here.sum()
    ore_count = is_ore_here.sum()

    new_wood = jnp.minimum(state.properties[pidx, 0] + wood_count, 5.0)
    new_ore = jnp.minimum(state.properties[pidx, 1] + ore_count, 5.0)

    state = state.replace(
        alive=state.alive & ~is_wood_here & ~is_ore_here,
        properties=state.properties
            .at[pidx, 0].set(jnp.where(wood_count > 0, new_wood, state.properties[pidx, 0]))
            .at[pidx, 1].set(jnp.where(ore_count > 0, new_ore, state.properties[pidx, 1])),
        reward_acc=state.reward_acc + (wood_count + ore_count) * 0.05,
        game_state=state.game_state
            .at[0].set(state.game_state[0] + wood_count)
            .at[1].set(state.game_state[1] + ore_count),
    )

    # --- Interact: craft at workbench or mine rubble (vectorized) ---
    adj_xs = new_px + ADJ_DX  # shape (4,)
    adj_ys = new_py + ADJ_DY  # shape (4,)

    # Check if any workbench is adjacent
    is_wb = state.alive & (state.entity_type == 6)
    wb_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_wb = (is_wb[:, None] & wb_match).any()

    can_craft = is_interact & has_adj_wb & \
                (state.properties[pidx, 0] >= 2) & (state.properties[pidx, 1] >= 2) & \
                (state.properties[pidx, 2] < 1)

    crafted_props = state.properties \
        .at[pidx, 0].set(state.properties[pidx, 0] - 2) \
        .at[pidx, 1].set(state.properties[pidx, 1] - 2) \
        .at[pidx, 2].set(1.0)
    state = state.replace(
        properties=jnp.where(can_craft, crafted_props, state.properties),
        reward_acc=jnp.where(can_craft, state.reward_acc + 0.3, state.reward_acc),
        game_state=jnp.where(can_craft, state.game_state.at[2].set(1.0), state.game_state),
    )

    # Check if any rubble is adjacent
    is_rubble = state.alive & (state.entity_type == 7)
    rubble_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_rubble = (is_rubble[:, None] & rubble_match).any()

    can_mine = is_interact & ~can_craft & has_adj_rubble & (state.properties[pidx, 2] >= 1)

    # Tombstone adjacent rubble
    rubble_to_kill = is_rubble & rubble_match.any(axis=1)
    state = state.replace(
        alive=jnp.where(can_mine, state.alive & ~rubble_to_kill, state.alive),
        properties=jnp.where(can_mine,
                             state.properties.at[pidx, 2].set(0.0),
                             state.properties),
        reward_acc=jnp.where(can_mine, state.reward_acc + 0.3, state.reward_acc),
        game_state=jnp.where(can_mine, state.game_state.at[3].set(1.0), state.game_state),
    )

    # --- Wave-based reward shaping ---
    wave = _compute_quest_wave(state, CONFIG)
    px_now, py_now = state.x[pidx], state.y[pidx]
    new_wave_dist = wave[py_now, px_now]
    old_wave_dist = state.game_state[4]

    has_valid = (old_wave_dist < 900) & (new_wave_dist < 900)
    delta = old_wave_dist - new_wave_dist
    shaping = jnp.where(has_valid & (delta > 0), 0.05, 0.0)
    state = state.replace(
        reward_acc=state.reward_acc + shaping,
        game_state=state.game_state.at[4].set(new_wave_dist),
    )

    # No rebuild_grid needed — get_obs builds obs directly from entity arrays
    scent = jnp.clip(wave / (CONFIG.grid_h + CONFIG.grid_w), 0.0, 1.0)
    scent = 1.0 - scent

    obs = get_obs(state, CONFIG)
    obs = {**obs, 'grid': obs['grid'].at[6].set(scent)}

    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    wb_count = int((state.alive & (state.entity_type == 6)).sum())
    results.append(("one_workbench", wb_count == 1))

    rubble_count = int((state.alive & (state.entity_type == 7)).sum())
    results.append(("one_rubble", rubble_count >= 1))

    wood_count = int((state.alive & (state.entity_type == 4)).sum())
    results.append(("enough_wood", wood_count >= 2))

    ore_count = int((state.alive & (state.entity_type == 5)).sum())
    results.append(("enough_ore", ore_count >= 2))

    pidx = int(state.player_idx)
    results.append(("player_empty_inventory",
                    float(state.properties[pidx, 0]) == 0 and
                    float(state.properties[pidx, 1]) == 0 and
                    float(state.properties[pidx, 2]) == 0))

    return results
