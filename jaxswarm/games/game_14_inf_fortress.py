"""Game 14: INF FORTRESS — the capstone sandbox combining all systems."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.movement import move_toward
from jaxswarm.systems.movement import DX, DY
from jaxswarm.systems.magma import magma_system

CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=512,
    max_stack=4,
    num_entity_types=16,   # see spec
    num_tags=10,           # 0=player,1=solid,2=hazard,3=pickup,4=exit,5=npc,6=mineable,7=pushable,8=trap,9=defense
    num_props=8,           # 0=food,1=stress,2=ore,3=stone,4=has_pickaxe,5=wealth,6=age/charges,7=direction
    num_actions=8,         # 0-3=move, 4=interact, 5=wait, 6=build_barricade, 7=build_trap
    max_turns=1000,
    step_penalty=-0.002,
    game_state_size=16,
    prop_maxes=(30.0, 20.0, 10.0, 20.0, 1.0, 100.0, 30.0, 4.0),
    max_behaviors=16,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait", "build_barricade", "build_trap"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Entity types
PLAYER = 1; ROCK = 2; ADAMANTINE = 3; MAGMA = 4; WALL = 5; SOIL = 6
SPROUT = 7; MATURE = 8; FOOD = 9; WORKBENCH = 10; KEG = 11
GOBLIN = 12; TRAP = 13; BARRICADE = 14; VAULT = 15

# Goblin config
GOBLIN_SPAWN_INTERVAL = 100
GOBLIN_COUNT_PER_WAVE = 3

# Deterministic trace — simplified route: farm, mine, deposit
_trace14 = (
    [2] * 10 + [1] * 2 +  # move to farm zone
    [4] +                  # interact with soil (plant)
    [2, 4] + [2, 4] +     # plant more
    [5] * 20 +             # wait for growth
    [3] * 3 +              # move to collect mature crops
    [4] * 3 +              # interact to collect food
    [2] * 8 + [1] * 8 +   # move to mine zone
    [2] * 4 +              # mine rock
    [2] * 4 +              # mine more + adamantine
    [3] * 12 + [0] * 16 +  # head to vault
    [4] * 5 +              # deposit wealth
    [5] * 50               # wait — won't win, but tests mechanics
)
DETERMINISTIC_TRACE = _trace14[:995]


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 20)

    n = CONFIG.max_entities
    alive = jnp.zeros(n, dtype=jnp.bool_)
    entity_type = jnp.zeros(n, dtype=jnp.int32)
    x = jnp.zeros(n, dtype=jnp.int32)
    y = jnp.zeros(n, dtype=jnp.int32)
    tags = jnp.zeros((n, CONFIG.num_tags), dtype=jnp.bool_)
    properties = jnp.zeros((n, CONFIG.num_props), dtype=jnp.float32)

    slot = 0

    # --- Player (slot 0) in safe zone ---
    k_px, k_py = jax.random.split(keys[0])
    player_x = jax.random.randint(k_px, (), 2, 6)
    player_y = jax.random.randint(k_py, (), 2, 6)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(PLAYER)
    x = x.at[slot].set(player_x)
    y = y.at[slot].set(player_y)
    tags = tags.at[slot, 0].set(True)  # player
    properties = properties.at[slot, 0].set(20.0)  # food=20
    slot += 1  # slot 1

    # --- Workbench (slot 1) at (5, 5) ---
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(WORKBENCH)
    x = x.at[slot].set(5)
    y = y.at[slot].set(5)
    tags = tags.at[slot, 5].set(True)  # npc
    slot += 1  # slot 2

    # --- Keg (slot 2) at (2, 2) ---
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(KEG)
    x = x.at[slot].set(2)
    y = y.at[slot].set(2)
    tags = tags.at[slot, 5].set(True)  # npc
    properties = properties.at[slot, 6].set(3.0)  # charges=3
    slot += 1  # slot 3

    # --- Vault (slot 3) at (4, 28) ---
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(VAULT)
    x = x.at[slot].set(4)
    y = y.at[slot].set(28)
    tags = tags.at[slot, 5].set(True)  # npc
    slot += 1  # slot 4

    # --- Soil patches (slots 4-18): Farm zone 5x3 at (10-14, 3-5) ---
    soil_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(10, 15, dtype=jnp.int32),
        jnp.arange(3, 6, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)  # 15 soil tiles
    for i in range(15):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(SOIL)
        x = x.at[slot].set(soil_coords[i, 0])
        y = y.at[slot].set(soil_coords[i, 1])
        tags = tags.at[slot, 5].set(True)  # npc (non-passable terrain marker)
        slot += 1  # slots 4-18 → next is 19

    # --- Pre-allocated goblin slots (slots 19-27): 3 waves × 3 goblins = 9 ---
    goblin_spawn_xs = jnp.full(9, 31, dtype=jnp.int32)
    goblin_spawn_ys = jnp.array([4, 16, 28, 4, 16, 28, 4, 16, 28], dtype=jnp.int32)
    for i in range(9):
        entity_type = entity_type.at[slot].set(GOBLIN)
        x = x.at[slot].set(goblin_spawn_xs[i])
        y = y.at[slot].set(goblin_spawn_ys[i])
        tags = tags.at[slot, 2].set(True)  # hazard
        properties = properties.at[slot, 0].set(3.0)  # HP
        # alive=False — spawned later
        slot += 1  # slots 19-27 → next is 28

    # --- Magma pocket (slots 28-36): 3x3 at (22-24, 14-16) ---
    magma_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(22, 25, dtype=jnp.int32),
        jnp.arange(14, 17, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)  # 9 magma
    for i in range(9):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(MAGMA)
        x = x.at[slot].set(magma_coords[i, 0])
        y = y.at[slot].set(magma_coords[i, 1])
        tags = tags.at[slot, 2].set(True)  # hazard
        slot += 1  # slots 28-36 → next is 37

    # --- Adamantine (slots 37-41): 5 blocks near magma ---
    adam_pos = jnp.array([
        [21, 14], [21, 15], [21, 16], [25, 14], [25, 15]
    ], dtype=jnp.int32)
    for i in range(5):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(ADAMANTINE)
        x = x.at[slot].set(adam_pos[i, 0])
        y = y.at[slot].set(adam_pos[i, 1])
        tags = tags.at[slot, 1].set(True)  # solid
        tags = tags.at[slot, 6].set(True)  # mineable
        slot += 1  # slots 37-41 → next is 42

    # --- Rock fill in mine zone (slots 42+): x=16-28, y=8-24 ---
    mine_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(16, 29, dtype=jnp.int32),
        jnp.arange(8, 25, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)  # 13*17=221

    mx = mine_coords[:, 0]
    my = mine_coords[:, 1]

    # Exclude magma (22-24, 14-16) and adamantine positions
    is_magma_pos = (mx >= 22) & (mx <= 24) & (my >= 14) & (my <= 16)
    is_adam_pos = jnp.zeros(221, dtype=jnp.bool_)
    for i in range(5):
        is_adam_pos = is_adam_pos | ((mx == adam_pos[i, 0]) & (my == adam_pos[i, 1]))
    is_rock = ~is_magma_pos & ~is_adam_pos

    rock_count = is_rock.sum()
    rock_offsets = jnp.cumsum(is_rock.astype(jnp.int32)) - 1
    rock_slots = slot + rock_offsets  # starting from slot 42

    alive = alive.at[rock_slots].set(jnp.where(is_rock, True, alive[rock_slots]))
    entity_type = entity_type.at[rock_slots].set(jnp.where(is_rock, ROCK, entity_type[rock_slots]))
    x = x.at[rock_slots].set(jnp.where(is_rock, mx, x[rock_slots]))
    y = y.at[rock_slots].set(jnp.where(is_rock, my, y[rock_slots]))
    tags = tags.at[rock_slots, 1].set(jnp.where(is_rock, True, tags[rock_slots, 1]))  # solid
    tags = tags.at[rock_slots, 6].set(jnp.where(is_rock, True, tags[rock_slots, 6]))  # mineable

    # --- Game state init ---
    game_state = jnp.zeros(CONFIG.game_state_size, dtype=jnp.float32)
    game_state = game_state.at[9].set(float(GOBLIN_SPAWN_INTERVAL))  # turn_next_invasion

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
    tantrum_turns = state.properties[pidx, 7]  # reuse direction prop for tantrum counter
    # Actually, use a game_state slot for tantrum since prop 7 is direction for goblins
    # Use game_state[5] for tantrum_turns_remaining
    tantrum_turns_remaining = state.game_state[5]
    is_tantrum = tantrum_turns_remaining > 0

    # --- Tantrum override ---
    key, k_tantrum = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)
    tantrum_action = jax.random.randint(k_tantrum, (), 0, 4)
    effective_action = jnp.where(is_tantrum, tantrum_action, action)

    is_move = effective_action < 4
    is_interact = (effective_action == 4)
    is_build_barricade = (effective_action == 6)
    is_build_trap = (effective_action == 7)

    # Decrement tantrum
    new_tantrum_turns = jnp.where(is_tantrum, tantrum_turns_remaining - 1, tantrum_turns_remaining)
    state = state.replace(
        game_state=state.game_state.at[5].set(new_tantrum_turns),
    )

    # --- Movement / Mining ---
    target_x = px + DX[effective_action]
    target_y = py + DY[effective_action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)

    at_target = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_mineable = (at_target & state.tags[:, 6]).any()
    has_solid = (at_target & state.tags[:, 1]).any()
    has_hazard = (at_target & state.tags[:, 2]).any()

    # Mining
    mineable_mask = at_target & state.tags[:, 6]
    mineable_slot = jnp.argmax(mineable_mask)
    mineable_etype = state.entity_type[mineable_slot]
    can_mine = is_move & in_bounds & has_mineable & ~is_tantrum

    # Destroy mined entity
    mined_alive = state.alive.at[mineable_slot].set(False)
    mined_etype_arr = state.entity_type.at[mineable_slot].set(0)
    mined_tags = state.tags.at[mineable_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))
    mined_props = state.properties.at[mineable_slot].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))

    is_rock_mined = can_mine & (mineable_etype == ROCK)
    is_adam_mined = can_mine & (mineable_etype == ADAMANTINE)
    new_stone = jnp.minimum(state.properties[pidx, 3] + 1.0, 20.0)
    new_ore = jnp.minimum(state.properties[pidx, 2] + 1.0, 10.0)
    mine_props = mined_props
    mine_props = mine_props.at[pidx, 3].set(jnp.where(is_rock_mined, new_stone, state.properties[pidx, 3]))
    mine_props = mine_props.at[pidx, 2].set(jnp.where(is_adam_mined, new_ore, state.properties[pidx, 2]))

    state = state.replace(
        alive=jnp.where(can_mine, mined_alive, state.alive),
        entity_type=jnp.where(can_mine, mined_etype_arr, state.entity_type),
        tags=jnp.where(can_mine, mined_tags, state.tags),
        properties=jnp.where(can_mine, mine_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(is_adam_mined, 0.2, 0.0),
        game_state=state.game_state.at[7].set(
            jnp.where(is_adam_mined, state.game_state[7] + 1, state.game_state[7])
        ),
    )

    # Normal movement
    at_target2 = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_solid2 = (at_target2 & state.tags[:, 1]).any()
    has_hazard2 = (at_target2 & state.tags[:, 2]).any()
    can_move = is_move & in_bounds & ~has_solid2 & ~has_hazard2 & ~can_mine
    new_x = state.x.at[pidx].set(jnp.where(can_move, target_x, state.x[pidx]))
    new_y = state.y.at[pidx].set(jnp.where(can_move, target_y, state.y[pidx]))
    state = state.replace(x=new_x, y=new_y)

    # --- Pickup food at new position ---
    new_px, new_py = state.x[pidx], state.y[pidx]
    at_player = state.alive & (state.x == new_px) & (state.y == new_py)
    food_mask = at_player & (state.entity_type == FOOD) & state.tags[:, 3]
    has_food = food_mask.any()
    food_count = food_mask.sum().astype(jnp.float32)
    new_food = jnp.minimum(state.properties[pidx, 0] + food_count * 5.0, 30.0)
    state = state.replace(
        alive=jnp.where(has_food, state.alive & ~food_mask, state.alive),
        properties=jnp.where(has_food,
                             state.properties.at[pidx, 0].set(new_food),
                             state.properties),
        game_state=state.game_state.at[2].set(
            jnp.where(has_food, state.game_state[2] + food_count, state.game_state[2])
        ),
    )

    # --- Pickup mature crops ---
    at_player2 = state.alive & (state.x == new_px) & (state.y == new_py)
    mature_mask = at_player2 & (state.entity_type == MATURE)
    has_mature = mature_mask.any()
    mature_count = mature_mask.sum().astype(jnp.float32)
    new_food2 = jnp.minimum(state.properties[pidx, 0] + mature_count * 3.0, 30.0)
    state = state.replace(
        alive=jnp.where(has_mature, state.alive & ~mature_mask, state.alive),
        properties=jnp.where(has_mature,
                             state.properties.at[pidx, 0].set(new_food2),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_mature, 0.1 * mature_count, 0.0),
        game_state=state.game_state.at[8].set(
            jnp.where(has_mature, state.game_state[8] + mature_count, state.game_state[8])
        ),
    )

    # --- Interact: context-dependent ---
    adj_xs = new_px + ADJ_DX
    adj_ys = new_py + ADJ_DY

    # Workbench: craft pickaxe (2 ore + 2 stone)
    is_wb = state.alive & (state.entity_type == WORKBENCH)
    wb_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_wb = (is_wb[:, None] & wb_match).any()
    can_craft = is_interact & ~is_tantrum & has_adj_wb & \
                (state.properties[pidx, 2] >= 2) & (state.properties[pidx, 3] >= 2) & \
                (state.properties[pidx, 4] < 1)
    craft_props = state.properties \
        .at[pidx, 2].set(state.properties[pidx, 2] - 2) \
        .at[pidx, 3].set(state.properties[pidx, 3] - 2) \
        .at[pidx, 4].set(1.0)
    state = state.replace(
        properties=jnp.where(can_craft, craft_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_craft, 0.5, 0.0),
    )

    # Vault: deposit ore as wealth (each ore = 10 wealth)
    is_vault = state.alive & (state.entity_type == VAULT)
    vault_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_vault = (is_vault[:, None] & vault_match).any()
    player_ore = state.properties[pidx, 2]
    can_deposit = is_interact & ~is_tantrum & ~can_craft & has_adj_vault & (player_ore > 0)
    deposit_wealth = player_ore * 10.0
    new_wealth = state.properties[pidx, 5] + deposit_wealth
    deposit_props = state.properties \
        .at[pidx, 2].set(0.0) \
        .at[pidx, 5].set(new_wealth)
    state = state.replace(
        properties=jnp.where(can_deposit, deposit_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_deposit, 0.3 * player_ore, 0.0),
        game_state=state.game_state.at[0].set(
            jnp.where(can_deposit, state.game_state[0] + deposit_wealth, state.game_state[0])
        ),
    )

    # Soil: plant seed (player on soil, no existing sprout/mature)
    at_cell = state.alive & (state.x == new_px) & (state.y == new_py)
    on_soil = (at_cell & (state.entity_type == SOIL)).any()
    has_plant = (at_cell & ((state.entity_type == SPROUT) | (state.entity_type == MATURE))).any()
    can_plant = is_interact & ~is_tantrum & ~can_craft & ~can_deposit & on_soil & ~has_plant

    free_slot = jnp.argmin(state.alive)
    has_free = ~state.alive.all()
    do_plant = can_plant & has_free

    planted_alive = state.alive.at[free_slot].set(True)
    planted_etype = state.entity_type.at[free_slot].set(SPROUT)
    planted_x = state.x.at[free_slot].set(new_px)
    planted_y = state.y.at[free_slot].set(new_py)
    planted_tags = state.tags.at[free_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))
    planted_props = state.properties.at[free_slot].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))

    state = state.replace(
        alive=jnp.where(do_plant, planted_alive, state.alive),
        entity_type=jnp.where(do_plant, planted_etype, state.entity_type),
        x=jnp.where(do_plant, planted_x, state.x),
        y=jnp.where(do_plant, planted_y, state.y),
        tags=jnp.where(do_plant, planted_tags, state.tags),
        properties=jnp.where(do_plant, planted_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(do_plant, 0.02, 0.0),
    )

    # Keg: drink (adjacent interact)
    is_keg = state.alive & (state.entity_type == KEG)
    keg_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_keg = (is_keg[:, None] & keg_match).any()
    keg_slot = jnp.argmax(is_keg)
    keg_charges = state.properties[keg_slot, 6]
    can_drink = is_interact & ~is_tantrum & ~can_craft & ~can_deposit & ~can_plant & \
                has_adj_keg & (keg_charges > 0)
    drink_stress = jnp.maximum(state.properties[pidx, 1] - 10.0, 0.0)
    drink_props = state.properties.at[pidx, 1].set(drink_stress).at[keg_slot, 6].set(keg_charges - 1.0)
    state = state.replace(
        properties=jnp.where(can_drink, drink_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_drink, 0.02, 0.0),
    )

    # --- Build barricade (action 6): costs 3 stone ---
    can_barricade = is_build_barricade & ~is_tantrum & (state.properties[pidx, 3] >= 3)
    free_slot2 = jnp.argmin(state.alive)
    has_free2 = ~state.alive.all()
    do_barricade = can_barricade & has_free2

    barr_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[1].set(True).at[9].set(True)
    barr_alive = state.alive.at[free_slot2].set(True)
    barr_etype = state.entity_type.at[free_slot2].set(BARRICADE)
    barr_x = state.x.at[free_slot2].set(new_px)
    barr_y = state.y.at[free_slot2].set(new_py)
    barr_tags_arr = state.tags.at[free_slot2].set(barr_tags)
    barr_props = state.properties.at[free_slot2].set(
        jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    ).at[pidx, 3].set(state.properties[pidx, 3] - 3.0)

    state = state.replace(
        alive=jnp.where(do_barricade, barr_alive, state.alive),
        entity_type=jnp.where(do_barricade, barr_etype, state.entity_type),
        x=jnp.where(do_barricade, barr_x, state.x),
        y=jnp.where(do_barricade, barr_y, state.y),
        tags=jnp.where(do_barricade, barr_tags_arr, state.tags),
        properties=jnp.where(do_barricade, barr_props, state.properties),
        game_state=state.game_state.at[3].set(
            jnp.where(do_barricade, state.game_state[3] + 1, state.game_state[3])
        ),
    )

    # --- Build trap (action 7): costs 2 ore ---
    can_trap = is_build_trap & ~is_tantrum & (state.properties[pidx, 2] >= 2)
    free_slot3 = jnp.argmin(state.alive)
    has_free3 = ~state.alive.all()
    do_trap = can_trap & has_free3

    trap_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[8].set(True)
    trap_alive = state.alive.at[free_slot3].set(True)
    trap_etype = state.entity_type.at[free_slot3].set(TRAP)
    trap_x = state.x.at[free_slot3].set(new_px)
    trap_y = state.y.at[free_slot3].set(new_py)
    trap_tags_arr = state.tags.at[free_slot3].set(trap_tags)
    trap_props = state.properties.at[free_slot3].set(
        jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    ).at[pidx, 2].set(state.properties[pidx, 2] - 2.0)

    state = state.replace(
        alive=jnp.where(do_trap, trap_alive, state.alive),
        entity_type=jnp.where(do_trap, trap_etype, state.entity_type),
        x=jnp.where(do_trap, trap_x, state.x),
        y=jnp.where(do_trap, trap_y, state.y),
        tags=jnp.where(do_trap, trap_tags_arr, state.tags),
        properties=jnp.where(do_trap, trap_props, state.properties),
        game_state=state.game_state.at[4].set(
            jnp.where(do_trap, state.game_state[4] + 1, state.game_state[4])
        ),
    )

    # === Phase 2: Run Behaviors ===

    # --- Magma CA spread ---
    state = magma_system(state, CONFIG, magma_type=MAGMA, spread_chance=0.2)

    # --- Sprout growth (vectorized) ---
    is_sprout = state.alive & (state.entity_type == SPROUT)
    sprout_age = state.properties[:, 6] + 1.0
    aged_props = state.properties.at[:, 6].set(
        jnp.where(is_sprout, sprout_age, state.properties[:, 6])
    )
    is_now_mature = is_sprout & (sprout_age >= 20)
    new_etype = jnp.where(is_now_mature, MATURE, state.entity_type)
    new_tags_mature = state.tags.at[:, 3].set(state.tags[:, 3] | is_now_mature)  # pickup
    matured_props = aged_props.at[:, 6].set(
        jnp.where(is_now_mature, 0.0, aged_props[:, 6])
    )
    state = state.replace(
        entity_type=new_etype, tags=new_tags_mature, properties=matured_props,
    )

    # --- Keg regen (every 30 turns) ---
    keg_regen = (state.turn_number % 30 == 0) & is_keg.any()
    keg_slot2 = jnp.argmax(is_keg)
    keg_c = state.properties[keg_slot2, 6]
    state = state.replace(
        properties=jnp.where(
            keg_regen,
            state.properties.at[keg_slot2, 6].set(jnp.minimum(keg_c + 1.0, 3.0)),
            state.properties
        ),
    )

    # --- Goblin spawning: 3 goblins every 100 turns ---
    # Goblin slots: 19-21 (wave 1, turn 100), 22-24 (wave 2, turn 200), 25-27 (wave 3, turn 300)
    spawn_wave_1 = state.turn_number == 100
    spawn_wave_2 = state.turn_number == 200
    spawn_wave_3 = state.turn_number == 300

    spawn_alive = state.alive
    spawn_alive = jnp.where(spawn_wave_1, spawn_alive.at[19].set(True).at[20].set(True).at[21].set(True), spawn_alive)
    spawn_alive = jnp.where(spawn_wave_2, spawn_alive.at[22].set(True).at[23].set(True).at[24].set(True), spawn_alive)
    spawn_alive = jnp.where(spawn_wave_3, spawn_alive.at[25].set(True).at[26].set(True).at[27].set(True), spawn_alive)
    state = state.replace(alive=spawn_alive)

    # --- Goblin movement (move toward player every 2 turns) ---
    goblins_should_move = state.turn_number % 2 == 0

    def move_one_goblin(carry, goblin_slot):
        state = carry
        is_alive_goblin = state.alive[goblin_slot] & (state.entity_type[goblin_slot] == GOBLIN)
        should_move = goblins_should_move & is_alive_goblin

        key, subkey = jax.random.split(state.rng_key)
        state = state.replace(rng_key=key)

        moved_state, _ = move_toward(state, CONFIG, goblin_slot, state.x[pidx], state.y[pidx], subkey)
        state = jax.tree.map(
            lambda n, o: jnp.where(should_move, n, o), moved_state, state
        )
        return state, None

    goblin_slots = jnp.arange(19, 28, dtype=jnp.int32)
    state, _ = jax.lax.scan(move_one_goblin, state, goblin_slots)

    # --- Goblin-trap collision ---
    def check_goblin_trap(carry, goblin_slot):
        state = carry
        gx, gy = state.x[goblin_slot], state.y[goblin_slot]
        is_alive_g = state.alive[goblin_slot] & (state.entity_type[goblin_slot] == GOBLIN)

        trap_mask = state.alive & (state.entity_type == TRAP) & (state.x == gx) & (state.y == gy)
        has_trap = trap_mask.any() & is_alive_g
        trap_slot_found = jnp.argmax(trap_mask)

        killed_alive = state.alive.at[goblin_slot].set(False).at[trap_slot_found].set(False)
        killed_etype = state.entity_type.at[goblin_slot].set(0).at[trap_slot_found].set(0)
        killed_tags = state.tags.at[goblin_slot].set(
            jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_)
        ).at[trap_slot_found].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))

        killed_state = state.replace(
            alive=killed_alive, entity_type=killed_etype, tags=killed_tags,
            reward_acc=state.reward_acc + 1.0,
            game_state=state.game_state.at[1].set(state.game_state[1] + 1),
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(has_trap, n, o), killed_state, state
        )
        return state, None

    state, _ = jax.lax.scan(check_goblin_trap, state, goblin_slots)

    # --- Goblin-player collision ---
    def check_goblin_player(carry, goblin_slot):
        state = carry
        is_alive_g = state.alive[goblin_slot] & (state.entity_type[goblin_slot] == GOBLIN)
        same_cell = (state.x[goblin_slot] == state.x[pidx]) & (state.y[goblin_slot] == state.y[pidx])
        killed = is_alive_g & same_cell & (state.status == 0)
        state = state.replace(
            status=jnp.where(killed, jnp.int32(-1), state.status),
        )
        return state, None

    state, _ = jax.lax.scan(check_goblin_player, state, goblin_slots)

    # --- Magma-player check ---
    magma_at_p = state.alive & (state.entity_type == MAGMA) & \
                 (state.x == state.x[pidx]) & (state.y == state.y[pidx])
    magma_death = magma_at_p.any() & (state.status == 0)
    state = state.replace(
        status=jnp.where(magma_death, jnp.int32(-1), state.status),
    )

    # === Phase 3: Turn End ===

    # Hunger tick: -1 food per turn
    food = state.properties[pidx, 0] - 1.0
    state = state.replace(
        properties=state.properties.at[pidx, 0].set(food),
    )
    starved = (food <= 0) & (state.status == 0)
    state = state.replace(
        status=jnp.where(starved, jnp.int32(-1), state.status),
    )

    # Stress tick: +1 every 10 turns
    stress_tick = (state.turn_number % 10 == 0)
    cur_stress = state.properties[pidx, 1]
    new_stress = jnp.minimum(cur_stress + 1.0, 20.0)
    state = state.replace(
        properties=jnp.where(
            stress_tick,
            state.properties.at[pidx, 1].set(new_stress),
            state.properties
        ),
    )

    # Tantrum check
    final_stress = state.properties[pidx, 1]
    enter_tantrum = (final_stress >= 20) & (state.game_state[5] <= 0)
    state = state.replace(
        game_state=jnp.where(
            enter_tantrum,
            state.game_state.at[5].set(10.0),
            state.game_state
        ),
    )

    # Win check: wealth >= 100
    wealth = state.properties[pidx, 5]
    win = (wealth >= 100) & (state.status == 0)
    state = state.replace(
        status=jnp.where(win, jnp.int32(1), state.status),
    )

    # Update magma count
    magma_count = (state.alive & (state.entity_type == MAGMA)).sum().astype(jnp.float32)
    state = state.replace(
        game_state=state.game_state.at[6].set(magma_count),
    )

    # Rebuild grid
    state = rebuild_grid(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 20.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []

    # Player exists with food > 0
    pidx = int(state.player_idx)
    results.append(("player_has_food", float(state.properties[pidx, 0]) > 0))

    # Workbench exists
    wb_count = int((state.alive & (state.entity_type == WORKBENCH)).sum())
    results.append(("one_workbench", wb_count == 1))

    # Keg exists
    keg_count = int((state.alive & (state.entity_type == KEG)).sum())
    results.append(("one_keg", keg_count == 1))

    # Vault exists
    vault_count = int((state.alive & (state.entity_type == VAULT)).sum())
    results.append(("one_vault", vault_count == 1))

    # Soil exists (15 tiles)
    soil_count = int((state.alive & (state.entity_type == SOIL)).sum())
    results.append(("soil_tiles", soil_count == 15))

    # Magma pocket exists (9 cells)
    magma_count = int((state.alive & (state.entity_type == MAGMA)).sum())
    results.append(("nine_magma", magma_count == 9))

    # Adamantine exists (5 blocks)
    adam_count = int((state.alive & (state.entity_type == ADAMANTINE)).sum())
    results.append(("five_adamantine", adam_count == 5))

    # No goblins at start
    goblin_count = int((state.alive & (state.entity_type == GOBLIN)).sum())
    results.append(("no_goblins_at_start", goblin_count == 0))

    # Player starts in safe zone
    px, py = int(state.x[pidx]), int(state.y[pidx])
    results.append(("player_in_safe_zone", px < 8 and py < 8))

    return results
