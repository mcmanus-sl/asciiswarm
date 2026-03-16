"""Game 18: The Oven -- all of Game 17 + Pierre NPC baker, mill, bread, mouse, golden seed."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.wave import compute_distance_field
from jaxswarm.systems.movement import DX, DY
from jaxswarm.systems.fertility import fertility_system
from jaxswarm.systems.processing import processing_system

CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=256,
    max_stack=3,
    num_entity_types=18,   # 0=unused..17=golden_seed
    num_tags=12,
    num_props=10,          # 0=food,1=stamina,2=thirst,3=age/fuel,4=sticks,5=water,6=wheat_held,7=wool_held,8=bread_held,9=has_golden_seed
    num_actions=8,         # 0-3=move,4=interact,5=wait,6=feed_sheep,7=plant
    max_turns=1000,
    step_penalty=-0.001,
    game_state_size=12,    # 0=wheat_harvested,1=wool_collected,2=manure_placed,3=sheep_fed,
                           # 4=sticks_gathered,5=fires_fed,6=water_drawn,7=berries_eaten,
                           # 8=bread_baked,9=golden_seed_found,10=wheat_at_mill,11=mouse_fed
    prop_maxes=(10.0, 20.0, 10.0, 20.0, 5.0, 3.0, 20.0, 10.0, 10.0, 1.0),
    max_behaviors=4,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait", "feed_sheep", "plant"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Entity types
PLAYER = 1; WATER = 2; STICK = 3; CAMPFIRE = 4; WELL = 5; BERRY = 6
SOIL = 7; SPROUT = 8; MATURE_WHEAT = 9; SHEEP = 10; WOOL = 11; MANURE = 12
PIERRE = 13; MILL = 14; BREAD = 15; MOUSE = 16; GOLDEN_SEED = 17

DAY_LENGTH = 50

# Deterministic trace
_trace18 = (
    [2] * 3 + [1] * 2 +       # gather sticks
    [4] +                       # eat berry
    [7] + [2] + [7] + [2] + [7] +  # plant seeds
    [5] * 20 +                  # wait for growth
    [4] * 3 +                   # harvest wheat
    [2] * 5 +                   # move to sheep
    [6] * 3 +                   # feed sheep
    [5] * 15 +                  # wait for wool
    [2] * 5 +                   # move to mill
    [4] +                       # drop wheat at mill
    [5] * 15 +                  # wait for bread
    [4] +                       # pick up bread
    [2] * 3 +                   # move toward mouse
    [4] +                       # drop bread near mouse
    [5] * 300 +                 # keep farming
    [5] * 500                   # padding
)
DETERMINISTIC_TRACE = _trace18[:995]

# Mill slot (fixed for processing_system)
MILL_SLOT = 5
PIERRE_SLOT = 4


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

    slot = 0

    # Player (slot 0)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(PLAYER)
    x = x.at[slot].set(8)
    y = y.at[slot].set(8)
    tags = tags.at[slot, 0].set(True)
    properties = properties.at[slot, 0].set(8.0)   # food
    properties = properties.at[slot, 1].set(15.0)  # stamina
    properties = properties.at[slot, 2].set(8.0)   # thirst
    slot += 1

    # Campfire (slot 1)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(CAMPFIRE)
    x = x.at[slot].set(8)
    y = y.at[slot].set(9)
    tags = tags.at[slot, 6].set(True)  # warmth
    tags = tags.at[slot, 5].set(True)  # npc
    properties = properties.at[slot, 3].set(3.0)
    slot += 1

    # Well (slot 2)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(WELL)
    x = x.at[slot].set(10)
    y = y.at[slot].set(7)
    tags = tags.at[slot, 7].set(True)
    tags = tags.at[slot, 5].set(True)
    slot += 1

    # Berry bush (slot 3)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(BERRY)
    x = x.at[slot].set(6)
    y = y.at[slot].set(10)
    tags = tags.at[slot, 5].set(True)
    slot += 1

    # Pierre NPC (slot 4 = PIERRE_SLOT)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(PIERRE)
    x = x.at[slot].set(20)
    y = y.at[slot].set(8)
    tags = tags.at[slot, 5].set(True)  # npc
    slot += 1

    # Mill (slot 5 = MILL_SLOT)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(MILL)
    x = x.at[slot].set(20)
    y = y.at[slot].set(10)
    tags = tags.at[slot, 5].set(True)  # npc
    # prop 3 = busy timer (0 = idle)
    slot += 1

    # Mouse (slot 6)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(MOUSE)
    x = x.at[slot].set(25)
    y = y.at[slot].set(25)
    tags = tags.at[slot, 5].set(True)  # npc
    slot += 1

    # Sticks (5)
    stick_pos = jnp.array([[5, 6], [6, 5], [9, 5], [11, 9], [7, 11]], dtype=jnp.int32)
    for i in range(5):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(STICK)
        x = x.at[slot].set(stick_pos[i, 0])
        y = y.at[slot].set(stick_pos[i, 1])
        tags = tags.at[slot, 3].set(True)
        slot += 1

    # Farm soil (3x3 at 6-8, 6-8)
    soil_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(6, 9, dtype=jnp.int32),
        jnp.arange(6, 9, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)
    for i in range(9):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(SOIL)
        x = x.at[slot].set(soil_coords[i, 0])
        y = y.at[slot].set(soil_coords[i, 1])
        tags = tags.at[slot, 5].set(True)
        slot += 1

    # Sheep (3, in pasture)
    sheep_pos = jnp.array([[14, 7], [15, 8], [16, 9]], dtype=jnp.int32)
    for i in range(3):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(SHEEP)
        x = x.at[slot].set(sheep_pos[i, 0])
        y = y.at[slot].set(sheep_pos[i, 1])
        tags = tags.at[slot, 5].set(True)
        slot += 1

    # Water borders (simplified ring around starting area)
    for wy in range(4, 13):
        for wx in range(4, 13):
            on_island = (5 <= wx <= 11) and (5 <= wy <= 11)
            if not on_island and slot < n:
                alive = alive.at[slot].set(True)
                entity_type = entity_type.at[slot].set(WATER)
                x = x.at[slot].set(wx)
                y = y.at[slot].set(wy)
                tags = tags.at[slot, 1].set(True)
                slot += 1

    game_state = jnp.zeros(CONFIG.game_state_size, dtype=jnp.float32)

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
    is_feed_sheep = (action == 6)
    is_plant = (action == 7)

    # --- Movement ---
    target_x = px + DX[action]
    target_y = py + DY[action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)

    at_target = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_solid = (at_target & state.tags[:, 1]).any()

    can_move = is_move & in_bounds & ~has_solid
    new_x = state.x.at[pidx].set(jnp.where(can_move, target_x, px))
    new_y = state.y.at[pidx].set(jnp.where(can_move, target_y, py))
    state = state.replace(x=new_x, y=new_y)

    new_px, new_py = state.x[pidx], state.y[pidx]

    # --- Pickup sticks ---
    at_player = state.alive & (state.x == new_px) & (state.y == new_py)
    stick_mask = at_player & (state.entity_type == STICK)
    has_sticks = stick_mask.any()
    new_sticks = jnp.minimum(state.properties[pidx, 4] + stick_mask.sum().astype(jnp.float32), 5.0)
    state = state.replace(
        alive=jnp.where(has_sticks, state.alive & ~stick_mask, state.alive),
        properties=jnp.where(has_sticks,
                             state.properties.at[pidx, 4].set(new_sticks),
                             state.properties),
        game_state=state.game_state.at[4].set(
            jnp.where(has_sticks, state.game_state[4] + stick_mask.sum().astype(jnp.float32),
                       state.game_state[4])
        ),
    )

    # --- Pickup mature wheat ---
    at_player2 = state.alive & (state.x == new_px) & (state.y == new_py)
    wheat_mask = at_player2 & (state.entity_type == MATURE_WHEAT)
    has_wheat = wheat_mask.any()
    wheat_count = wheat_mask.sum().astype(jnp.float32)
    new_wheat = jnp.minimum(state.properties[pidx, 6] + wheat_count, 20.0)
    state = state.replace(
        alive=jnp.where(has_wheat, state.alive & ~wheat_mask, state.alive),
        properties=jnp.where(has_wheat,
                             state.properties.at[pidx, 6].set(new_wheat),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_wheat, 0.2 * wheat_count, 0.0),
        game_state=state.game_state.at[0].set(
            jnp.where(has_wheat, state.game_state[0] + wheat_count, state.game_state[0])
        ),
    )

    # --- Pickup wool ---
    at_player3 = state.alive & (state.x == new_px) & (state.y == new_py)
    wool_mask = at_player3 & (state.entity_type == WOOL)
    has_wool = wool_mask.any()
    wool_count = wool_mask.sum().astype(jnp.float32)
    new_wool = jnp.minimum(state.properties[pidx, 7] + wool_count, 10.0)
    state = state.replace(
        alive=jnp.where(has_wool, state.alive & ~wool_mask, state.alive),
        properties=jnp.where(has_wool,
                             state.properties.at[pidx, 7].set(new_wool),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_wool, 0.5 * wool_count, 0.0),
        game_state=state.game_state.at[1].set(
            jnp.where(has_wool, state.game_state[1] + wool_count, state.game_state[1])
        ),
    )

    # --- Pickup bread ---
    at_player4 = state.alive & (state.x == new_px) & (state.y == new_py)
    bread_mask = at_player4 & (state.entity_type == BREAD)
    has_bread = bread_mask.any()
    bread_count = bread_mask.sum().astype(jnp.float32)
    new_bread = jnp.minimum(state.properties[pidx, 8] + bread_count, 10.0)
    state = state.replace(
        alive=jnp.where(has_bread, state.alive & ~bread_mask, state.alive),
        properties=jnp.where(has_bread,
                             state.properties.at[pidx, 8].set(new_bread),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_bread, 1.0 * bread_count, 0.0),
        game_state=state.game_state.at[8].set(
            jnp.where(has_bread, state.game_state[8] + bread_count, state.game_state[8])
        ),
    )

    # --- Pickup golden seed ---
    at_player5 = state.alive & (state.x == new_px) & (state.y == new_py)
    gseed_mask = at_player5 & (state.entity_type == GOLDEN_SEED)
    has_gseed = gseed_mask.any()
    state = state.replace(
        alive=jnp.where(has_gseed, state.alive & ~gseed_mask, state.alive),
        properties=jnp.where(has_gseed,
                             state.properties.at[pidx, 9].set(1.0),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_gseed, 5.0, 0.0),
        game_state=state.game_state.at[9].set(
            jnp.where(has_gseed, 1.0, state.game_state[9])
        ),
    )

    # --- Interact: context-dependent ---
    adj_xs = new_px + ADJ_DX
    adj_ys = new_py + ADJ_DY

    # Feed fire
    is_fire = state.alive & (state.entity_type == CAMPFIRE)
    fire_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_fire = (is_fire[:, None] & fire_match).any()
    fire_slot = jnp.argmax(is_fire)
    can_feed_fire = is_interact & has_adj_fire & (state.properties[pidx, 4] >= 1)
    feed_props = state.properties \
        .at[pidx, 4].set(state.properties[pidx, 4] - 1.0) \
        .at[fire_slot, 3].set(jnp.minimum(state.properties[fire_slot, 3] + 2.0, 5.0))
    state = state.replace(
        properties=jnp.where(can_feed_fire, feed_props, state.properties),
        game_state=state.game_state.at[5].set(
            jnp.where(can_feed_fire, state.game_state[5] + 1, state.game_state[5])
        ),
    )

    # Draw water
    is_well = state.alive & (state.entity_type == WELL)
    well_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_well = (is_well[:, None] & well_match).any()
    can_draw = is_interact & ~can_feed_fire & has_adj_well & (state.properties[pidx, 5] < 3)
    state = state.replace(
        properties=jnp.where(can_draw,
                             state.properties.at[pidx, 5].set(
                                 jnp.minimum(state.properties[pidx, 5] + 1.0, 3.0)),
                             state.properties),
        game_state=state.game_state.at[6].set(
            jnp.where(can_draw, state.game_state[6] + 1, state.game_state[6])
        ),
    )

    # Eat berries
    is_bush = state.alive & (state.entity_type == BERRY)
    bush_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_bush = (is_bush[:, None] & bush_match).any()
    can_eat = is_interact & ~can_feed_fire & ~can_draw & has_adj_bush
    state = state.replace(
        properties=jnp.where(can_eat,
                             state.properties.at[pidx, 0].set(
                                 jnp.minimum(state.properties[pidx, 0] + 3.0, 10.0)),
                             state.properties),
        game_state=state.game_state.at[7].set(
            jnp.where(can_eat, state.game_state[7] + 1, state.game_state[7])
        ),
    )

    # Eat bread (interact with no other target, has bread)
    can_eat_bread = is_interact & ~can_feed_fire & ~can_draw & ~can_eat & (state.properties[pidx, 8] >= 1)
    eat_bread_props = state.properties \
        .at[pidx, 8].set(state.properties[pidx, 8] - 1.0) \
        .at[pidx, 1].set(jnp.minimum(state.properties[pidx, 1] + 5.0, 20.0))
    state = state.replace(
        properties=jnp.where(can_eat_bread, eat_bread_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_eat_bread, 0.1, 0.0),
    )

    # Drop wheat at mill (adjacent to mill, have wheat)
    is_mill = state.alive & (state.entity_type == MILL)
    mill_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_mill = (is_mill[:, None] & mill_match).any()
    mill_idle = state.properties[MILL_SLOT, 3] <= 0
    can_mill = is_interact & ~can_feed_fire & ~can_draw & ~can_eat & ~can_eat_bread & \
               has_adj_mill & mill_idle & (state.properties[pidx, 6] >= 3)
    mill_props = state.properties \
        .at[pidx, 6].set(state.properties[pidx, 6] - 3.0) \
        .at[MILL_SLOT, 3].set(10.0)  # busy for 10 turns
    state = state.replace(
        properties=jnp.where(can_mill, mill_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_mill, 0.2, 0.0),
        game_state=state.game_state.at[10].set(
            jnp.where(can_mill, state.game_state[10] + 3, state.game_state[10])
        ),
    )

    # Drop bread near mouse (adjacent to mouse, have bread) -> golden seed
    is_mouse = state.alive & (state.entity_type == MOUSE)
    mouse_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_mouse = (is_mouse[:, None] & mouse_match).any()
    gseed_not_found = state.game_state[9] < 1
    can_mouse = is_interact & ~can_feed_fire & ~can_draw & ~can_eat & ~can_eat_bread & ~can_mill & \
                has_adj_mouse & (state.properties[pidx, 8] >= 1) & gseed_not_found
    mouse_props = state.properties.at[pidx, 8].set(state.properties[pidx, 8] - 1.0)

    # Spawn golden seed at mouse location
    mouse_slot_idx = jnp.argmax(is_mouse)
    free_gs = jnp.argmin(state.alive)
    has_free_gs = ~state.alive.all()
    do_gseed = can_mouse & has_free_gs
    gs_alive = state.alive.at[free_gs].set(True)
    gs_etype = state.entity_type.at[free_gs].set(GOLDEN_SEED)
    gs_x = state.x.at[free_gs].set(state.x[mouse_slot_idx])
    gs_y = state.y.at[free_gs].set(state.y[mouse_slot_idx])
    gs_tags = state.tags.at[free_gs].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True))

    state = state.replace(
        alive=jnp.where(do_gseed, gs_alive, state.alive),
        entity_type=jnp.where(do_gseed, gs_etype, state.entity_type),
        x=jnp.where(do_gseed, gs_x, state.x),
        y=jnp.where(do_gseed, gs_y, state.y),
        tags=jnp.where(do_gseed, gs_tags, state.tags),
        properties=jnp.where(can_mouse, mouse_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(do_gseed, 2.0, 0.0),
        game_state=state.game_state.at[11].set(
            jnp.where(can_mouse, state.game_state[11] + 1, state.game_state[11])
        ),
    )

    # --- Feed sheep action (action 6): adjacent sheep, costs 1 wheat ---
    is_sheep = state.alive & (state.entity_type == SHEEP)
    sheep_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_sheep = (is_sheep[:, None] & sheep_match).any()
    adj_sheep_mask = is_sheep & ((state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])).any(axis=1)
    sheep_eidx = jnp.argmax(adj_sheep_mask)
    can_feed_sheep = is_feed_sheep & has_adj_sheep & (state.properties[pidx, 6] >= 1)

    sheep_feed_props = state.properties \
        .at[pidx, 6].set(state.properties[pidx, 6] - 1.0) \
        .at[sheep_eidx, 3].set(10.0)
    state = state.replace(
        properties=jnp.where(can_feed_sheep, sheep_feed_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_feed_sheep, 0.1, 0.0),
        game_state=state.game_state.at[3].set(
            jnp.where(can_feed_sheep, state.game_state[3] + 1, state.game_state[3])
        ),
    )

    # --- Plant action: plant on soil ---
    at_cell = state.alive & (state.x == new_px) & (state.y == new_py)
    on_soil = (at_cell & (state.entity_type == SOIL)).any()
    has_existing_plant = (at_cell & ((state.entity_type == SPROUT) | (state.entity_type == MATURE_WHEAT))).any()
    can_plant = is_plant & on_soil & ~has_existing_plant

    free_slot = jnp.argmin(state.alive)
    has_free = ~state.alive.all()
    do_plant = can_plant & has_free

    planted_alive = state.alive.at[free_slot].set(True)
    planted_etype = state.entity_type.at[free_slot].set(SPROUT)
    planted_x = state.x.at[free_slot].set(new_px)
    planted_y = state.y.at[free_slot].set(new_py)
    planted_tags = state.tags.at[free_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))

    state = state.replace(
        alive=jnp.where(do_plant, planted_alive, state.alive),
        entity_type=jnp.where(do_plant, planted_etype, state.entity_type),
        x=jnp.where(do_plant, planted_x, state.x),
        y=jnp.where(do_plant, planted_y, state.y),
        tags=jnp.where(do_plant, planted_tags, state.tags),
        reward_acc=state.reward_acc + jnp.where(do_plant, 0.05, 0.0),
    )

    # === Phase 2: Systems ===

    # --- Mill processing (bread production) ---
    bread_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)  # pickup
    bread_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state = processing_system(
        state, CONFIG,
        npc_slot=MILL_SLOT,
        busy_prop=3,
        busy_duration=10,
        output_type=BREAD,
        output_tags=bread_tags,
        output_props=bread_props,
    )

    # --- Fertility field from manure ---
    state, fertility = fertility_system(state, CONFIG, manure_type=MANURE, spread_chance=0.3)

    # --- Sprout growth (vectorized, manure halves threshold) ---
    is_sprout = state.alive & (state.entity_type == SPROUT)
    sprout_age = state.properties[:, 3] + 1.0
    aged_props = state.properties.at[:, 3].set(
        jnp.where(is_sprout, sprout_age, state.properties[:, 3])
    )
    H, W = CONFIG.grid_h, CONFIG.grid_w
    sprout_fert = fertility[
        jnp.clip(state.y, 0, H - 1),
        jnp.clip(state.x, 0, W - 1)
    ]
    growth_threshold = jnp.where(sprout_fert > 0.5, 8.0, 15.0)
    is_now_mature = is_sprout & (sprout_age >= growth_threshold)
    new_etype = jnp.where(is_now_mature, MATURE_WHEAT, state.entity_type)
    new_tags = state.tags.at[:, 3].set(state.tags[:, 3] | is_now_mature)
    matured_props = aged_props.at[:, 3].set(jnp.where(is_now_mature, 0.0, aged_props[:, 3]))
    state = state.replace(entity_type=new_etype, tags=new_tags, properties=matured_props)

    # --- Sheep production (wool + manure) ---
    is_sheep_alive = state.alive & (state.entity_type == SHEEP)

    def process_sheep(carry, eidx):
        st = carry
        is_this = is_sheep_alive[eidx]
        fed = st.properties[eidx, 3]
        is_fed = is_this & (fed > 0)
        new_fed = jnp.where(is_fed, fed - 1.0, fed)
        just_produced = is_fed & (new_fed <= 0)
        reset_fed = jnp.where(just_produced, 0.0, new_fed)
        st = st.replace(properties=st.properties.at[eidx, 3].set(reset_fed))

        # Spawn wool
        free_w = jnp.argmin(st.alive)
        has_free_w = ~st.alive.all()
        can_wool = just_produced & has_free_w
        wool_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)
        w_alive = st.alive.at[free_w].set(True)
        w_etype = st.entity_type.at[free_w].set(WOOL)
        w_x = st.x.at[free_w].set(st.x[eidx])
        w_y = st.y.at[free_w].set(st.y[eidx])
        w_tags = st.tags.at[free_w].set(wool_tags)
        w_props = st.properties.at[free_w].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))
        wool_st = st.replace(alive=w_alive, entity_type=w_etype, x=w_x, y=w_y, tags=w_tags, properties=w_props)
        st = jax.tree.map(lambda n, o: jnp.where(can_wool, n, o), wool_st, st)

        # Spawn manure
        free_m = jnp.argmin(st.alive)
        has_free_m = ~st.alive.all()
        can_manure = just_produced & has_free_m
        m_alive = st.alive.at[free_m].set(True)
        m_etype = st.entity_type.at[free_m].set(MANURE)
        m_x = st.x.at[free_m].set(st.x[eidx])
        m_y = st.y.at[free_m].set(st.y[eidx])
        m_tags = st.tags.at[free_m].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))
        m_props = st.properties.at[free_m].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))
        manure_st = st.replace(alive=m_alive, entity_type=m_etype, x=m_x, y=m_y, tags=m_tags, properties=m_props)
        st = jax.tree.map(lambda n, o: jnp.where(can_manure, n, o), manure_st, st)

        return st, None

    sheep_indices = jnp.arange(CONFIG.max_entities, dtype=jnp.int32)
    state, _ = jax.lax.scan(process_sheep, state, sheep_indices)

    # --- Fire decay ---
    fire_fuel = state.properties[fire_slot, 3]
    new_fuel = jnp.where(is_fire.any(), jnp.maximum(fire_fuel - 0.1, 0.0), fire_fuel)
    state = state.replace(properties=state.properties.at[fire_slot, 3].set(new_fuel))

    # --- Warmth field ---
    fire_alive = state.alive & (state.entity_type == CAMPFIRE) & (state.properties[:, 3] > 0)
    fire_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    ent_indices = state.y * W + state.x
    fire_grid = fire_grid.at[ent_indices].set(fire_grid[ent_indices] | fire_alive)
    fire_grid = fire_grid.reshape(H, W)

    wall_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    wall_grid = wall_grid.at[ent_indices].set(wall_grid[ent_indices] | (state.alive & state.tags[:, 1]))
    wall_grid = wall_grid.reshape(H, W)

    dist = compute_distance_field(wall_grid, fire_grid)
    warmth_at_player = jnp.clip(1.0 - dist[new_py, new_px] / 3.0, 0.0, 1.0)
    warmth_at_player = jnp.where(dist[new_py, new_px] >= 999.0, 0.0, warmth_at_player)

    # --- Time cycle ---
    cycle_pos = state.turn_number % (DAY_LENGTH * 2)
    is_night = cycle_pos >= DAY_LENGTH

    food = jnp.maximum(state.properties[pidx, 0] - 0.10, 0.0)
    stamina_drain = jnp.where(is_night, 0.5 * (1.0 - warmth_at_player * 0.8), 0.08)
    stamina = jnp.maximum(state.properties[pidx, 1] - stamina_drain, 0.0)
    thirst = jnp.maximum(state.properties[pidx, 2] - 0.12, 0.0)

    # Auto-drink water
    has_water = state.properties[pidx, 5] > 0
    thirst_low = thirst < 4
    thirst_restore = jnp.where(has_water & thirst_low, 2.0, 0.0)
    water_used = jnp.where(thirst_restore > 0, 1.0, 0.0)

    state = state.replace(
        properties=state.properties
            .at[pidx, 0].set(food)
            .at[pidx, 1].set(stamina)
            .at[pidx, 2].set(jnp.minimum(thirst + thirst_restore, 10.0))
            .at[pidx, 5].set(jnp.maximum(state.properties[pidx, 5] - water_used, 0.0)),
    )

    # Death
    dead = ((food <= 0) | (stamina <= 0)) & (state.status == 0)
    state = state.replace(status=jnp.where(dead, jnp.int32(-1), state.status))

    # Win: 5 bread baked + golden seed found
    total_bread = state.game_state[8]
    has_golden = state.properties[pidx, 9] >= 1
    win = (total_bread >= 5) & has_golden & (state.status == 0)
    state = state.replace(
        status=jnp.where(win, jnp.int32(1), state.status),
        reward_acc=state.reward_acc + jnp.where(win, 10.0, 0.0),
    )

    # Survival bonus
    state = state.replace(
        reward_acc=state.reward_acc + jnp.where(state.status == 0, 0.003, 0.0),
    )

    state = rebuild_grid(state, CONFIG)
    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []

    pidx = int(state.player_idx)
    results.append(("player_has_food", float(state.properties[pidx, 0]) > 0))
    results.append(("player_has_stamina", float(state.properties[pidx, 1]) > 0))

    fire_count = int((state.alive & (state.entity_type == CAMPFIRE)).sum())
    results.append(("one_campfire", fire_count == 1))

    well_count = int((state.alive & (state.entity_type == WELL)).sum())
    results.append(("one_well", well_count == 1))

    mill_count = int((state.alive & (state.entity_type == MILL)).sum())
    results.append(("one_mill", mill_count == 1))

    pierre_count = int((state.alive & (state.entity_type == PIERRE)).sum())
    results.append(("one_pierre", pierre_count == 1))

    mouse_count = int((state.alive & (state.entity_type == MOUSE)).sum())
    results.append(("one_mouse", mouse_count == 1))

    sheep_count = int((state.alive & (state.entity_type == SHEEP)).sum())
    results.append(("sheep_exist", sheep_count >= 1))

    return results
