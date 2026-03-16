"""Game 20: Fishing for Islands -- capstone. 6x6 shell, 4 chunks to unmask, Atlantean Monument."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.wave import compute_distance_field
from jaxswarm.systems.movement import DX, DY
from jaxswarm.systems.fertility import fertility_system
from jaxswarm.systems.processing import processing_system
from jaxswarm.systems.island_effects import compute_island_effects

CONFIG = EnvConfig(
    grid_w=32, grid_h=32,
    max_entities=512,
    max_stack=4,
    num_entity_types=22,   # 0=unused..21=monument
    num_tags=14,
    num_props=12,          # 0=food,1=stamina,2=thirst,3=age/fuel,4=sticks,5=water,
                           # 6=wheat_held,7=wool_held,8=bread_held,9=has_golden_seed,10=bait,11=deep_fish_held
    num_actions=10,        # 0-3=move,4=interact,5=wait,6=fish,7=plant,8=fish_deep,9=build_bridge
    max_turns=3000,
    step_penalty=-0.001,
    game_state_size=16,    # 0=farm_revealed,1=pasture_revealed,2=bakery_revealed,3=pier_revealed,
                           # 4=wheat_harvested,5=wool_collected,6=bread_baked,7=deep_fish_caught,
                           # 8=monument_found,9=sticks_gathered,10=fires_fed,11=water_drawn,
                           # 12=chunks_total,13=sheep_fed,14=reserved,15=reserved
    prop_maxes=(10.0, 20.0, 10.0, 20.0, 5.0, 3.0, 20.0, 10.0, 10.0, 1.0, 5.0, 5.0),
    max_behaviors=4,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait", "fish", "plant", "fish_deep", "build_bridge"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Entity types
PLAYER = 1; WATER = 2; STICK = 3; CAMPFIRE = 4; WELL = 5; BERRY = 6
SOIL = 7; SPROUT = 8; MATURE_WHEAT = 9; SHEEP = 10; WOOL = 11; MANURE = 12
PIERRE = 13; MILL = 14; BREAD = 15; MOUSE = 16; GOLDEN_SEED = 17
PIER = 18; DEEP_FISH = 19; BRIDGE = 20; MONUMENT = 21

DAY_LENGTH = 60

# Fixed slots
MILL_SLOT = 5

# Chunk zones (all hidden initially except starting shell)
# Starting shell: 4-9, 13-18 (6x6 center)
# Farm chunk: 12-18, 4-10
# Pasture chunk: 12-18, 20-26
# Bakery chunk: 22-28, 13-18
# Pier chunk: 22-28, 4-10
CHUNK_TAGS = [11, 11, 11, 11]  # tag 11 = chunk marker per entity

# Deterministic trace
_trace20 = (
    [2] * 3 + [1] * 2 +       # gather sticks
    [4] +                       # eat berry
    [4] +                       # trade for bait
    [6] +                       # fish: reveal farm chunk
    [2] * 5 + [7] * 3 +        # walk to farm + plant
    [5] * 20 +                  # wait
    [4] * 3 +                   # harvest
    [6] +                       # fish: reveal pasture
    [2] * 5 + [6] * 2 +        # walk + feed sheep
    [5] * 15 +                  # wait
    [6] +                       # fish: reveal bakery
    [2] * 5 + [4] +             # walk to mill, drop wheat
    [5] * 15 +                  # wait for bread
    [6] +                       # fish: reveal pier
    [2] * 10 +                  # walk to pier
    [8] * 3 +                   # fish deep (monument)
    [5] * 500 +                 # keep cycling
    [5] * 1000 +                # more padding
    [5] * 1300                  # final padding
)
DETERMINISTIC_TRACE = _trace20[:2995]


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
    x = x.at[slot].set(6)
    y = y.at[slot].set(15)
    tags = tags.at[slot, 0].set(True)
    properties = properties.at[slot, 0].set(8.0)
    properties = properties.at[slot, 1].set(18.0)
    properties = properties.at[slot, 2].set(8.0)
    slot += 1

    # Campfire (slot 1)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(CAMPFIRE)
    x = x.at[slot].set(7)
    y = y.at[slot].set(16)
    tags = tags.at[slot, 6].set(True)
    tags = tags.at[slot, 5].set(True)
    properties = properties.at[slot, 3].set(3.0)
    slot += 1

    # Well (slot 2)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(WELL)
    x = x.at[slot].set(8)
    y = y.at[slot].set(14)
    tags = tags.at[slot, 7].set(True)
    tags = tags.at[slot, 5].set(True)
    slot += 1

    # Berry bush (slot 3)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(BERRY)
    x = x.at[slot].set(5)
    y = y.at[slot].set(17)
    tags = tags.at[slot, 5].set(True)
    slot += 1

    # Bait trader (slot 4) — trades sticks for bait
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(PIERRE)
    x = x.at[slot].set(9)
    y = y.at[slot].set(15)
    tags = tags.at[slot, 5].set(True)
    tags = tags.at[slot, 8].set(True)  # bait trader marker
    slot += 1

    # Mill (slot 5 = MILL_SLOT) — pre-placed on bakery chunk, alive=False
    entity_type = entity_type.at[slot].set(MILL)
    x = x.at[slot].set(25)
    y = y.at[slot].set(15)
    tags = tags.at[slot, 5].set(True)
    tags = tags.at[slot, 11].set(True)  # chunk: bakery
    # alive=False until bakery chunk revealed
    slot += 1

    # Fishing spots (4, one per chunk reveal direction)
    # Each adjacent to starting shell edge
    fish_spots = [
        (10, 7, 0),   # east-north -> reveals farm chunk
        (10, 23, 1),  # east-south -> reveals pasture chunk
        (10, 15, 2),  # east-mid -> reveals bakery chunk
        (10, 11, 3),  # east -> reveals pier chunk
    ]
    fish_spot_start = slot
    for fx, fy, chunk_id in fish_spots:
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(WATER)
        x = x.at[slot].set(fx)
        y = y.at[slot].set(fy)
        tags = tags.at[slot, 5].set(True)
        tags = tags.at[slot, 9].set(True)  # fishing spot marker
        properties = properties.at[slot, 3].set(float(chunk_id))  # which chunk this reveals
        slot += 1

    # Sticks (6) on starting shell
    stick_pos = jnp.array([[4, 14], [5, 13], [8, 13], [9, 17], [6, 18], [4, 16]], dtype=jnp.int32)
    for i in range(6):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(STICK)
        x = x.at[slot].set(stick_pos[i, 0])
        y = y.at[slot].set(stick_pos[i, 1])
        tags = tags.at[slot, 3].set(True)
        slot += 1

    # === Farm chunk entities (alive=False until revealed) ===
    # Soil 3x3 at (14-16, 5-7)
    farm_chunk_start = slot
    soil_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(14, 17, dtype=jnp.int32),
        jnp.arange(5, 8, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)
    for i in range(9):
        entity_type = entity_type.at[slot].set(SOIL)
        x = x.at[slot].set(soil_coords[i, 0])
        y = y.at[slot].set(soil_coords[i, 1])
        tags = tags.at[slot, 5].set(True)
        tags = tags.at[slot, 11].set(True)  # chunk marker
        properties = properties.at[slot, 3].set(0.0)  # chunk 0 = farm
        slot += 1
    farm_chunk_end = slot

    # === Pasture chunk entities (alive=False) ===
    pasture_chunk_start = slot
    sheep_pos = jnp.array([[14, 22], [15, 23], [16, 24]], dtype=jnp.int32)
    for i in range(3):
        entity_type = entity_type.at[slot].set(SHEEP)
        x = x.at[slot].set(sheep_pos[i, 0])
        y = y.at[slot].set(sheep_pos[i, 1])
        tags = tags.at[slot, 5].set(True)
        tags = tags.at[slot, 11].set(True)
        properties = properties.at[slot, 3].set(1.0)  # chunk 1 = pasture
        slot += 1
    pasture_chunk_end = slot

    # === Bakery chunk entities (alive=False) ===
    bakery_chunk_start = slot
    # Pierre baker
    entity_type = entity_type.at[slot].set(PIERRE)
    x = x.at[slot].set(24)
    y = y.at[slot].set(16)
    tags = tags.at[slot, 5].set(True)
    tags = tags.at[slot, 11].set(True)
    properties = properties.at[slot, 3].set(2.0)  # chunk 2 = bakery
    slot += 1
    # Mouse
    entity_type = entity_type.at[slot].set(MOUSE)
    x = x.at[slot].set(26)
    y = y.at[slot].set(17)
    tags = tags.at[slot, 5].set(True)
    tags = tags.at[slot, 11].set(True)
    properties = properties.at[slot, 3].set(2.0)
    slot += 1
    bakery_chunk_end = slot

    # === Pier chunk entities (alive=False) ===
    pier_chunk_start = slot
    # Pier tiles (3x3 at 24-26, 5-7)
    pier_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(24, 27, dtype=jnp.int32),
        jnp.arange(5, 8, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)
    for i in range(9):
        entity_type = entity_type.at[slot].set(PIER)
        x = x.at[slot].set(pier_coords[i, 0])
        y = y.at[slot].set(pier_coords[i, 1])
        tags = tags.at[slot, 11].set(True)
        properties = properties.at[slot, 3].set(3.0)  # chunk 3 = pier
        slot += 1
    # Deep fishing spot on pier
    entity_type = entity_type.at[slot].set(WATER)
    x = x.at[slot].set(25)
    y = y.at[slot].set(6)
    tags = tags.at[slot, 9].set(True)   # fishing marker
    tags = tags.at[slot, 10].set(True)  # deep water marker
    tags = tags.at[slot, 5].set(True)
    tags = tags.at[slot, 11].set(True)
    properties = properties.at[slot, 3].set(3.0)
    slot += 1
    pier_chunk_end = slot

    # === Bridge tiles (pre-allocated, alive=False) ===
    # Bridges connect shell to chunks (activated by build_bridge action)
    bridge_start = slot
    # Bridge to farm (y=10-12, x=10)
    for by in range(10, 13):
        entity_type = entity_type.at[slot].set(BRIDGE)
        x = x.at[slot].set(10)
        y = y.at[slot].set(by)
        tags = tags.at[slot, 11].set(True)
        properties = properties.at[slot, 3].set(0.0)  # farm bridge
        slot += 1
    # Bridge to pasture (y=19, x=10-12)
    for bx in range(10, 13):
        entity_type = entity_type.at[slot].set(BRIDGE)
        x = x.at[slot].set(bx)
        y = y.at[slot].set(19)
        tags = tags.at[slot, 11].set(True)
        properties = properties.at[slot, 3].set(1.0)
        slot += 1
    # Bridge to bakery (y=15, x=12-14)
    for bx in range(12, 15):
        entity_type = entity_type.at[slot].set(BRIDGE)
        x = x.at[slot].set(bx)
        y = y.at[slot].set(15)
        tags = tags.at[slot, 11].set(True)
        properties = properties.at[slot, 3].set(2.0)
        slot += 1
    # Bridge to pier (y=7, x=12-14)
    for bx in range(12, 15):
        entity_type = entity_type.at[slot].set(BRIDGE)
        x = x.at[slot].set(bx)
        y = y.at[slot].set(7)
        tags = tags.at[slot, 11].set(True)
        properties = properties.at[slot, 3].set(3.0)
        slot += 1
    bridge_end = slot

    # Water borders around starting shell
    for wy in range(12, 20):
        for wx in range(3, 11):
            on_shell = (4 <= wx <= 9) and (13 <= wy <= 18)
            if not on_shell and slot < n:
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
    is_fish = (action == 6)
    is_plant = (action == 7)
    is_fish_deep = (action == 8)
    is_build_bridge = (action == 9)

    # --- Compute island effects ---
    farm_active = state.game_state[0] >= 1
    pasture_active = state.game_state[1] >= 1
    bakery_active = state.game_state[2] >= 1
    pier_active = state.game_state[3] >= 1
    chunks_unlocked = jnp.array([True, farm_active, pasture_active, bakery_active, pier_active], dtype=jnp.bool_)
    # Effects: [food_mult, water_mult, stamina_mult, fish_mult]
    effect_mults = jnp.array([
        [1.0, 1.0, 1.0, 1.0],   # base shell
        [1.5, 1.0, 1.0, 1.0],   # farm: 1.5x food
        [1.0, 1.0, 1.2, 1.0],   # pasture: 1.2x stamina
        [1.0, 1.0, 1.0, 1.0],   # bakery: no global effect
        [1.0, 2.0, 1.0, 2.0],   # pier: 2x water, 2x fishing
    ], dtype=jnp.float32)
    multipliers = compute_island_effects(state, chunks_unlocked, effect_mults)
    food_mult = multipliers[0]
    water_mult = multipliers[1]
    fish_mult = multipliers[3]

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
        game_state=state.game_state.at[9].set(
            jnp.where(has_sticks, state.game_state[9] + stick_mask.sum().astype(jnp.float32),
                       state.game_state[9])
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
        reward_acc=state.reward_acc + jnp.where(has_wheat, 0.15 * wheat_count, 0.0),
        game_state=state.game_state.at[4].set(
            jnp.where(has_wheat, state.game_state[4] + wheat_count, state.game_state[4])
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
        game_state=state.game_state.at[5].set(
            jnp.where(has_wool, state.game_state[5] + wool_count, state.game_state[5])
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
        game_state=state.game_state.at[6].set(
            jnp.where(has_bread, state.game_state[6] + bread_count, state.game_state[6])
        ),
    )

    # --- Pickup monument ---
    at_player5 = state.alive & (state.x == new_px) & (state.y == new_py)
    monument_mask = at_player5 & (state.entity_type == MONUMENT)
    has_monument = monument_mask.any()
    state = state.replace(
        alive=jnp.where(has_monument, state.alive & ~monument_mask, state.alive),
        reward_acc=state.reward_acc + jnp.where(has_monument, 10.0, 0.0),
        game_state=state.game_state.at[8].set(
            jnp.where(has_monument, 1.0, state.game_state[8])
        ),
    )

    # --- Pickup deep fish ---
    at_player6 = state.alive & (state.x == new_px) & (state.y == new_py)
    dfish_mask = at_player6 & (state.entity_type == DEEP_FISH)
    has_dfish = dfish_mask.any()
    dfish_count = dfish_mask.sum().astype(jnp.float32)
    new_dfish = jnp.minimum(state.properties[pidx, 11] + dfish_count, 5.0)
    state = state.replace(
        alive=jnp.where(has_dfish, state.alive & ~dfish_mask, state.alive),
        properties=jnp.where(has_dfish,
                             state.properties.at[pidx, 11].set(new_dfish),
                             state.properties),
        game_state=state.game_state.at[7].set(
            jnp.where(has_dfish, state.game_state[7] + dfish_count, state.game_state[7])
        ),
    )

    # --- Interact: context-dependent ---
    adj_xs = new_px + ADJ_DX
    adj_ys = new_py + ADJ_DY

    # Trade sticks for bait (adjacent to bait trader = tag 8)
    is_bait_trader = state.alive & state.tags[:, 8]
    trader_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_trader = (is_bait_trader[:, None] & trader_match).any()
    can_trade = is_interact & has_adj_trader & (state.properties[pidx, 4] >= 2)
    trade_props = state.properties \
        .at[pidx, 4].set(state.properties[pidx, 4] - 2.0) \
        .at[pidx, 10].set(jnp.minimum(state.properties[pidx, 10] + 1.0, 5.0))
    state = state.replace(
        properties=jnp.where(can_trade, trade_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_trade, 0.1, 0.0),
    )

    # Feed fire
    is_fire = state.alive & (state.entity_type == CAMPFIRE)
    fire_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_fire = (is_fire[:, None] & fire_match).any()
    fire_slot = jnp.argmax(is_fire)
    can_feed_fire = is_interact & ~can_trade & has_adj_fire & (state.properties[pidx, 4] >= 1)
    feed_props = state.properties \
        .at[pidx, 4].set(state.properties[pidx, 4] - 1.0) \
        .at[fire_slot, 3].set(jnp.minimum(state.properties[fire_slot, 3] + 2.0, 5.0))
    state = state.replace(
        properties=jnp.where(can_feed_fire, feed_props, state.properties),
        game_state=state.game_state.at[10].set(
            jnp.where(can_feed_fire, state.game_state[10] + 1, state.game_state[10])
        ),
    )

    # Draw water (enhanced by multiplier)
    is_well = state.alive & (state.entity_type == WELL)
    well_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_well = (is_well[:, None] & well_match).any()
    can_draw = is_interact & ~can_trade & ~can_feed_fire & has_adj_well & (state.properties[pidx, 5] < 3)
    water_gain = jnp.minimum(1.0 * water_mult, 3.0 - state.properties[pidx, 5])
    state = state.replace(
        properties=jnp.where(can_draw,
                             state.properties.at[pidx, 5].set(
                                 jnp.minimum(state.properties[pidx, 5] + water_gain, 3.0)),
                             state.properties),
        game_state=state.game_state.at[11].set(
            jnp.where(can_draw, state.game_state[11] + 1, state.game_state[11])
        ),
    )

    # Eat berries
    is_bush = state.alive & (state.entity_type == BERRY)
    bush_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_bush = (is_bush[:, None] & bush_match).any()
    can_eat = is_interact & ~can_trade & ~can_feed_fire & ~can_draw & has_adj_bush
    food_gain = jnp.minimum(3.0 * food_mult, 10.0 - state.properties[pidx, 0])
    state = state.replace(
        properties=jnp.where(can_eat,
                             state.properties.at[pidx, 0].set(
                                 jnp.minimum(state.properties[pidx, 0] + food_gain, 10.0)),
                             state.properties),
    )

    # Eat bread
    can_eat_bread = is_interact & ~can_trade & ~can_feed_fire & ~can_draw & ~can_eat & (state.properties[pidx, 8] >= 1)
    eat_bread_props = state.properties \
        .at[pidx, 8].set(state.properties[pidx, 8] - 1.0) \
        .at[pidx, 1].set(jnp.minimum(state.properties[pidx, 1] + 5.0, 20.0))
    state = state.replace(
        properties=jnp.where(can_eat_bread, eat_bread_props, state.properties),
    )

    # Drop wheat at mill (if mill alive)
    is_mill = state.alive & (state.entity_type == MILL)
    mill_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_mill = (is_mill[:, None] & mill_match).any()
    mill_idle = state.properties[MILL_SLOT, 3] <= 0
    can_mill = is_interact & ~can_trade & ~can_feed_fire & ~can_draw & ~can_eat & ~can_eat_bread & \
               has_adj_mill & mill_idle & (state.properties[pidx, 6] >= 3)
    mill_props = state.properties \
        .at[pidx, 6].set(state.properties[pidx, 6] - 3.0) \
        .at[MILL_SLOT, 3].set(10.0)
    state = state.replace(
        properties=jnp.where(can_mill, mill_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_mill, 0.2, 0.0),
    )

    # Feed sheep (interact adjacent to sheep, costs wheat)
    is_sheep = state.alive & (state.entity_type == SHEEP)
    sheep_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_sheep = (is_sheep[:, None] & sheep_match).any()
    adj_sheep_mask = is_sheep & ((state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])).any(axis=1)
    sheep_eidx = jnp.argmax(adj_sheep_mask)
    can_feed_sheep = is_interact & ~can_trade & ~can_feed_fire & ~can_draw & ~can_eat & ~can_eat_bread & ~can_mill & \
                     has_adj_sheep & (state.properties[pidx, 6] >= 1)
    sheep_feed_props = state.properties \
        .at[pidx, 6].set(state.properties[pidx, 6] - 1.0) \
        .at[sheep_eidx, 3].set(10.0)
    state = state.replace(
        properties=jnp.where(can_feed_sheep, sheep_feed_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_feed_sheep, 0.1, 0.0),
        game_state=state.game_state.at[13].set(
            jnp.where(can_feed_sheep, state.game_state[13] + 1, state.game_state[13])
        ),
    )

    # --- Fish action (6): reveal chunks ---
    is_fish_spot = state.alive & state.tags[:, 9] & ~state.tags[:, 10]
    fish_adj_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_fishspot = (is_fish_spot[:, None] & fish_adj_match).any()
    fish_spot_mask = is_fish_spot & (state.x[:, None] == adj_xs[None, :]).any(axis=1) & (state.y[:, None] == adj_ys[None, :]).any(axis=1)
    fish_spot_idx = jnp.argmax(fish_spot_mask)
    target_chunk = state.properties[fish_spot_idx, 3].astype(jnp.int32)
    chunk_already_revealed = state.game_state[target_chunk] >= 1
    has_bait = state.properties[pidx, 10] > 0
    can_fish_reveal = is_fish & has_adj_fishspot & has_bait & ~chunk_already_revealed

    # Reveal all entities in target chunk (match by tag 11 + prop 3 == chunk_id)
    # Also reveal bridges for that chunk
    chunk_match = state.tags[:, 11] & (state.properties[:, 3].astype(jnp.int32) == target_chunk) & ~state.alive
    bridge_match = (state.entity_type == BRIDGE) & (state.properties[:, 3].astype(jnp.int32) == target_chunk) & ~state.alive
    revealed_alive = state.alive | chunk_match | bridge_match
    # Also reveal mill for bakery chunk
    mill_reveal = (state.entity_type == MILL) & ~state.alive & (target_chunk == 2)
    revealed_alive = revealed_alive | mill_reveal

    fish_bait_props = state.properties.at[pidx, 10].set(state.properties[pidx, 10] - 1.0)

    state = state.replace(
        alive=jnp.where(can_fish_reveal, revealed_alive, state.alive),
        properties=jnp.where(can_fish_reveal, fish_bait_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_fish_reveal, 3.0, 0.0),
        game_state=state.game_state.at[target_chunk].set(
            jnp.where(can_fish_reveal, 1.0, state.game_state[target_chunk])
        ).at[12].set(
            jnp.where(can_fish_reveal, state.game_state[12] + 1, state.game_state[12])
        ),
    )

    # --- Fish deep action (8): catch deep fish or monument ---
    is_deep_spot = state.alive & state.tags[:, 9] & state.tags[:, 10]
    deep_adj = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_deep = (is_deep_spot[:, None] & deep_adj).any()
    has_bait_deep = state.properties[pidx, 10] > 0
    all_chunks = (state.game_state[0] >= 1) & (state.game_state[1] >= 1) & (state.game_state[2] >= 1) & (state.game_state[3] >= 1)
    monument_not_found = state.game_state[8] < 1
    can_fish_deep = is_fish_deep & has_adj_deep & has_bait_deep

    # If all chunks unlocked + monument not found -> catch monument, else catch deep fish
    catch_monument = can_fish_deep & all_chunks & monument_not_found

    # Spawn entity at player position
    free_catch = jnp.argmin(state.alive)
    has_free_catch = ~state.alive.all()
    do_catch = can_fish_deep & has_free_catch

    catch_type = jnp.where(catch_monument, MONUMENT, DEEP_FISH)
    c_alive = state.alive.at[free_catch].set(True)
    c_etype = state.entity_type.at[free_catch].set(catch_type)
    c_x = state.x.at[free_catch].set(new_px)
    c_y = state.y.at[free_catch].set(new_py)
    c_tags = state.tags.at[free_catch].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True))
    deep_bait_props = state.properties.at[pidx, 10].set(state.properties[pidx, 10] - 1.0)

    state = state.replace(
        alive=jnp.where(do_catch, c_alive, state.alive),
        entity_type=jnp.where(do_catch, c_etype, state.entity_type),
        x=jnp.where(do_catch, c_x, state.x),
        y=jnp.where(do_catch, c_y, state.y),
        tags=jnp.where(do_catch, c_tags, state.tags),
        properties=jnp.where(can_fish_deep, deep_bait_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(catch_monument, 5.0, jnp.where(do_catch, 1.0, 0.0)),
    )

    # --- Plant action ---
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

    # --- Mill processing ---
    bread_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)
    bread_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state = processing_system(
        state, CONFIG, npc_slot=MILL_SLOT, busy_prop=3, busy_duration=10,
        output_type=BREAD, output_tags=bread_tags, output_props=bread_props,
    )

    # --- Fertility ---
    state, fertility = fertility_system(state, CONFIG, manure_type=MANURE, spread_chance=0.3)

    # --- Sprout growth ---
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

    # --- Sheep production ---
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
    new_fuel = jnp.where(is_fire.any(), jnp.maximum(fire_fuel - 0.08, 0.0), fire_fuel)
    state = state.replace(properties=state.properties.at[fire_slot, 3].set(new_fuel))

    # --- Warmth ---
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

    food = jnp.maximum(state.properties[pidx, 0] - 0.08, 0.0)
    stamina_drain = jnp.where(is_night, 0.4 * (1.0 - warmth_at_player * 0.8), 0.06)
    stamina = jnp.maximum(state.properties[pidx, 1] - stamina_drain, 0.0)
    thirst = jnp.maximum(state.properties[pidx, 2] - 0.10, 0.0)

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

    # Death: any meter hits 0
    dead = ((food <= 0) | (stamina <= 0) | (thirst <= 0)) & (state.status == 0)
    state = state.replace(status=jnp.where(dead, jnp.int32(-1), state.status))

    # Win: all 4 chunks unmasked + monument found
    all_chunks_revealed = (state.game_state[0] >= 1) & (state.game_state[1] >= 1) & \
                          (state.game_state[2] >= 1) & (state.game_state[3] >= 1)
    has_monument = state.game_state[8] >= 1
    win = all_chunks_revealed & has_monument & (state.status == 0)
    state = state.replace(
        status=jnp.where(win, jnp.int32(1), state.status),
        reward_acc=state.reward_acc + jnp.where(win, 20.0, 0.0),
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
    results.append(("player_has_thirst", float(state.properties[pidx, 2]) > 0))

    fire_count = int((state.alive & (state.entity_type == CAMPFIRE)).sum())
    results.append(("one_campfire", fire_count == 1))

    well_count = int((state.alive & (state.entity_type == WELL)).sum())
    results.append(("one_well", well_count == 1))

    # No chunks revealed at start
    farm = float(state.game_state[0])
    pasture = float(state.game_state[1])
    bakery = float(state.game_state[2])
    pier = float(state.game_state[3])
    results.append(("no_chunks_at_start", farm == 0 and pasture == 0 and bakery == 0 and pier == 0))

    # Monument not found at start
    monument = float(state.game_state[8])
    results.append(("no_monument_at_start", monument == 0.0))

    return results
