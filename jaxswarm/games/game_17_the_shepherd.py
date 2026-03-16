"""Game 17: The Shepherd -- farm + pasture with sheep husbandry."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.wave import compute_distance_field
from jaxswarm.systems.movement import DX, DY
from jaxswarm.systems.fertility import fertility_system

CONFIG = EnvConfig(
    grid_w=24, grid_h=24,
    max_entities=192,
    max_stack=3,
    num_entity_types=13,   # 0=unused..12=manure
    num_tags=10,
    num_props=8,           # 0=food,1=stamina,2=thirst,3=age/fuel,4=sticks,5=water,6=wheat_held,7=wool_held
    num_actions=8,         # 0-3=move,4=interact,5=wait,6=feed_sheep,7=plant
    max_turns=800,
    step_penalty=-0.001,
    game_state_size=8,     # 0=wheat_harvested,1=wool_collected,2=manure_placed,3=sheep_fed,
                           # 4=sticks_gathered,5=fires_fed,6=water_drawn,7=berries_eaten
    prop_maxes=(10.0, 20.0, 10.0, 20.0, 5.0, 3.0, 20.0, 10.0),
    max_behaviors=2,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait", "feed_sheep", "plant"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Entity types
PLAYER = 1; WATER = 2; STICK = 3; CAMPFIRE = 4; WELL = 5; BERRY = 6
SOIL = 7; SPROUT = 8; MATURE_WHEAT = 9; SHEEP = 10; WOOL = 11; MANURE = 12

DAY_LENGTH = 50

# Island layout: Farm (left) 4-10 x 4-10, Pasture (right) 14-20 x 4-10
FARM_ZONE = (4, 4, 10, 10)
PASTURE_ZONE = (14, 4, 20, 10)

# Deterministic trace
_trace17 = (
    [2] * 3 + [1] * 2 +       # gather sticks
    [4] +                       # interact: eat berry
    [2] * 3 +                   # move toward pasture
    [7] + [2] + [7] + [2] + [7] +  # plant seeds on soil
    [5] * 20 +                  # wait for growth
    [4] * 3 +                   # harvest wheat
    [2] * 5 +                   # move to sheep
    [6] * 3 +                   # feed sheep wheat
    [5] * 15 +                  # wait for wool production
    [4] * 3 +                   # pick up wool
    [5] * 200 +                 # keep farming/feeding cycles
    [5] * 400                   # padding
)
DETERMINISTIC_TRACE = _trace17[:795]


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

    # Player
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(PLAYER)
    x = x.at[slot].set(7)
    y = y.at[slot].set(7)
    tags = tags.at[slot, 0].set(True)
    properties = properties.at[slot, 0].set(8.0)   # food
    properties = properties.at[slot, 1].set(15.0)  # stamina
    properties = properties.at[slot, 2].set(8.0)   # thirst
    slot += 1

    # Campfire
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(CAMPFIRE)
    x = x.at[slot].set(7)
    y = y.at[slot].set(8)
    tags = tags.at[slot, 6].set(True)  # warmth
    tags = tags.at[slot, 5].set(True)  # npc
    properties = properties.at[slot, 3].set(3.0)   # fuel
    slot += 1

    # Well
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(WELL)
    x = x.at[slot].set(9)
    y = y.at[slot].set(6)
    tags = tags.at[slot, 7].set(True)
    tags = tags.at[slot, 5].set(True)
    slot += 1

    # Berry bush
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(BERRY)
    x = x.at[slot].set(5)
    y = y.at[slot].set(9)
    tags = tags.at[slot, 5].set(True)
    slot += 1

    # Sticks (5)
    stick_pos = jnp.array([[4, 5], [5, 4], [8, 4], [10, 8], [6, 10]], dtype=jnp.int32)
    for i in range(5):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(STICK)
        x = x.at[slot].set(stick_pos[i, 0])
        y = y.at[slot].set(stick_pos[i, 1])
        tags = tags.at[slot, 3].set(True)  # pickup
        slot += 1

    # Farm soil (3x3 at 5-7, 5-7)
    soil_coords = jnp.stack(jnp.meshgrid(
        jnp.arange(5, 8, dtype=jnp.int32),
        jnp.arange(5, 8, dtype=jnp.int32),
        indexing='xy'
    ), axis=-1).reshape(-1, 2)
    for i in range(9):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(SOIL)
        x = x.at[slot].set(soil_coords[i, 0])
        y = y.at[slot].set(soil_coords[i, 1])
        tags = tags.at[slot, 5].set(True)
        slot += 1

    # Sheep (3, in pasture zone)
    sheep_pos = jnp.array([[16, 6], [17, 7], [18, 8]], dtype=jnp.int32)
    for i in range(3):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(SHEEP)
        x = x.at[slot].set(sheep_pos[i, 0])
        y = y.at[slot].set(sheep_pos[i, 1])
        tags = tags.at[slot, 5].set(True)  # npc
        # prop 3 = fed counter (0 = not fed)
        slot += 1

    # Water borders around farm island
    for wy in range(3, 12):
        for wx in range(3, 12):
            on_island = (FARM_ZONE[0] <= wx <= FARM_ZONE[2]) and (FARM_ZONE[1] <= wy <= FARM_ZONE[3])
            if not on_island and slot < n:
                alive = alive.at[slot].set(True)
                entity_type = entity_type.at[slot].set(WATER)
                x = x.at[slot].set(wx)
                y = y.at[slot].set(wy)
                tags = tags.at[slot, 1].set(True)  # solid
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

    # --- Interact: eat/draw water/feed fire ---
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

    # --- Feed sheep action (action 6): adjacent sheep, costs 1 wheat ---
    is_sheep = state.alive & (state.entity_type == SHEEP)
    sheep_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_sheep = (is_sheep[:, None] & sheep_match).any()
    sheep_slot = jnp.argmax(is_sheep[:, None] & sheep_match)  # flatten index
    # Get actual sheep entity index
    adj_sheep_mask = is_sheep & ((state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])).any(axis=1)
    sheep_eidx = jnp.argmax(adj_sheep_mask)
    can_feed_sheep = is_feed_sheep & has_adj_sheep & (state.properties[pidx, 6] >= 1)

    # Set sheep fed counter to 10 (production_interval)
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

    # --- Fertility field from manure ---
    state, fertility = fertility_system(state, CONFIG, manure_type=MANURE, spread_chance=0.3)

    # --- Sprout growth (vectorized, manure halves threshold) ---
    is_sprout = state.alive & (state.entity_type == SPROUT)
    sprout_age = state.properties[:, 3] + 1.0
    aged_props = state.properties.at[:, 3].set(
        jnp.where(is_sprout, sprout_age, state.properties[:, 3])
    )
    # Check fertility at each sprout position
    H, W = CONFIG.grid_h, CONFIG.grid_w
    sprout_fert = fertility[
        jnp.clip(state.y, 0, H - 1),
        jnp.clip(state.x, 0, W - 1)
    ]
    growth_threshold = jnp.where(sprout_fert > 0.5, 8.0, 15.0)
    is_now_mature = is_sprout & (sprout_age >= growth_threshold)
    new_etype = jnp.where(is_now_mature, MATURE_WHEAT, state.entity_type)
    new_tags = state.tags.at[:, 3].set(state.tags[:, 3] | is_now_mature)  # pickup
    matured_props = aged_props.at[:, 3].set(jnp.where(is_now_mature, 0.0, aged_props[:, 3]))
    state = state.replace(entity_type=new_etype, tags=new_tags, properties=matured_props)

    # --- Sheep production (wool + manure) ---
    # Decrement fed counter for each sheep, produce on reaching 0
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
    fire_slot_idx = jnp.argmax(is_fire)
    fire_fuel = state.properties[fire_slot_idx, 3]
    new_fuel = jnp.where(is_fire.any(), jnp.maximum(fire_fuel - 0.1, 0.0), fire_fuel)
    state = state.replace(properties=state.properties.at[fire_slot_idx, 3].set(new_fuel))

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

    food = jnp.maximum(state.properties[pidx, 0] - 0.12, 0.0)
    stamina_drain = jnp.where(is_night, 0.6 * (1.0 - warmth_at_player * 0.8), 0.1)
    stamina = jnp.maximum(state.properties[pidx, 1] - stamina_drain, 0.0)
    thirst = jnp.maximum(state.properties[pidx, 2] - 0.15, 0.0)

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

    # Win: 5 wool + 20 wheat_held
    wool_held = state.properties[pidx, 7]
    wheat_held = state.properties[pidx, 6]
    win = (wool_held >= 5) & (wheat_held >= 20) & (state.status == 0)
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

    sheep_count = int((state.alive & (state.entity_type == SHEEP)).sum())
    results.append(("sheep_exist", sheep_count >= 1))

    soil_count = int((state.alive & (state.entity_type == SOIL)).sum())
    results.append(("soil_exists", soil_count >= 1))

    return results
