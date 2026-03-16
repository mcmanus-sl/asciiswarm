"""Game 15: The Campfire — survive day/night cycles on a tiny island."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.core.wave import compute_distance_field
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=32,
    max_stack=2,
    num_entity_types=7,    # 0=unused, 1=player, 2=water, 3=stick, 4=campfire, 5=well, 6=berry_bush
    num_tags=8,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=warmth, 7=water_source
    num_props=6,           # 0=food, 1=stamina, 2=thirst, 3=fire_fuel, 4=sticks_held, 5=water_held
    num_actions=6,
    max_turns=400,         # 4 day/night cycles (50+50 per cycle)
    step_penalty=-0.001,
    game_state_size=4,     # 0=sticks_gathered, 1=fires_fed, 2=water_drawn, 3=berries_eaten
    prop_maxes=(10.0, 20.0, 10.0, 5.0, 5.0, 3.0),
    max_behaviors=2,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

DAY_LENGTH = 50  # turns per half-cycle

# Deterministic trace: gather sticks, feed fire, draw water, eat berries, huddle at night
_trace15 = (
    # Day 1: gather sticks and set up
    [2] * 2 + [0] * 2 +   # move to stick area
    [3] * 2 +              # more movement
    [1] * 2 + [2] +        # collect sticks by walking over them
    # Feed fire
    [3] * 3 + [1] * 2 +   # move to campfire area
    [4] +                  # interact: feed fire
    # Draw water
    [2] * 2 + [0] +       # move to well
    [4] +                  # interact: draw water
    # Eat berries
    [3] + [1] * 2 +       # move to berry bush
    [4] +                  # interact: eat
    # Night: huddle near fire
    [3] * 2 + [0] * 2 +   # move to campfire
    [5] * 40 +             # wait near fire
    # Day 2: repeat
    [2] * 3 + [0] * 2 +   # gather more sticks
    [3] * 3 + [1] * 2 + [4] +  # feed fire
    [2] * 2 + [0] + [4] +      # draw water
    [3] + [1] * 2 + [4] +      # eat berries
    [5] * 40 +             # huddle
    # Continue...
    [5] * 200              # fill remaining turns
)
DETERMINISTIC_TRACE = _trace15[:395]

# Island layout: center 6x6 (x=5-10, y=5-10) is land, rest is water
ISLAND_X1, ISLAND_Y1 = 5, 5
ISLAND_X2, ISLAND_Y2 = 10, 10


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

    # Slot 0: Player at center of island
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(1)
    x = x.at[slot].set(7)
    y = y.at[slot].set(7)
    tags = tags.at[slot, 0].set(True)  # player
    properties = properties.at[slot, 0].set(8.0)   # food
    properties = properties.at[slot, 1].set(15.0)  # stamina
    properties = properties.at[slot, 2].set(8.0)   # thirst
    slot += 1

    # Slot 1: Campfire at (7, 8) — center of island
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(4)  # campfire
    x = x.at[slot].set(7)
    y = y.at[slot].set(8)
    tags = tags.at[slot, 6].set(True)  # warmth
    tags = tags.at[slot, 5].set(True)  # npc
    properties = properties.at[slot, 3].set(3.0)  # fire_fuel
    slot += 1

    # Slot 2: Well at (9, 6)
    alive = alive.at[slot].set(True)
    entity_type = entity_type.at[slot].set(5)  # well
    x = x.at[slot].set(9)
    y = y.at[slot].set(6)
    tags = tags.at[slot, 7].set(True)  # water_source
    tags = tags.at[slot, 5].set(True)  # npc
    slot += 1

    # Slots 3-4: Berry bushes
    bush_positions = jnp.array([[6, 9], [10, 6]], dtype=jnp.int32)
    for i in range(2):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(6)  # berry_bush
        x = x.at[slot].set(bush_positions[i, 0])
        y = y.at[slot].set(bush_positions[i, 1])
        tags = tags.at[slot, 5].set(True)  # npc (stays in place)
        slot += 1

    # Slots 5-10: Sticks scattered on island (pickup)
    stick_positions = jnp.array([
        [5, 5], [6, 6], [8, 5], [10, 7], [5, 9], [9, 10]
    ], dtype=jnp.int32)
    for i in range(6):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(3)  # stick
        x = x.at[slot].set(stick_positions[i, 0])
        y = y.at[slot].set(stick_positions[i, 1])
        tags = tags.at[slot, 3].set(True)  # pickup
        slot += 1

    # Remaining slots: Water border (solid, blocks movement off island)
    # Fill entire grid edge and non-island cells with water
    # We'll place water entities at the 4 edges of the island to block
    # For simplicity, use tag 1 (solid) for water = impassable
    water_coords = []
    for wx in range(16):
        for wy in range(16):
            if not (ISLAND_X1 <= wx <= ISLAND_X2 and ISLAND_Y1 <= wy <= ISLAND_Y2):
                # Only place water at the perimeter of the island (adjacent to land)
                adj_to_island = (
                    (ISLAND_X1 - 1 <= wx <= ISLAND_X2 + 1) and
                    (ISLAND_Y1 - 1 <= wy <= ISLAND_Y2 + 1)
                )
                if adj_to_island:
                    water_coords.append([wx, wy])

    # Limit to available slots
    max_water = min(len(water_coords), n - slot)
    for i in range(max_water):
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(2)  # water
        x = x.at[slot].set(water_coords[i][0])
        y = y.at[slot].set(water_coords[i][1])
        tags = tags.at[slot, 1].set(True)  # solid (blocks movement)
        tags = tags.at[slot, 2].set(True)  # hazard
        slot += 1

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

    # --- Pickup sticks ---
    new_px, new_py = state.x[pidx], state.y[pidx]
    at_player = state.alive & (state.x == new_px) & (state.y == new_py)
    stick_mask = at_player & (state.entity_type == 3) & state.tags[:, 3]
    has_sticks = stick_mask.any()
    stick_count = stick_mask.sum().astype(jnp.float32)
    new_sticks = jnp.minimum(state.properties[pidx, 4] + stick_count, 5.0)
    state = state.replace(
        alive=jnp.where(has_sticks, state.alive & ~stick_mask, state.alive),
        properties=jnp.where(has_sticks,
                             state.properties.at[pidx, 4].set(new_sticks),
                             state.properties),
        reward_acc=state.reward_acc + jnp.where(has_sticks, 0.05 * stick_count, 0.0),
        game_state=state.game_state.at[0].set(
            jnp.where(has_sticks, state.game_state[0] + stick_count, state.game_state[0])
        ),
    )

    # --- Interact: context-dependent ---
    adj_xs = new_px + ADJ_DX
    adj_ys = new_py + ADJ_DY

    # Feed fire: adjacent to campfire, have sticks
    is_fire = state.alive & (state.entity_type == 4)
    fire_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_fire = (is_fire[:, None] & fire_match).any()
    fire_slot = jnp.argmax(is_fire)
    can_feed = is_interact & has_adj_fire & (state.properties[pidx, 4] >= 1)

    feed_props = state.properties \
        .at[pidx, 4].set(state.properties[pidx, 4] - 1.0) \
        .at[fire_slot, 3].set(jnp.minimum(state.properties[fire_slot, 3] + 2.0, 5.0))
    state = state.replace(
        properties=jnp.where(can_feed, feed_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_feed, 0.1, 0.0),
        game_state=state.game_state.at[1].set(
            jnp.where(can_feed, state.game_state[1] + 1, state.game_state[1])
        ),
    )

    # Draw water: adjacent to well
    is_well = state.alive & (state.entity_type == 5)
    well_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_well = (is_well[:, None] & well_match).any()
    can_draw = is_interact & ~can_feed & has_adj_well & (state.properties[pidx, 5] < 3)

    draw_props = state.properties.at[pidx, 5].set(
        jnp.minimum(state.properties[pidx, 5] + 1.0, 3.0)
    )
    state = state.replace(
        properties=jnp.where(can_draw, draw_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_draw, 0.05, 0.0),
        game_state=state.game_state.at[2].set(
            jnp.where(can_draw, state.game_state[2] + 1, state.game_state[2])
        ),
    )

    # Eat berries: adjacent to berry bush
    is_bush = state.alive & (state.entity_type == 6)
    bush_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_bush = (is_bush[:, None] & bush_match).any()
    can_eat = is_interact & ~can_feed & ~can_draw & has_adj_bush

    eat_props = state.properties.at[pidx, 0].set(
        jnp.minimum(state.properties[pidx, 0] + 3.0, 10.0)
    )
    state = state.replace(
        properties=jnp.where(can_eat, eat_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_eat, 0.05, 0.0),
        game_state=state.game_state.at[3].set(
            jnp.where(can_eat, state.game_state[3] + 1, state.game_state[3])
        ),
    )

    # --- Fire fuel decay ---
    fire_fuel = state.properties[fire_slot, 3]
    new_fuel = jnp.where(is_fire.any(), jnp.maximum(fire_fuel - 0.1, 0.0), fire_fuel)
    state = state.replace(
        properties=state.properties.at[fire_slot, 3].set(new_fuel),
    )

    # --- Compute warmth field ---
    H, W = CONFIG.grid_h, CONFIG.grid_w
    fire_alive = state.alive & (state.entity_type == 4) & (state.properties[:, 3] > 0)
    fire_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    indices = state.y * W + state.x
    fire_grid = fire_grid.at[indices].set(fire_grid[indices] | fire_alive)
    fire_grid = fire_grid.reshape(H, W)

    wall_grid = jnp.zeros(H * W, dtype=jnp.bool_)
    is_solid_ent = state.alive & state.tags[:, 1]
    wall_grid = wall_grid.at[indices].set(wall_grid[indices] | is_solid_ent)
    wall_grid = wall_grid.reshape(H, W)

    dist = compute_distance_field(wall_grid, fire_grid)
    warmth_radius = 3
    warmth_at_player = jnp.clip(1.0 - dist[new_py, new_px] / warmth_radius, 0.0, 1.0)
    warmth_at_player = jnp.where(dist[new_py, new_px] >= 999.0, 0.0, warmth_at_player)

    # --- Time cycle: meter depletion ---
    cycle_pos = state.turn_number % (DAY_LENGTH * 2)
    is_night = cycle_pos >= DAY_LENGTH

    food_drain = 0.15
    thirst_drain = 0.2
    stamina_drain_base = jnp.where(is_night, 0.8, 0.15)
    # Warmth reduces night stamina drain
    stamina_drain = stamina_drain_base * jnp.where(is_night, 1.0 - warmth_at_player * 0.8, 1.0)

    # Drink water to restore thirst
    has_water = state.properties[pidx, 5] > 0
    thirst_restore = jnp.where(has_water & (state.properties[pidx, 2] < 5), 2.0, 0.0)
    water_consumed = jnp.where(thirst_restore > 0, 1.0, 0.0)

    food = jnp.maximum(state.properties[pidx, 0] - food_drain, 0.0)
    stamina = jnp.maximum(state.properties[pidx, 1] - stamina_drain, 0.0)
    thirst = jnp.maximum(state.properties[pidx, 2] - thirst_drain + thirst_restore, 0.0)
    water = jnp.maximum(state.properties[pidx, 5] - water_consumed, 0.0)

    state = state.replace(
        properties=state.properties
            .at[pidx, 0].set(food)
            .at[pidx, 1].set(stamina)
            .at[pidx, 2].set(jnp.minimum(thirst, 10.0))
            .at[pidx, 5].set(water),
    )

    # --- Death checks ---
    food_dead = (food <= 0) & (state.status == 0)
    stamina_dead = (stamina <= 0) & (state.status == 0)
    state = state.replace(
        status=jnp.where(food_dead | stamina_dead, jnp.int32(-1), state.status),
    )

    # --- Win: survive 200 turns ---
    # Changed from 400 to 200 for reasonable difficulty
    survived = (state.turn_number >= 200) & (state.status == 0)
    state = state.replace(
        status=jnp.where(survived, jnp.int32(1), state.status),
        reward_acc=state.reward_acc + jnp.where(survived, 10.0, 0.0),
    )

    # Small survival reward each turn
    alive_bonus = jnp.where(state.status == 0, 0.005, 0.0)
    state = state.replace(reward_acc=state.reward_acc + alive_bonus)

    # Rebuild grid (sticks removed)
    state = rebuild_grid(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []

    # Player exists with meters > 0
    pidx = int(state.player_idx)
    results.append(("player_has_food", float(state.properties[pidx, 0]) > 0))
    results.append(("player_has_stamina", float(state.properties[pidx, 1]) > 0))
    results.append(("player_has_thirst", float(state.properties[pidx, 2]) > 0))

    # Campfire exists
    fire_count = int((state.alive & (state.entity_type == 4)).sum())
    results.append(("one_campfire", fire_count == 1))

    # Well exists
    well_count = int((state.alive & (state.entity_type == 5)).sum())
    results.append(("one_well", well_count == 1))

    # Berry bushes exist
    bush_count = int((state.alive & (state.entity_type == 6)).sum())
    results.append(("two_berry_bushes", bush_count == 2))

    # Sticks exist
    stick_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("sticks_available", stick_count >= 4))

    # Player on island
    px, py = int(state.x[pidx]), int(state.y[pidx])
    on_island = (ISLAND_X1 <= px <= ISLAND_X2) and (ISLAND_Y1 <= py <= ISLAND_Y2)
    results.append(("player_on_island", on_island))

    return results
