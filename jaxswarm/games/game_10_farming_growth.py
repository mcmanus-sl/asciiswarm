"""Game 10: Farming & Growth — plant seeds, wait for crops, harvest, deliver."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=14, grid_h=14,
    max_entities=24,       # 0=player, 1-9=soil, 10=seedbag, 11=bin, 12-23=sprouts/mature
    max_stack=3,           # sprout/mature on soil
    num_entity_types=8,    # 0=unused, 1=player, 2=wall, 3=soil, 4=seedbag, 5=sprout, 6=mature, 7=bin
    num_tags=6,
    num_props=4,           # 0=seeds, 1=crops, 2=delivered, 3=age(sprout)
    num_actions=6,
    max_turns=300,
    step_penalty=-0.005,
    game_state_size=4,     # 0=seeds_planted, 1=crops_harvested, 2=crops_delivered, 3=unused
    prop_maxes=(10.0, 10.0, 5.0, 1.0),
    max_behaviors=2,       # player only; farming_system uses fori_loop over max_entities for sprouts
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Deterministic trace for seed 0: Player(1,3), Seedbag(2,5), Soil(5-7,5-7), Bin(12,1).
# N=0,S=1,E=2,W=3,I=4,WAIT=5
_trace10 = (
    [1]*2 + [2] +               # (1,3)->(1,5)->(2,5) pickup seeds=6
    [2]*3 +                      # (2,5)->(5,5) on soil
    [4] +                        # plant at (5,5)
    [2, 4] +                     # plant at (6,5)
    [2, 4] +                     # plant at (7,5)
    [1, 4] +                     # plant at (7,6)
    [3, 4] +                     # plant at (6,6)
    [5]*16 +                     # wait 16 turns for growth
    [2] +                        # (6,6)->(7,6) harvest
    [0] +                        # (7,6)->(7,5) harvest
    [3] +                        # (7,5)->(6,5) harvest
    [3] +                        # (6,5)->(5,5) harvest
    [3] +                        # leave (5,5) to (4,5)
    [2] +                        # back to (5,5) — not needed, already harvested
    # Actually: 4 mature + 1 auto-collected = 5 crops
    [0]*4 + [2]*6 +              # (5,5)->(5,1)->(11,1) adjacent to bin
    [4]*5                         # deliver 5 crops = WIN
)
DETERMINISTIC_TRACE = _trace10[:295]


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 4)

    # --- Vectorized bulk init: no create_entity loops ---

    # Slot 0: Player in farmhouse area (top-left)
    k_px, k_py = jax.random.split(keys[0])
    player_x = jax.random.randint(k_px, (), 1, 4)
    player_y = jax.random.randint(k_py, (), 1, 4)

    # Slots 1-9: Soil patch 3x3 at (5-7, 5-7)
    soil_idx = jnp.arange(9, dtype=jnp.int32)
    soil_xs = 5 + (soil_idx % 3)
    soil_ys = 5 + (soil_idx // 3)

    # Assemble all entity arrays
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
    tags = tags.at[0, 0].set(True)  # player tag

    # Slots 1-9: soil (type 3, tag 5 = npc)
    alive = alive.at[1:10].set(True)
    entity_type = entity_type.at[1:10].set(3)
    x = x.at[1:10].set(soil_xs)
    y = y.at[1:10].set(soil_ys)
    tags = tags.at[1:10, 5].set(True)  # npc tag

    # Slot 10: seedbag (type 4, tag 3 = pickup) at (2, 5)
    alive = alive.at[10].set(True)
    entity_type = entity_type.at[10].set(4)
    x = x.at[10].set(2)
    y = y.at[10].set(5)
    tags = tags.at[10, 3].set(True)  # pickup tag

    # Slot 11: bin (type 7, tag 5 = npc) at (12, 1)
    alive = alive.at[11].set(True)
    entity_type = entity_type.at[11].set(7)
    x = x.at[11].set(12)
    y = y.at[11].set(1)
    tags = tags.at[11, 5].set(True)  # npc tag

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

    # --- Movement: vectorized solid check ---
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

    # --- Pickup at new position ---
    new_px, new_py = state.x[pidx], state.y[pidx]
    at_player = state.alive & (state.x == new_px) & (state.y == new_py)

    # Seedbag pickup (type 4): gives seeds=6, tombstone the seedbag
    seedbag_mask = at_player & (state.entity_type == 4)
    has_seedbag = seedbag_mask.any()
    alive_after_seedbag = state.alive & ~seedbag_mask
    props_after_seedbag = jnp.where(has_seedbag, state.properties.at[pidx, 0].set(6.0), state.properties)
    reward_after_seedbag = state.reward_acc + jnp.where(has_seedbag, 0.1, 0.0)
    state = state.replace(
        alive=jnp.where(has_seedbag, alive_after_seedbag, state.alive),
        properties=props_after_seedbag,
        reward_acc=reward_after_seedbag,
    )

    # Mature crop pickup (type 6): crops += 1, tombstone
    at_player2 = state.alive & (state.x == new_px) & (state.y == new_py)
    mature_mask = at_player2 & (state.entity_type == 6)
    num_mature = mature_mask.sum().astype(jnp.float32)
    has_mature = num_mature > 0
    alive_after_mature = state.alive & ~mature_mask
    new_crops = jnp.minimum(state.properties[pidx, 1] + num_mature, 10.0)
    props_after_mature = state.properties.at[pidx, 1].set(jnp.where(has_mature, new_crops, state.properties[pidx, 1]))
    reward_after_mature = state.reward_acc + jnp.where(has_mature, 0.15 * num_mature, 0.0)
    gs_after_mature = state.game_state.at[1].set(
        jnp.where(has_mature, state.game_state[1] + num_mature, state.game_state[1])
    )
    state = state.replace(
        alive=jnp.where(has_mature, alive_after_mature, state.alive),
        properties=props_after_mature,
        reward_acc=reward_after_mature,
        game_state=gs_after_mature,
    )

    # --- Interact: plant seed or deliver to bin ---
    # Check cell for soil and existing plants
    at_cell = state.alive & (state.x == new_px) & (state.y == new_py)
    on_soil = (at_cell & (state.entity_type == 3)).any()
    has_plant = (at_cell & ((state.entity_type == 5) | (state.entity_type == 6))).any()

    can_plant = is_interact & on_soil & ~has_plant & (state.properties[pidx, 0] >= 1)

    # Create sprout: find first free slot, assign directly
    free_slot = jnp.argmin(state.alive)  # first False = first free
    planted_alive = state.alive.at[free_slot].set(True)
    planted_etype = state.entity_type.at[free_slot].set(5)
    planted_x = state.x.at[free_slot].set(new_px)
    planted_y = state.y.at[free_slot].set(new_py)
    planted_tags = state.tags.at[free_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))
    planted_props_row = state.properties.at[free_slot].set(jnp.zeros(CONFIG.num_props, dtype=jnp.float32))
    # Also decrement seeds
    planted_props_row = planted_props_row.at[pidx, 0].set(state.properties[pidx, 0] - 1)

    state = state.replace(
        alive=jnp.where(can_plant, planted_alive, state.alive),
        entity_type=jnp.where(can_plant, planted_etype, state.entity_type),
        x=jnp.where(can_plant, planted_x, state.x),
        y=jnp.where(can_plant, planted_y, state.y),
        tags=jnp.where(can_plant, planted_tags, state.tags),
        properties=jnp.where(can_plant, planted_props_row, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_plant, 0.05, 0.0),
        game_state=state.game_state.at[0].set(
            jnp.where(can_plant, state.game_state[0] + 1, state.game_state[0])
        ),
    )

    # Deliver to bin: vectorized adjacent check
    adj_xs = new_px + ADJ_DX  # shape (4,)
    adj_ys = new_py + ADJ_DY  # shape (4,)
    is_bin_entity = state.alive & (state.entity_type == 7)
    bin_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])  # [max_entities, 4]
    has_adj_bin = (is_bin_entity[:, None] & bin_match).any()

    can_deliver = is_interact & ~can_plant & has_adj_bin & (state.properties[pidx, 1] >= 1)
    new_delivered = state.properties[pidx, 2] + 1.0
    deliver_props = state.properties.at[pidx, 1].set(state.properties[pidx, 1] - 1).at[pidx, 2].set(new_delivered)
    win = (new_delivered >= 5) & (state.status == 0)
    state = state.replace(
        properties=jnp.where(can_deliver, deliver_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_deliver, 0.3, 0.0),
        game_state=state.game_state.at[2].set(
            jnp.where(can_deliver, state.game_state[2] + 1, state.game_state[2])
        ),
        status=jnp.where(can_deliver & win, jnp.int32(1), state.status),
    )

    # --- Growth: vectorized inline (replaces farming_system) ---
    is_sprout = state.alive & (state.entity_type == 5)
    new_age = state.properties[:, 3] + 1.0
    # Increment age for all sprouts
    aged_props = state.properties.at[:, 3].set(
        jnp.where(is_sprout, new_age, state.properties[:, 3])
    )
    # Mature: sprouts that hit threshold become type 6, get pickup tag (3), reset age
    is_now_mature = is_sprout & (new_age >= 15)
    new_etype = jnp.where(is_now_mature, 6, state.entity_type)
    new_tags = state.tags.at[:, 3].set(state.tags[:, 3] | is_now_mature)  # add pickup tag
    matured_props = aged_props.at[:, 3].set(
        jnp.where(is_now_mature, 0.0, aged_props[:, 3])
    )
    state = state.replace(
        entity_type=new_etype,
        tags=new_tags,
        properties=matured_props,
    )

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    soil_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("nine_soil_tiles", soil_count == 9))

    seedbag_count = int((state.alive & (state.entity_type == 4)).sum())
    results.append(("one_seedbag", seedbag_count == 1))

    bin_count = int((state.alive & (state.entity_type == 7)).sum())
    results.append(("one_bin", bin_count == 1))

    pidx = int(state.player_idx)
    results.append(("player_empty_inventory",
                    float(state.properties[pidx, 0]) == 0 and
                    float(state.properties[pidx, 1]) == 0 and
                    float(state.properties[pidx, 2]) == 0))

    sprout_count = int((state.alive & (state.entity_type == 5)).sum())
    mature_count = int((state.alive & (state.entity_type == 6)).sum())
    results.append(("no_plants_at_start", sprout_count == 0 and mature_count == 0))

    return results
