"""Game 12: Tantrum Spiral — haul boulders, manage stress, avoid tantrum destruction."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=12, grid_h=12,
    max_entities=16,       # player + 5 boulders + 9 stockpile + keg
    max_stack=3,           # boulder on stockpile
    num_entity_types=5,    # 0=unused, 1=player, 2=boulder, 3=stockpile, 4=keg
    num_tags=8,            # 0=player, 1=solid, 2=hazard, 3=pickup, 4=exit, 5=npc, 6=pushable, 7=target
    num_props=4,           # 0=stress, 1=boulders_hauled, 2=charges (keg), 3=tantrum_turns
    num_actions=6,
    max_turns=400,
    step_penalty=-0.005,
    game_state_size=4,     # 0=boulders_hauled, 1=drinks_taken, 2=tantrum_count, 3=items_smashed
    prop_maxes=(20.0, 5.0, 3.0, 10.0),
    max_behaviors=2,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

# Deterministic trace for seed 0:
# Player at (6,6), boulders scattered, stockpile 3x3 at (1-3, 9-11), keg at (10, 1)
# Strategy: push boulders onto stockpile, drink keg when stress gets high
# N=0, S=1, E=2, W=3, I=4, WAIT=5
_trace12 = (
    # Go to nearest boulder and push it toward stockpile
    [3] * 3 + [0] * 2 +    # move to position behind boulder
    [1] * 3 +               # push boulder south
    [3] * 2 +               # push boulder west toward stockpile
    [1] * 2 +               # push onto stockpile row
    # Stress ~8 (4 from haul + passive). Go drink at keg (10,1)
    [0] * 8 + [2] * 6 +    # go to keg area
    [4] +                   # drink — stress drops
    # Go back and push another boulder
    [3] * 6 + [1] * 4 +    # reposition
    [1] * 2 + [3] * 2 +    # push boulder toward stockpile
    # Push more boulders
    [1] * 2 + [3] * 2 +    # push 3rd
    # Drink again
    [0] * 6 + [2] * 6 + [4] +
    # Final boulders
    [3] * 4 + [1] * 4 +
    [3] * 3 + [1] * 3
)
DETERMINISTIC_TRACE = _trace12[:395]


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

    # Slot 0: Player at center (6, 6)
    alive = alive.at[0].set(True)
    entity_type = entity_type.at[0].set(1)
    x = x.at[0].set(6)
    y = y.at[0].set(6)
    tags = tags.at[0, 0].set(True)  # player tag
    # stress=0, boulders_hauled=0, charges=0, tantrum_turns=0

    # Slots 1-5: Boulders at scattered positions
    boulder_positions = jnp.array([
        [4, 3], [8, 4], [3, 5], [7, 7], [9, 8]
    ], dtype=jnp.int32)
    for i in range(5):
        slot = 1 + i
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(2)  # boulder
        x = x.at[slot].set(boulder_positions[i, 0])
        y = y.at[slot].set(boulder_positions[i, 1])
        tags = tags.at[slot, 1].set(True)   # solid
        tags = tags.at[slot, 6].set(True)   # pushable

    # Slots 6-14: Stockpile 3x3 at (1-3, 9-11)
    stockpile_idx = jnp.arange(9, dtype=jnp.int32)
    sp_xs = 1 + (stockpile_idx % 3)
    sp_ys = 9 + (stockpile_idx // 3)
    for i in range(9):
        slot = 6 + i
        alive = alive.at[slot].set(True)
        entity_type = entity_type.at[slot].set(3)  # stockpile
        x = x.at[slot].set(sp_xs[i])
        y = y.at[slot].set(sp_ys[i])
        tags = tags.at[slot, 7].set(True)   # target tag

    # Slot 15: Keg at (10, 1) — tavern zone
    alive = alive.at[15].set(True)
    entity_type = entity_type.at[15].set(4)  # keg
    x = x.at[15].set(10)
    y = y.at[15].set(1)
    tags = tags.at[15, 5].set(True)   # npc tag
    properties = properties.at[15, 2].set(3.0)  # 3 charges

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
    tantrum_turns = state.properties[pidx, 3]
    is_tantrum = tantrum_turns > 0

    # --- Tantrum override ---
    # If in tantrum, replace action with random direction
    key, k_tantrum = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)
    tantrum_action = jax.random.randint(k_tantrum, (), 0, 4)  # random move
    effective_action = jnp.where(is_tantrum, tantrum_action, action)

    is_move = effective_action < 4
    is_interact = (effective_action == 4)

    target_x = px + DX[effective_action]
    target_y = py + DY[effective_action]
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)

    # Check target cell
    at_target = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_pushable = (at_target & state.tags[:, 6]).any()
    has_solid = (at_target & state.tags[:, 1]).any()

    # --- Tantrum: destroy items on bump ---
    # If in tantrum and bump into boulder/keg, destroy it
    bump_target_mask = at_target & (state.tags[:, 6] | state.tags[:, 5])  # pushable or npc
    any_bump = is_tantrum & is_move & in_bounds & bump_target_mask.any()
    bump_slot = jnp.argmax(bump_target_mask)  # first match

    tantrum_destroyed = state.alive.at[bump_slot].set(False)
    tantrum_etype = state.entity_type.at[bump_slot].set(0)
    tantrum_tags_arr = state.tags.at[bump_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))

    state = state.replace(
        alive=jnp.where(any_bump, tantrum_destroyed, state.alive),
        entity_type=jnp.where(any_bump, tantrum_etype, state.entity_type),
        tags=jnp.where(any_bump, tantrum_tags_arr, state.tags),
        game_state=state.game_state.at[3].set(
            jnp.where(any_bump, state.game_state[3] + 1, state.game_state[3])
        ),
    )

    # Decrement tantrum turns
    new_tantrum = jnp.where(is_tantrum, tantrum_turns - 1, tantrum_turns)
    state = state.replace(
        properties=state.properties.at[pidx, 3].set(new_tantrum),
    )

    # --- Push mechanics (only when NOT in tantrum) ---
    # Re-check target after potential tantrum destruction
    at_target2 = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    pushable_mask = at_target2 & state.tags[:, 6]
    push_slot = jnp.argmax(pushable_mask)
    has_pushable2 = pushable_mask.any()

    # Push direction = same as player movement
    push_x = safe_tx + DX[effective_action]
    push_y = safe_ty + DY[effective_action]
    push_in_bounds = (push_x >= 0) & (push_x < CONFIG.grid_w) & (push_y >= 0) & (push_y < CONFIG.grid_h)
    safe_push_x = jnp.clip(push_x, 0, CONFIG.grid_w - 1)
    safe_push_y = jnp.clip(push_y, 0, CONFIG.grid_h - 1)

    # Check push destination is clear (no solid)
    at_push_dest = state.alive & (state.x == safe_push_x) & (state.y == safe_push_y)
    push_blocked = (at_push_dest & state.tags[:, 1]).any()

    can_push = ~is_tantrum & is_move & in_bounds & has_pushable2 & push_in_bounds & ~push_blocked

    # Move boulder to push destination
    pushed_x = state.x.at[push_slot].set(safe_push_x)
    pushed_y = state.y.at[push_slot].set(safe_push_y)
    state = state.replace(
        x=jnp.where(can_push, pushed_x, state.x),
        y=jnp.where(can_push, pushed_y, state.y),
    )

    # Check if boulder landed on stockpile
    at_push_dest2 = state.alive & (state.x == safe_push_x) & (state.y == safe_push_y)
    on_stockpile = (at_push_dest2 & state.tags[:, 7]).any()
    boulder_hauled = can_push & on_stockpile

    # Remove boulder that's been hauled (it's on the stockpile now — consider it scored)
    hauled_alive = state.alive.at[push_slot].set(False)
    hauled_etype = state.entity_type.at[push_slot].set(0)
    hauled_tags = state.tags.at[push_slot].set(jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_))

    new_hauled = state.properties[pidx, 1] + 1.0
    hauled_props = state.properties.at[pidx, 1].set(new_hauled)
    # Stress +4 for hauling
    new_stress = jnp.minimum(state.properties[pidx, 0] + 4.0, 20.0)
    hauled_props = hauled_props.at[pidx, 0].set(new_stress)

    state = state.replace(
        alive=jnp.where(boulder_hauled, hauled_alive, state.alive),
        entity_type=jnp.where(boulder_hauled, hauled_etype, state.entity_type),
        tags=jnp.where(boulder_hauled, hauled_tags, state.tags),
        properties=jnp.where(boulder_hauled, hauled_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(boulder_hauled, 0.5, 0.0),
        game_state=state.game_state.at[0].set(
            jnp.where(boulder_hauled, state.game_state[0] + 1, state.game_state[0])
        ),
    )

    # Win check: 5 boulders hauled
    win = boulder_hauled & (new_hauled >= 5) & (state.status == 0)
    state = state.replace(
        status=jnp.where(win, jnp.int32(1), state.status),
    )

    # --- Player movement (after push clears the way or if nothing to push) ---
    at_target3 = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    has_solid3 = (at_target3 & state.tags[:, 1]).any()
    can_move = is_move & in_bounds & ~has_solid3 & ~is_tantrum
    # In tantrum: move freely (items already destroyed above)
    tantrum_can_move = is_tantrum & is_move & in_bounds & ~(at_target3 & state.tags[:, 1]).any()
    actually_move = can_move | tantrum_can_move

    new_x = state.x.at[pidx].set(jnp.where(actually_move, target_x, state.x[pidx]))
    new_y = state.y.at[pidx].set(jnp.where(actually_move, target_y, state.y[pidx]))
    state = state.replace(x=new_x, y=new_y)

    # --- Interact: drink at keg ---
    new_px, new_py = state.x[pidx], state.y[pidx]
    adj_xs = new_px + ADJ_DX
    adj_ys = new_py + ADJ_DY
    is_keg = state.alive & (state.entity_type == 4)
    keg_match = (state.x[:, None] == adj_xs[None, :]) & (state.y[:, None] == adj_ys[None, :])
    has_adj_keg = (is_keg[:, None] & keg_match).any()
    keg_slot = jnp.argmax(is_keg)  # there's only one keg

    keg_charges = state.properties[keg_slot, 2]
    can_drink = is_interact & ~is_tantrum & has_adj_keg & (keg_charges > 0)

    drink_stress = jnp.maximum(state.properties[pidx, 0] - 10.0, 0.0)
    drink_props = state.properties.at[pidx, 0].set(drink_stress)
    drink_props = drink_props.at[keg_slot, 2].set(keg_charges - 1.0)

    state = state.replace(
        properties=jnp.where(can_drink, drink_props, state.properties),
        reward_acc=state.reward_acc + jnp.where(can_drink, 0.05, 0.0),
        game_state=state.game_state.at[1].set(
            jnp.where(can_drink, state.game_state[1] + 1, state.game_state[1])
        ),
    )

    # --- Phase 2: Keg charge regeneration ---
    keg_regen = (state.turn_number % 20 == 0) & is_keg.any()
    keg_slot2 = jnp.argmax(is_keg)
    current_charges = state.properties[keg_slot2, 2]
    new_charges = jnp.minimum(current_charges + 1.0, 3.0)
    state = state.replace(
        properties=jnp.where(
            keg_regen,
            state.properties.at[keg_slot2, 2].set(new_charges),
            state.properties
        ),
    )

    # --- Phase 3: Turn End ---
    # Passive stress: +1 every 5 turns
    stress_tick = (state.turn_number % 5 == 0)
    current_stress = state.properties[pidx, 0]
    new_stress2 = jnp.minimum(current_stress + 1.0, 20.0)
    state = state.replace(
        properties=jnp.where(
            stress_tick,
            state.properties.at[pidx, 0].set(new_stress2),
            state.properties
        ),
    )

    # Tantrum trigger: stress >= 20 and not already in tantrum
    final_stress = state.properties[pidx, 0]
    enter_tantrum = (final_stress >= 20) & (state.properties[pidx, 3] <= 0)
    state = state.replace(
        properties=jnp.where(
            enter_tantrum,
            state.properties.at[pidx, 3].set(10.0),
            state.properties
        ),
        game_state=state.game_state.at[2].set(
            jnp.where(enter_tantrum, state.game_state[2] + 1, state.game_state[2])
        ),
    )

    # Rebuild grid
    state = rebuild_grid(state, CONFIG)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []

    # 1. Exactly 5 boulders at start
    boulder_count = int((state.alive & (state.entity_type == 2)).sum())
    results.append(("five_boulders", boulder_count == 5))

    # 2. Exactly 9 stockpile tiles (3x3)
    stockpile_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("nine_stockpile", stockpile_count == 9))

    # 3. Exactly 1 keg
    keg_count = int((state.alive & (state.entity_type == 4)).sum())
    results.append(("one_keg", keg_count == 1))

    # 4. Player starts with stress=0, boulders_hauled=0
    pidx = int(state.player_idx)
    results.append(("player_no_stress",
                    float(state.properties[pidx, 0]) == 0 and
                    float(state.properties[pidx, 1]) == 0))

    # 5. Keg starts with charges=3
    keg_mask = state.alive & (state.entity_type == 4)
    keg_slot = int(jnp.argmax(keg_mask))
    results.append(("keg_three_charges", float(state.properties[keg_slot, 2]) == 3.0))

    return results
