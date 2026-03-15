"""Game 04: Dungeon Crawl — multi-room combat with enemies and potions."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import (
    create_entity, destroy_entity, get_entities_at, move_entity, rebuild_grid,
)
from jaxswarm.core.obs import get_obs
from jaxswarm.core.movement import move_toward
from jaxswarm.systems.movement import DX, DY
from jaxswarm.systems.combat import combat_system, enemy_attack_player

CONFIG = EnvConfig(
    grid_w=16, grid_h=16,
    max_entities=128,
    max_stack=3,
    num_entity_types=8,    # 0=unused, 1=player, 2=exit, 3=wall, 4=wanderer, 5=chaser, 6=sentinel, 7=potion
    num_tags=6,
    num_props=3,           # 0=health, 1=attack, 2=direction
    num_actions=6,
    max_turns=500,
    step_penalty=-0.005,
    game_state_size=2,     # 0=enemies_killed, 1=potions_used
    prop_maxes=(10.0, 5.0, 4.0),
    max_behaviors=16,
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace: Room 1 → corridor (3,7-8) → Room 3 → corridor (7-8,11) → Room 4.
# Player random in Room 1 (1-6, 1-6), exit random in Room 4 (9-14, 9-14).
# Enemies move every other turn (gated). Must reach exit before HP runs out.
_trace04 = []
# Phase 1: Go to x=3, then rush south through corridor.
# Interleave: go west while going south to minimize time.
_trace04 += [3] * 5  # west to x=1 (at most 5 west from x=6)
_trace04 += [2] * 2  # east to x=3
_trace04 += [1] * 13 # south: y=1→6(room1), 7→8(corridor), 9→14(room3) max overshoot OK
# Phase 2: East through corridor (7-8,11) and into Room 4
# First go north if overshot past y=11
_trace04 += [0] * 3  # north back to y=11 if overshot
_trace04 += [2] * 8  # east through corridor and across Room 4
# Phase 3: Sweep Room 4 (9-14, 9-14) — go to (9,9) corner, boustrophedon
# Player at ~(11,11) after east. Go west to x=9, north to y=9 (stay in Room 4!)
_trace04 += [3] * 2 + [0] * 2  # to (9,9) — careful not to exit Room 4
for row in range(6):
    if row % 2 == 0:
        _trace04 += [2] * 5
    else:
        _trace04 += [3] * 5
    if row < 5:
        _trace04 += [1]
DETERMINISTIC_TRACE = _trace04[:495]


# Room layout: 4 rooms in quadrants connected by corridors
# Room 1: (1,1)-(6,6)   Room 2: (9,1)-(14,6)
# Room 3: (1,9)-(6,14)  Room 4: (9,9)-(14,14)
# Corridors: horizontal at y=3 from x=6 to x=9, vertical at x=3 from y=6 to y=9
#            horizontal at y=11 from x=6 to x=9, vertical at x=11 from y=6 to y=9

ROOMS = [
    (1, 1, 6, 6),     # (x1, y1, x2, y2) inclusive
    (9, 1, 14, 6),
    (1, 9, 6, 14),
    (9, 9, 14, 14),
]

CORRIDORS = [
    # (x_start, y_start, x_end, y_end) — single-tile wide paths
    (7, 3, 8, 3),    # room 1 to room 2 (horizontal, gap at x=7-8)
    (3, 7, 3, 8),    # room 1 to room 3 (vertical, gap at y=7-8)
    (11, 7, 11, 8),  # room 2 to room 4 (vertical, gap at y=7-8)
    (7, 11, 8, 11),  # room 3 to room 4 (horizontal, gap at x=7-8)
]


def _is_in_room(x, y):
    """Check if (x,y) is inside any room (not on the wall)."""
    in_any = jnp.bool_(False)
    for x1, y1, x2, y2 in ROOMS:
        in_any = in_any | ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2))
    return in_any


def _is_in_corridor(x, y):
    """Check if (x,y) is in a corridor."""
    in_any = jnp.bool_(False)
    for x1, y1, x2, y2 in CORRIDORS:
        in_any = in_any | ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2))
    return in_any


def _is_open(x, y):
    return _is_in_room(x, y) | _is_in_corridor(x, y)


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 20)
    # Key assignments preserved from original to keep deterministic trace valid:
    # keys[0]=player, keys[1]=exit, keys[2..17]=rooms loop, keys[18]=state.rng_key

    # --- Behavior entities first (player + NPCs) so they occupy low slots ---

    # Player in random position within room 1 (x=1-6, y=1-6)
    player_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
    player_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    player_props = player_props.at[0].set(10.0)  # health
    player_props = player_props.at[1].set(2.0)   # attack
    k_px, k_py, k_rest = jax.random.split(keys[0], 3)
    player_x = jax.random.randint(k_px, (), 1, 7)
    player_y = jax.random.randint(k_py, (), 1, 7)
    state, player_slot = create_entity(
        state, CONFIG, jnp.int32(1), player_x, player_y, player_tags, player_props
    )
    state = state.replace(player_idx=player_slot)

    # Helper to place entity in random room position
    hazard_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[2].set(True)
    pickup_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)

    def _place_in_room(state, key, room_idx, etype, tags, props):
        rx1, ry1, rx2, ry2 = ROOMS[room_idx]
        k1, k2 = jax.random.split(key)
        x = jax.random.randint(k1, (), rx1, rx2 + 1)
        y = jax.random.randint(k2, (), ry1, ry2 + 1)
        state, _ = create_entity(state, CONFIG, jnp.int32(etype), x, y, tags, props)
        return state

    # Place NPCs (wanderers, chasers, sentinels) — use same keys as original
    # Original loop: ki=2 + room_idx*4 + {0=wanderer, 1=chaser, 2=sentinel, 3=potion}
    for room_idx in range(4):
        ki_base = 2 + room_idx * 4
        # Wanderer in each room
        w_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
        w_props = w_props.at[0].set(1.0).at[1].set(1.0).at[2].set(1.0)
        state = _place_in_room(state, keys[ki_base], room_idx, 4, hazard_tags, w_props)

        # Chaser in rooms 2+ (idx >= 1)
        c_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
        c_props = c_props.at[0].set(2.0).at[1].set(2.0)
        new_state = _place_in_room(state, keys[ki_base + 1], room_idx, 5, hazard_tags, c_props)
        state = jax.tree.map(
            lambda n, o: jnp.where(room_idx >= 1, n, o), new_state, state
        )

        # Sentinel in rooms 3+ (idx >= 2)
        s_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
        s_props = s_props.at[0].set(3.0).at[1].set(1.0)
        new_state = _place_in_room(state, keys[ki_base + 2], room_idx, 6, hazard_tags, s_props)
        state = jax.tree.map(
            lambda n, o: jnp.where(room_idx >= 2, n, o), new_state, state
        )

    # --- Static entities below (walls, exit, potions) ---

    # Place walls: fill entire grid with walls, then carve rooms and corridors
    wall_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[1].set(True)
    wall_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

    def place_walls(carry, idx):
        state = carry
        y = idx // CONFIG.grid_w
        x = idx % CONFIG.grid_w
        is_open = _is_open(x, y)
        new_state, _ = create_entity(
            state, CONFIG, jnp.int32(3), x, y, wall_tags, wall_props
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(~is_open, n, o), new_state, state
        )
        return state, None

    state, _ = jax.lax.scan(
        place_walls, state, jnp.arange(CONFIG.grid_w * CONFIG.grid_h, dtype=jnp.int32)
    )

    # Exit in random position within room 4 (x=9-14, y=9-14)
    exit_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[4].set(True)
    exit_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    k_ex, k_ey = jax.random.split(keys[1])
    exit_x = jax.random.randint(k_ex, (), 9, 15)
    exit_y = jax.random.randint(k_ey, (), 9, 15)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(2), exit_x, exit_y, exit_tags, exit_props
    )

    # Potions in each room (keys[ki_base + 3] for each room)
    for room_idx in range(4):
        ki_base = 2 + room_idx * 4
        p_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
        state = _place_in_room(state, keys[ki_base + 3], room_idx, 7, pickup_tags, p_props)

    state = state.replace(rng_key=keys[18])
    state = rebuild_grid(state, CONFIG)
    obs = get_obs(state, CONFIG)
    return state, obs


def _movement_with_combat(state: EnvState, action: jnp.int32) -> EnvState:
    """Move player. If target has hazard: bump-attack. If pickup: collect. If solid: block."""
    pidx = state.player_idx
    px = state.x[pidx]
    py = state.y[pidx]
    is_move = action < 4

    target_x = px + DX[action]
    target_y = py + DY[action]
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)

    slots, count = get_entities_at(state, safe_tx, safe_ty)

    # Check target cell contents
    def classify_cell(i, carry):
        has_solid, has_hazard, has_pickup = carry
        slot = slots[i]
        is_valid = (i < count) & in_bounds & is_move
        alive = state.alive[slot]
        has_solid = has_solid | (is_valid & alive & state.tags[slot, 1])
        has_hazard = has_hazard | (is_valid & alive & state.tags[slot, 2])
        has_pickup = has_pickup | (is_valid & alive & state.tags[slot, 3])
        return (has_solid, has_hazard, has_pickup)

    has_solid, has_hazard, has_pickup = jax.lax.fori_loop(
        0, CONFIG.max_stack, classify_cell,
        (jnp.bool_(False), jnp.bool_(False), jnp.bool_(False))
    )

    # If hazard: do combat (handled by combat_system called separately)
    # If solid or hazard: don't move
    can_move = is_move & in_bounds & ~has_solid & ~has_hazard

    # Move player
    new_state, _ = move_entity(state, CONFIG, pidx, target_x, target_y)
    state = jax.tree.map(
        lambda n, o: jnp.where(can_move, n, o), new_state, state
    )

    # Pick up potion if moved to cell with pickup
    moved_slots, moved_count = get_entities_at(state, state.x[pidx], state.y[pidx])

    def pickup_potion(j, state):
        slot = moved_slots[j]
        is_valid = j < moved_count
        is_potion = is_valid & state.alive[slot] & (state.entity_type[slot] == 7)

        new_hp = jnp.minimum(state.properties[pidx, 0] + 3.0, 10.0)
        new_state = destroy_entity(state, CONFIG, slot)
        new_state = new_state.replace(
            properties=new_state.properties.at[pidx, 0].set(new_hp),
            reward_acc=new_state.reward_acc + 0.1,
            game_state=new_state.game_state.at[1].set(new_state.game_state[1] + 1),
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(is_potion, n, o), new_state, state
        )
        return state

    state = jax.lax.fori_loop(0, CONFIG.max_stack, pickup_potion, state)
    return state


def _npc_behavior(state, slot, config):
    return state

def _wanderer_behavior(state, slot, config):
    """Random walk."""
    key, subkey = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)
    direction = jax.random.randint(subkey, (), 0, 4)
    new_x = state.x[slot] + DX[direction]
    new_y = state.y[slot] + DY[direction]
    # Check for solid at target
    safe_nx = jnp.clip(new_x, 0, config.grid_w - 1)
    safe_ny = jnp.clip(new_y, 0, config.grid_h - 1)
    in_bounds = (new_x >= 0) & (new_x < config.grid_w) & (new_y >= 0) & (new_y < config.grid_h)
    target_slots, tc = get_entities_at(state, safe_nx, safe_ny)
    def cs(i, hs):
        s = target_slots[i]
        v = (i < tc) & in_bounds & state.alive[s] & state.tags[s, 1]
        return hs | v
    blocked = jax.lax.fori_loop(0, config.max_stack, cs, jnp.bool_(False))
    new_state, _ = move_entity(state, config, slot, new_x, new_y)
    state = jax.tree.map(
        lambda n, o: jnp.where(~blocked & in_bounds, n, o), new_state, state
    )
    return state

def _chaser_behavior(state, slot, config):
    """Chase player if within Manhattan 5, else random walk."""
    pidx = state.player_idx
    dist = jnp.abs(state.x[slot] - state.x[pidx]) + jnp.abs(state.y[slot] - state.y[pidx])
    in_range = dist <= 5

    key, subkey = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)

    # Chase: move toward player
    chase_state, _ = move_toward(state, config, slot, state.x[pidx], state.y[pidx], subkey)
    # Wander: random walk
    wander_state = _wanderer_behavior(state, slot, config)

    state = jax.tree.map(
        lambda c, w: jnp.where(in_range, c, w), chase_state, wander_state
    )
    return state

def _sentinel_behavior(state, slot, config):
    """Stationary — does not move."""
    return state

BEHAVIOR_TABLE = [
    _npc_behavior,       # 0: unused
    _npc_behavior,       # 1: player
    _npc_behavior,       # 2: exit
    _npc_behavior,       # 3: wall
    _wanderer_behavior,  # 4: wanderer
    _chaser_behavior,    # 5: chaser
    _sentinel_behavior,  # 6: sentinel
    _npc_behavior,       # 7: potion
]


def _run_behaviors(state: EnvState) -> EnvState:
    """Run NPC behaviors and check for enemy-on-player collisions."""
    def loop_body(i, state):
        is_npc = state.alive[i] & (state.entity_type[i] >= 4) & (state.entity_type[i] <= 6)
        type_idx = state.entity_type[i]
        branches = [lambda s, sl=i, c=CONFIG, f=fn: f(s, sl, c) for fn in BEHAVIOR_TABLE]
        new_state = jax.lax.switch(type_idx, branches, state)
        # Check if enemy landed on player
        new_state = enemy_attack_player(new_state, jnp.int32(i), CONFIG)
        state = jax.tree.map(
            lambda n, o: jnp.where(is_npc, n, o), new_state, state
        )
        return state

    return jax.lax.fori_loop(0, CONFIG.max_behaviors, loop_body, state)


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    # Phase 1: Player movement + combat + pickup
    state = combat_system(state, action, CONFIG)  # bump-attack damage
    state = _movement_with_combat(state, action)   # move/collect

    # Check exit collision after movement
    pidx = state.player_idx
    exit_slots, exit_count = get_entities_at(state, state.x[pidx], state.y[pidx])
    def check_exit(i, found):
        slot = exit_slots[i]
        is_exit = (i < exit_count) & state.alive[slot] & state.tags[slot, 4]
        return found | is_exit
    on_exit = jax.lax.fori_loop(0, CONFIG.max_stack, check_exit, jnp.bool_(False))
    state = state.replace(
        status=jnp.where(on_exit & (state.status == 0), jnp.int32(1), state.status)
    )

    # Phase 2: NPC behaviors (move every other turn to give player breathing room)
    should_move = (state.turn_number % 2 == 0)
    new_state = _run_behaviors(state)
    state = jax.tree.map(
        lambda n, o: jnp.where(should_move, n, o), new_state, state
    )

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    # Player health > 0
    pidx = int(state.player_idx)
    results.append(("player_health_positive", float(state.properties[pidx, 0]) > 0))

    # Total enemies between 5 and 20
    enemy_mask = state.alive & ((state.entity_type == 4) | (state.entity_type == 5) | (state.entity_type == 6))
    enemy_count = int(enemy_mask.sum())
    results.append(("enemy_count_valid", 5 <= enemy_count <= 20))

    # At least one potion
    potion_count = int((state.alive & (state.entity_type == 7)).sum())
    results.append(("has_potions", potion_count >= 1))

    # Player exists
    player_count = int((state.alive & (state.entity_type == 1)).sum())
    results.append(("exactly_one_player", player_count == 1))

    # Exit exists
    exit_count = int((state.alive & (state.entity_type == 2)).sum())
    results.append(("exactly_one_exit", exit_count == 1))

    return results
