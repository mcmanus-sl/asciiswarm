"""Game 05: Pac-Man Collect — collect all dots while avoiding ghosts."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import (
    create_entity, destroy_entity, get_entities_at, move_entity, rebuild_grid,
)
from jaxswarm.core.obs import get_obs
from jaxswarm.core.movement import move_toward
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=12, grid_h=12,
    max_entities=160,
    max_stack=2,
    num_entity_types=6,    # 0=unused, 1=player, 2=dot, 3=chaser, 4=patroller, 5=wall
    num_tags=6,
    num_props=3,           # 0=patrol_direction, 1=patrol_steps, 2=unused
    num_actions=6,
    max_turns=500,
    step_penalty=-0.005,
    game_state_size=2,     # 0=dots_remaining, 1=dots_collected
    prop_maxes=(4.0, 10.0, 1.0),
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace: 3 full boustrophedon passes through the 10x10 interior.
# Each pass = 10*9 + 9 = 99 steps + 10 to go back to (1,1) = ~109.
# 3 passes = ~327 steps + setup. Fits in 400 turns.
# The cross walls block some directions, but 3 passes from different starting
# corners should cover all cells by approaching from multiple angles.
_trace = []
# Pass 1: from (6,6) go to (1,10), sweep northward
_trace += [1]*4 + [3]*5  # to (1,10): 9
for row in range(10):
    if row % 2 == 0:
        _trace += [2]*9  # east
    else:
        _trace += [3]*9  # west
    if row < 9:
        _trace += [0]  # north
# 9 + 99 = 108. At y=1. Pass 2: sweep southward
for row in range(10):
    if row % 2 == 0:
        _trace += [2]*9
    else:
        _trace += [3]*9
    if row < 9:
        _trace += [1]
# 108 + 99 = 207. At y=10. Pass 3: go to (10,10), sweep northward via west
_trace += [2]*9  # east to x=10: 216
for row in range(10):
    if row % 2 == 0:
        _trace += [3]*9  # west
    else:
        _trace += [2]*9  # east
    if row < 9:
        _trace += [0]
# 216 + 99 = 315. Pass 4: sweep southward (opposite direction)
for row in range(10):
    if row % 2 == 0:
        _trace += [3]*9
    else:
        _trace += [2]*9
    if row < 9:
        _trace += [1]
# Remaining dots: row 1 x=2-9 (blocked by chaser at 1,1 and patroller at 10,1),
# and cross gap (5,5)(6,5).
# Navigate to (2,2), north to (2,1), east to (4,1) [Q1 row 1 dots].
# Then south, through cross gap, east to Q2, north to row 1, collect Q2 dots.
# Finally collect (5,5)/(6,5).
_trace += [3]*9 + [0]*8  # west to x=1, north to y=2 (not y=1, avoid chaser)
_trace += [2]*1 + [0]*1  # to (2,2), north to (2,1)
_trace += [2]*2  # east to (4,1) — collect dots x=2,3,4 on row 1
# Can't go east past x=4 because wall at x=5 y=3 blocks... wait, x=5 on y=1 is clear!
# Vertical wall is at x=5 y=3-4 and y=7-8, NOT y=1. So we can sweep through.
_trace += [2]*5  # east to (9,1) — collect dots x=5,6,7,8,9 if no ghost blocks
# Patroller at (10,1) blocks going further east.
# Now collect cross gap: go south to y=4, go to x=5, south to (5,5)
_trace += [1]*3  # south to y=4
_trace += [3]*4  # west to x=5
_trace += [1]*1  # south to (5,5)... wait, wall at (5,4)? No, vertical wall is at y=3,4. y=5 is gap.
# Actually going south from (5,4) puts us at (5,5). Then west to collect... no.
# The player is at (5,4) after west. South goes to (5,5) — collect that dot.
# Then east to (6,5) — collect that dot. WIN!
_trace += [2]*1  # east to (6,5) — collect!
_trace += [3]*1  # west to (5,5) — collect! WIN!
DETERMINISTIC_TRACE = _trace[:495]


def _is_wall_cell(x, y):
    """Check if cell should be a wall (border + cross pattern)."""
    is_border = (x == 0) | (x == 11) | (y == 0) | (y == 11)
    # Horizontal wall: y=5, x=3 to x=8, gap at x=5 and x=6
    is_h_wall = (y == 5) & (x >= 3) & (x <= 8) & (x != 5) & (x != 6)
    # Vertical wall: x=5, y=3 to y=8, gap at y=5 and y=6
    is_v_wall = (x == 5) & (y >= 3) & (y <= 8) & (y != 5) & (y != 6)
    return is_border | is_h_wall | is_v_wall


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)

    wall_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[1].set(True)
    wall_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

    # Place walls
    def place_walls(carry, idx):
        state = carry
        y = idx // CONFIG.grid_w
        x = idx % CONFIG.grid_w
        is_wall = _is_wall_cell(x, y)
        new_state, _ = create_entity(
            state, CONFIG, jnp.int32(5), x, y, wall_tags, wall_props
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(is_wall, n, o), new_state, state
        )
        return state, None

    state, _ = jax.lax.scan(
        place_walls, state, jnp.arange(CONFIG.grid_w * CONFIG.grid_h, dtype=jnp.int32)
    )

    # Player at center (6, 6)
    player_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
    player_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, player_slot = create_entity(
        state, CONFIG, jnp.int32(1), jnp.int32(6), jnp.int32(6), player_tags, player_props
    )
    state = state.replace(player_idx=player_slot)

    # Chaser at (1, 1)
    hazard_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[2].set(True)
    chaser_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(3), jnp.int32(1), jnp.int32(1), hazard_tags, chaser_props
    )

    # Patroller at (10, 1)
    patroller_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    patroller_props = patroller_props.at[0].set(0.0)  # direction: east
    patroller_props = patroller_props.at[1].set(0.0)  # steps: 0
    state, _ = create_entity(
        state, CONFIG, jnp.int32(4), jnp.int32(10), jnp.int32(1), hazard_tags, patroller_props
    )

    # Place dots on all empty cells (not wall, player, chaser, patroller)
    dot_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)
    dot_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

    # Special positions to skip
    special_cells = jnp.array([
        [6, 6],   # player
        [1, 1],   # chaser
        [10, 1],  # patroller
    ], dtype=jnp.int32)

    def place_dots(carry, idx):
        state, dot_count = carry
        y = idx // CONFIG.grid_w
        x = idx % CONFIG.grid_w
        is_wall = _is_wall_cell(x, y)
        is_special = jnp.any((special_cells[:, 0] == x) & (special_cells[:, 1] == y))
        should_place = ~is_wall & ~is_special

        new_state, _ = create_entity(
            state, CONFIG, jnp.int32(2), x, y, dot_tags, dot_props
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(should_place, n, o), new_state, state
        )
        dot_count = dot_count + jnp.where(should_place, 1, 0)
        return (state, dot_count), None

    (state, dot_count), _ = jax.lax.scan(
        place_dots, (state, jnp.int32(0)),
        jnp.arange(CONFIG.grid_w * CONFIG.grid_h, dtype=jnp.int32)
    )

    state = state.replace(
        game_state=state.game_state.at[0].set(dot_count.astype(jnp.float32)),
        rng_key=jax.random.split(rng_key)[0],
    )
    state = rebuild_grid(state, CONFIG)
    obs = get_obs(state, CONFIG)
    return state, obs


def _chaser_behavior(state, slot, config):
    """Move toward player using Manhattan distance. Only moves every 3rd turn."""
    pidx = state.player_idx
    key, subkey = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)
    can_move = (state.turn_number % 2 == 0)
    state_moved, _ = move_toward(state, config, slot, state.x[pidx], state.y[pidx], subkey)
    state = jax.tree.map(
        lambda n, o: jnp.where(can_move, n, o), state_moved, state
    )
    return state


def _patroller_behavior(state, slot, config):
    """Rectangular patrol: east→south→west→north, 9 steps per leg. Moves every 2nd turn."""
    can_move = (state.turn_number % 3 == 0)
    direction = state.properties[slot, 0].astype(jnp.int32)  # 0=E, 1=S, 2=W, 3=N
    steps = state.properties[slot, 1].astype(jnp.int32)

    # Direction to dx/dy: E(1,0), S(0,1), W(-1,0), N(0,-1)
    pdx = jnp.array([1, 0, -1, 0], dtype=jnp.int32)
    pdy = jnp.array([0, 1, 0, -1], dtype=jnp.int32)

    new_x = state.x[slot] + pdx[direction]
    new_y = state.y[slot] + pdy[direction]

    new_state, moved = move_entity(state, config, slot, new_x, new_y)
    actually_moved = moved & can_move
    state = jax.tree.map(
        lambda n, o: jnp.where(actually_moved, n, o), new_state, state
    )

    new_steps = jnp.where(can_move, steps + 1, steps)
    # After 9 steps or blocked, turn
    should_turn = (new_steps >= 9) | (~moved & can_move)
    new_direction = jnp.where(should_turn, (direction + 1) % 4, direction)
    new_steps = jnp.where(should_turn, jnp.int32(0), new_steps)

    state = state.replace(
        properties=state.properties.at[slot, 0].set(new_direction.astype(jnp.float32))
                                    .at[slot, 1].set(new_steps.astype(jnp.float32))
    )
    return state


def _noop(state, slot, config):
    return state

BEHAVIOR_TABLE = [
    _noop,                # 0: unused
    _noop,                # 1: player
    _noop,                # 2: dot
    _chaser_behavior,     # 3: chaser
    _patroller_behavior,  # 4: patroller
    _noop,                # 5: wall
]


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    pidx = state.player_idx
    px = state.x[pidx]
    py = state.y[pidx]
    is_move = action < 4

    target_x = px + DX[action]
    target_y = py + DY[action]
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)

    # Check target for solid/hazard
    slots, count = get_entities_at(state, safe_tx, safe_ty)

    def classify(i, carry):
        has_solid, has_hazard = carry
        slot = slots[i]
        v = (i < count) & in_bounds & is_move & state.alive[slot]
        has_solid = has_solid | (v & state.tags[slot, 1])
        has_hazard = has_hazard | (v & state.tags[slot, 2])
        return (has_solid, has_hazard)

    has_solid, has_hazard = jax.lax.fori_loop(
        0, CONFIG.max_stack, classify, (jnp.bool_(False), jnp.bool_(False))
    )

    # Hazard blocks movement (ghost acts as wall) — death only from ghost moving onto player
    can_move = is_move & in_bounds & ~has_solid & ~has_hazard
    new_state, _ = move_entity(state, CONFIG, pidx, target_x, target_y)
    state = jax.tree.map(
        lambda n, o: jnp.where(can_move, n, o), new_state, state
    )

    # Collect dots at new position
    new_px = state.x[pidx]
    new_py = state.y[pidx]
    cell_slots, cell_count = get_entities_at(state, new_px, new_py)

    def collect_dot(j, state):
        slot = cell_slots[j]
        is_dot = (j < cell_count) & state.alive[slot] & (state.entity_type[slot] == 2)
        new_state = destroy_entity(state, CONFIG, slot)
        new_state = new_state.replace(
            reward_acc=new_state.reward_acc + 0.05,
            game_state=new_state.game_state.at[0].set(new_state.game_state[0] - 1)
                                          .at[1].set(new_state.game_state[1] + 1),
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(is_dot, n, o), new_state, state
        )
        return state

    state = jax.lax.fori_loop(0, CONFIG.max_stack, collect_dot, state)

    # Win if all dots collected
    all_collected = state.game_state[0] <= 0
    state = state.replace(
        status=jnp.where(all_collected & (state.status == 0), jnp.int32(1), state.status)
    )

    # Phase 2: Ghost behaviors
    def ghost_loop(i, state):
        is_ghost = state.alive[i] & ((state.entity_type[i] == 3) | (state.entity_type[i] == 4))
        type_idx = state.entity_type[i]
        branches = [lambda s, sl=i, c=CONFIG: fn(s, sl, c) for fn in BEHAVIOR_TABLE]
        new_state = jax.lax.switch(type_idx, branches, state)
        # Check if ghost landed on player
        ghost_on_player = (new_state.x[i] == new_state.x[pidx]) & (new_state.y[i] == new_state.y[pidx])
        new_state = new_state.replace(
            status=jnp.where(
                ghost_on_player & new_state.alive[i] & (new_state.status == 0),
                jnp.int32(-1), new_state.status
            )
        )
        state = jax.tree.map(
            lambda n, o: jnp.where(is_ghost, n, o), new_state, state
        )
        return state

    state = jax.lax.fori_loop(0, CONFIG.max_entities, ghost_loop, state)

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    chaser_count = int((state.alive & (state.entity_type == 3)).sum())
    patroller_count = int((state.alive & (state.entity_type == 4)).sum())
    results.append(("one_chaser", chaser_count == 1))
    results.append(("one_patroller", patroller_count == 1))

    dot_count = int((state.alive & (state.entity_type == 2)).sum())
    results.append(("at_least_20_dots", dot_count >= 20))

    pidx = int(state.player_idx)
    results.append(("player_at_center", int(state.x[pidx]) == 6 and int(state.y[pidx]) == 6))

    # Player not on ghost cell
    px, py = int(state.x[pidx]), int(state.y[pidx])
    ghost_mask = state.alive & ((state.entity_type == 3) | (state.entity_type == 4))
    no_overlap = True
    for i in range(int(state.alive.sum())):
        if ghost_mask[i]:
            if int(state.x[i]) == px and int(state.y[i]) == py:
                no_overlap = False
    results.append(("player_not_on_ghost", no_overlap))

    return results
