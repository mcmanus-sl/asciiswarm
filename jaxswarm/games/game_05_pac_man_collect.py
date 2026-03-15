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
    game_state_size=3,     # 0=dots_remaining, 1=dots_collected, 2=prev_nearest_dot_dist
    prop_maxes=(4.0, 10.0, 1.0),
    max_behaviors=4,       # slots 0-3: player, chaser, patroller (+ 1 spare). Created first in reset().
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace for seed 0. Ghosts move every 4th turn (outer gate).
# Player starts at (6,6), chaser at (2,3), patroller at (7,4).
# Strategy: sweep right half first (away from chaser), then left half,
# then two full-width passes, then targeted cleanup of wall-adjacent cells.
# Actions: 0=N, 1=S, 2=E, 3=W
_trace = []
# Phase 1: (6,6) -> (10,10) — move away from chaser
_trace += [1]*4 + [2]*4
# Phase 2: right half sweep northward (x=6..10, y=10..1)
for row in range(10):
    _trace += [3]*4 if row % 2 == 0 else [2]*4
    if row < 9: _trace += [0]
# Phase 3: cross to left side
_trace += [3]*9
# Phase 4: left half sweep southward (x=1..5, y=1..10)
for row in range(10):
    _trace += [2]*4 if row % 2 == 0 else [3]*4
    if row < 9: _trace += [1]
# Phase 5: go to (10,1) for full-width sweep
_trace += [0]*9 + [2]*9
# Phase 6: full-width sweep southward (x=1..10)
for row in range(10):
    _trace += [3]*9 if row % 2 == 0 else [2]*9
    if row < 9: _trace += [1]
# Phase 7: full-width sweep northward
for row in range(10):
    _trace += [2]*9 if row % 2 == 0 else [3]*9
    if row < 9: _trace += [0]
# Phase 8: cleanup from (1,1) — remaining dots near walls
_trace += [1, 1, 2, 0]           # (1,2)→(1,3)→(2,3)→(2,2) — 3 dots
_trace += [2]*6 + [1]*2           # (2,2)→(8,2)→(8,4) — 1 dot
_trace += [2] + [1]*2 + [3]*4 + [0]  # (8,4)→(9,4)→(9,6)→(5,6)→(5,5) — 1 dot, WIN
DETERMINISTIC_TRACE = _trace[:495]


def _is_wall_cell(x, y):
    """Check if cell should be a wall (border + cross pattern)."""
    is_border = (x == 0) | (x == 11) | (y == 0) | (y == 11)
    # Horizontal wall: y=5, x=3 to x=8, gap at x=5 and x=6
    is_h_wall = (y == 5) & (x >= 3) & (x <= 8) & (x != 5) & (x != 6)
    # Vertical wall: x=5, y=3 to y=8, gap at y=5 and y=6
    is_v_wall = (x == 5) & (y >= 3) & (y <= 8) & (y != 5) & (y != 6)
    return is_border | is_h_wall | is_v_wall


def _nearest_dot_manhattan(state, config):
    """Min Manhattan distance from player to any alive dot. Fully vectorized."""
    pidx = state.player_idx
    px, py = state.x[pidx], state.y[pidx]
    dists = jnp.abs(state.x - px) + jnp.abs(state.y - py)
    # INF for non-dots or dead entities
    INF = config.grid_h + config.grid_w
    dists = jnp.where(state.alive & (state.entity_type == 2), dists, INF)
    return jnp.min(dists).astype(jnp.float32)


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)

    # === CRITICAL: Create behavior entities FIRST (slots 0-2) ===
    # This guarantees player/ghosts occupy the lowest slots so the
    # behavior fori_loop(0, max_behaviors=4) covers them all.

    # Compute all positions upfront to avoid overlaps
    k_px, k_py, k_rest = jax.random.split(rng_key, 3)
    player_x = jax.random.randint(k_px, (), 1, 11)
    player_y = jax.random.randint(k_py, (), 1, 11)
    is_wall = _is_wall_cell(player_x, player_y)
    player_x = jnp.where(is_wall, jnp.int32(6), player_x)
    player_y = jnp.where(is_wall, jnp.int32(6), player_y)

    k_cx, k_cy, k_rest = jax.random.split(k_rest, 3)
    chaser_x = jax.random.randint(k_cx, (), 1, 5)
    chaser_y = jax.random.randint(k_cy, (), 1, 5)

    k_ptx, k_pty, k_rest = jax.random.split(k_rest, 3)
    patroller_x = jax.random.randint(k_ptx, (), 7, 11)
    patroller_y = jax.random.randint(k_pty, (), 1, 5)

    on_ghost = ((player_x == chaser_x) & (player_y == chaser_y)) | \
               ((player_x == patroller_x) & (player_y == patroller_y))
    player_x = jnp.where(on_ghost, jnp.int32(6), player_x)
    player_y = jnp.where(on_ghost, jnp.int32(6), player_y)

    # Slot 0: player
    player_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
    player_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, player_slot = create_entity(
        state, CONFIG, jnp.int32(1), player_x, player_y, player_tags, player_props
    )
    state = state.replace(player_idx=player_slot)

    # Slot 1: chaser
    hazard_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[2].set(True)
    chaser_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    state, _ = create_entity(
        state, CONFIG, jnp.int32(3), chaser_x, chaser_y, hazard_tags, chaser_props
    )

    # Slot 2: patroller
    patroller_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
    patroller_props = patroller_props.at[0].set(0.0)  # direction: east
    patroller_props = patroller_props.at[1].set(0.0)  # steps: 0
    state, _ = create_entity(
        state, CONFIG, jnp.int32(4), patroller_x, patroller_y, hazard_tags, patroller_props
    )

    # === Now place static entities (walls, dots) in remaining slots ===

    wall_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[1].set(True)
    wall_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

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

    # Place dots on all empty cells (not wall, player, chaser, patroller)
    dot_tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[3].set(True)
    dot_props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

    special_cells = jnp.stack([
        jnp.array([player_x, player_y], dtype=jnp.int32),
        jnp.array([chaser_x, chaser_y], dtype=jnp.int32),
        jnp.array([patroller_x, patroller_y], dtype=jnp.int32),
    ])

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
        rng_key=k_rest,
    )
    state = rebuild_grid(state, CONFIG)
    init_dist = _nearest_dot_manhattan(state, CONFIG)
    state = state.replace(
        game_state=state.game_state.at[2].set(init_dist),
    )
    obs = get_obs(state, CONFIG)
    return state, obs


def _chaser_behavior(state, slot, config):
    """Chase player within Manhattan 3, else random walk. Paced by outer % 4 gate."""
    pidx = state.player_idx
    key, subkey = jax.random.split(state.rng_key)
    state = state.replace(rng_key=key)

    # Chase if within Manhattan 3, else random walk
    dist = jnp.abs(state.x[slot] - state.x[pidx]) + jnp.abs(state.y[slot] - state.y[pidx])
    in_range = dist <= 3

    k1, k2 = jax.random.split(subkey)
    # Chase path (greedy Manhattan)
    chase_state, _ = move_toward(state, config, slot, state.x[pidx], state.y[pidx], k1)
    # Random walk
    direction = jax.random.randint(k2, (), 0, 4)
    new_x = state.x[slot] + DX[direction]
    new_y = state.y[slot] + DY[direction]
    walk_state, _ = move_entity(state, config, slot, new_x, new_y)

    # Select chase or walk
    state = jax.tree.map(
        lambda c, w: jnp.where(in_range, c, w), chase_state, walk_state
    )
    return state


def _patroller_behavior(state, slot, config):
    """Rectangular patrol: east→south→west→north, 9 steps per leg. Outer gate handles pacing."""
    direction = state.properties[slot, 0].astype(jnp.int32)  # 0=E, 1=S, 2=W, 3=N
    steps = state.properties[slot, 1].astype(jnp.int32)

    # Direction to dx/dy: E(1,0), S(0,1), W(-1,0), N(0,-1)
    pdx = jnp.array([1, 0, -1, 0], dtype=jnp.int32)
    pdy = jnp.array([0, 1, 0, -1], dtype=jnp.int32)

    new_x = state.x[slot] + pdx[direction]
    new_y = state.y[slot] + pdy[direction]

    new_state, moved = move_entity(state, config, slot, new_x, new_y)
    state = jax.tree.map(
        lambda n, o: jnp.where(moved, n, o), new_state, state
    )

    new_steps = steps + 1
    # After 9 steps or blocked, turn
    should_turn = (new_steps >= 9) | ~moved
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

    # Manhattan reward shaping: +0.01 for decreasing distance to nearest dot
    old_dist = state.game_state[2]
    new_dist = _nearest_dot_manhattan(state, CONFIG)
    has_dots = state.game_state[0] > 0
    shaping = jnp.where(has_dots & (old_dist > new_dist), 0.01, 0.0)
    state = state.replace(
        reward_acc=state.reward_acc + shaping,
        game_state=state.game_state.at[2].set(jnp.where(has_dots, new_dist, 0.0)),
    )

    # Win if all dots collected
    all_collected = state.game_state[0] <= 0
    state = state.replace(
        status=jnp.where(all_collected & (state.status == 0), jnp.int32(1), state.status)
    )

    # Phase 2: Ghost behaviors (all ghosts move every 4th turn via outer gate)
    should_move_ghosts = (state.turn_number % 4 == 0)
    def ghost_loop(i, state):
        is_ghost = state.alive[i] & ((state.entity_type[i] == 3) | (state.entity_type[i] == 4))
        type_idx = state.entity_type[i]
        branches = [lambda s, sl=i, c=CONFIG, f=fn: f(s, sl, c) for fn in BEHAVIOR_TABLE]
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

    new_state = jax.lax.fori_loop(0, CONFIG.max_behaviors, ghost_loop, state)
    state = jax.tree.map(
        lambda n, o: jnp.where(should_move_ghosts, n, o), new_state, state
    )

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
    px, py = int(state.x[pidx]), int(state.y[pidx])
    results.append(("player_in_interior", 1 <= px <= 10 and 1 <= py <= 10))

    # Player not on ghost cell
    ghost_mask = state.alive & ((state.entity_type == 3) | (state.entity_type == 4))
    no_overlap = True
    for i in range(int(state.alive.sum())):
        if ghost_mask[i]:
            if int(state.x[i]) == px and int(state.y[i]) == py:
                no_overlap = False
    results.append(("player_not_on_ghost", no_overlap))

    return results
