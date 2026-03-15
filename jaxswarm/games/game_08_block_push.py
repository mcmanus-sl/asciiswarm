"""Game 08: Block Push — push 2 blocks onto 2 target tiles."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, EnvState, init_state
from jaxswarm.core.grid_ops import rebuild_grid
from jaxswarm.core.obs import get_obs
from jaxswarm.systems.movement import DX, DY

CONFIG = EnvConfig(
    grid_w=8, grid_h=8,
    max_entities=10,       # player + 2 blocks + 2 targets + margin
    max_stack=3,           # block can stack on target
    num_entity_types=5,    # 0=unused, 1=player, 2=block, 3=target, 4=wall
    num_tags=8,            # +pushable(6), +target(7)
    num_props=1,
    num_actions=6,
    max_turns=300,
    step_penalty=-0.005,
    game_state_size=4,     # 0=blocks_on_target, 1=total_pushes, 2=prev_block_dist_sum, 3=unused
    prop_maxes=(1.0,),
    max_behaviors=2,       # player only (no NPCs)
)

ACTION_NAMES = ["move_n", "move_s", "move_e", "move_w", "interact", "wait"]

# Deterministic trace for seed 0: Player(4,6), Blocks(3,2)(3,4), Targets(2,1)(3,1).
# Push block(3,2) west to target(2,1), then push block(3,4) north to target(3,1).
_trace08 = (
    [3]*3 + [0]*4 +     # (4,6)->(1,6)->(1,2), west of block(3,2)
    [2] +                # push block(3,2)->(2,2)? No, need to push it to (2,1).
    # Correction: go to (4,2) east of block, push W. Then push N from (2,3).
    []
)
# N=0,S=1,E=2,W=3
_trace08 = (
    [0]*4 +              # (4,6)->(4,2)
    [3] +                # push block(3,2)->(2,2), player->(3,2)
    [1] + [3] +          # (3,2)->(3,3)->(2,3)
    [0] +                # push block(2,2)->(2,1)=TARGET, player->(2,2)
    [1]*3 + [2] +        # (2,2)->(2,5)->(3,5)
    [0]*3                # push block(3,4)->(3,3)->(3,2)->(3,1)=TARGET, WIN
)
DETERMINISTIC_TRACE = _trace08[:295]


def reset(rng_key: jax.Array) -> tuple[EnvState, dict]:
    state = init_state(CONFIG, rng_key)
    keys = jax.random.split(rng_key, 12)

    # --- Generate random positions using SAME key splits as before ---

    # Player in bottom half
    k_px, k_py = jax.random.split(keys[0])
    player_x = jax.random.randint(k_px, (), 1, 7)
    player_y = jax.random.randint(k_py, (), 4, 7)

    # Block 1 in center area
    k_b1x, k_b1y = jax.random.split(keys[1])
    b1x = jax.random.randint(k_b1x, (), 2, 6)
    b1y = jax.random.randint(k_b1y, (), 2, 6)

    # Block 2 in center area (avoid same cell as block 1)
    k_b2x, k_b2y = jax.random.split(keys[2])
    b2x = jax.random.randint(k_b2x, (), 2, 6)
    b2y = jax.random.randint(k_b2y, (), 2, 6)
    same = (b2x == b1x) & (b2y == b1y)
    b2x = jnp.where(same, jnp.clip(b2x + 1, 2, 5), b2x)

    # Target 1
    k_t1x, k_t1y = jax.random.split(keys[3])
    t1x = jax.random.randint(k_t1x, (), 1, 7)
    t1y = jnp.int32(1)

    # Target 2 (avoid same cell as target 1)
    k_t2x, k_t2y = jax.random.split(keys[4])
    t2x = jax.random.randint(k_t2x, (), 1, 7)
    same_t = (t2x == t1x) & (jnp.int32(1) == t1y)
    t2x = jnp.where(same_t, jnp.clip(t2x + 1, 1, 6), t2x)
    t2y = jnp.int32(1)

    # --- Vectorized bulk initialization ---
    # Slot layout: 0=player, 1=block1, 2=block2, 3=target1, 4=target2
    N = 5

    alive = state.alive.at[:N].set(True)
    entity_type = state.entity_type.at[0].set(1).at[1].set(2).at[2].set(2).at[3].set(3).at[4].set(3)
    x = state.x.at[0].set(player_x).at[1].set(b1x).at[2].set(b2x).at[3].set(t1x).at[4].set(t2x)
    y = state.y.at[0].set(player_y).at[1].set(b1y).at[2].set(b2y).at[3].set(t1y).at[4].set(t2y)

    # Tags: player=[tag0], blocks=[tag1,tag6], targets=[tag7]
    tags = state.tags
    tags = tags.at[0, 0].set(True)           # player: tag 0
    tags = tags.at[1, 1].set(True)           # block1: tag 1 (solid)
    tags = tags.at[1, 6].set(True)           # block1: tag 6 (pushable)
    tags = tags.at[2, 1].set(True)           # block2: tag 1 (solid)
    tags = tags.at[2, 6].set(True)           # block2: tag 6 (pushable)
    tags = tags.at[3, 7].set(True)           # target1: tag 7 (target)
    tags = tags.at[4, 7].set(True)           # target2: tag 7 (target)

    state = state.replace(
        alive=alive,
        entity_type=entity_type,
        x=x,
        y=y,
        tags=tags,
        player_idx=jnp.int32(0),
    )

    # Compute initial block-to-target distance sum for reward shaping
    dist_sum = _block_target_dist_sum(state)
    state = state.replace(
        game_state=state.game_state.at[2].set(dist_sum),
        rng_key=keys[5],
    )
    state = rebuild_grid(state, CONFIG)
    obs = get_obs(state, CONFIG)
    return state, obs


def _block_target_dist_sum(state):
    """Sum of Manhattan distances from each block to nearest target."""
    # Find blocks and targets
    is_block = state.alive & (state.entity_type == 2)
    is_target = state.alive & (state.entity_type == 3)

    # For each block, find min distance to any target
    # Simple: sum of (block_i to nearest target)
    block_xs = jnp.where(is_block, state.x, 0)
    block_ys = jnp.where(is_block, state.y, 0)
    target_xs = jnp.where(is_target, state.x, 0)
    target_ys = jnp.where(is_target, state.y, 0)

    # Compute pairwise distances (all entities x all entities)
    dx = jnp.abs(state.x[:, None] - state.x[None, :])
    dy = jnp.abs(state.y[:, None] - state.y[None, :])
    dists = dx + dy  # [max_entities, max_entities]

    # For each block, min distance to any target
    # Mask: block_i to target_j
    block_mask = is_block[:, None]  # [max_entities, 1]
    target_mask = is_target[None, :]  # [1, max_entities]
    valid = block_mask & target_mask

    masked_dists = jnp.where(valid, dists, 999)
    min_per_block = jnp.min(masked_dists, axis=1)  # [max_entities]
    total = jnp.where(is_block, min_per_block, 0).sum()
    return total.astype(jnp.float32)


def _count_blocks_on_targets(state):
    """Count how many target cells have a block on them."""
    is_target = state.alive & (state.entity_type == 3)
    is_block = state.alive & (state.entity_type == 2)

    # For each target, check if any block shares its position
    tx = jnp.where(is_target, state.x, -1)
    ty = jnp.where(is_target, state.y, -1)
    bx = jnp.where(is_block, state.x, -2)
    by = jnp.where(is_block, state.y, -2)

    # target_i matched if any block_j has same (x,y)
    match = (tx[:, None] == bx[None, :]) & (ty[:, None] == by[None, :])
    target_has_block = match.any(axis=1) & is_target
    return target_has_block.sum().astype(jnp.int32)


def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    state = state.replace(reward_acc=jnp.float32(0.0))
    state = state.replace(turn_number=state.turn_number + 1)

    pidx = state.player_idx
    px, py = state.x[pidx], state.y[pidx]
    is_move = action < 4

    dx = DX[action]
    dy = DY[action]
    target_x = px + dx
    target_y = py + dy
    safe_tx = jnp.clip(target_x, 0, CONFIG.grid_w - 1)
    safe_ty = jnp.clip(target_y, 0, CONFIG.grid_h - 1)
    in_bounds = (target_x >= 0) & (target_x < CONFIG.grid_w) & (target_y >= 0) & (target_y < CONFIG.grid_h)

    # --- Vectorized target cell classification (no get_entities_at / fori_loop) ---
    at_target = state.alive & (state.x == safe_tx) & (state.y == safe_ty)
    is_solid_at_target = at_target & state.tags[:, 1]
    is_pushable_at_target = at_target & state.tags[:, 6]
    has_wall = (is_solid_at_target & ~is_pushable_at_target).any() & is_move & in_bounds
    has_pushable = is_pushable_at_target.any() & is_move & in_bounds
    push_slot = jnp.argmax(is_pushable_at_target)  # 0 if none found, guarded by has_pushable

    # --- Push destination check (vectorized) ---
    push_dest_x = jnp.clip(target_x + dx, 0, CONFIG.grid_w - 1)
    push_dest_y = jnp.clip(target_y + dy, 0, CONFIG.grid_h - 1)
    push_in_bounds = ((target_x + dx) >= 0) & ((target_x + dx) < CONFIG.grid_w) & \
                     ((target_y + dy) >= 0) & ((target_y + dy) < CONFIG.grid_h)

    at_dest = state.alive & (state.x == push_dest_x) & (state.y == push_dest_y)
    dest_blocked = (at_dest & state.tags[:, 1]).any()

    can_push = has_pushable & push_in_bounds & ~dest_blocked & is_move & in_bounds

    # --- Execute push: update block x,y directly (no move_entity) ---
    new_x = state.x.at[push_slot].set(
        jnp.where(can_push, push_dest_x, state.x[push_slot])
    )
    new_y = state.y.at[push_slot].set(
        jnp.where(can_push, push_dest_y, state.y[push_slot])
    )
    state = state.replace(x=new_x, y=new_y)

    # Increment push count
    state = state.replace(
        game_state=jnp.where(can_push, state.game_state.at[1].set(state.game_state[1] + 1), state.game_state)
    )

    # --- Move player directly (no move_entity) ---
    can_move_normal = is_move & in_bounds & ~has_wall & ~has_pushable
    can_move = can_move_normal | can_push

    new_px = state.x.at[pidx].set(
        jnp.where(can_move, safe_tx, state.x[pidx])
    )
    new_py = state.y.at[pidx].set(
        jnp.where(can_move, safe_ty, state.y[pidx])
    )
    state = state.replace(x=new_px, y=new_py)

    # --- Check win: blocks on targets ---
    blocks_on = _count_blocks_on_targets(state)
    state = state.replace(
        game_state=state.game_state.at[0].set(blocks_on.astype(jnp.float32)),
        status=jnp.where((blocks_on >= 2) & (state.status == 0), jnp.int32(1), state.status),
    )

    # --- Reward shaping: distance decrease ---
    old_dist = state.game_state[2]
    new_dist = _block_target_dist_sum(state)
    shaping = (old_dist - new_dist) * 0.02
    state = state.replace(
        reward_acc=state.reward_acc + shaping,
        game_state=state.game_state.at[2].set(new_dist),
    )

    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)
    return state, obs, reward, done


def invariant_checks(state: EnvState) -> list[tuple[str, bool]]:
    results = []
    block_count = int((state.alive & (state.entity_type == 2)).sum())
    results.append(("two_blocks", block_count == 2))

    target_count = int((state.alive & (state.entity_type == 3)).sum())
    results.append(("two_targets", target_count == 2))

    pidx = int(state.player_idx)
    py = int(state.y[pidx])
    results.append(("player_bottom_half", py >= 4))

    # Blocks not on targets initially
    on_target = int(_count_blocks_on_targets(state))
    results.append(("not_pre_solved", on_target == 0))

    return results
