"""Game 08: Block Push — push 2 blocks onto 2 targets (simplified Sokoban)."""

from collections import deque
from asciiswarm.kernel.invariants import invariant, InvariantError

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc', 'pushable', 'target'],
    'grid': (8, 8),
    'max_turns': 300,
    'step_penalty': -0.005,
    'player_properties': [],
}

# Known-good fallback layout (guaranteed solvable)
_FALLBACK = {
    'player': (3, 5),
    'blocks': [(3, 3), (4, 4)],
    'targets': [(3, 1), (4, 1)],
}

DIRS = [(0, -1), (0, 1), (1, 0), (-1, 0)]


def _is_solvable(walls, player, blocks, targets, max_moves=50):
    """BFS over (player, block1, block2) state space to check solvability."""
    target_set = frozenset(targets)

    # Normalize block positions as sorted tuple for canonical state
    def state(p, bs):
        return (p, tuple(sorted(bs)))

    start = state(player, blocks)
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (pp, bbs), depth = queue.popleft()
        if depth >= max_moves:
            continue

        for dx, dy in DIRS:
            nx, ny = pp[0] + dx, pp[1] + dy
            if (nx, ny) in walls:
                continue
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue

            new_blocks = list(bbs)
            push_blocked = False

            # Check if player pushes a block
            for i, (bx, by) in enumerate(new_blocks):
                if (nx, ny) == (bx, by):
                    # Push this block
                    nbx, nby = bx + dx, by + dy
                    if (nbx, nby) in walls or not (0 <= nbx < 8 and 0 <= nby < 8):
                        push_blocked = True
                        break
                    # Check if another block is in the way
                    for j, (obx, oby) in enumerate(new_blocks):
                        if j != i and (obx, oby) == (nbx, nby):
                            push_blocked = True
                            break
                    if push_blocked:
                        break
                    new_blocks[i] = (nbx, nby)
                    break

            if push_blocked:
                continue

            new_state = state((nx, ny), new_blocks)
            if new_state in visited:
                continue
            visited.add(new_state)

            # Check win
            if frozenset(new_blocks) == target_set:
                return True

            queue.append((new_state, depth + 1))

    return False


def setup(env):
    w, h = env.config['grid']

    # Build border walls
    wall_cells = set()
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
        wall_cells.add((x, 0))
        wall_cells.add((x, h - 1))
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)
        wall_cells.add((0, y))
        wall_cells.add((w - 1, y))

    # Generate solvable layout
    player_pos = None
    block_positions = None
    target_positions = None
    solved = False

    for attempt in range(100):
        # Random player in bottom half (y >= 4), interior cells
        px = 1 + int(env.random() * 6)
        py = 4 + int(env.random() * 3)  # y in 4,5,6
        pp = (px, py)

        # 2 blocks in center (2 <= x <= 5, 2 <= y <= 5)
        blocks = []
        for _ in range(2):
            for _try in range(20):
                bx = 2 + int(env.random() * 4)
                by = 2 + int(env.random() * 4)
                bp = (bx, by)
                if bp != pp and bp not in blocks and bp not in wall_cells:
                    blocks.append(bp)
                    break

        if len(blocks) != 2:
            continue

        # 2 targets in top half (y <= 3), interior cells
        targets = []
        for _ in range(2):
            for _try in range(20):
                tx = 1 + int(env.random() * 6)
                ty = 1 + int(env.random() * 3)  # y in 1,2,3
                tp = (tx, ty)
                if tp not in targets and tp not in blocks and tp != pp and tp not in wall_cells:
                    targets.append(tp)
                    break

        if len(targets) != 2:
            continue

        # Blocks must not start on targets
        if set(blocks) & set(targets):
            continue

        if _is_solvable(wall_cells, pp, blocks, targets, max_moves=50):
            player_pos = pp
            block_positions = blocks
            target_positions = targets
            solved = True
            break

    if not solved:
        player_pos = _FALLBACK['player']
        block_positions = list(_FALLBACK['blocks'])
        target_positions = list(_FALLBACK['targets'])

    # Create entities
    # Targets first (lower z-order, blocks go on top)
    for tx, ty in target_positions:
        env.create_entity('target', tx, ty, 'X', ['target'], z_order=3)

    for bx, by in block_positions:
        env.create_entity('block', bx, by, 'B', ['solid', 'pushable'], z_order=5)

    player = env.create_entity('player', player_pos[0], player_pos[1], '@', ['player'], z_order=10)

    # Track previous block-to-target distance for reward shaping
    prev_dist = [None]

    def _total_distance():
        """Sum of Manhattan distances from each block to nearest unoccupied target."""
        blks = env.get_entities_by_type('block')
        tgts = env.get_entities_by_type('target')
        if not blks or not tgts:
            return 0
        total = 0
        for b in blks:
            min_d = min(abs(b.x - t.x) + abs(b.y - t.y) for t in tgts)
            total += min_d
        return total

    prev_dist[0] = _total_distance()

    # --- Input handler ---
    def on_input(event):
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]
        action = event.payload['action']
        moves = {
            'move_n': (0, -1),
            'move_s': (0, 1),
            'move_e': (1, 0),
            'move_w': (-1, 0),
        }
        if action in moves:
            dx, dy = moves[action]
            env.move_entity(p.id, p.x + dx, p.y + dy)

    env.on('input', on_input)

    # --- Before move: solids block movement, but allow pushable ---
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        mover = event.payload['entity']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                if ent.has_tag('pushable') and mover.has_tag('player'):
                    # Let collision handler deal with the push
                    return
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # --- Collision handler: push blocks ---
    def on_collision(event):
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        if not mover.has_tag('player'):
            return

        for occ in occupants:
            if occ.has_tag('pushable'):
                # At collision time, mover is still at its original position.
                # The block (occ) is at the destination cell.
                # Push direction = destination - source = occ.pos - mover.pos
                dx = occ.x - mover.x
                dy = occ.y - mover.y

                # Try to push the block
                push_result = env.move_entity(occ.id, occ.x + dx, occ.y + dy)
                if not push_result:
                    # Block can't be pushed, cancel player move
                    event.cancel()
                    return

                # Block pushed successfully - emit distance-based reward
                new_dist = _total_distance()
                delta = prev_dist[0] - new_dist
                if delta > 0:
                    env.emit('reward', {'amount': 0.1 * delta})
                prev_dist[0] = new_dist

                # Check win: all targets have a block on them
                targets = env.get_entities_by_type('target')
                all_covered = True
                for t in targets:
                    has_block = False
                    for e in env.get_entities_at(t.x, t.y):
                        if e.has_tag('pushable'):
                            has_block = True
                            break
                    if not has_block:
                        all_covered = False
                        break

                if all_covered:
                    env.emit('reward', {'amount': 1.0})
                    env.end_game('won')
                return

    env.on('collision', on_collision)


# ---- Game-specific invariants ----

@invariant('two_blocks')
def check_two_blocks(env):
    blocks = env.get_entities_by_type('block')
    if len(blocks) != 2:
        raise InvariantError(f"Expected 2 blocks, found {len(blocks)}")


@invariant('two_targets')
def check_two_targets(env):
    targets = env.get_entities_by_type('target')
    if len(targets) != 2:
        raise InvariantError(f"Expected 2 targets, found {len(targets)}")


@invariant('player_in_bottom_half')
def check_player_bottom(env):
    p = env.get_entities_by_tag('player')[0]
    if p.y < 4:
        raise InvariantError(f"Player at y={p.y}, expected y >= 4")


@invariant('blocks_in_center')
def check_blocks_center(env):
    for b in env.get_entities_by_type('block'):
        if not (2 <= b.x <= 5 and 2 <= b.y <= 5):
            raise InvariantError(f"Block at ({b.x},{b.y}), expected 2<=x<=5, 2<=y<=5")


@invariant('targets_in_top_half')
def check_targets_top(env):
    for t in env.get_entities_by_type('target'):
        if t.y > 3:
            raise InvariantError(f"Target at y={t.y}, expected y <= 3")


@invariant('no_block_on_target')
def check_no_presolved(env):
    targets = {(t.x, t.y) for t in env.get_entities_by_type('target')}
    for b in env.get_entities_by_type('block'):
        if (b.x, b.y) in targets:
            raise InvariantError(f"Block at ({b.x},{b.y}) starts on a target")


@invariant('no_duplicate_blocks')
def check_unique_blocks(env):
    blocks = env.get_entities_by_type('block')
    positions = [(b.x, b.y) for b in blocks]
    if len(set(positions)) != len(positions):
        raise InvariantError("Two blocks share the same cell")


@invariant('no_duplicate_targets')
def check_unique_targets(env):
    targets = env.get_entities_by_type('target')
    positions = [(t.x, t.y) for t in targets]
    if len(set(positions)) != len(positions):
        raise InvariantError("Two targets share the same cell")


@invariant('puzzle_solvable')
def check_solvable(env):
    w, h = env.config['grid']
    walls = set()
    for e in env.get_entities_by_tag('solid'):
        if not e.has_tag('pushable'):
            walls.add((e.x, e.y))
    p = env.get_entities_by_tag('player')[0]
    blocks = [(b.x, b.y) for b in env.get_entities_by_type('block')]
    targets = [(t.x, t.y) for t in env.get_entities_by_type('target')]
    if not _is_solvable(walls, (p.x, p.y), blocks, targets, max_moves=50):
        raise InvariantError("Puzzle is not solvable within 50 moves")
