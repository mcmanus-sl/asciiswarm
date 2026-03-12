"""Game 06: Ice Sliding — slide in a direction until hitting a wall or rock."""

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, Invariant, EXIT_INVARIANTS,
)

INVARIANTS = list(EXIT_INVARIANTS)

GAME_CONFIG = {
    'grid': (10, 10),
    'max_turns': 200,
    'step_penalty': -0.01,
    'player_properties': [],
}


def _ice_bfs_reachable(env, start_x, start_y, target_x, target_y):
    """BFS over ice-sliding states to check if target is reachable."""
    w, h = env.config['grid']

    def is_solid(x, y):
        for e in env.get_entities_at(x, y):
            if e.has_tag('solid'):
                return True
        return False

    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    visited = set()
    visited.add((start_x, start_y))
    queue = [(start_x, start_y)]

    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in directions:
            # Simulate sliding
            nx, ny = cx, cy
            while True:
                nnx, nny = nx + dx, ny + dy
                if nnx < 0 or nnx >= w or nny < 0 or nny >= h:
                    break
                if is_solid(nnx, nny):
                    break
                nx, ny = nnx, nny
                # Check if we pass through or land on the target
                if nx == target_x and ny == target_y:
                    return True
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def _hardcoded_layout():
    """Known-good fallback rock positions for a 10x10 grid.
    Player at (1, 8), exit at (8, 1).
    Rocks placed to create a solvable sliding puzzle."""
    return [
        (3, 8), (1, 5), (5, 5), (8, 5),
        (3, 3), (6, 3), (5, 1), (8, 3),
    ]


def setup(env):
    w, h = env.config['grid']

    # Player in bottom-left area (x in 0-2, y in 7-9)
    px = int(env.random() * 3)
    py = 7 + int(env.random() * 3)

    # Exit in top-right area (x in 7-9, y in 0-2)
    ex = 7 + int(env.random() * 3)
    ey = int(env.random() * 3)

    player = env.create_entity('player', px, py, '@', ['player'], z_order=10)
    exit_ent = env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)

    # Place rocks with solvability check
    occupied = {(px, py), (ex, ey)}
    rocks_placed = False

    for attempt in range(100):
        num_rocks = 8 + int(env.random() * 5)  # 8-12
        rock_positions = []
        positions_set = set(occupied)
        valid = True

        for _ in range(num_rocks):
            placed = False
            for _try in range(50):
                rx = int(env.random() * w)
                ry = int(env.random() * h)
                if (rx, ry) not in positions_set:
                    rock_positions.append((rx, ry))
                    positions_set.add((rx, ry))
                    placed = True
                    break
            if not placed:
                # consume random to stay deterministic, then skip
                valid = False
                break

        if not valid:
            continue

        # Temporarily create rocks to test solvability
        rock_ids = []
        for rx, ry in rock_positions:
            r = env.create_entity('rock', rx, ry, 'O', ['solid'], z_order=5)
            rock_ids.append(r.id)

        solvable = _ice_bfs_reachable(env, px, py, ex, ey)

        if solvable:
            rocks_placed = True
            break
        else:
            # Remove rocks and try again
            for rid in rock_ids:
                env.destroy_entity(rid)

    if not rocks_placed:
        # Fallback to hardcoded layout
        # Reset player and exit to known positions
        env.move_entity(player.id, 1, 8)
        env.move_entity(exit_ent.id, 8, 1)

        for rx, ry in _hardcoded_layout():
            env.create_entity('rock', rx, ry, 'O', ['solid'], z_order=5)

    # Input handler — ice sliding
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
        if action not in moves:
            return

        dx, dy = moves[action]
        # Slide until blocked
        while True:
            moved = env.move_entity(p.id, p.x + dx, p.y + dy)
            if not moved:
                break
            # Check if game ended (collision with exit)
            if env.status != 'playing':
                break

    env.on('input', on_input)

    # Before_move — solids block movement
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # Collision — player slides onto exit
    def on_collision(event):
        mover = event.payload['mover']
        if not mover.has_tag('player'):
            return
        for occ in event.payload['occupants']:
            if occ.has_tag('exit'):
                env.end_game('won')

    env.on('collision', on_collision)


# ---- Game-specific invariants ----

@invariant('rocks_count_8_to_12')
def check_rocks_count(env):
    rocks = env.get_entities_by_type('rock')
    if not (8 <= len(rocks) <= 12):
        raise InvariantError(f"Expected 8-12 rocks, found {len(rocks)}")


@invariant('player_in_bottom_left')
def check_player_position(env):
    p = env.get_entities_by_tag('player')[0]
    if p.x > 2 or p.y < 7:
        raise InvariantError(
            f"Player at ({p.x}, {p.y}), expected x<=2 and y>=7"
        )


@invariant('exit_in_top_right')
def check_exit_position(env):
    e = env.get_entities_by_tag('exit')[0]
    if e.x < 7 or e.y > 2:
        raise InvariantError(
            f"Exit at ({e.x}, {e.y}), expected x>=7 and y<=2"
        )


@invariant('no_rock_on_player_or_exit')
def check_no_rock_overlap(env):
    p = env.get_entities_by_tag('player')[0]
    e = env.get_entities_by_tag('exit')[0]
    for rock in env.get_entities_by_type('rock'):
        if (rock.x, rock.y) == (p.x, p.y):
            raise InvariantError(f"Rock at player position ({p.x}, {p.y})")
        if (rock.x, rock.y) == (e.x, e.y):
            raise InvariantError(f"Rock at exit position ({e.x}, {e.y})")


@invariant('exit_reachable_by_ice_sliding')
def check_ice_reachable(env):
    p = env.get_entities_by_tag('player')[0]
    e = env.get_entities_by_tag('exit')[0]
    if not _ice_bfs_reachable(env, p.x, p.y, e.x, e.y):
        raise InvariantError("Exit not reachable from player via ice sliding")
