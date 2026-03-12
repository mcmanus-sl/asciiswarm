"""Game 05: Pac-Man Collect — collect all dots while avoiding ghosts."""

from asciiswarm.kernel.invariants import invariant, InvariantError

GAME_CONFIG = {
    'grid': (12, 12),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [],
}


def setup(env):
    w, h = env.config['grid']

    # Border walls
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    # Interior walls: cross pattern
    # Horizontal wall: y=5, x=3..8 with gaps at x=5,6
    for x in range(3, 9):
        if x not in (5, 6):
            env.create_entity('wall', x, 5, '#', ['solid'], z_order=1)

    # Vertical wall: x=5, y=3..8 with gaps at y=5,6
    for y in range(3, 9):
        if y not in (5, 6):
            env.create_entity('wall', 5, y, '#', ['solid'], z_order=1)

    # Track occupied cells
    wall_cells = set()
    for ent in env.get_entities_by_tag('solid'):
        wall_cells.add((ent.x, ent.y))

    # Player at center (6, 6)
    player = env.create_entity('player', 6, 6, '@', ['player'], z_order=10)

    # Chaser ghost at (1, 1)
    chaser = env.create_entity('chaser', 1, 1, 'C', ['hazard'], z_order=5)

    # Patroller ghost at (10, 1)
    patroller = env.create_entity('patroller', 10, 1, 'P', ['hazard'], z_order=5,
                                   properties={'patrol_direction': 0, 'patrol_steps': 0})

    # Occupied cells (no dots here)
    occupied = wall_cells | {(6, 6), (1, 1), (10, 1)}

    # Place dots on all remaining empty interior cells
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if (x, y) not in occupied:
                env.create_entity('dot', x, y, '.', ['pickup'], z_order=3)

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

    # --- Before move: solids block movement ---
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # --- Collision handler ---
    def on_collision(event):
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        # Player walks into pickup (dot)
        if mover.has_tag('player'):
            for occ in occupants:
                if occ.has_tag('pickup'):
                    env.destroy_entity(occ.id)
                    env.emit('reward', {'amount': 0.05})
                    # Check win: no more dots
                    if not env.get_entities_by_tag('pickup'):
                        env.end_game('won')
                    return

        # Player walks into hazard
        if mover.has_tag('player'):
            for occ in occupants:
                if occ.has_tag('hazard'):
                    env.end_game('lost')
                    return

        # Hazard walks into player
        if mover.has_tag('hazard'):
            for occ in occupants:
                if occ.has_tag('player'):
                    env.end_game('lost')
                    return

    env.on('collision', on_collision)

    # --- Chaser behavior ---
    def chaser_behavior(entity, env):
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]

        dx = p.x - entity.x
        dy = p.y - entity.y

        # Prefer axis with greater distance, break ties by preferring horizontal
        if abs(dx) >= abs(dy):
            # Try horizontal first
            step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
            step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
            if step_x != 0:
                if not env.move_entity(entity.id, entity.x + step_x, entity.y):
                    if step_y != 0:
                        env.move_entity(entity.id, entity.x, entity.y + step_y)
            elif step_y != 0:
                env.move_entity(entity.id, entity.x, entity.y + step_y)
        else:
            # Try vertical first
            step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
            step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
            if step_y != 0:
                if not env.move_entity(entity.id, entity.x, entity.y + step_y):
                    if step_x != 0:
                        env.move_entity(entity.id, entity.x + step_x, entity.y)
            elif step_x != 0:
                env.move_entity(entity.id, entity.x + step_x, entity.y)

    env.register_behavior('chaser', chaser_behavior)

    # --- Patroller behavior ---
    # Patrol path: east along y=1, south along x=10, west along y=10, north along x=1
    PATROL_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # east, south, west, north

    def patroller_behavior(entity, env):
        direction = entity.get('patrol_direction', 0)
        steps = entity.get('patrol_steps', 0)

        dx, dy = PATROL_DIRS[direction]
        moved = env.move_entity(entity.id, entity.x + dx, entity.y + dy)

        if not moved:
            # Blocked, advance to next direction
            direction = (direction + 1) % 4
            entity.set('patrol_direction', direction)
            entity.set('patrol_steps', 0)
        else:
            steps += 1
            if steps >= 9:
                direction = (direction + 1) % 4
                steps = 0
            entity.set('patrol_direction', direction)
            entity.set('patrol_steps', steps)

    env.register_behavior('patroller', patroller_behavior)


# ---- Game-specific invariants ----

@invariant('one_chaser_one_patroller')
def check_ghosts(env):
    chasers = env.get_entities_by_type('chaser')
    patrollers = env.get_entities_by_type('patroller')
    if len(chasers) != 1:
        raise InvariantError(f"Expected 1 chaser, found {len(chasers)}")
    if len(patrollers) != 1:
        raise InvariantError(f"Expected 1 patroller, found {len(patrollers)}")


@invariant('enough_dots')
def check_dots(env):
    dots = env.get_entities_by_tag('pickup')
    if len(dots) < 20:
        raise InvariantError(f"Expected at least 20 dots, found {len(dots)}")


@invariant('player_at_center')
def check_player_center(env):
    p = env.get_entities_by_tag('player')[0]
    if (p.x, p.y) != (6, 6):
        raise InvariantError(f"Player at ({p.x},{p.y}), expected (6,6)")


@invariant('no_dot_on_ghost_or_wall')
def check_dot_placement(env):
    wall_cells = {(e.x, e.y) for e in env.get_entities_by_tag('solid')}
    ghost_cells = {(e.x, e.y) for e in env.get_entities_by_tag('hazard')}
    for dot in env.get_entities_by_tag('pickup'):
        if (dot.x, dot.y) in wall_cells:
            raise InvariantError(f"Dot at ({dot.x},{dot.y}) is on a wall")
        if (dot.x, dot.y) in ghost_cells:
            raise InvariantError(f"Dot at ({dot.x},{dot.y}) is on a ghost")


@invariant('player_not_on_ghost')
def check_player_not_on_ghost(env):
    p = env.get_entities_by_tag('player')[0]
    for g in env.get_entities_by_tag('hazard'):
        if (p.x, p.y) == (g.x, g.y):
            raise InvariantError(f"Player starts on ghost at ({p.x},{p.y})")


@invariant('all_cells_reachable')
def check_reachable(env):
    """All non-wall cells reachable from player via BFS."""
    w, h = env.config['grid']
    wall_cells = {(e.x, e.y) for e in env.get_entities_by_tag('solid')}
    p = env.get_entities_by_tag('player')[0]

    visited = set()
    queue = [(p.x, p.y)]
    visited.add((p.x, p.y))
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in wall_cells and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    # Check all non-wall interior cells are reachable
    for x in range(w):
        for y in range(h):
            if (x, y) not in wall_cells and (x, y) not in visited:
                raise InvariantError(f"Cell ({x},{y}) is not reachable from player")
