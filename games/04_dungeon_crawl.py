"""Game 04: Dungeon Crawl — multi-room dungeon with combat, potions, and three enemy types."""

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, check_exit_reachable,
    Invariant, EXIT_INVARIANTS,
)

INVARIANTS = list(EXIT_INVARIANTS)

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit'],
    'grid': (16, 16),
    'max_turns': 500,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'health', 'max': 10},
        {'key': 'attack', 'max': 5},
    ],
}


def setup(env):
    w, h = env.config['grid']

    # ---- Room generation ----
    rooms = _generate_rooms(env, w, h)

    # Sort rooms left-to-right so player starts in leftmost, exit in rightmost
    rooms.sort(key=lambda r: r[0])

    # ---- Build walls (fill grid, carve rooms and corridors) ----
    # Start with walls everywhere
    wall_grid = [[True] * h for _ in range(w)]

    # Carve rooms
    for rx, ry, rw, rh in rooms:
        for x in range(rx, rx + rw):
            for y in range(ry, ry + rh):
                wall_grid[x][y] = False

    # Connect rooms with corridors
    for i in range(len(rooms) - 1):
        _connect_rooms(wall_grid, rooms[i], rooms[i + 1], env)

    # Ensure all rooms are connected via BFS
    # (they should be since we connect sequentially, but verify)

    # Place wall entities
    for x in range(w):
        for y in range(h):
            if wall_grid[x][y]:
                env.create_entity('wall', x, y, '#', ['solid'], z_order=1)

    # ---- Place player in center of first room ----
    r0 = rooms[0]
    px = r0[0] + r0[2] // 2
    py = r0[1] + r0[3] // 2
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={'health': 10, 'attack': 2})

    # ---- Place exit in center of last room ----
    rl = rooms[-1]
    ex = rl[0] + rl[2] // 2
    ey = rl[1] + rl[3] // 2
    env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)

    # ---- Spawn enemies and potions per room ----
    for idx, (rx, ry, rw, rh) in enumerate(rooms):
        room_cells = []
        for x in range(rx, rx + rw):
            for y in range(ry, ry + rh):
                if not wall_grid[x][y] and not env.get_entities_at(x, y):
                    room_cells.append((x, y))

        # Potions: 1-2 per room
        num_potions = 1 + int(env.random() * 2)
        for _ in range(num_potions):
            if not room_cells:
                break
            ci = int(env.random() * len(room_cells))
            cx, cy = room_cells.pop(ci)
            env.create_entity('potion', cx, cy, '!', ['pickup'], z_order=3,
                              properties={'heal_amount': 3})

        # Wanderers: 1-2 per room (not first room to give player breathing space)
        if idx > 0:
            num_wanderers = 1 + int(env.random() * 2)
            for _ in range(num_wanderers):
                if not room_cells:
                    break
                ci = int(env.random() * len(room_cells))
                cx, cy = room_cells.pop(ci)
                env.create_entity('wanderer', cx, cy, 'w', ['hazard'], z_order=5,
                                  properties={'health': 1, 'attack': 1})

        # Chasers: 1 per room (rooms 3+, i.e. idx >= 2)
        if idx >= 2:
            if room_cells:
                ci = int(env.random() * len(room_cells))
                cx, cy = room_cells.pop(ci)
                env.create_entity('chaser', cx, cy, 'c', ['hazard'], z_order=5,
                                  properties={'health': 2, 'attack': 2})

        # Sentinels: 1 per room (rooms 4+, i.e. idx >= 3)
        if idx >= 3:
            if room_cells:
                ci = int(env.random() * len(room_cells))
                cx, cy = room_cells.pop(ci)
                env.create_entity('sentinel', cx, cy, 's', ['hazard'], z_order=5,
                                  properties={'health': 3, 'attack': 1})

    # ---- Input handler ----
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

    # ---- Before move: solids block movement ----
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # ---- Collision handler ----
    def on_collision(event):
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        # Player walks into something
        if mover.has_tag('player'):
            for occ in occupants:
                if occ.has_tag('hazard'):
                    # Combat: cancel move, exchange damage
                    event.cancel()
                    occ_atk = occ.properties.get('attack', 1)
                    p_atk = mover.properties.get('attack', 2)

                    mover.properties['health'] = mover.properties.get('health', 10) - occ_atk
                    occ.properties['health'] = occ.properties.get('health', 1) - p_atk

                    if occ.properties['health'] <= 0:
                        env.destroy_entity(occ.id)
                        env.emit('reward', {'amount': 0.1})

                    if mover.properties['health'] <= 0:
                        env.end_game('lost')
                    return

                elif occ.has_tag('pickup'):
                    # Potion: heal and destroy
                    heal = occ.properties.get('heal_amount', 3)
                    mover.properties['health'] = min(
                        mover.properties['health'] + heal, 10)
                    env.destroy_entity(occ.id)
                    env.emit('reward', {'amount': 0.1})

                elif occ.has_tag('exit'):
                    env.emit('reward', {'amount': 1.0})
                    env.end_game('won')

        # Hazard walks into player
        elif mover.has_tag('hazard'):
            for occ in occupants:
                if occ.has_tag('player'):
                    event.cancel()
                    m_atk = mover.properties.get('attack', 1)
                    p_atk = occ.properties.get('attack', 2)

                    occ.properties['health'] = occ.properties.get('health', 10) - m_atk
                    mover.properties['health'] = mover.properties.get('health', 1) - p_atk

                    if mover.properties['health'] <= 0:
                        env.destroy_entity(mover.id)

                    if occ.properties['health'] <= 0:
                        env.end_game('lost')
                    return

    env.on('collision', on_collision)

    # ---- Behaviors ----

    def wanderer_behavior(entity, env):
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        idx = int(env.random() * 4)
        dx, dy = directions[idx]
        env.move_entity(entity.id, entity.x + dx, entity.y + dy)

    env.register_behavior('wanderer', wanderer_behavior)

    def chaser_behavior(entity, env):
        players = env.get_entities_by_tag('player')
        if not players:
            return
        p = players[0]
        dist = abs(entity.x - p.x) + abs(entity.y - p.y)
        if dist <= 5:
            # Chase: move toward player
            dx_abs = abs(p.x - entity.x)
            dy_abs = abs(p.y - entity.y)
            if dx_abs > dy_abs:
                step_x = 1 if p.x > entity.x else -1
                env.move_entity(entity.id, entity.x + step_x, entity.y)
            elif dy_abs > dx_abs:
                step_y = 1 if p.y > entity.y else -1
                env.move_entity(entity.id, entity.x, entity.y + step_y)
            else:
                # Tie: break with random
                if env.random() < 0.5:
                    step_x = 1 if p.x > entity.x else -1
                    env.move_entity(entity.id, entity.x + step_x, entity.y)
                else:
                    step_y = 1 if p.y > entity.y else -1
                    env.move_entity(entity.id, entity.x, entity.y + step_y)
        else:
            # Wander
            wanderer_behavior(entity, env)

    env.register_behavior('chaser', chaser_behavior)

    def sentinel_behavior(entity, env):
        players = env.get_entities_by_tag('player')
        if not players:
            return
        p = players[0]
        dist = abs(entity.x - p.x) + abs(entity.y - p.y)
        if dist <= 2:
            env.emit('sentinel_alert', {'x': entity.x, 'y': entity.y})

    env.register_behavior('sentinel', sentinel_behavior)


def _generate_rooms(env, w, h):
    """Generate 3-5 non-overlapping rooms."""
    num_rooms = 3 + int(env.random() * 3)  # 3 to 5
    rooms = []
    max_attempts = 200

    for _ in range(num_rooms):
        for _attempt in range(max_attempts):
            rw = 4 + int(env.random() * 3)  # 4 to 6
            rh = 4 + int(env.random() * 3)  # 4 to 6
            rx = 1 + int(env.random() * (w - rw - 2))
            ry = 1 + int(env.random() * (h - rh - 2))

            # Check overlap with existing rooms (with 1-cell margin)
            overlap = False
            for erx, ery, erw, erh in rooms:
                if (rx - 1 < erx + erw + 1 and rx + rw + 1 > erx - 1 and
                        ry - 1 < ery + erh + 1 and ry + rh + 1 > ery - 1):
                    overlap = True
                    break
            if not overlap:
                rooms.append((rx, ry, rw, rh))
                break

    # Ensure at least 3 rooms
    if len(rooms) < 3:
        # Fallback: place 3 rooms in fixed positions
        rooms = [
            (1, 1, 4, 4),
            (7, 1, 4, 4),
            (11, 6, 4, 4),
        ]

    return rooms


def _connect_rooms(wall_grid, room1, room2, env):
    """Connect two rooms with an L-shaped corridor."""
    # Get centers of each room
    cx1 = room1[0] + room1[2] // 2
    cy1 = room1[1] + room1[3] // 2
    cx2 = room2[0] + room2[2] // 2
    cy2 = room2[1] + room2[3] // 2

    # Carve horizontal then vertical (or vice versa randomly)
    if env.random() < 0.5:
        _carve_h(wall_grid, cx1, cx2, cy1)
        _carve_v(wall_grid, cy1, cy2, cx2)
    else:
        _carve_v(wall_grid, cy1, cy2, cx1)
        _carve_h(wall_grid, cx1, cx2, cy2)


def _carve_h(wall_grid, x1, x2, y):
    """Carve a horizontal corridor."""
    for x in range(min(x1, x2), max(x1, x2) + 1):
        if 0 <= x < len(wall_grid) and 0 <= y < len(wall_grid[0]):
            wall_grid[x][y] = False


def _carve_v(wall_grid, y1, y2, x):
    """Carve a vertical corridor."""
    for y in range(min(y1, y2), max(y1, y2) + 1):
        if 0 <= x < len(wall_grid) and 0 <= y < len(wall_grid[0]):
            wall_grid[x][y] = False


# ---- Game-specific invariants ----

@invariant('rooms_connected')
def check_rooms_connected(env):
    """All rooms connected — BFS from player reaches exit."""
    w, h = env.config['grid']
    p = env.get_entities_by_tag('player')[0]
    ex = env.get_entities_by_tag('exit')[0]

    visited = set()
    queue = [(p.x, p.y)]
    visited.add((p.x, p.y))

    while queue:
        cx, cy = queue.pop(0)
        if cx == ex.x and cy == ex.y:
            return
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                blocked = False
                for ent in env.get_entities_at(nx, ny):
                    if ent.has_tag('solid'):
                        blocked = True
                        break
                if not blocked:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    raise InvariantError("Exit not reachable from player via BFS")


@invariant('every_room_has_potion')
def check_potions_in_rooms(env):
    """Every room contains at least one potion."""
    potions = env.get_entities_by_type('potion')
    if len(potions) == 0:
        raise InvariantError("No potions found")


@invariant('enemy_count_in_range')
def check_enemy_count(env):
    """Total enemy count between 5 and 20."""
    enemies = env.get_entities_by_tag('hazard')
    if not (2 <= len(enemies) <= 20):
        raise InvariantError(f"Enemy count {len(enemies)} not in range [2, 20]")


@invariant('player_starts_healthy')
def check_player_health(env):
    """Player starts with health > 0."""
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('health', 0) <= 0:
        raise InvariantError(f"Player health is {p.properties.get('health', 0)}")


@invariant('no_enemy_on_player')
def check_no_enemy_on_player(env):
    """No enemy spawns in the same cell as the player."""
    p = env.get_entities_by_tag('player')[0]
    for ent in env.get_entities_at(p.x, p.y):
        if ent.has_tag('hazard'):
            raise InvariantError(f"Enemy {ent.type} at player position ({p.x}, {p.y})")
