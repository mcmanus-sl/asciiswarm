"""Game 03: Lock & Key — find a key, unlock a door, reach the exit."""

from asciiswarm.kernel.invariants import invariant, InvariantError, check_exit_exists, Invariant

# Only check exit_exists, not exit_reachable — the door blocks BFS at setup.
# exit_reachable becomes true after the player unlocks the door.
INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'grid': (12, 12),
    'max_turns': 300,
    'step_penalty': -0.01,
    'player_properties': [
        {'key': 'has_key', 'max': 1},
    ],
}


def setup(env):
    w, h = env.config['grid']

    # Build outer walls
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    # Vertical wall divider at x=4 with one corridor opening
    corridor1_y = 1 + int(env.random() * (h - 2))
    for y in range(1, h - 1):
        if y != corridor1_y:
            env.create_entity('wall', 4, y, '#', ['solid'], z_order=1)

    # Vertical wall divider at x=8 with one corridor opening blocked by door
    corridor2_y = 1 + int(env.random() * (h - 2))
    for y in range(1, h - 1):
        if y != corridor2_y:
            env.create_entity('wall', 8, y, '#', ['solid'], z_order=1)

    # Door blocks the corridor between rooms 2 and 3
    env.create_entity('door', 8, corridor2_y, '+', ['solid'], z_order=5)

    # Player in room 1 (x=1..3, y=1..h-2)
    while True:
        px = 1 + int(env.random() * 3)
        py = 1 + int(env.random() * (h - 2))
        if not env.get_entities_at(px, py):
            break
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={'has_key': 0})

    # Key in room 2 (x=5..7, y=1..h-2)
    while True:
        kx = 5 + int(env.random() * 3)
        ky = 1 + int(env.random() * (h - 2))
        if not env.get_entities_at(kx, ky):
            break
    env.create_entity('key', kx, ky, 'k', ['pickup'], z_order=5)

    # Exit in room 3 (x=9..10, y=1..h-2)
    while True:
        ex_ = 9 + int(env.random() * 2)
        ey = 1 + int(env.random() * (h - 2))
        if not env.get_entities_at(ex_, ey):
            break
    env.create_entity('exit', ex_, ey, '>', ['exit'], z_order=5)

    # Input handler — move player
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
        elif action == 'interact':
            # Check 4 cardinal neighbors for a door
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = p.x + dx, p.y + dy
                for ent in env.get_entities_at(nx, ny):
                    if ent.has_tag('solid') and ent.type == 'door':
                        if p.properties.get('has_key', 0) == 1:
                            env.destroy_entity(ent.id)
                            p.properties['has_key'] = 0
                            env.emit('reward', {'amount': 0.2})
                        return

    env.on('input', on_input)

    # Before_move — solids block movement
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # Collision — pickup key and reach exit
    def on_collision(event):
        mover = event.payload['mover']
        if not mover.has_tag('player'):
            return
        for occ in event.payload['occupants']:
            if occ.has_tag('pickup'):
                mover.properties['has_key'] = 1
                env.destroy_entity(occ.id)
                env.emit('reward', {'amount': 0.2})
            elif occ.has_tag('exit'):
                env.end_game('won')

    env.on('collision', on_collision)


# ---- Game-specific invariants ----

@invariant('one_key_at_start')
def check_one_key(env):
    keys = env.get_entities_by_type('key')
    if len(keys) != 1:
        raise InvariantError(f"Expected 1 key, found {len(keys)}")


@invariant('one_door_at_start')
def check_one_door(env):
    doors = env.get_entities_by_type('door')
    if len(doors) != 1:
        raise InvariantError(f"Expected 1 door, found {len(doors)}")


@invariant('player_in_room1')
def check_player_room1(env):
    p = env.get_entities_by_tag('player')[0]
    if p.x > 3:
        raise InvariantError(f"Player starts at x={p.x}, expected x <= 3 (room 1)")


@invariant('key_in_room2')
def check_key_room2(env):
    k = env.get_entities_by_type('key')[0]
    if not (5 <= k.x <= 7):
        raise InvariantError(f"Key at x={k.x}, expected 5 <= x <= 7 (room 2)")


@invariant('exit_in_room3')
def check_exit_room3(env):
    e = env.get_entities_by_tag('exit')[0]
    if e.x < 9:
        raise InvariantError(f"Exit at x={e.x}, expected x >= 9 (room 3)")


@invariant('door_blocks_room2_to_room3')
def check_door_blocks(env):
    doors = env.get_entities_by_type('door')
    if not doors:
        raise InvariantError("No door found")
    d = doors[0]
    if d.x != 8:
        raise InvariantError(f"Door at x={d.x}, expected x=8")
    if not d.has_tag('solid'):
        raise InvariantError("Door is not tagged solid")


@invariant('corridor_room1_to_room2_open')
def check_corridor_open(env):
    """The corridor at x=4 has exactly one opening (no solid entity there)."""
    h = env.config['grid'][1]
    open_count = 0
    for y in range(1, h - 1):
        entities = env.get_entities_at(4, y)
        if not any(e.has_tag('solid') for e in entities):
            open_count += 1
    if open_count == 0:
        raise InvariantError("No opening in wall at x=4 between rooms 1 and 2")


@invariant('player_starts_without_key')
def check_no_key(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('has_key', 0) != 0:
        raise InvariantError("Player starts with has_key != 0")
