"""Game 09: Inventory & Crafting — collect wood/ore, craft pickaxe, mine rubble, reach exit."""

from asciiswarm.kernel.invariants import invariant, InvariantError, check_exit_exists, Invariant

# Exit is blocked by rubble at setup, so only check exit_exists (not reachable).
INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'tags': ['player', 'solid', 'pickup', 'exit', 'npc'],
    'grid': (16, 16),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'wood', 'max': 5},
        {'key': 'ore', 'max': 5},
        {'key': 'has_pickaxe', 'max': 1},
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

    # Vertical wall at x=12, floor-to-ceiling, with one gap for rubble
    rubble_y = 1 + int(env.random() * (h - 2))
    for y in range(1, h - 1):
        if y != rubble_y:
            env.create_entity('wall', 12, y, '#', ['solid'], z_order=1)

    # Place rubble entity at the gap
    env.create_entity('rubble', 12, rubble_y, '%', ['solid'], z_order=5)

    def _find_empty(x_min, x_max, y_min, y_max):
        """Find a random empty cell in the given range."""
        while True:
            x = x_min + int(env.random() * (x_max - x_min + 1))
            y = y_min + int(env.random() * (y_max - y_min + 1))
            if not env.get_entities_at(x, y):
                return x, y

    # Player in left third (x < 5)
    px, py = _find_empty(1, 4, 1, h - 2)
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={'wood': 0, 'ore': 0, 'has_pickaxe': 0})

    # Workbench in center area (6 <= x <= 9, 6 <= y <= 9)
    wx, wy = _find_empty(6, 9, 6, 9)
    env.create_entity('workbench', wx, wy, 'W', ['npc'], z_order=5)

    # Wood: 4-6 scattered in interior (not on wall at x=12 or right section)
    num_wood = 4 + int(env.random() * 3)
    for _ in range(num_wood):
        x, y = _find_empty(1, 11, 1, h - 2)
        env.create_entity('wood', x, y, 't', ['pickup'], z_order=3)

    # Ore: 3-5 scattered in interior
    num_ore = 3 + int(env.random() * 3)
    for _ in range(num_ore):
        x, y = _find_empty(1, 11, 1, h - 2)
        env.create_entity('ore', x, y, 'o', ['pickup'], z_order=3)

    # Exit in right section (x >= 13)
    ex, ey = _find_empty(13, w - 2, 1, h - 2)
    env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)

    # ---- Event Handlers ----

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
            # Check 4 cardinal neighbors for workbench
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = p.x + dx, p.y + dy
                for ent in env.get_entities_at(nx, ny):
                    if ent.type == 'workbench':
                        if p.properties.get('wood', 0) >= 2 and p.properties.get('ore', 0) >= 2:
                            p.properties['wood'] -= 2
                            p.properties['ore'] -= 2
                            p.properties['has_pickaxe'] = 1
                            env.emit('reward', {'amount': 0.3})
                        return

            # Check 4 cardinal neighbors for rubble
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = p.x + dx, p.y + dy
                for ent in env.get_entities_at(nx, ny):
                    if ent.type == 'rubble':
                        if p.properties.get('has_pickaxe', 0) == 1:
                            env.destroy_entity(ent.id)
                            p.properties['has_pickaxe'] = 0
                            env.emit('reward', {'amount': 0.3})
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

    # Collision — pickup items and reach exit
    def on_collision(event):
        mover = event.payload['mover']
        if not mover.has_tag('player'):
            return
        for occ in event.payload['occupants']:
            if occ.has_tag('pickup'):
                if occ.type == 'wood':
                    current = mover.properties.get('wood', 0)
                    mover.properties['wood'] = min(current + 1, 5)
                elif occ.type == 'ore':
                    current = mover.properties.get('ore', 0)
                    mover.properties['ore'] = min(current + 1, 5)
                env.destroy_entity(occ.id)
                env.emit('reward', {'amount': 0.05})
            elif occ.has_tag('exit'):
                env.end_game('won')

    env.on('collision', on_collision)


# ---- Game-specific invariants ----

@invariant('one_workbench')
def check_one_workbench(env):
    wb = env.get_entities_by_type('workbench')
    if len(wb) != 1:
        raise InvariantError(f"Expected 1 workbench, found {len(wb)}")


@invariant('one_rubble')
def check_one_rubble(env):
    rubble = env.get_entities_by_type('rubble')
    if len(rubble) != 1:
        raise InvariantError(f"Expected 1 rubble, found {len(rubble)}")


@invariant('enough_wood_and_ore')
def check_enough_materials(env):
    wood = env.get_entities_by_type('wood')
    ore = env.get_entities_by_type('ore')
    if len(wood) < 2:
        raise InvariantError(f"Expected at least 2 wood, found {len(wood)}")
    if len(ore) < 2:
        raise InvariantError(f"Expected at least 2 ore, found {len(ore)}")


@invariant('player_inventory_zero')
def check_player_inventory_zero(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('wood', 0) != 0:
        raise InvariantError("Player starts with wood != 0")
    if p.properties.get('ore', 0) != 0:
        raise InvariantError("Player starts with ore != 0")
    if p.properties.get('has_pickaxe', 0) != 0:
        raise InvariantError("Player starts with has_pickaxe != 0")


@invariant('exit_behind_rubble')
def check_exit_behind_rubble(env):
    exits = env.get_entities_by_tag('exit')
    if not exits:
        raise InvariantError("No exit found")
    e = exits[0]
    if e.x < 13:
        raise InvariantError(f"Exit at x={e.x}, expected x >= 13")


@invariant('rubble_at_x12')
def check_rubble_at_x12(env):
    rubble = env.get_entities_by_type('rubble')
    if not rubble:
        raise InvariantError("No rubble found")
    r = rubble[0]
    if r.x != 12:
        raise InvariantError(f"Rubble at x={r.x}, expected x=12")
