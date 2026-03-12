"""Game 10: Farming & Growth — plant seeds, grow crops, harvest, deliver quota."""

from asciiswarm.kernel.invariants import invariant, InvariantError

GAME_CONFIG = {
    'tags': ['player', 'solid', 'pickup', 'exit', 'npc'],
    'grid': (14, 14),
    'max_turns': 300,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'seeds', 'max': 10},
        {'key': 'crops', 'max': 10},
        {'key': 'delivered', 'max': 5},
    ],
}


def setup(env):
    w, h = env.config['grid']

    # Outer walls
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    # Farmhouse walls (small room top-left, x=0..4, y=0..4)
    # Bottom wall of farmhouse at y=4, x=1..3 with opening at x=2
    for x in range(1, 4):
        if x != 2:
            env.create_entity('wall', x, 4, '#', ['solid'], z_order=1)
    # Right wall of farmhouse at x=4, y=1..3 with opening at y=2
    for y in range(1, 4):
        if y != 2:
            env.create_entity('wall', 4, y, '#', ['solid'], z_order=1)

    # Player in farmhouse area (x=1..3, y=1..3)
    while True:
        px = 1 + int(env.random() * 3)
        py = 1 + int(env.random() * 3)
        if not env.get_entities_at(px, py):
            break
    env.create_entity('player', px, py, '@', ['player'], z_order=10,
                       properties={'seeds': 0, 'crops': 0, 'delivered': 0})

    # Soil patch: 3x3 at center (x=5..7, y=5..7)
    for sx in range(5, 8):
        for sy in range(5, 8):
            env.create_entity('soil', sx, sy, '~', ['npc'], z_order=2)

    # Seedbag near farmhouse
    env.create_entity('seedbag', 2, 5, 's', ['pickup'], z_order=3)

    # Collection bin
    env.create_entity('bin', 12, 1, 'B', ['npc'], z_order=5)

    # --- Event Handlers ---

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
            # Check if standing on soil with seeds and no sprout/mature here
            on_soil = False
            for ent in env.get_entities_at(p.x, p.y):
                if ent.type == 'soil':
                    on_soil = True
                    break
            has_sprout_or_mature = False
            for ent in env.get_entities_at(p.x, p.y):
                if ent.type in ('sprout', 'mature'):
                    has_sprout_or_mature = True
                    break

            if on_soil and p.properties['seeds'] >= 1 and not has_sprout_or_mature:
                env.create_entity('sprout', p.x, p.y, ',', ['npc'], z_order=3,
                                  properties={'age': 0})
                p.properties['seeds'] -= 1
                env.emit('reward', {'amount': 0.05})
                return

            # Check 4 cardinal neighbors for bin, deliver crops
            if p.properties['crops'] >= 1:
                for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                    nx, ny = p.x + dx, p.y + dy
                    for ent in env.get_entities_at(nx, ny):
                        if ent.type == 'bin':
                            p.properties['crops'] -= 1
                            p.properties['delivered'] += 1
                            env.emit('reward', {'amount': 0.3})
                            if p.properties['delivered'] >= 5:
                                env.end_game('won')
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

    # Collision — pickup seedbag and mature crops
    def on_collision(event):
        mover = event.payload['mover']
        if not mover.has_tag('player'):
            return
        for occ in list(event.payload['occupants']):
            if occ.type == 'seedbag':
                mover.properties['seeds'] = 6
                env.destroy_entity(occ.id)
                env.emit('reward', {'amount': 0.1})
            elif occ.type == 'mature':
                mover.properties['crops'] = min(mover.properties['crops'] + 1, 10)
                env.destroy_entity(occ.id)
                env.emit('reward', {'amount': 0.15})

    env.on('collision', on_collision)

    # Sprout behavior — age and grow into mature
    def sprout_behavior(entity, env):
        age = entity.get('age', 0) + 1
        entity.set('age', age)
        if age >= 15:
            x, y = entity.x, entity.y
            env.destroy_entity(entity.id)
            env.create_entity('mature', x, y, '*', ['pickup'], z_order=3)

    env.register_behavior('sprout', sprout_behavior)


# ---- Game-specific invariants ----

@invariant('nine_soil_tiles')
def check_nine_soil(env):
    soils = env.get_entities_by_type('soil')
    if len(soils) != 9:
        raise InvariantError(f"Expected 9 soil tiles, found {len(soils)}")


@invariant('one_seedbag_at_start')
def check_one_seedbag(env):
    bags = env.get_entities_by_type('seedbag')
    if len(bags) != 1:
        raise InvariantError(f"Expected 1 seedbag, found {len(bags)}")


@invariant('one_bin_exists')
def check_one_bin(env):
    bins = env.get_entities_by_type('bin')
    if len(bins) != 1:
        raise InvariantError(f"Expected 1 bin, found {len(bins)}")


@invariant('player_starts_empty')
def check_player_starts_empty(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('seeds', 0) != 0:
        raise InvariantError(f"Player starts with seeds={p.properties['seeds']}")
    if p.properties.get('crops', 0) != 0:
        raise InvariantError(f"Player starts with crops={p.properties['crops']}")
    if p.properties.get('delivered', 0) != 0:
        raise InvariantError(f"Player starts with delivered={p.properties['delivered']}")


@invariant('no_sprouts_or_mature_at_start')
def check_no_plants_at_start(env):
    sprouts = env.get_entities_by_type('sprout')
    mature = env.get_entities_by_type('mature')
    if sprouts:
        raise InvariantError(f"Found {len(sprouts)} sprouts at start")
    if mature:
        raise InvariantError(f"Found {len(mature)} mature plants at start")
