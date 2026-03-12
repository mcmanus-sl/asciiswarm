"""Game 17: Siege & Defense — defend a fortress from enemy waves by building walls and traps."""

from asciiswarm.kernel.invariants import invariant, InvariantError

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'build_wall', 'build_trap', 'order_archer'],
    'grid': (24, 20),
    'max_turns': 800,
    'step_penalty': -0.002,
    'player_properties': [
        {'key': 'health', 'max': 10},
        {'key': 'stone', 'max': 20},
        {'key': 'wave', 'max': 5},
        {'key': 'archers_alive', 'max': 3},
        {'key': 'fort_hp', 'max': 10},
    ],
}

# Wave schedule: (turn_number, composition)
WAVE_SCHEDULE = [
    (120, {'grunt': 4}),
    (240, {'grunt': 6}),
    (360, {'grunt': 5, 'brute': 2}),
    (480, {'grunt': 4, 'brute': 2, 'sapper': 2}),
    (600, {'grunt': 6, 'brute': 3, 'sapper': 3}),
]

ENEMY_STATS = {
    'grunt': {'hp': 2, 'atk': 1, 'glyph': 'g'},
    'brute': {'hp': 5, 'atk': 2, 'glyph': 'B'},
    'sapper': {'hp': 1, 'atk': 1, 'glyph': 's'},
}

ENEMY_TYPES = ('grunt', 'brute', 'sapper')


def setup(env):
    w, h = 24, 20

    # ---- Outer boundary walls ----
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    occupied = set()
    for ent in env.get_entities_by_tag('solid'):
        occupied.add((ent.x, ent.y))

    # ---- Fort core at (4, 10) ----
    env.create_entity('fort_core', 4, 10, 'C', ['npc'], z_order=5,
                       properties={'hp': 10})
    occupied.add((4, 10))

    # ---- Player inside fortress ----
    env.create_entity('player', 3, 10, '@', ['player'], z_order=10,
                       properties={'health': 10, 'stone': 5, 'wave': 0,
                                   'archers_alive': 3, 'fort_hp': 10})
    occupied.add((3, 10))

    # ---- 3 archers inside fortress (x < 9) ----
    archer_positions = [(2, 8), (6, 10), (2, 12)]
    for ax, ay in archer_positions:
        env.create_entity('archer', ax, ay, 'A', ['npc'], z_order=8,
                          properties={})
        occupied.add((ax, ay))

    # ---- 6-10 stone deposits in no-man's-land (x=9-15) ----
    num_stones = 6 + int(env.random() * 5)
    placed = 0
    while placed < num_stones:
        for _attempt in range(100):
            sx = 9 + int(env.random() * 7)
            sy = 1 + int(env.random() * (h - 2))
            if (sx, sy) not in occupied:
                env.create_entity('stone_deposit', sx, sy, 'o', ['pickup'], z_order=3)
                occupied.add((sx, sy))
                placed += 1
                break

    # ---- Track active wave state ----
    wave_state = {'current_wave': 0}

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

        elif action == 'build_wall':
            if p.properties['stone'] >= 3:
                bx, by = p.x + 1, p.y
                if bx > 8 and 0 < bx < w - 1 and 0 < by < h - 1:
                    blocked = False
                    for ent in env.get_entities_at(bx, by):
                        if ent.has_tag('solid') or ent.type in ('fort_core', 'player', 'archer',
                                                                 'built_wall', 'trap'):
                            blocked = True
                            break
                    if not blocked:
                        p.properties['stone'] -= 3
                        env.create_entity('built_wall', bx, by, '=', ['solid'], z_order=5,
                                          properties={'hp': 3})
                        env.emit('reward', {'amount': 0.1})

        elif action == 'build_trap':
            if p.properties['stone'] >= 2:
                bx, by = p.x + 1, p.y
                if bx > 8 and 0 < bx < w - 1 and 0 < by < h - 1:
                    blocked = False
                    for ent in env.get_entities_at(bx, by):
                        if ent.has_tag('solid') or ent.type in ('fort_core', 'player', 'archer',
                                                                 'built_wall', 'trap'):
                            blocked = True
                            break
                    if not blocked:
                        p.properties['stone'] -= 2
                        env.create_entity('trap', bx, by, '^', ['npc'], z_order=3,
                                          properties={})
                        env.emit('reward', {'amount': 0.1})

        elif action == 'order_archer':
            archers = env.get_entities_by_type('archer')
            if archers:
                nearest = _nearest_entity(p, archers)
                if nearest:
                    _swap_positions(env, p, nearest)

    env.on('input', on_input)

    # ---- Before move: solids block, with exceptions for enemies vs built_wall ----
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        mover = event.payload['entity']

        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                # Sappers pass through built_wall entirely
                if mover.type == 'sapper' and ent.type == 'built_wall':
                    continue
                # Other enemies can "reach" built_wall (collision will handle damage + cancel)
                if mover.type in ENEMY_TYPES and ent.type == 'built_wall':
                    continue
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # ---- Collision handler ----
    def on_collision(event):
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        for target in list(occupants):
            # Check entities still exist (may have been destroyed in previous iteration)
            if env.get_entity(mover.id) is None:
                return
            if env.get_entity(target.id) is None:
                continue

            # Player picks up stone deposit
            if mover.type == 'player' and target.type == 'stone_deposit':
                mover.properties['stone'] = min(20, mover.properties['stone'] + 3)
                env.destroy_entity(target.id)
                env.emit('reward', {'amount': 0.1})

            # Enemy steps on trap
            elif mover.type in ENEMY_TYPES and target.type == 'trap':
                env.destroy_entity(target.id)
                env.destroy_entity(mover.id)
                env.emit('reward', {'amount': 0.2})
                event.cancel()
                return  # mover destroyed

            # Player collides with enemy (mutual damage, cancel move)
            elif mover.type == 'player' and target.type in ENEMY_TYPES:
                atk = ENEMY_STATS[target.type]['atk']
                mover.properties['health'] -= atk
                target.properties['hp'] -= 1
                if target.properties['hp'] <= 0:
                    env.destroy_entity(target.id)
                    env.emit('reward', {'amount': 0.3})
                event.cancel()

            # Enemy collides with player
            elif mover.type in ENEMY_TYPES and target.type == 'player':
                atk = ENEMY_STATS[mover.type]['atk']
                target.properties['health'] -= atk
                mover.properties['hp'] -= 1
                if mover.properties['hp'] <= 0:
                    env.destroy_entity(mover.id)
                    env.emit('reward', {'amount': 0.3})
                    event.cancel()
                    return  # mover destroyed
                event.cancel()

            # Enemy reaches fort_core
            elif mover.type in ENEMY_TYPES and target.type == 'fort_core':
                atk = ENEMY_STATS[mover.type]['atk']
                target.properties['hp'] -= atk
                p = env.get_entities_by_tag('player')
                if p:
                    p[0].properties['fort_hp'] = target.properties['hp']
                event.cancel()

            # Enemy hits built_wall (sappers pass through, others damage it)
            elif mover.type in ENEMY_TYPES and target.type == 'built_wall':
                if mover.type == 'sapper':
                    # Sappers pass through built_wall — don't cancel
                    continue
                atk = ENEMY_STATS[mover.type]['atk']
                target.properties['hp'] -= atk
                if target.properties['hp'] <= 0:
                    env.destroy_entity(target.id)
                event.cancel()

    env.on('collision', on_collision)

    # ---- Turn end: wave spawning, archer attacks, win/lose ----
    def on_turn_end(event):
        if env.status != 'playing':
            return
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]
        turn = env.turn_number

        # 1. Check wave spawning
        for i, (wave_turn, composition) in enumerate(WAVE_SCHEDULE):
            wave_num = i + 1
            if turn == wave_turn and wave_state['current_wave'] < wave_num:
                wave_state['current_wave'] = wave_num
                p.properties['wave'] = wave_num
                _spawn_wave(env, composition)

        # 2. Archer attacks (each archer attacks nearest enemy within range 6)
        enemies = _get_all_enemies(env)
        if enemies:
            archers = env.get_entities_by_type('archer')
            for archer in list(archers):
                nearest = _nearest_entity(archer, enemies)
                if nearest:
                    dist = abs(nearest.x - archer.x) + abs(nearest.y - archer.y)
                    if dist <= 6:
                        nearest.properties['hp'] -= 1
                        if nearest.properties['hp'] <= 0:
                            env.destroy_entity(nearest.id)
                            env.emit('reward', {'amount': 0.2})
                            # Refresh enemies list
                            enemies = _get_all_enemies(env)

        # 3. Update player properties
        archers = env.get_entities_by_type('archer')
        p.properties['archers_alive'] = len(archers)
        fort_cores = env.get_entities_by_type('fort_core')
        if fort_cores:
            p.properties['fort_hp'] = fort_cores[0].properties['hp']

        # 4. Win/lose checks
        if p.properties['health'] <= 0:
            env.end_game('lost')
            return
        if p.properties['fort_hp'] <= 0:
            env.end_game('lost')
            return

        # Wave 5 cleared: all wave 5 enemies dead and wave 5 has been triggered
        if wave_state['current_wave'] >= 5:
            remaining = _get_all_enemies(env)
            if not remaining:
                env.end_game('won')
                return

    env.on('turn_end', on_turn_end)

    # ---- Enemy behaviors ----
    def enemy_behavior(entity, env):
        # Entity may have been destroyed earlier this turn (snapshot iteration)
        if env.get_entity(entity.id) is None:
            return
        fort_cores = env.get_entities_by_type('fort_core')
        if fort_cores:
            _move_toward(env, entity, fort_cores[0].x, fort_cores[0].y)

    env.register_behavior('grunt', enemy_behavior)
    env.register_behavior('brute', enemy_behavior)
    env.register_behavior('sapper', enemy_behavior)


def _swap_positions(env, a, b):
    """Swap two entities' positions directly on the grid."""
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    # Remove from old grid cells
    env._grid[ay][ax].remove(a)
    env._grid[by][bx].remove(b)
    # Update positions
    a.x, a.y = bx, by
    b.x, b.y = ax, ay
    # Add to new grid cells
    env._grid[by][bx].append(a)
    env._grid[ay][ax].append(b)


def _spawn_wave(env, composition):
    """Spawn enemies in the spawn zone (x=16-22)."""
    h = 20
    occupied = set()
    for ent in env.get_all_entities():
        occupied.add((ent.x, ent.y))

    for enemy_type, count in composition.items():
        stats = ENEMY_STATS[enemy_type]
        for _ in range(count):
            for _attempt in range(100):
                sx = 16 + int(env.random() * 7)
                sy = 1 + int(env.random() * (h - 2))
                if (sx, sy) not in occupied:
                    env.create_entity(enemy_type, sx, sy, stats['glyph'], ['hazard'], z_order=7,
                                      properties={'hp': stats['hp'], 'atk': stats['atk']})
                    occupied.add((sx, sy))
                    break


def _get_all_enemies(env):
    enemies = []
    for etype in ENEMY_TYPES:
        enemies.extend(env.get_entities_by_type(etype))
    return enemies


def _nearest_entity(source, entities):
    best = None
    best_dist = float('inf')
    for e in entities:
        d = abs(e.x - source.x) + abs(e.y - source.y)
        if d < best_dist:
            best_dist = d
            best = e
    return best


def _move_toward(env, entity, tx, ty):
    dx = tx - entity.x
    dy = ty - entity.y
    if dx == 0 and dy == 0:
        return
    dx_abs, dy_abs = abs(dx), abs(dy)
    moves = []
    if dx_abs >= dy_abs and dx != 0:
        moves.append((1 if dx > 0 else -1, 0))
        if dy != 0:
            moves.append((0, 1 if dy > 0 else -1))
    elif dy != 0:
        moves.append((0, 1 if dy > 0 else -1))
        if dx != 0:
            moves.append((1 if dx > 0 else -1, 0))
    elif dx != 0:
        moves.append((1 if dx > 0 else -1, 0))
    for mx, my in moves:
        if env.move_entity(entity.id, entity.x + mx, entity.y + my):
            return


# ---- Game-specific invariants ----

@invariant('fort_core_exists_at_start')
def check_fort_core(env):
    cores = env.get_entities_by_type('fort_core')
    if len(cores) != 1:
        raise InvariantError(f"Expected 1 fort_core, found {len(cores)}")
    if cores[0].properties.get('hp') != 10:
        raise InvariantError(f"Fort core hp is {cores[0].properties.get('hp')}, expected 10")
    if cores[0].x != 4 or cores[0].y != 10:
        raise InvariantError(f"Fort core at ({cores[0].x},{cores[0].y}), expected (4,10)")


@invariant('three_archers_at_start')
def check_archers(env):
    archers = env.get_entities_by_type('archer')
    if len(archers) != 3:
        raise InvariantError(f"Expected 3 archers, found {len(archers)}")


@invariant('stone_deposits_exist')
def check_stone_deposits(env):
    deposits = env.get_entities_by_type('stone_deposit')
    if len(deposits) < 6 or len(deposits) > 10:
        raise InvariantError(f"Expected 6-10 stone deposits, found {len(deposits)}")


@invariant('no_enemies_at_start')
def check_no_enemies(env):
    enemies = _get_all_enemies(env)
    if enemies:
        raise InvariantError(f"Expected no enemies at start, found {len(enemies)}")


@invariant('player_starts_inside_fortress')
def check_player_position(env):
    p = env.get_entities_by_tag('player')
    if not p:
        raise InvariantError("No player found")
    if p[0].x >= 9:
        raise InvariantError(f"Player x={p[0].x}, expected < 9 (inside fortress)")


@invariant('wave_zero_at_start')
def check_wave_zero(env):
    p = env.get_entities_by_tag('player')
    if not p:
        raise InvariantError("No player found")
    if p[0].properties.get('wave', 0) != 0:
        raise InvariantError(f"Player wave is {p[0].properties.get('wave')}, expected 0")


@invariant('build_zone_is_east')
def check_build_zone(env):
    for bw in env.get_entities_by_type('built_wall'):
        if bw.x <= 8:
            raise InvariantError(f"Built wall at x={bw.x}, expected x > 8")
