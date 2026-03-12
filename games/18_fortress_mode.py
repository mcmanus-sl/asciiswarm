"""Game 18: Fortress Mode (capstone) — manage dwarves, farm, trade, and survive siege waves."""

from asciiswarm.kernel.invariants import invariant, InvariantError

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'order_eat', 'order_sleep', 'order_mine', 'order_farm',
                'order_craft', 'order_build', 'order_guard', 'order_haul'],
    'grid': (40, 30),
    'max_turns': 2000,
    'step_penalty': 0.0,
    'player_properties': [
        {'key': 'population', 'max': 12},
        {'key': 'food_stock', 'max': 50},
        {'key': 'stone_stock', 'max': 50},
        {'key': 'wood_stock', 'max': 50},
        {'key': 'wealth', 'max': 100},
        {'key': 'avg_mood', 'max': 10},
        {'key': 'wave', 'max': 5},
        {'key': 'score', 'max': 100},
    ],
}


def setup(env):
    w, h = 40, 30
    occupied = set()

    # ---- Fortress outer walls (x=0-12) with entrance gap at east wall around y=15 ----
    # North wall of fortress
    for x in range(13):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        occupied.add((x, 0))
    # South wall of fortress
    for x in range(13):
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
        occupied.add((x, h - 1))
    # West wall of fortress
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        occupied.add((0, y))
    # East wall of fortress with entrance gap at y=14,15,16
    for y in range(1, h - 1):
        if y in (14, 15, 16):
            continue  # entrance gap
        env.create_entity('wall', 12, y, '#', ['solid'], z_order=1)
        occupied.add((12, y))

    # Boundary walls for the rest of the map (top/bottom rows, right column)
    for x in range(13, w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        occupied.add((x, 0))
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
        occupied.add((x, h - 1))
    for y in range(1, h - 1):
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)
        occupied.add((w - 1, y))

    # ---- Player at center of fortress ----
    px, py = 6, 15
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={
                                   'population': 6, 'food_stock': 10,
                                   'stone_stock': 5, 'wood_stock': 5,
                                   'wealth': 0, 'avg_mood': 10,
                                   'wave': 0, 'score': 0,
                               })
    occupied.add((px, py))

    # ---- Fortress interior: dormitory with 6 beds (top-left area) ----
    bed_positions = [(2, 3), (3, 3), (4, 3), (2, 4), (3, 4), (4, 4)]
    for bx, by in bed_positions:
        env.create_entity('bed', bx, by, 'b', ['npc'], z_order=5)
        occupied.add((bx, by))

    # ---- 2 food stores ----
    for fx, fy in [(2, 8), (3, 8)]:
        env.create_entity('food_store', fx, fy, 'F', ['npc'], z_order=5)
        occupied.add((fx, fy))

    # ---- 1 workshop ----
    env.create_entity('workshop', 8, 8, 'W', ['npc'], z_order=5)
    occupied.add((8, 8))

    # ---- 1 tavern ----
    env.create_entity('tavern', 6, 22, 'T', ['npc'], z_order=5)
    occupied.add((6, 22))

    # ---- 6 dwarves near player ----
    dwarf_spots = [(5, 14), (7, 14), (5, 16), (7, 16), (5, 15), (7, 15)]
    for dx, dy in dwarf_spots:
        if (dx, dy) not in occupied:
            env.create_entity('dwarf', dx, dy, 'D', ['npc'], z_order=8,
                              properties={'hunger': 10, 'rest': 10, 'social': 10,
                                          'mood': 10, 'task': 'idle', 'hp': 5,
                                          'tantrum_turns': 0})
            occupied.add((dx, dy))

    # ---- Farm zone (x=13-19): 4x4 soil patch (16 tiles) ----
    for sx in range(14, 18):
        for sy in range(10, 14):
            env.create_entity('soil', sx, sy, '.', ['npc'], z_order=2,
                              properties={'has_crop': False, 'crop_age': 0,
                                          'watered': False})
            occupied.add((sx, sy))

    # ---- Water source and pump near farm ----
    env.create_entity('water_source', 16, 8, '~', ['npc'], z_order=3,
                      properties={'timer': 0})
    occupied.add((16, 8))
    env.create_entity('pump', 17, 8, 'P', ['npc'], z_order=4,
                      properties={'on': False})
    occupied.add((17, 8))

    # ---- Wilderness (x=20-32): trees, ore, rabbits, wolves, bushes ----
    def _place_random(etype, glyph, tags, z, props, count_min, count_max, x_lo, x_hi, y_lo, y_hi):
        count = count_min + int(env.random() * (count_max - count_min + 1))
        placed = 0
        for _ in range(count * 50):
            if placed >= count:
                break
            rx = x_lo + int(env.random() * (x_hi - x_lo))
            ry = y_lo + int(env.random() * (y_hi - y_lo))
            if (rx, ry) not in occupied:
                env.create_entity(etype, rx, ry, glyph, tags, z_order=z,
                                  properties=dict(props) if props else {})
                occupied.add((rx, ry))
                placed += 1

    _place_random('tree', 't', ['npc'], 3, {'wood': 1}, 8, 12, 20, 33, 1, 29)
    _place_random('ore', 'o', ['npc'], 3, {'stone': 1}, 6, 8, 20, 33, 1, 29)
    _place_random('rabbit', 'r', ['npc'], 5, {'hp': 1}, 6, 8, 20, 33, 1, 29)
    _place_random('wolf', 'w', ['hazard'], 6, {'hp': 3, 'attack': 2}, 2, 3, 20, 33, 1, 29)
    _place_random('bush', 'u', ['npc'], 3, {}, 4, 5, 20, 33, 1, 29)

    # ---- Siege approach (x=33-38): 8 build sites along fortress eastern approach ----
    build_y_start = 11
    build_positions = []
    for i in range(8):
        bx = 13 + (i % 4)
        by = build_y_start + (i // 4) * 2
        if (bx, by) in occupied:
            for ddx, ddy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nbx, nby = bx + ddx, by + ddy
                if (nbx, nby) not in occupied and 1 <= nbx < w - 1 and 1 <= nby < h - 1:
                    bx, by = nbx, nby
                    break
        build_positions.append((bx, by))
        env.create_entity('build_site', bx, by, '_', ['npc'], z_order=2)
        occupied.add((bx, by))

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
            return

        if action == 'interact':
            _handle_interact(env, p)
            return

        if action == 'wait':
            return

        # Order commands
        order_tasks = {
            'order_eat': 'eat', 'order_sleep': 'sleep',
            'order_mine': 'mine', 'order_farm': 'farm',
            'order_craft': 'craft', 'order_build': 'build',
            'order_guard': 'guard', 'order_haul': 'haul',
        }
        if action in order_tasks:
            task = order_tasks[action]
            dwarves = env.get_entities_by_type('dwarf')
            best_dwarf = None
            best_dist = float('inf')
            for d in dwarves:
                if d.properties.get('tantrum_turns', 0) > 0:
                    continue
                dist = abs(d.x - p.x) + abs(d.y - p.y)
                if dist <= 5 and dist < best_dist:
                    best_dist = dist
                    best_dwarf = d
            if best_dwarf:
                best_dwarf.properties['task'] = task
                env.emit('reward', {'amount': 0.05})

    env.on('input', on_input)

    # ---- Before move: solids block ----
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # ---- Collision handler ----
    def on_collision(event):
        pass

    env.on('collision', on_collision)

    # ---- Turn end ----
    def on_turn_end(event):
        if env.status != 'playing':
            return
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]
        turn = env.turn_number

        # Per-dwarf reward
        dwarves = env.get_entities_by_type('dwarf')
        env.emit('reward', {'amount': 0.01 * len(dwarves)})

        # 1. Needs decay
        for d in list(dwarves):
            if d.properties.get('tantrum_turns', 0) > 0:
                continue
            if turn % 5 == 0:
                d.properties['hunger'] = max(0, d.properties['hunger'] - 1)
            if turn % 6 == 0:
                d.properties['rest'] = max(0, d.properties['rest'] - 1)
            if turn % 8 == 0:
                d.properties['social'] = max(0, d.properties['social'] - 1)

        # 2. Process existing tantrums
        for d in list(dwarves):
            if d.properties.get('tantrum_turns', 0) > 0:
                d.properties['tantrum_turns'] -= 1
                if d.properties['tantrum_turns'] <= 0:
                    d.properties['hunger'] = 3
                    d.properties['rest'] = 3
                    d.properties['social'] = 3
                    d.properties['task'] = 'idle'
                    d.properties['tantrum_turns'] = 0

        # 3. Recalculate mood and tantrum cascade
        dwarves = env.get_entities_by_type('dwarf')
        new_tantrums = []
        for d in dwarves:
            if d.properties.get('tantrum_turns', 0) > 0:
                d.properties['mood'] = 0
                continue
            d.properties['mood'] = min(d.properties['hunger'],
                                       d.properties['rest'],
                                       d.properties['social'])
            if d.properties['mood'] <= 0:
                new_tantrums.append(d)

        while new_tantrums:
            next_wave = []
            for d in new_tantrums:
                if d.properties.get('tantrum_turns', 0) > 0:
                    continue
                d.properties['tantrum_turns'] = 3
                d.properties['task'] = 'tantrum'
                d.properties['mood'] = 0
                env.emit('reward', {'amount': -0.3})
                for other in env.get_entities_by_type('dwarf'):
                    if other.id == d.id or other.properties.get('tantrum_turns', 0) > 0:
                        continue
                    dist = abs(other.x - d.x) + abs(other.y - d.y)
                    if dist <= 3:
                        other.properties['hunger'] = max(0, other.properties['hunger'] - 1)
                        other.properties['rest'] = max(0, other.properties['rest'] - 1)
                        other.properties['social'] = max(0, other.properties['social'] - 1)
                        other.properties['mood'] = min(other.properties['hunger'],
                                                       other.properties['rest'],
                                                       other.properties['social'])
                        if other.properties['mood'] <= 0:
                            next_wave.append(other)
            new_tantrums = next_wave

        # 4. Crop growth
        for soil in env.get_entities_by_type('soil'):
            if soil.properties.get('has_crop'):
                growth_time = 8 if soil.properties.get('watered') else 12
                soil.properties['crop_age'] = soil.properties.get('crop_age', 0) + 1
                if soil.properties['crop_age'] >= growth_time:
                    soil.properties['crop_age'] = growth_time  # cap at mature

        # 5. Water mechanics
        _water_tick(env, p, turn)

        # 6. Ecology
        _ecology_tick(env, turn)

        # 7. Siege waves
        _siege_tick(env, p, turn)

        # 8. Enemy behavior
        _enemy_tick(env)

        # 9. Merchant
        _merchant_tick(env, p, turn)

        # 10. Population growth
        if turn % 150 == 0 and turn > 0:
            dwarves = env.get_entities_by_type('dwarf')
            if (p.properties['avg_mood'] > 7
                    and p.properties['food_stock'] > 20
                    and len(dwarves) < 12):
                # Spawn new dwarf inside fortress
                for _att in range(100):
                    nx = 1 + int(env.random() * 11)
                    ny = 1 + int(env.random() * 28)
                    blocked = False
                    for ent in env.get_entities_at(nx, ny):
                        if ent.has_tag('solid') or ent.type in ('dwarf', 'player'):
                            blocked = True
                            break
                    if not blocked:
                        env.create_entity('dwarf', nx, ny, 'D', ['npc'], z_order=8,
                                          properties={'hunger': 10, 'rest': 10,
                                                      'social': 10, 'mood': 10,
                                                      'task': 'idle', 'hp': 5,
                                                      'tantrum_turns': 0})
                        break

        # 11. Update player properties
        dwarves = env.get_entities_by_type('dwarf')
        p.properties['population'] = len(dwarves)
        if dwarves:
            total_mood = sum(d.properties['mood'] for d in dwarves)
            p.properties['avg_mood'] = total_mood // len(dwarves)
        else:
            p.properties['avg_mood'] = 0

        # 12. Win/lose
        if len(dwarves) == 0:
            env.emit('reward', {'amount': -5.0})
            env.end_game('lost')
            return

        if turn >= 2000:
            env.end_game('won')
            return

    env.on('turn_end', on_turn_end)

    # ---- Dwarf behavior ----
    def dwarf_behavior(entity, env):
        if env.status != 'playing':
            return
        if env.get_entity(entity.id) is None:
            return
        task = entity.properties.get('task', 'idle')
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]

        if task == 'idle':
            return

        if task == 'tantrum':
            _tantrum_move(env, entity)
            return

        if task == 'eat':
            food_stores = env.get_entities_by_type('food_store')
            if not food_stores or p.properties['food_stock'] <= 0:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, food_stores)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    p.properties['food_stock'] = max(0, p.properties['food_stock'] - 1)
                    entity.properties['hunger'] = 10
                    entity.properties['task'] = 'idle'
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'sleep':
            beds = env.get_entities_by_type('bed')
            if not beds:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, beds)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    entity.properties['rest'] = 10
                    entity.properties['task'] = 'idle'
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'mine':
            ores = env.get_entities_by_type('ore')
            if not ores:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, ores)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    env.destroy_entity(nearest.id)
                    p.properties['stone_stock'] = min(50, p.properties['stone_stock'] + 2)
                    entity.properties['task'] = 'idle'
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'farm':
            soils = env.get_entities_by_type('soil')
            # First check for mature crops to harvest
            mature = [s for s in soils if s.properties.get('has_crop')
                      and s.properties.get('crop_age', 0) >= (8 if s.properties.get('watered') else 12)]
            if mature:
                nearest = _nearest_entity(entity, mature)
                if nearest:
                    dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                    if dist <= 1:
                        nearest.properties['has_crop'] = False
                        nearest.properties['crop_age'] = 0
                        p.properties['food_stock'] = min(50, p.properties['food_stock'] + 2)
                        env.emit('reward', {'amount': 0.5})
                        entity.properties['task'] = 'idle'
                    else:
                        _move_toward(env, entity, nearest.x, nearest.y)
                return

            # Plant on empty soil
            empty = [s for s in soils if not s.properties.get('has_crop')]
            if empty and p.properties['food_stock'] >= 1:
                nearest = _nearest_entity(entity, empty)
                if nearest:
                    dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                    if dist <= 1:
                        p.properties['food_stock'] = max(0, p.properties['food_stock'] - 1)
                        nearest.properties['has_crop'] = True
                        nearest.properties['crop_age'] = 0
                        entity.properties['task'] = 'idle'
                    else:
                        _move_toward(env, entity, nearest.x, nearest.y)
            else:
                entity.properties['task'] = 'idle'

        elif task == 'craft':
            workshops = env.get_entities_by_type('workshop')
            if not workshops:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, workshops)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    if p.properties['stone_stock'] >= 2 and p.properties['wood_stock'] >= 1:
                        p.properties['stone_stock'] -= 2
                        p.properties['wood_stock'] -= 1
                        p.properties['wealth'] = min(100, p.properties['wealth'] + 3)
                    entity.properties['task'] = 'idle'
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'build':
            sites = env.get_entities_by_type('build_site')
            if not sites or p.properties['stone_stock'] < 2:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, sites)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    sx, sy = nearest.x, nearest.y
                    env.destroy_entity(nearest.id)
                    env.create_entity('built_wall', sx, sy, '=', ['solid'], z_order=5,
                                      properties={'hp': 3})
                    p.properties['stone_stock'] -= 2
                    env.emit('reward', {'amount': 0.5})
                    entity.properties['task'] = 'idle'
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'guard':
            # Move to fortress entrance area (x=12, y=14-16)
            target_x, target_y = 11, 15
            dist = abs(entity.x - target_x) + abs(entity.y - target_y)
            if dist <= 2:
                # Attack nearest enemy within 2 tiles
                enemies = env.get_entities_by_type('grunt') + env.get_entities_by_type('brute')
                for e in enemies:
                    edist = abs(e.x - entity.x) + abs(e.y - entity.y)
                    if edist <= 2:
                        e.properties['hp'] = e.properties.get('hp', 1) - 1
                        if e.properties['hp'] <= 0:
                            env.destroy_entity(e.id)
                            env.emit('reward', {'amount': 0.2})
                        break
            else:
                _move_toward(env, entity, target_x, target_y)

        elif task == 'haul':
            # Look for trees (wood pickup) or ore in wilderness
            trees = env.get_entities_by_type('tree')
            if trees:
                nearest = _nearest_entity(entity, trees)
                if nearest:
                    dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                    if dist <= 1:
                        env.destroy_entity(nearest.id)
                        p.properties['wood_stock'] = min(50, p.properties['wood_stock'] + 1)
                        entity.properties['task'] = 'idle'
                    else:
                        _move_toward(env, entity, nearest.x, nearest.y)
            else:
                entity.properties['task'] = 'idle'

    env.register_behavior('dwarf', dwarf_behavior)

    # ---- Enemy behaviors ----
    def grunt_behavior(entity, env):
        if env.status != 'playing':
            return
        if env.get_entity(entity.id) is None:
            return
        # Move toward fortress entrance (x=12, y=15)
        target_x, target_y = 12, 15
        dist = abs(entity.x - target_x) + abs(entity.y - target_y)
        if dist <= 1:
            # Inside fortress: attack dwarves or destroy furniture
            _enemy_attack_interior(env, entity)
        else:
            _move_toward(env, entity, target_x, target_y)

    def brute_behavior(entity, env):
        if env.status != 'playing':
            return
        if env.get_entity(entity.id) is None:
            return
        target_x, target_y = 12, 15
        dist = abs(entity.x - target_x) + abs(entity.y - target_y)
        if dist <= 1:
            _enemy_attack_interior(env, entity)
        else:
            _move_toward(env, entity, target_x, target_y)

    env.register_behavior('grunt', grunt_behavior)
    env.register_behavior('brute', brute_behavior)

    # ---- Rabbit behavior ----
    def rabbit_behavior(entity, env):
        if env.status != 'playing':
            return
        if env.get_entity(entity.id) is None:
            return
        # Random movement in wilderness
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        idx = int(env.random() * 4)
        dx, dy = dirs[idx]
        nx, ny = entity.x + dx, entity.y + dy
        if 20 <= nx <= 32 and 1 <= ny <= 28:
            env.move_entity(entity.id, nx, ny)

    env.register_behavior('rabbit', rabbit_behavior)

    # ---- Wolf behavior ----
    def wolf_behavior(entity, env):
        if env.status != 'playing':
            return
        if env.get_entity(entity.id) is None:
            return
        # Hunt nearest rabbit or attack dwarves in wilderness
        rabbits = env.get_entities_by_type('rabbit')
        dwarves_in_wild = [d for d in env.get_entities_by_type('dwarf')
                           if 20 <= d.x <= 32]

        targets = rabbits + dwarves_in_wild
        if targets:
            nearest = _nearest_entity(entity, targets)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    if nearest.type == 'rabbit':
                        env.destroy_entity(nearest.id)
                    elif nearest.type == 'dwarf':
                        nearest.properties['hp'] = nearest.properties.get('hp', 5) - 2
                        if nearest.properties['hp'] <= 0:
                            env.destroy_entity(nearest.id)
                            env.emit('reward', {'amount': -2.0})
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)
        else:
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            idx = int(env.random() * 4)
            dx, dy = dirs[idx]
            nx, ny = entity.x + dx, entity.y + dy
            if 20 <= nx <= 32 and 1 <= ny <= 28:
                env.move_entity(entity.id, nx, ny)

    env.register_behavior('wolf', wolf_behavior)

    # ---- Merchant behavior (stays still) ----
    def merchant_behavior(entity, env):
        pass

    env.register_behavior('merchant', merchant_behavior)


def _handle_interact(env, player):
    """Handle interact action: pump toggle, merchant trade, workshop craft."""
    # Check adjacent entities
    for dx, dy in [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]:
        nx, ny = player.x + dx, player.y + dy
        for ent in env.get_entities_at(nx, ny):
            if ent.type == 'pump':
                ent.properties['on'] = not ent.properties.get('on', False)
                return
            if ent.type == 'merchant':
                p = player
                # Trade: 5 food/stone/wood for +5 wealth
                if (p.properties['food_stock'] >= 5
                        or p.properties['stone_stock'] >= 5
                        or p.properties['wood_stock'] >= 5):
                    # Trade whichever resource is highest
                    resources = [
                        ('food_stock', p.properties['food_stock']),
                        ('stone_stock', p.properties['stone_stock']),
                        ('wood_stock', p.properties['wood_stock']),
                    ]
                    resources.sort(key=lambda r: r[1], reverse=True)
                    for rname, rval in resources:
                        if rval >= 5:
                            p.properties[rname] -= 5
                            p.properties['wealth'] = min(100, p.properties['wealth'] + 5)
                            env.emit('reward', {'amount': 0.3})
                            return
                return
            if ent.type == 'workshop':
                p = player
                if p.properties['stone_stock'] >= 2 and p.properties['wood_stock'] >= 1:
                    p.properties['stone_stock'] -= 2
                    p.properties['wood_stock'] -= 1
                    p.properties['wealth'] = min(100, p.properties['wealth'] + 3)
                return


def _water_tick(env, player, turn):
    """Handle water source spawning and pump draining."""
    ws = env.get_entities_by_type('water_source')
    if not ws:
        return
    ws = ws[0]

    pumps = env.get_entities_by_type('pump')
    pump_on = pumps[0].properties.get('on', False) if pumps else False

    if pump_on:
        # Drain water within 3 tiles of pump
        pump = pumps[0]
        for wt in list(env.get_entities_by_type('water')):
            dist = abs(wt.x - pump.x) + abs(wt.y - pump.y)
            if dist <= 3:
                env.destroy_entity(wt.id)
    else:
        # Spawn water every 10 turns
        if turn % 10 == 0:
            # Spread water near water source
            wx, wy = ws.x, ws.y
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = wx + dx, wy + dy
                if 0 < nx < 39 and 0 < ny < 29:
                    has_water = any(e.type == 'water' for e in env.get_entities_at(nx, ny))
                    has_solid = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                    if not has_water and not has_solid:
                        env.create_entity('water', nx, ny, '~', ['npc'], z_order=1)
                        break

    # Water effects on farm soil
    for soil in env.get_entities_by_type('soil'):
        has_water = any(e.type == 'water' for e in env.get_entities_at(soil.x, soil.y))
        soil.properties['watered'] = has_water

    # Water flooding fortress: reduce food_stock
    for wt in env.get_entities_by_type('water'):
        if wt.x <= 12:
            # Water in fortress
            food_stores = env.get_entities_by_type('food_store')
            for fs in food_stores:
                if abs(fs.x - wt.x) + abs(fs.y - wt.y) <= 2:
                    player.properties['food_stock'] = max(0, player.properties['food_stock'] - 1)
                    break
            break  # Only reduce once per turn


def _ecology_tick(env, turn):
    """Handle rabbit reproduction and wolf reproduction."""
    # Rabbits reproduce near bushes every 25 turns
    if turn % 25 == 0:
        rabbits = env.get_entities_by_type('rabbit')
        if len(rabbits) < 10:
            bushes = env.get_entities_by_type('bush')
            if bushes:
                bush = bushes[int(env.random() * len(bushes))]
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = bush.x + dx, bush.y + dy
                    if 20 <= nx <= 32 and 1 <= ny <= 28:
                        occupied_at = env.get_entities_at(nx, ny)
                        if not any(e.has_tag('solid') for e in occupied_at):
                            env.create_entity('rabbit', nx, ny, 'r', ['npc'], z_order=5,
                                              properties={'hp': 1})
                            break

    # Wolves reproduce every 50 turns
    if turn % 50 == 0:
        wolves = env.get_entities_by_type('wolf')
        if len(wolves) < 4 and wolves:
            wolf = wolves[int(env.random() * len(wolves))]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = wolf.x + dx, wolf.y + dy
                if 20 <= nx <= 32 and 1 <= ny <= 28:
                    occupied_at = env.get_entities_at(nx, ny)
                    if not any(e.has_tag('solid') for e in occupied_at):
                        env.create_entity('wolf', nx, ny, 'w', ['hazard'], z_order=6,
                                          properties={'hp': 3, 'attack': 2})
                        break


def _siege_tick(env, player, turn):
    """Spawn siege waves every 300 turns."""
    wave_turns = [300, 600, 900, 1200, 1500]
    wave_configs = [
        (3, 0),   # Wave 1: 3 grunts
        (5, 0),   # Wave 2: 5 grunts
        (4, 2),   # Wave 3: 4 grunts + 2 brutes
        (3, 3),   # Wave 4: 3 grunts + 3 brutes
        (5, 4),   # Wave 5: 5 grunts + 4 brutes
    ]

    for i, wt in enumerate(wave_turns):
        if turn == wt:
            player.properties['wave'] = i + 1
            grunts, brutes = wave_configs[i]
            # Spawn enemies at right edge (x=38)
            for j in range(grunts):
                sy = 10 + j * 2
                if sy >= 28:
                    sy = 28
                env.create_entity('grunt', 38, sy, 'g', ['hazard'], z_order=7,
                                  properties={'hp': 2})
            for j in range(brutes):
                sy = 11 + j * 2
                if sy >= 28:
                    sy = 28
                env.create_entity('brute', 38, sy, 'B', ['hazard'], z_order=7,
                                  properties={'hp': 4})

    # Check if wave survived (no enemies left after a wave)
    if player.properties['wave'] > 0:
        enemies = env.get_entities_by_type('grunt') + env.get_entities_by_type('brute')
        if not enemies and turn not in wave_turns:
            # Wave survived — only reward once
            wave_num = player.properties['wave']
            expected_clear_turn = wave_turns[wave_num - 1] if wave_num <= 5 else 0
            if turn > expected_clear_turn:
                # Check we haven't already rewarded (use score as tracker)
                wave_score_key = wave_num * 10
                if player.properties.get('score', 0) < wave_score_key:
                    player.properties['score'] = wave_score_key
                    env.emit('reward', {'amount': 1.0})


def _enemy_tick(env):
    """Process enemy attacks on built walls."""
    for enemy in list(env.get_entities_by_type('grunt') + env.get_entities_by_type('brute')):
        if env.get_entity(enemy.id) is None:
            continue
        # Check adjacent built walls and damage them
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = enemy.x + dx, enemy.y + dy
            for ent in list(env.get_entities_at(nx, ny)):
                if ent.type == 'built_wall':
                    ent.properties['hp'] = ent.properties.get('hp', 3) - 1
                    if ent.properties['hp'] <= 0:
                        env.destroy_entity(ent.id)
                    break


def _enemy_attack_interior(env, enemy):
    """Enemy attacks dwarves or destroys furniture inside fortress."""
    p = env.get_entities_by_tag('player')
    if not p:
        return
    # Attack nearest dwarf
    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]:
        nx, ny = enemy.x + dx, enemy.y + dy
        for ent in list(env.get_entities_at(nx, ny)):
            if ent.type == 'dwarf':
                ent.properties['hp'] = ent.properties.get('hp', 5) - 1
                if ent.properties['hp'] <= 0:
                    env.destroy_entity(ent.id)
                    env.emit('reward', {'amount': -2.0})
                return
            if ent.type in ('bed', 'food_store', 'workshop', 'tavern'):
                env.destroy_entity(ent.id)
                return


def _merchant_tick(env, player, turn):
    """Merchant arrives every 200 turns, stays 20 turns."""
    if turn % 200 == 0 and turn > 0:
        # Spawn merchant at fortress entrance
        existing = env.get_entities_by_type('merchant')
        if not existing:
            env.create_entity('merchant', 13, 15, 'M', ['npc'], z_order=7,
                              properties={'arrive_turn': turn})

    # Remove merchant after 20 turns
    for m in list(env.get_entities_by_type('merchant')):
        arrive = m.properties.get('arrive_turn', 0)
        if turn - arrive >= 20:
            env.destroy_entity(m.id)


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


def _tantrum_move(env, entity):
    dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    idx = int(env.random() * 4)
    dx, dy = dirs[idx]
    env.move_entity(entity.id, entity.x + dx, entity.y + dy)


# ---- Game-specific invariants ----

@invariant('six_dwarves_at_start')
def check_six_dwarves(env):
    dwarves = env.get_entities_by_type('dwarf')
    if len(dwarves) != 6:
        raise InvariantError(f"Expected 6 dwarves, found {len(dwarves)}")
    for d in dwarves:
        if d.properties.get('mood') != 10:
            raise InvariantError(
                f"Dwarf at ({d.x},{d.y}) mood is {d.properties.get('mood')}, expected 10")


@invariant('facilities_exist')
def check_facilities(env):
    beds = env.get_entities_by_type('bed')
    if len(beds) != 6:
        raise InvariantError(f"Expected 6 beds, found {len(beds)}")
    food_stores = env.get_entities_by_type('food_store')
    if len(food_stores) != 2:
        raise InvariantError(f"Expected 2 food stores, found {len(food_stores)}")
    workshops = env.get_entities_by_type('workshop')
    if len(workshops) != 1:
        raise InvariantError(f"Expected 1 workshop, found {len(workshops)}")
    taverns = env.get_entities_by_type('tavern')
    if len(taverns) != 1:
        raise InvariantError(f"Expected 1 tavern, found {len(taverns)}")


@invariant('sixteen_soil_tiles')
def check_soil(env):
    soils = env.get_entities_by_type('soil')
    if len(soils) != 16:
        raise InvariantError(f"Expected 16 soil tiles, found {len(soils)}")


@invariant('wilderness_resources')
def check_wilderness(env):
    trees = env.get_entities_by_type('tree')
    if len(trees) < 8:
        raise InvariantError(f"Expected >= 8 trees, found {len(trees)}")
    ores = env.get_entities_by_type('ore')
    if len(ores) < 6:
        raise InvariantError(f"Expected >= 6 ore, found {len(ores)}")


@invariant('no_enemies_at_start')
def check_no_enemies(env):
    grunts = env.get_entities_by_type('grunt')
    brutes = env.get_entities_by_type('brute')
    if len(grunts) + len(brutes) > 0:
        raise InvariantError(f"Expected 0 enemies, found {len(grunts) + len(brutes)}")


@invariant('fortress_entrance_exists')
def check_entrance(env):
    # Check that position (12, 15) is not blocked by a wall
    walls_at_entrance = [e for e in env.get_entities_at(12, 15) if e.type == 'wall']
    if walls_at_entrance:
        raise InvariantError("Fortress entrance at (12,15) is blocked by a wall")


@invariant('water_source_and_pump')
def check_water(env):
    ws = env.get_entities_by_type('water_source')
    if len(ws) != 1:
        raise InvariantError(f"Expected 1 water source, found {len(ws)}")
    pumps = env.get_entities_by_type('pump')
    if len(pumps) != 1:
        raise InvariantError(f"Expected 1 pump, found {len(pumps)}")


@invariant('eight_build_sites')
def check_build_sites(env):
    sites = env.get_entities_by_type('build_site')
    if len(sites) != 8:
        raise InvariantError(f"Expected 8 build sites, found {len(sites)}")
