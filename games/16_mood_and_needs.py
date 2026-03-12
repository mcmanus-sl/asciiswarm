"""Game 16: Mood & Needs — manage dwarf needs to prevent tantrum cascades while building walls."""

from asciiswarm.kernel.invariants import invariant, InvariantError

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'order_eat', 'order_sleep', 'order_socialize', 'order_build'],
    'grid': (16, 16),
    'max_turns': 500,
    'step_penalty': -0.003,
    'player_properties': [
        {'key': 'walls_built', 'max': 5},
        {'key': 'dwarves_alive', 'max': 4},
        {'key': 'avg_mood', 'max': 10},
    ],
}


def setup(env):
    w, h = 16, 16

    # ---- Outer boundary walls only ----
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    occupied = set()
    for ent in env.get_entities_by_tag('solid'):
        occupied.add((ent.x, ent.y))

    # ---- Player at center ----
    player = env.create_entity('player', 8, 8, '@', ['player'], z_order=10,
                               properties={'walls_built': 0, 'dwarves_alive': 4,
                                           'avg_mood': 10})
    occupied.add((8, 8))

    # ---- 4 dwarves near player ----
    dwarf_positions = [(7, 8), (9, 8), (8, 7), (8, 9)]
    for dx, dy in dwarf_positions:
        env.create_entity('dwarf', dx, dy, 'D', ['npc'], z_order=8,
                          properties={'hunger': 6, 'rest': 6, 'social': 6,
                                      'mood': 6, 'task': 'idle',
                                      'carrying': 0, 'tantrum_turns': 0})
        occupied.add((dx, dy))

    # ---- Facilities spread apart ----
    # 2 food stores (top-left)
    for fx, fy in [(2, 2), (3, 2)]:
        env.create_entity('food_store', fx, fy, 'F', ['npc'], z_order=5)
        occupied.add((fx, fy))

    # 4 beds (top-right)
    for bx, by in [(12, 2), (13, 2), (12, 3), (13, 3)]:
        env.create_entity('bed', bx, by, 'b', ['npc'], z_order=5)
        occupied.add((bx, by))

    # 1 tavern (bottom-center)
    env.create_entity('tavern', 8, 13, 'T', ['npc'], z_order=5)
    occupied.add((8, 13))

    # ---- 5 build sites at edges ----
    build_site_positions = [(2, 8), (14, 8), (8, 2), (8, 13), (13, 13)]
    for bsx, bsy in build_site_positions:
        if (bsx, bsy) in occupied:
            # Find adjacent empty spot
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = bsx + ddx, bsy + ddy
                if (nx, ny) not in occupied and 1 <= nx < w - 1 and 1 <= ny < h - 1:
                    bsx, bsy = nx, ny
                    break
        env.create_entity('build_site', bsx, bsy, '_', ['npc'], z_order=2)
        occupied.add((bsx, bsy))

    # ---- 8-10 stones scattered randomly ----
    num_stones = 8 + int(env.random() * 3)
    placed = 0
    while placed < num_stones:
        for _attempt in range(100):
            sx = 1 + int(env.random() * (w - 2))
            sy = 1 + int(env.random() * (h - 2))
            if (sx, sy) not in occupied:
                env.create_entity('stone', sx, sy, 'o', ['pickup'], z_order=3)
                occupied.add((sx, sy))
                placed += 1
                break

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

        elif action in ('order_eat', 'order_sleep', 'order_socialize', 'order_build'):
            task = action.replace('order_', '')
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
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]
        turn = env.turn_number

        dwarves = env.get_entities_by_type('dwarf')
        if not dwarves:
            return

        # 1. Decay needs (faster rates for more tantrum pressure)
        for d in dwarves:
            if d.properties.get('tantrum_turns', 0) > 0:
                continue
            if turn % 4 == 0:
                d.properties['hunger'] = max(0, d.properties['hunger'] - 1)
            if turn % 5 == 0:
                d.properties['rest'] = max(0, d.properties['rest'] - 1)
            if turn % 6 == 0:
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

        # 3. Recalculate mood
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

        # 4. Tantrum cascade
        while new_tantrums:
            next_wave = []
            for d in new_tantrums:
                if d.properties.get('tantrum_turns', 0) > 0:
                    continue
                d.properties['tantrum_turns'] = 3
                d.properties['task'] = 'tantrum'
                d.properties['mood'] = 0
                env.emit('reward', {'amount': -0.3})

                all_dwarves = env.get_entities_by_type('dwarf')
                for other in all_dwarves:
                    if other.id == d.id:
                        continue
                    if other.properties.get('tantrum_turns', 0) > 0:
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

        # 5. Update player properties
        dwarves = env.get_entities_by_type('dwarf')
        p.properties['dwarves_alive'] = len(dwarves)
        if dwarves:
            total_mood = sum(d.properties['mood'] for d in dwarves)
            p.properties['avg_mood'] = total_mood // len(dwarves)
        else:
            p.properties['avg_mood'] = 0

        # 6. Win/lose
        if p.properties['walls_built'] >= 5:
            env.end_game('won')
            return

        if len(dwarves) == 0:
            env.end_game('lost')
            return

    env.on('turn_end', on_turn_end)

    # ---- Dwarf behavior ----
    def dwarf_behavior(entity, env):
        task = entity.properties.get('task', 'idle')
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]

        if task == 'idle':
            return

        elif task == 'tantrum':
            _tantrum_move(env, entity)
            return

        elif task == 'eat':
            food_stores = env.get_entities_by_type('food_store')
            if not food_stores:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, food_stores)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    entity.properties['hunger'] = 10
                    entity.properties['task'] = 'idle'
                    env.emit('reward', {'amount': 0.1})
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
                    env.emit('reward', {'amount': 0.1})
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'socialize':
            taverns = env.get_entities_by_type('tavern')
            if not taverns:
                entity.properties['task'] = 'idle'
                return
            nearest = _nearest_entity(entity, taverns)
            if nearest:
                dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                if dist <= 1:
                    entity.properties['social'] = 10
                    entity.properties['task'] = 'idle'
                    env.emit('reward', {'amount': 0.1})
                else:
                    _move_toward(env, entity, nearest.x, nearest.y)

        elif task == 'build':
            carrying = entity.properties.get('carrying', 0)
            if carrying == 0:
                stones = env.get_entities_by_type('stone')
                if not stones:
                    entity.properties['task'] = 'idle'
                    return
                nearest = _nearest_entity(entity, stones)
                if nearest:
                    dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                    if dist <= 1:
                        env.destroy_entity(nearest.id)
                        entity.properties['carrying'] = 1
                        env.emit('reward', {'amount': 0.15})
                    else:
                        _move_toward(env, entity, nearest.x, nearest.y)
            else:
                sites = env.get_entities_by_type('build_site')
                if not sites:
                    entity.properties['task'] = 'idle'
                    return
                nearest = _nearest_entity(entity, sites)
                if nearest:
                    dist = abs(nearest.x - entity.x) + abs(nearest.y - entity.y)
                    if dist <= 1:
                        bx, by = nearest.x, nearest.y
                        env.destroy_entity(nearest.id)
                        env.create_entity('built_wall', bx, by, '=', ['solid'],
                                          z_order=5)
                        entity.properties['carrying'] = 0
                        p.properties['walls_built'] += 1
                        env.emit('reward', {'amount': 0.8})
                    else:
                        _move_toward(env, entity, nearest.x, nearest.y)

    env.register_behavior('dwarf', dwarf_behavior)


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
    nx, ny = entity.x + dx, entity.y + dy
    for ent in list(env.get_entities_at(nx, ny)):
        if ent.has_tag('solid') and ent.type == 'wall':
            continue
        if ent.has_tag('player'):
            continue
        if ent.type == 'dwarf':
            continue
        if ent.type in ('food_store', 'bed', 'build_site', 'stone', 'tavern', 'built_wall'):
            env.destroy_entity(ent.id)
    env.move_entity(entity.id, nx, ny)


# ---- Game-specific invariants ----

@invariant('four_dwarves_at_start')
def check_four_dwarves(env):
    dwarves = env.get_entities_by_type('dwarf')
    if len(dwarves) != 4:
        raise InvariantError(f"Expected 4 dwarves, found {len(dwarves)}")
    for d in dwarves:
        if d.properties.get('mood') != 6:
            raise InvariantError(
                f"Dwarf at ({d.x},{d.y}) mood is {d.properties.get('mood')}, expected 6")


@invariant('facilities_exist')
def check_facilities(env):
    food_stores = env.get_entities_by_type('food_store')
    if len(food_stores) < 2:
        raise InvariantError(f"Expected >= 2 food stores, found {len(food_stores)}")
    beds = env.get_entities_by_type('bed')
    if len(beds) < 4:
        raise InvariantError(f"Expected >= 4 beds, found {len(beds)}")
    taverns = env.get_entities_by_type('tavern')
    if len(taverns) < 1:
        raise InvariantError(f"Expected >= 1 tavern, found {len(taverns)}")


@invariant('five_build_sites')
def check_build_sites(env):
    sites = env.get_entities_by_type('build_site')
    if len(sites) != 5:
        raise InvariantError(f"Expected 5 build sites, found {len(sites)}")


@invariant('enough_stones')
def check_stones(env):
    stones = env.get_entities_by_type('stone')
    if len(stones) < 8:
        raise InvariantError(f"Expected >= 8 stones, found {len(stones)}")


@invariant('dwarves_idle_at_start')
def check_dwarves_idle(env):
    for d in env.get_entities_by_type('dwarf'):
        if d.properties.get('task') != 'idle':
            raise InvariantError(
                f"Dwarf at ({d.x},{d.y}) task is {d.properties.get('task')}, expected 'idle'")


@invariant('walls_built_zero_at_start')
def check_walls_built_zero(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('walls_built', 0) != 0:
        raise InvariantError(
            f"Player starts with walls_built={p.properties['walls_built']}")
