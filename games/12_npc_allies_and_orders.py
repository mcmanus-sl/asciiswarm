"""Game 12: NPC Allies & Orders — command allies to harvest, guard, and build while reaching the exit."""

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, Invariant,
)

# Exit exists but may be behind raiders, so only check existence.
INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'order_follow', 'order_guard', 'order_harvest'],
    'grid': (20, 20),
    'max_turns': 500,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'wood_stockpile', 'max': 10},
        {'key': 'allies_alive', 'max': 3},
    ],
}


def setup(env):
    w, h = 20, 20

    # ---- Outer boundary walls ----
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    # ---- Base room: small room at left side (x=1..4, y=8..12) ----
    base_x_min, base_x_max = 1, 4
    base_y_min, base_y_max = 8, 12
    # Walls around base room (except outer boundary already placed and entrance)
    for x in range(base_x_min, base_x_max + 1):
        if not env.get_entities_at(x, base_y_min):
            env.create_entity('wall', x, base_y_min, '#', ['solid'], z_order=1)
        if not env.get_entities_at(x, base_y_max):
            env.create_entity('wall', x, base_y_max, '#', ['solid'], z_order=1)
    for y in range(base_y_min + 1, base_y_max):
        if not env.get_entities_at(base_x_max, y):
            env.create_entity('wall', base_x_max, y, '#', ['solid'], z_order=1)
    # Entrance: open the wall at (4, 10) - center of right wall
    for ent in env.get_entities_at(base_x_max, 10):
        if ent.has_tag('solid'):
            env.destroy_entity(ent.id)

    # ---- Player at (2, 10) ----
    player = env.create_entity('player', 2, 10, '@', ['player'], z_order=10,
                               properties={'health': 5, 'attack': 1,
                                           'wood_stockpile': 0, 'allies_alive': 3})

    # ---- Stockpile inside base ----
    env.create_entity('stockpile', 1, 9, 'S', ['npc'], z_order=5,
                      properties={})

    # ---- 3 allies inside base room ----
    ally_positions = [(2, 9), (3, 9), (3, 10)]
    for i, (ax, ay) in enumerate(ally_positions):
        # Clear if occupied
        for ent in env.get_entities_at(ax, ay):
            if ent.has_tag('solid'):
                env.destroy_entity(ent.id)
        env.create_entity('ally', ax, ay, 'A', ['npc'], z_order=8,
                          properties={'mode': 'follow', 'health': 3,
                                      'attack': 1, 'carrying': 0})

    # ---- Barricade slots at x=14, y=9,10,11 ----
    for by in [9, 10, 11]:
        env.create_entity('barricade_slot', 14, by, '_', ['npc'], z_order=2,
                          properties={})

    # ---- Trees in middle zone (5 <= x <= 13) ----
    num_trees = 6 + int(env.random() * 3)  # 6-8
    occupied = set()
    occupied.add((2, 10))  # player
    for ax, ay in ally_positions:
        occupied.add((ax, ay))
    occupied.add((1, 9))  # stockpile
    for by in [9, 10, 11]:
        occupied.add((14, by))  # barricade slots
    occupied.add((19 - 1, 10))  # exit area

    # Add all wall positions to occupied
    for ent in env.get_entities_by_tag('solid'):
        occupied.add((ent.x, ent.y))

    tree_count = 0
    for _ in range(num_trees):
        for _attempt in range(100):
            tx = 5 + int(env.random() * 9)  # 5 to 13
            ty = 1 + int(env.random() * (h - 2))  # 1 to 18
            if (tx, ty) not in occupied:
                env.create_entity('tree', tx, ty, 'T', ['pickup'], z_order=3,
                                  properties={})
                occupied.add((tx, ty))
                tree_count += 1
                break

    # ---- Raiders on right side (x >= 15) ----
    num_raiders = 2 + int(env.random() * 2)  # 2-3
    for _ in range(num_raiders):
        for _attempt in range(100):
            rx = 15 + int(env.random() * 4)  # 15-18
            ry = 1 + int(env.random() * (h - 2))
            if (rx, ry) not in occupied:
                env.create_entity('raider', rx, ry, 'r', ['hazard'], z_order=5,
                                  properties={'health': 2, 'attack': 1})
                occupied.add((rx, ry))
                break

    # ---- Exit at far right ----
    # Clear any entity at exit position
    exit_x, exit_y = 18, 10
    for ent in env.get_entities_at(exit_x, exit_y):
        if not ent.has_tag('solid') or ent.type == 'wall':
            pass
    env.create_entity('exit', exit_x, exit_y, '>', ['exit'], z_order=5)

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

        elif action == 'interact':
            # Check if adjacent to barricade_slot and have enough wood
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = p.x + dx, p.y + dy
                for ent in env.get_entities_at(nx, ny):
                    if ent.type == 'barricade_slot':
                        if p.properties.get('wood_stockpile', 0) >= 2:
                            # Check no barricade already there
                            has_barricade = False
                            for e2 in env.get_entities_at(nx, ny):
                                if e2.type == 'barricade':
                                    has_barricade = True
                                    break
                            if not has_barricade:
                                p.properties['wood_stockpile'] -= 2
                                env.create_entity('barricade', nx, ny, '=',
                                                  ['solid'], z_order=5)
                                env.emit('reward', {'amount': 0.2})
                                return

        elif action in ('order_follow', 'order_guard', 'order_harvest'):
            mode = action.replace('order_', '')
            # Find nearest ally within 5 tiles
            allies = env.get_entities_by_type('ally')
            best_ally = None
            best_dist = float('inf')
            for a in allies:
                d = abs(a.x - p.x) + abs(a.y - p.y)
                if d <= 5 and d < best_dist:
                    best_dist = d
                    best_ally = a
            if best_ally:
                best_ally.properties['mode'] = mode

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
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        for occ in occupants:
            # Player reaches exit
            if mover.has_tag('player') and occ.has_tag('exit'):
                env.emit('reward', {'amount': 1.0})
                env.end_game('won')
                return

            # Player vs raider combat
            if mover.has_tag('player') and occ.has_tag('hazard'):
                event.cancel()
                _combat(env, mover, occ)
                return

            # Raider vs player combat
            if mover.has_tag('hazard') and occ.has_tag('player'):
                event.cancel()
                _combat_raider_vs_player(env, mover, occ)
                return

            # Raider vs ally combat
            if mover.has_tag('hazard') and occ.type == 'ally':
                event.cancel()
                _combat_raider_vs_ally(env, mover, occ)
                return

            # Ally vs raider combat
            if mover.type == 'ally' and occ.has_tag('hazard'):
                event.cancel()
                _combat_ally_vs_raider(env, mover, occ)
                return

    env.on('collision', on_collision)

    # ---- Behaviors ----

    def ally_behavior(entity, env):
        mode = entity.properties.get('mode', 'follow')
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]

        if mode == 'follow':
            _move_toward(env, entity, p.x, p.y)

        elif mode == 'guard':
            # If raider within 3 tiles, move toward it
            raiders = env.get_entities_by_type('raider')
            nearest_raider = None
            nearest_dist = float('inf')
            for r in raiders:
                d = abs(r.x - entity.x) + abs(r.y - entity.y)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_raider = r

            if nearest_raider and nearest_dist <= 3:
                _move_toward(env, entity, nearest_raider.x, nearest_raider.y)
            else:
                # Move toward barricade zone center (14, 10)
                _move_toward(env, entity, 14, 10)

        elif mode == 'harvest':
            carrying = entity.properties.get('carrying', 0)
            if carrying == 0:
                # Move toward nearest tree
                trees = env.get_entities_by_type('tree')
                if trees:
                    nearest_tree = None
                    nearest_dist = float('inf')
                    for t in trees:
                        d = abs(t.x - entity.x) + abs(t.y - entity.y)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_tree = t

                    if nearest_tree:
                        if nearest_dist <= 1:
                            # Adjacent: destroy tree and pick up
                            env.destroy_entity(nearest_tree.id)
                            entity.properties['carrying'] = 1
                        else:
                            _move_toward(env, entity, nearest_tree.x, nearest_tree.y)
            else:
                # Move toward stockpile
                stockpiles = env.get_entities_by_type('stockpile')
                if stockpiles:
                    s = stockpiles[0]
                    d = abs(s.x - entity.x) + abs(s.y - entity.y)
                    if d <= 1:
                        # Adjacent: deposit wood
                        entity.properties['carrying'] = 0
                        p.properties['wood_stockpile'] = min(
                            p.properties.get('wood_stockpile', 0) + 1, 10)
                        env.emit('reward', {'amount': 0.1})
                    else:
                        _move_toward(env, entity, s.x, s.y)

    env.register_behavior('ally', ally_behavior)

    def raider_behavior(entity, env):
        # Chase nearest player or ally
        targets = []
        players = env.get_entities_by_tag('player')
        allies = env.get_entities_by_type('ally')
        targets.extend(players)
        targets.extend(allies)

        if not targets:
            # Random movement
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            idx = int(env.random() * 4)
            dx, dy = dirs[idx]
            env.move_entity(entity.id, entity.x + dx, entity.y + dy)
            return

        nearest = None
        nearest_dist = float('inf')
        for t in targets:
            d = abs(t.x - entity.x) + abs(t.y - entity.y)
            if d < nearest_dist:
                nearest_dist = d
                nearest = t

        if nearest:
            _move_toward(env, entity, nearest.x, nearest.y)

    env.register_behavior('raider', raider_behavior)


def _move_toward(env, entity, tx, ty):
    """Move entity one step toward target (Manhattan, prefer axis with larger gap)."""
    dx_abs = abs(tx - entity.x)
    dy_abs = abs(ty - entity.y)

    if dx_abs == 0 and dy_abs == 0:
        return

    if dx_abs > dy_abs:
        step_x = 1 if tx > entity.x else -1
        env.move_entity(entity.id, entity.x + step_x, entity.y)
    elif dy_abs > dx_abs:
        step_y = 1 if ty > entity.y else -1
        env.move_entity(entity.id, entity.x, entity.y + step_y)
    else:
        # Tie: try x first, then y
        if dx_abs > 0:
            step_x = 1 if tx > entity.x else -1
            result = env.move_entity(entity.id, entity.x + step_x, entity.y)
            if not result:
                step_y = 1 if ty > entity.y else -1
                env.move_entity(entity.id, entity.x, entity.y + step_y)
        else:
            step_y = 1 if ty > entity.y else -1
            env.move_entity(entity.id, entity.x, entity.y + step_y)


def _combat(env, player, raider):
    """Player attacks raider: mutual damage."""
    p_atk = player.properties.get('attack', 1)
    r_atk = raider.properties.get('attack', 1)

    player.properties['health'] -= r_atk
    raider.properties['health'] -= p_atk

    if raider.properties['health'] <= 0:
        env.destroy_entity(raider.id)
        env.emit('reward', {'amount': 0.15})

    if player.properties['health'] <= 0:
        env.end_game('lost')


def _combat_raider_vs_player(env, raider, player):
    """Raider walks into player."""
    p_atk = player.properties.get('attack', 1)
    r_atk = raider.properties.get('attack', 1)

    player.properties['health'] -= r_atk
    raider.properties['health'] -= p_atk

    if raider.properties['health'] <= 0:
        env.destroy_entity(raider.id)

    if player.properties['health'] <= 0:
        env.end_game('lost')


def _combat_raider_vs_ally(env, raider, ally):
    """Raider walks into ally: mutual damage."""
    raider.properties['health'] -= 1
    ally.properties['health'] -= 1

    if raider.properties['health'] <= 0:
        env.destroy_entity(raider.id)

    if ally.properties['health'] <= 0:
        env.destroy_entity(ally.id)
        # Update allies_alive
        p = env.get_entities_by_tag('player')
        if p:
            p[0].properties['allies_alive'] = len(env.get_entities_by_type('ally'))


def _combat_ally_vs_raider(env, ally, raider):
    """Ally walks into raider: mutual damage."""
    raider.properties['health'] -= 1
    ally.properties['health'] -= 1

    if raider.properties['health'] <= 0:
        env.destroy_entity(raider.id)

    if ally.properties['health'] <= 0:
        env.destroy_entity(ally.id)
        p = env.get_entities_by_tag('player')
        if p:
            p[0].properties['allies_alive'] = len(env.get_entities_by_type('ally'))


# ---- Game-specific invariants ----

@invariant('three_allies_at_start')
def check_allies(env):
    allies = env.get_entities_by_type('ally')
    if len(allies) != 3:
        raise InvariantError(f"Expected 3 allies, found {len(allies)}")
    for a in allies:
        if a.properties.get('mode') != 'follow':
            raise InvariantError(f"Ally at ({a.x},{a.y}) mode is {a.properties.get('mode')}, expected 'follow'")


@invariant('allies_in_base')
def check_allies_in_base(env):
    for a in env.get_entities_by_type('ally'):
        if not (1 <= a.x <= 4 and 8 <= a.y <= 12):
            raise InvariantError(f"Ally at ({a.x},{a.y}) not in base room")


@invariant('raiders_exist')
def check_raiders(env):
    raiders = env.get_entities_by_type('raider')
    if len(raiders) < 2:
        raise InvariantError(f"Expected >= 2 raiders, found {len(raiders)}")


@invariant('trees_exist')
def check_trees(env):
    trees = env.get_entities_by_type('tree')
    if len(trees) < 6:
        raise InvariantError(f"Expected >= 6 trees, found {len(trees)}")


@invariant('stockpile_and_barricade_slots')
def check_stockpile_and_slots(env):
    stockpiles = env.get_entities_by_type('stockpile')
    if len(stockpiles) != 1:
        raise InvariantError(f"Expected 1 stockpile, found {len(stockpiles)}")
    slots = env.get_entities_by_type('barricade_slot')
    if len(slots) != 3:
        raise InvariantError(f"Expected 3 barricade_slots, found {len(slots)}")


@invariant('exit_at_x_18')
def check_exit_position(env):
    exits = env.get_entities_by_tag('exit')
    if not exits:
        raise InvariantError("No exit found")
    if exits[0].x != 18:
        raise InvariantError(f"Exit at x={exits[0].x}, expected x=18")
