"""Game 11: Multi-Floor Dungeon — 3-floor side-by-side dungeon with stairs, combat, keys."""

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, Invariant,
)

# Exit exists but is blocked by locked_exit at setup, so only check existence.
INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (36, 12),
    'max_turns': 600,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'health', 'max': 10},
        {'key': 'keys_held', 'max': 3},
        {'key': 'floor', 'max': 3},
    ],
}

# Floor zone x-ranges
FLOOR_ZONES = [(0, 11), (12, 23), (24, 35)]


def _get_floor(x):
    """Return floor number (1-3) for a given x coordinate."""
    if x <= 11:
        return 1
    elif x <= 23:
        return 2
    else:
        return 3


def _empty_cells(env, x_min, x_max, y_min, y_max):
    """Get list of empty (non-occupied) cells in a rectangle."""
    cells = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if not env.get_entities_at(x, y):
                cells.append((x, y))
    return cells


def _pick_cell(env, cells):
    """Pick and remove a random cell from a list. Returns (x, y) or None."""
    if not cells:
        return None
    idx = int(env.random() * len(cells))
    return cells.pop(idx)


def setup(env):
    w, h = 36, 12

    # ---- Build walls: boundaries for each floor ----
    for floor_idx in range(3):
        x_min = floor_idx * 12
        x_max = x_min + 11
        for x in range(x_min, x_max + 1):
            env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
            env.create_entity('wall', x, 11, '#', ['solid'], z_order=1)
        for y in range(0, 12):
            env.create_entity('wall', x_min, y, '#', ['solid'], z_order=1)
            env.create_entity('wall', x_max, y, '#', ['solid'], z_order=1)

    # ---- Internal walls: random clusters per floor ----
    for floor_idx in range(3):
        x_min = floor_idx * 12 + 1
        x_max = floor_idx * 12 + 10
        num_clusters = 2 + int(env.random() * 2)  # 2-3 clusters
        for _ in range(num_clusters):
            # Random cluster position in interior
            cx = x_min + 1 + int(env.random() * (x_max - x_min - 2))
            cy = 2 + int(env.random() * 7)  # y in [2, 8]
            # Small L or T shaped cluster (2-4 walls)
            cluster_size = 2 + int(env.random() * 2)
            placed = [(cx, cy)]
            for _ in range(cluster_size - 1):
                if not placed:
                    break
                base = placed[int(env.random() * len(placed))]
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                d = dirs[int(env.random() * 4)]
                nx, ny = base[0] + d[0], base[1] + d[1]
                if x_min < nx < x_max and 1 < ny < 10 and (nx, ny) not in placed:
                    placed.append((nx, ny))
            for wx, wy in placed:
                if not env.get_entities_at(wx, wy):
                    env.create_entity('wall', wx, wy, '#', ['solid'], z_order=1)

    # ---- Place stairs ----
    # Floor 1 -> Floor 2: stairs_down on right side of floor 1, stairs_up on left side of floor 2
    # Floor 2 -> Floor 3: stairs_down on right side of floor 2, stairs_up on left side of floor 3

    # Stairs between floor 1 and floor 2
    stair_pairs = []

    # Floor 1 stairs_down: right side (x around 9-10)
    f1_stair_cells = _empty_cells(env, 8, 10, 2, 9)
    pos = _pick_cell(env, f1_stair_cells)
    if not pos:
        pos = (9, 5)
        for e in env.get_entities_at(9, 5):
            env.destroy_entity(e.id)
    stairs_down_1 = env.create_entity('stairs_down', pos[0], pos[1], '<', ['npc'], z_order=5,
                                       properties={'target_floor': 2})

    # Floor 2 stairs_up: left side (x around 13-14)
    f2_stair_up_cells = _empty_cells(env, 13, 15, 2, 9)
    pos = _pick_cell(env, f2_stair_up_cells)
    if not pos:
        pos = (13, 5)
        for e in env.get_entities_at(13, 5):
            env.destroy_entity(e.id)
    stairs_up_2 = env.create_entity('stairs_up', pos[0], pos[1], '>', ['npc'], z_order=5,
                                     properties={'target_floor': 1})
    stair_pairs.append((stairs_down_1, stairs_up_2))

    # Floor 2 stairs_down: right side (x around 21-22)
    f2_stair_down_cells = _empty_cells(env, 20, 22, 2, 9)
    pos = _pick_cell(env, f2_stair_down_cells)
    if not pos:
        pos = (21, 5)
        for e in env.get_entities_at(21, 5):
            env.destroy_entity(e.id)
    stairs_down_2 = env.create_entity('stairs_down', pos[0], pos[1], '<', ['npc'], z_order=5,
                                       properties={'target_floor': 3})

    # Floor 3 stairs_up: left side (x around 25-26)
    f3_stair_up_cells = _empty_cells(env, 25, 27, 2, 9)
    pos = _pick_cell(env, f3_stair_up_cells)
    if not pos:
        pos = (25, 5)
        for e in env.get_entities_at(25, 5):
            env.destroy_entity(e.id)
    stairs_up_3 = env.create_entity('stairs_up', pos[0], pos[1], '>', ['npc'], z_order=5,
                                     properties={'target_floor': 2})
    stair_pairs.append((stairs_down_2, stairs_up_3))

    # Store stair pair mapping: stairs_down.id -> stairs_up entity, stairs_up.id -> stairs_down entity
    stair_map = {}
    for sd, su in stair_pairs:
        stair_map[sd.id] = su
        stair_map[su.id] = sd

    # ---- Place player in floor 1 ----
    f1_cells = _empty_cells(env, 1, 10, 1, 10)
    pos = _pick_cell(env, f1_cells)
    if not pos:
        pos = (5, 5)
    player = env.create_entity('player', pos[0], pos[1], '@', ['player'], z_order=10,
                               properties={'health': 10, 'keys_held': 0, 'floor': 1})

    # ---- Place potions (1 per floor) ----
    for floor_idx in range(3):
        x_min = floor_idx * 12 + 1
        x_max = floor_idx * 12 + 10
        cells = _empty_cells(env, x_min, x_max, 1, 10)
        pos = _pick_cell(env, cells)
        if pos:
            env.create_entity('potion', pos[0], pos[1], '!', ['pickup'], z_order=3,
                              properties={'heal_amount': 3})

    # ---- Place key on floor 2 ----
    f2_cells = _empty_cells(env, 13, 22, 1, 10)
    pos = _pick_cell(env, f2_cells)
    if pos:
        env.create_entity('floor_key', pos[0], pos[1], 'k', ['pickup'], z_order=3)

    # ---- Place locked exit and exit on floor 3 ----
    # Exit on the right side of floor 3, locked_exit blocks access
    # Place exit at x=34, locked_exit at x=33 (same y)
    exit_y = 2 + int(env.random() * 7)  # y in [2, 8]
    # Clear cells for exit placement
    for e in env.get_entities_at(34, exit_y):
        env.destroy_entity(e.id)
    for e in env.get_entities_at(33, exit_y):
        env.destroy_entity(e.id)
    env.create_entity('locked_exit', 33, exit_y, '+', ['solid'], z_order=5)
    env.create_entity('exit', 34, exit_y, 'E', ['exit'], z_order=5)

    # ---- Place enemies ----
    # Floor 1: 1-2 wanderers
    f1_enemy_cells = _empty_cells(env, 1, 10, 1, 10)
    num_wanderers = 1 + int(env.random() * 2)
    for _ in range(num_wanderers):
        pos = _pick_cell(env, f1_enemy_cells)
        if pos:
            env.create_entity('wanderer', pos[0], pos[1], 'w', ['hazard'], z_order=5,
                              properties={'health': 1, 'attack': 1})

    # Floor 2: 2-3 chasers
    f2_enemy_cells = _empty_cells(env, 13, 22, 1, 10)
    num_chasers_f2 = 2 + int(env.random() * 2)
    for _ in range(num_chasers_f2):
        pos = _pick_cell(env, f2_enemy_cells)
        if pos:
            env.create_entity('chaser', pos[0], pos[1], 'c', ['hazard'], z_order=5,
                              properties={'health': 2, 'attack': 2})

    # Floor 3: 1 sentinel + 2 chasers
    f3_enemy_cells = _empty_cells(env, 25, 34, 1, 10)
    pos = _pick_cell(env, f3_enemy_cells)
    if pos:
        env.create_entity('sentinel', pos[0], pos[1], 's', ['hazard'], z_order=5,
                          properties={'health': 3, 'attack': 1})
    for _ in range(2):
        pos = _pick_cell(env, f3_enemy_cells)
        if pos:
            env.create_entity('chaser', pos[0], pos[1], 'c', ['hazard'], z_order=5,
                              properties={'health': 2, 'attack': 2})

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
            _handle_interact(env, p, stair_map)

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
                    # Pickup handling
                    if occ.type == 'potion':
                        heal = occ.properties.get('heal_amount', 3)
                        mover.properties['health'] = min(
                            mover.properties['health'] + heal, 10)
                    elif occ.type == 'floor_key':
                        mover.properties['keys_held'] = mover.properties.get('keys_held', 0) + 1
                    env.destroy_entity(occ.id)
                    env.emit('reward', {'amount': 0.1})

                elif occ.has_tag('exit'):
                    env.emit('reward', {'amount': 1.0})
                    env.end_game('won')

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
        # Only chase if on same floor
        entity_floor = _get_floor(entity.x)
        player_floor = p.properties.get('floor', 1)
        dist = abs(entity.x - p.x) + abs(entity.y - p.y)
        if entity_floor == player_floor and dist <= 6:
            dx_abs = abs(p.x - entity.x)
            dy_abs = abs(p.y - entity.y)
            if dx_abs > dy_abs:
                step_x = 1 if p.x > entity.x else -1
                env.move_entity(entity.id, entity.x + step_x, entity.y)
            elif dy_abs > dx_abs:
                step_y = 1 if p.y > entity.y else -1
                env.move_entity(entity.id, entity.x, entity.y + step_y)
            else:
                if env.random() < 0.5:
                    step_x = 1 if p.x > entity.x else -1
                    env.move_entity(entity.id, entity.x + step_x, entity.y)
                else:
                    step_y = 1 if p.y > entity.y else -1
                    env.move_entity(entity.id, entity.x, entity.y + step_y)
        else:
            wanderer_behavior(entity, env)

    env.register_behavior('chaser', chaser_behavior)

    def sentinel_behavior(entity, env):
        players = env.get_entities_by_tag('player')
        if not players:
            return
        p = players[0]
        entity_floor = _get_floor(entity.x)
        player_floor = p.properties.get('floor', 1)
        dist = abs(entity.x - p.x) + abs(entity.y - p.y)
        if entity_floor == player_floor and dist <= 3:
            env.emit('sentinel_alert', {'x': entity.x, 'y': entity.y})

    env.register_behavior('sentinel', sentinel_behavior)


def _handle_interact(env, player, stair_map):
    """Handle interact action: stairs and locked_exit."""
    px, py = player.x, player.y

    # Check if standing on stairs
    for ent in env.get_entities_at(px, py):
        if ent.type == 'stairs_down' and ent.id in stair_map:
            target = stair_map[ent.id]
            # Verify target still exists
            if env.get_entity(target.id) is None:
                continue
            # Teleport: remove from grid, set coords, add to new grid
            env._grid[player.y][player.x].remove(player)
            player.x = target.x
            player.y = target.y
            env._grid[player.y][player.x].append(player)
            player.properties['floor'] = _get_floor(player.x)
            env.emit('reward', {'amount': 0.2})
            return

        if ent.type == 'stairs_up' and ent.id in stair_map:
            target = stair_map[ent.id]
            if env.get_entity(target.id) is None:
                continue
            env._grid[player.y][player.x].remove(player)
            player.x = target.x
            player.y = target.y
            env._grid[player.y][player.x].append(player)
            player.properties['floor'] = _get_floor(player.x)
            return

    # Check 4 cardinal neighbors for locked_exit
    if player.properties.get('keys_held', 0) >= 1:
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            for ent in env.get_entities_at(nx, ny):
                if ent.type == 'locked_exit':
                    env.destroy_entity(ent.id)
                    player.properties['keys_held'] -= 1
                    env.emit('reward', {'amount': 0.3})
                    return


# ---- Game-specific invariants ----

@invariant('grid_36x12')
def check_grid_size(env):
    w, h = env.config['grid']
    if w != 36 or h != 12:
        raise InvariantError(f"Grid is {w}x{h}, expected 36x12")


@invariant('player_in_floor1')
def check_player_floor1(env):
    p = env.get_entities_by_tag('player')[0]
    if p.x > 11:
        raise InvariantError(f"Player starts at x={p.x}, expected x < 12 (floor 1)")


@invariant('floor_boundaries')
def check_floor_boundaries(env):
    """Each floor has wall boundaries."""
    walls = {(e.x, e.y) for e in env.get_entities_by_tag('solid')}
    for floor_idx in range(3):
        x_min = floor_idx * 12
        x_max = x_min + 11
        for y in range(12):
            if (x_min, y) not in walls:
                raise InvariantError(f"Missing wall at floor {floor_idx+1} left boundary ({x_min}, {y})")
            if (x_max, y) not in walls:
                raise InvariantError(f"Missing wall at floor {floor_idx+1} right boundary ({x_max}, {y})")
        for x in range(x_min, x_max + 1):
            if (x, 0) not in walls:
                raise InvariantError(f"Missing wall at top boundary ({x}, 0)")
            if (x, 11) not in walls:
                raise InvariantError(f"Missing wall at bottom boundary ({x}, 11)")


@invariant('stair_connections')
def check_stairs(env):
    sd = env.get_entities_by_type('stairs_down')
    su = env.get_entities_by_type('stairs_up')
    if len(sd) < 2:
        raise InvariantError(f"Expected at least 2 stairs_down, found {len(sd)}")
    if len(su) < 2:
        raise InvariantError(f"Expected at least 2 stairs_up, found {len(su)}")


@invariant('one_floor_key')
def check_one_key(env):
    keys = env.get_entities_by_type('floor_key')
    if len(keys) != 1:
        raise InvariantError(f"Expected exactly 1 floor_key, found {len(keys)}")
    k = keys[0]
    if not (12 <= k.x <= 23):
        raise InvariantError(f"floor_key at x={k.x}, expected on floor 2 (x in [12,23])")


@invariant('locked_exit_and_exit_on_floor3')
def check_exit_floor3(env):
    exits = env.get_entities_by_tag('exit')
    if len(exits) != 1:
        raise InvariantError(f"Expected 1 exit, found {len(exits)}")
    ex = exits[0]
    if not (24 <= ex.x <= 35):
        raise InvariantError(f"Exit at x={ex.x}, expected on floor 3")
    locked = env.get_entities_by_type('locked_exit')
    if len(locked) != 1:
        raise InvariantError(f"Expected 1 locked_exit, found {len(locked)}")
    le = locked[0]
    if not (24 <= le.x <= 35):
        raise InvariantError(f"locked_exit at x={le.x}, expected on floor 3")
