"""Game 14: Fluid & Pressure — navigate a flooding mine, activate pumps, and reach the exit."""

from collections import deque

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, Invariant,
)

# Exit exists but may be initially behind flooded area — only check existence
INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (20, 16),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'air', 'max': 10},
    ],
}

# Maximum water tiles on the map — prevents runaway entity growth
_MAX_WATER = 60


def _bfs_reachable(w, h, solid_set, start, goal):
    """BFS check ignoring water (hazard), only blocked by solids."""
    visited = {start}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            return True
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and (nx, ny) not in solid_set:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def setup(env):
    w, h = env.config['grid']  # 20, 16

    # Build mine layout: border walls + internal tunnel walls
    occupied = set()

    # Border walls
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
        occupied.add((x, 0))
        occupied.add((x, h - 1))
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)
        occupied.add((0, y))
        occupied.add((w - 1, y))

    # Internal walls: create mine tunnels with some horizontal barriers
    wall_positions = set()
    num_barriers = 3 + int(env.random() * 3)  # 3-5

    for i in range(num_barriers):
        bx_base = 4 + int(env.random() * (w - 8))
        by = 3 + int(env.random() * (h - 6))
        barrier_len = 3 + int(env.random() * 4)  # 3-6 length
        gap_pos = int(env.random() * barrier_len)

        for j in range(barrier_len):
            if j == gap_pos:
                continue
            if env.random() < 0.5:
                wx, wy = bx_base + j, by
            else:
                wx, wy = bx_base, by + j

            # Consume random for determinism even if unused
            env.random()

            if 1 <= wx < w - 1 and 1 <= wy < h - 1 and (wx, wy) not in occupied:
                wall_positions.add((wx, wy))

    # Player on left side (dry area)
    px, py = 1, h // 2
    occupied.add((px, py))

    # Exit on far right
    ex, ey = w - 2, h // 2
    occupied.add((ex, ey))

    # Remove walls near player/exit
    wall_positions.discard((px, py))
    wall_positions.discard((ex, ey))
    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
        wall_positions.discard((px + dx, py + dy))
        wall_positions.discard((ex + dx, ey + dy))

    # Ensure path exists
    solid_set = set(occupied) | wall_positions
    if not _bfs_reachable(w, h, solid_set, (px, py), (ex, ey)):
        walls_list = list(wall_positions)
        for _ in range(len(walls_list)):
            idx = int(env.random() * len(walls_list))
            removed = walls_list.pop(idx)
            test_solid = set(occupied) | set(walls_list)
            if _bfs_reachable(w, h, test_solid, (px, py), (ex, ey)):
                wall_positions = set(walls_list)
                break
        else:
            wall_positions = set()

    for wx, wy in wall_positions:
        env.create_entity('wall', wx, wy, '#', ['solid'], z_order=1)
        occupied.add((wx, wy))

    # Create player
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={'air': 10})

    # Create exit
    env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)

    # Water sources: 1-2 in upper area
    num_sources = 1 + int(env.random() * 2)  # 1-2
    source_positions = []
    for _ in range(num_sources):
        for _attempt in range(100):
            sx = 5 + int(env.random() * (w - 10))
            sy = 1 + int(env.random() * (h // 3))
            if (sx, sy) not in occupied:
                source_positions.append((sx, sy))
                occupied.add((sx, sy))
                break

    for sx, sy in source_positions:
        env.create_entity('water_source', sx, sy, 'S', ['npc'], z_order=5)

    # Pumps: 2-3
    num_pumps = 2 + int(env.random() * 2)  # 2-3
    for _ in range(num_pumps):
        for _attempt in range(100):
            pump_x = 3 + int(env.random() * (w - 6))
            pump_y = 2 + int(env.random() * (h - 4))
            if (pump_x, pump_y) not in occupied:
                env.create_entity('pump', pump_x, pump_y, 'P', ['npc'], z_order=5,
                                  properties={'active': 0})
                occupied.add((pump_x, pump_y))
                break

    # Drains: 2-3
    num_drains = 2 + int(env.random() * 2)  # 2-3
    for _ in range(num_drains):
        for _attempt in range(100):
            drain_x = 3 + int(env.random() * (w - 6))
            drain_y = 2 + int(env.random() * (h - 4))
            if (drain_x, drain_y) not in occupied:
                env.create_entity('drain', drain_x, drain_y, 'D', ['npc'], z_order=2)
                occupied.add((drain_x, drain_y))
                break

    # Valve: 1
    for _attempt in range(100):
        vx = w // 3 + int(env.random() * (w // 3))
        vy = 1 + int(env.random() * (h - 2))
        if (vx, vy) not in occupied:
            env.create_entity('valve', vx, vy, 'V', ['npc'], z_order=5)
            occupied.add((vx, vy))
            break

    # ---- Helper: check if cell is open for water ----
    def _can_place_water(nx, ny):
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return False
        for ent in env.get_entities_at(nx, ny):
            if ent.has_tag('solid') or ent.type == 'water':
                return False
        return True

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
            for ddx, ddy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = p.x + ddx, p.y + ddy
                for ent in env.get_entities_at(nx, ny):
                    if ent.type == 'pump':
                        ent.properties['active'] = 1 - ent.properties.get('active', 0)
                        env.emit('reward', {'amount': 0.1})
                        return
                    elif ent.type == 'valve':
                        sources = env.get_entities_by_type('water_source')
                        if sources:
                            nearest = min(sources, key=lambda s: abs(s.x - p.x) + abs(s.y - p.y))
                            env.destroy_entity(nearest.id)
                            env.emit('reward', {'amount': 0.3})
                        env.destroy_entity(ent.id)
                        return

    env.on('input', on_input)

    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    def on_collision(event):
        mover = event.payload['mover']
        if not mover.has_tag('player'):
            return
        for occ in event.payload['occupants']:
            if occ.has_tag('exit'):
                env.end_game('won')

    env.on('collision', on_collision)

    def on_turn_end(event):
        if env.status != 'playing':
            return

        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]

        # Drowning check
        on_water = False
        for ent in env.get_entities_at(p.x, p.y):
            if ent.type == 'water':
                on_water = True
                break

        if on_water:
            p.properties['air'] = p.properties.get('air', 10) - 1
            if p.properties['air'] <= 0:
                env.end_game('lost')
        else:
            p.properties['air'] = 10

        # Drain cleanup
        for drain in env.get_entities_by_type('drain'):
            for ent in list(env.get_entities_at(drain.x, drain.y)):
                if ent.type == 'water':
                    env.destroy_entity(ent.id)

    env.on('turn_end', on_turn_end)

    # ---- Behaviors ----

    # Water source: spawns 1 water per turn in adjacent empty cell
    def water_source_behavior(entity, env):
        if env.status != 'playing':
            return
        if len(env.get_entities_by_type('water')) >= _MAX_WATER:
            return
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = entity.x + dx, entity.y + dy
            if _can_place_water(nx, ny):
                neighbors.append((nx, ny))
        if neighbors:
            idx = int(env.random() * len(neighbors))
            nx, ny = neighbors[idx]
            env.create_entity('water', nx, ny, '~', ['hazard'], z_order=3,
                              properties={'depth': 1, 'born': env.turn_number})

    env.register_behavior('water_source', water_source_behavior)

    # Water: spreads to 1 neighbor, but only every 3 turns after creation
    # and only with 30% probability — this makes flooding gradual, not exponential
    def water_behavior(entity, env):
        if env.status != 'playing':
            return
        if len(env.get_entities_by_type('water')) >= _MAX_WATER:
            return
        # Only spread if water is at least 3 turns old
        born = entity.properties.get('born', 0)
        if env.turn_number - born < 3:
            return
        # 30% chance to spread per turn (makes flooding gradual)
        if env.random() > 0.3:
            return
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = entity.x + dx, entity.y + dy
            if _can_place_water(nx, ny):
                neighbors.append((nx, ny))
        if neighbors:
            idx = int(env.random() * len(neighbors))
            nx, ny = neighbors[idx]
            env.create_entity('water', nx, ny, '~', ['hazard'], z_order=3,
                              properties={'depth': 1, 'born': env.turn_number})

    env.register_behavior('water', water_behavior)

    # Pump: destroy water within Manhattan distance 2 when active
    def pump_behavior(entity, env):
        if env.status != 'playing':
            return
        if entity.properties.get('active', 0) != 1:
            return
        for water in list(env.get_entities_by_type('water')):
            dist = abs(water.x - entity.x) + abs(water.y - entity.y)
            if dist <= 2:
                env.destroy_entity(water.id)

    env.register_behavior('pump', pump_behavior)


# ---- Game-specific invariants ----

@invariant('water_sources_1_to_2')
def check_water_sources(env):
    sources = env.get_entities_by_type('water_source')
    if not (1 <= len(sources) <= 2):
        raise InvariantError(f"Expected 1-2 water sources, found {len(sources)}")


@invariant('pumps_2_to_3_all_inactive')
def check_pumps(env):
    pumps = env.get_entities_by_type('pump')
    if not (2 <= len(pumps) <= 3):
        raise InvariantError(f"Expected 2-3 pumps, found {len(pumps)}")
    for p in pumps:
        if p.properties.get('active', 0) != 0:
            raise InvariantError(f"Pump at ({p.x}, {p.y}) is active at start")


@invariant('drains_2_to_3')
def check_drains(env):
    drains = env.get_entities_by_type('drain')
    if not (2 <= len(drains) <= 3):
        raise InvariantError(f"Expected 2-3 drains, found {len(drains)}")


@invariant('valve_exists')
def check_valve(env):
    valves = env.get_entities_by_type('valve')
    if len(valves) != 1:
        raise InvariantError(f"Expected 1 valve, found {len(valves)}")


@invariant('player_starts_dry')
def check_player_dry(env):
    p = env.get_entities_by_tag('player')[0]
    for ent in env.get_entities_at(p.x, p.y):
        if ent.type == 'water':
            raise InvariantError("Player starts on water")


@invariant('player_starts_with_air_10')
def check_player_air(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('air') != 10:
        raise InvariantError(f"Player air is {p.properties.get('air')}, expected 10")
