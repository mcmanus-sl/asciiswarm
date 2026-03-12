"""Game 13: Ecology & Spawning — survival on a living map with rabbit/wolf population dynamics."""

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, Invariant,
)

INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (20, 20),
    'max_turns': 500,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'food', 'max': 20},
        {'key': 'health', 'max': 10},
    ],
}


def setup(env):
    w, h = env.config['grid']

    # Occupied positions tracker
    occupied = set()

    # Outer boundary walls
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

    # Scattered rock clusters (3-5 clusters of 2-3 walls)
    num_clusters = 3 + int(env.random() * 3)
    for _ in range(num_clusters):
        cx = 3 + int(env.random() * (w - 6))
        cy = 3 + int(env.random() * (h - 6))
        cluster_size = 2 + int(env.random() * 2)
        horizontal = env.random() < 0.5
        for i in range(cluster_size):
            if horizontal:
                wx, wy = cx + i, cy
            else:
                wx, wy = cx, cy + i
            if 1 <= wx < w - 1 and 1 <= wy < h - 1 and (wx, wy) not in occupied:
                env.create_entity('wall', wx, wy, '#', ['solid'], z_order=1)
                occupied.add((wx, wy))

    # Player at bottom-left corner area
    px, py = 2, h - 3
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={'food': 15, 'health': 10})
    occupied.add((px, py))

    # Exit at top-right corner area
    ex, ey = w - 3, 2
    env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)
    occupied.add((ex, ey))

    # BFS reachability check
    def bfs_reachable(start, goal):
        visited = {start}
        queue = [start]
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) == goal:
                return True
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    # Check for walls
                    is_wall = False
                    for ent in env.get_entities_at(nx, ny):
                        if ent.has_tag('solid'):
                            is_wall = True
                            break
                    if not is_wall:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    # Verify exit is reachable; if not, remove rock clusters and retry
    if not bfs_reachable((px, py), (ex, ey)):
        # Remove all non-boundary walls
        for ent in list(env.get_entities_by_type('wall')):
            if 1 <= ent.x < w - 1 and 1 <= ent.y < h - 1:
                env.destroy_entity(ent.id)
                occupied.discard((ent.x, ent.y))

    def _random_empty_cell(avoid=None):
        """Find a random empty cell inside the boundary."""
        if avoid is None:
            avoid = set()
        for _ in range(500):
            rx = 1 + int(env.random() * (w - 2))
            ry = 1 + int(env.random() * (h - 2))
            if (rx, ry) not in occupied and (rx, ry) not in avoid:
                has_solid = False
                for ent in env.get_entities_at(rx, ry):
                    if ent.has_tag('solid'):
                        has_solid = True
                        break
                if not has_solid:
                    return rx, ry
        return None

    # Bushes: 5-7 scattered
    num_bushes = 5 + int(env.random() * 3)
    for _ in range(num_bushes):
        pos = _random_empty_cell()
        if pos:
            env.create_entity('bush', pos[0], pos[1], '*', ['npc'], z_order=2)
            occupied.add(pos)

    # Rabbits: 8-10 scattered
    num_rabbits = 8 + int(env.random() * 3)
    for _ in range(num_rabbits):
        pos = _random_empty_cell()
        if pos:
            env.create_entity('rabbit', pos[0], pos[1], 'r', ['npc'], z_order=5,
                              properties={'age': 0})
            occupied.add(pos)

    # Wolves: 2-3, distance >= 8 from player
    num_wolves = 2 + int(env.random() * 2)
    for _ in range(num_wolves):
        for _attempt in range(500):
            pos = _random_empty_cell()
            if pos:
                dist = abs(pos[0] - px) + abs(pos[1] - py)
                if dist >= 8:
                    env.create_entity('wolf', pos[0], pos[1], 'W', ['hazard'], z_order=5,
                                      properties={'health': 3, 'age': 0})
                    occupied.add(pos)
                    break

    # ---- Input handler ----
    def on_input(event):
        players = env.get_entities_by_tag('player')
        if not players:
            return
        p = players[0]
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

    # ---- Before move: solid blocking ----
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
                env.end_game('won')
                return

            # Player hunts rabbit
            if mover.has_tag('player') and occ.type == 'rabbit':
                env.destroy_entity(occ.id)
                mover.properties['food'] = min(mover.properties['food'] + 3, 20)
                env.emit('reward', {'amount': 0.1})

            # Wolf attacks player (wolf moves into player)
            elif mover.type == 'wolf' and occ.has_tag('player'):
                event.cancel()
                occ.properties['health'] -= 2
                mover.properties['health'] -= 1
                if mover.properties['health'] <= 0:
                    env.destroy_entity(mover.id)
                    env.emit('reward', {'amount': 0.2})
                if occ.properties['health'] <= 0:
                    env.end_game('lost')
                    return

            # Player attacks wolf (player moves into wolf)
            elif mover.has_tag('player') and occ.type == 'wolf':
                event.cancel()
                mover.properties['health'] -= 2
                occ.properties['health'] -= 1
                if occ.properties['health'] <= 0:
                    env.destroy_entity(occ.id)
                    env.emit('reward', {'amount': 0.2})
                if mover.properties['health'] <= 0:
                    env.end_game('lost')
                    return

            # Wolf hunts rabbit
            elif mover.type == 'wolf' and occ.type == 'rabbit':
                env.destroy_entity(occ.id)
                mover.properties['age'] = 0  # fed, reset age

    env.on('collision', on_collision)

    # ---- Rabbit behavior ----
    def rabbit_behavior(entity, env):
        rabbits = env.get_entities_by_type('rabbit')
        wolves = env.get_entities_by_type('wolf')

        # Increment age
        entity.properties['age'] = entity.properties.get('age', 0) + 1

        # Check for nearby wolves (flee)
        nearest_wolf = None
        nearest_wolf_dist = float('inf')
        for wolf in wolves:
            dist = abs(wolf.x - entity.x) + abs(wolf.y - entity.y)
            if dist <= 3 and dist < nearest_wolf_dist:
                nearest_wolf = wolf
                nearest_wolf_dist = dist

        if nearest_wolf:
            # Flee: move away from nearest wolf
            dx = entity.x - nearest_wolf.x
            dy = entity.y - nearest_wolf.y
            # Pick the larger axis to flee along
            if abs(dx) >= abs(dy):
                move_dx = 1 if dx >= 0 else -1
                move_dy = 0
            else:
                move_dx = 0
                move_dy = 1 if dy >= 0 else -1
            env.move_entity(entity.id, entity.x + move_dx, entity.y + move_dy)
        else:
            # Random movement
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            idx = int(env.random() * 4)
            dx, dy = dirs[idx]
            env.move_entity(entity.id, entity.x + dx, entity.y + dy)

        # Reproduction
        if entity.properties.get('age', 0) >= 20 and len(rabbits) < 15:
            # Check for nearby bush
            bushes = env.get_entities_by_type('bush')
            near_bush = False
            for bush in bushes:
                if abs(bush.x - entity.x) + abs(bush.y - entity.y) <= 2:
                    near_bush = True
                    break

            if near_bush:
                # Find adjacent empty cell
                dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                # Shuffle using env.random
                for i in range(3, 0, -1):
                    j = int(env.random() * (i + 1))
                    dirs[i], dirs[j] = dirs[j], dirs[i]

                for ddx, ddy in dirs:
                    nx, ny = entity.x + ddx, entity.y + ddy
                    if 0 <= nx < w and 0 <= ny < h:
                        entities_at = env.get_entities_at(nx, ny)
                        if not entities_at:
                            env.create_entity('rabbit', nx, ny, 'r', ['npc'],
                                              z_order=5, properties={'age': 0})
                            entity.properties['age'] = 0
                            break

    env.register_behavior('rabbit', rabbit_behavior)

    # ---- Wolf behavior ----
    def wolf_behavior(entity, env):
        wolves = env.get_entities_by_type('wolf')
        rabbits = env.get_entities_by_type('rabbit')

        # Increment age
        entity.properties['age'] = entity.properties.get('age', 0) + 1

        # Hunt: find nearest rabbit within distance 5
        nearest_rabbit = None
        nearest_dist = float('inf')
        for rabbit in rabbits:
            dist = abs(rabbit.x - entity.x) + abs(rabbit.y - entity.y)
            if dist <= 5 and dist < nearest_dist:
                nearest_rabbit = rabbit
                nearest_dist = dist

        if nearest_rabbit:
            # Move toward nearest rabbit
            dx = nearest_rabbit.x - entity.x
            dy = nearest_rabbit.y - entity.y
            if abs(dx) >= abs(dy):
                move_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
                move_dy = 0
            else:
                move_dx = 0
                move_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
            env.move_entity(entity.id, entity.x + move_dx, entity.y + move_dy)
        else:
            # Random movement
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            idx = int(env.random() * 4)
            dx, dy = dirs[idx]
            env.move_entity(entity.id, entity.x + dx, entity.y + dy)

        # Wolf reproduction: every 40 turns, if wolf count < 5
        if entity.properties.get('age', 0) >= 40 and len(wolves) < 5:
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            for i in range(3, 0, -1):
                j = int(env.random() * (i + 1))
                dirs[i], dirs[j] = dirs[j], dirs[i]

            for ddx, ddy in dirs:
                nx, ny = entity.x + ddx, entity.y + ddy
                if 0 <= nx < w and 0 <= ny < h:
                    entities_at = env.get_entities_at(nx, ny)
                    if not entities_at:
                        env.create_entity('wolf', nx, ny, 'W', ['hazard'],
                                          z_order=5, properties={'health': 3, 'age': 0})
                        entity.properties['age'] = 0
                        break

    env.register_behavior('wolf', wolf_behavior)

    # ---- Turn end: hunger clock ----
    def on_turn_end(event):
        players = env.get_entities_by_tag('player')
        if not players:
            return
        p = players[0]
        if env.turn_number % 3 == 0:
            p.properties['food'] -= 1
            if p.properties['food'] <= 0:
                env.end_game('lost')

    env.on('turn_end', on_turn_end)


# ---- Game-specific invariants ----

@invariant('rabbit_count_8_to_10')
def check_rabbit_count(env):
    rabbits = env.get_entities_by_type('rabbit')
    if not (8 <= len(rabbits) <= 10):
        raise InvariantError(f"Found {len(rabbits)} rabbits, expected 8-10")


@invariant('wolf_count_2_to_3')
def check_wolf_count(env):
    wolves = env.get_entities_by_type('wolf')
    if not (2 <= len(wolves) <= 3):
        raise InvariantError(f"Found {len(wolves)} wolves, expected 2-3")


@invariant('wolf_distance_from_player')
def check_wolf_distance(env):
    p = env.get_entities_by_tag('player')[0]
    wolves = env.get_entities_by_type('wolf')
    for wolf in wolves:
        dist = abs(wolf.x - p.x) + abs(wolf.y - p.y)
        if dist < 8:
            raise InvariantError(
                f"Wolf at ({wolf.x}, {wolf.y}) is {dist} from player, need >= 8")


@invariant('bush_count_5_to_7')
def check_bush_count(env):
    bushes = env.get_entities_by_type('bush')
    if not (5 <= len(bushes) <= 7):
        raise InvariantError(f"Found {len(bushes)} bushes, expected 5-7")


@invariant('player_starts_with_food_15_health_10')
def check_player_props(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('food') != 15:
        raise InvariantError(f"Player food is {p.properties.get('food')}, expected 15")
    if p.properties.get('health') != 10:
        raise InvariantError(f"Player health is {p.properties.get('health')}, expected 10")
