"""Game 07: Hunger Clock — reach the exit before starving, collect food to survive."""

from asciiswarm.kernel.invariants import (
    EXIT_INVARIANTS, invariant, InvariantError,
)

INVARIANTS = list(EXIT_INVARIANTS)

GAME_CONFIG = {
    'grid': (14, 14),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'food', 'max': 20},
    ],
}


def setup(env):
    w, h = env.config['grid']

    # Player at bottom-left corner (0, 13)
    player = env.create_entity('player', 0, 13, '@', ['player'], z_order=10,
                               properties={'food': 20})

    # Exit at top-right corner (13, 0)
    env.create_entity('exit', 13, 0, '>', ['exit'], z_order=5)

    # Place wall clusters: 3-5 clusters of 2-4 walls each
    occupied = {(0, 13), (13, 0)}  # player and exit
    num_clusters = 3 + int(env.random() * 3)  # 3 to 5

    def try_place_walls():
        walls = []
        for _ in range(num_clusters):
            cluster_size = 2 + int(env.random() * 3)  # 2 to 4
            horizontal = env.random() < 0.5
            # Pick starting cell
            sx = 1 + int(env.random() * (w - 2))
            sy = 1 + int(env.random() * (h - 2))
            for i in range(cluster_size):
                if horizontal:
                    wx, wy = sx + i, sy
                else:
                    wx, wy = sx, sy + i
                if 0 <= wx < w and 0 <= wy < h and (wx, wy) not in occupied:
                    walls.append((wx, wy))
                    occupied.add((wx, wy))
        return walls

    wall_positions = try_place_walls()

    # BFS to verify exit reachable from player
    def bfs_reachable(walls_set):
        start = (0, 13)
        goal = (13, 0)
        visited = {start}
        queue = [start]
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) == goal:
                return True
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and (nx, ny) not in walls_set:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    walls_set = set(wall_positions)
    if not bfs_reachable(walls_set):
        # Regenerate up to 100 times
        for _ in range(100):
            occupied.clear()
            occupied.add((0, 13))
            occupied.add((13, 0))
            wall_positions = try_place_walls()
            walls_set = set(wall_positions)
            if bfs_reachable(walls_set):
                break
        else:
            # Fallback: no walls
            wall_positions = []
            walls_set = set()

    # Create wall entities
    for wx, wy in wall_positions:
        env.create_entity('wall', wx, wy, '#', ['solid'], z_order=1)

    # Place 10-15 food entities on random empty cells
    food_positions = set()
    num_food = 10 + int(env.random() * 6)  # 10 to 15
    for _ in range(num_food):
        for _attempt in range(200):
            fx = int(env.random() * w)
            fy = int(env.random() * h)
            if (fx, fy) not in occupied and (fx, fy) not in food_positions:
                food_positions.add((fx, fy))
                break

    for fx, fy in food_positions:
        env.create_entity('food', fx, fy, 'f', ['pickup'], z_order=5)

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

    env.on('input', on_input)

    # Before_move — solids block movement
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # Collision handler
    def on_collision(event):
        mover = event.payload['mover']
        if not mover.has_tag('player'):
            return
        for occ in event.payload['occupants']:
            if occ.has_tag('pickup'):
                # Collect food: heal 5, cap at 20
                mover.properties['food'] = min(mover.properties['food'] + 5, 20)
                env.destroy_entity(occ.id)
                env.emit('reward', {'amount': 0.05})
            elif occ.has_tag('exit'):
                env.end_game('won')

    env.on('collision', on_collision)

    # Turn end — hunger tick
    def on_turn_end(event):
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]
        p.properties['food'] -= 1
        if p.properties['food'] <= 0:
            env.end_game('lost')

    env.on('turn_end', on_turn_end)


# ---- Game-specific invariants ----

@invariant('player_starts_with_food_20')
def check_food(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('food') != 20:
        raise InvariantError(f"Player food is {p.properties.get('food')}, expected 20")


@invariant('food_count_10_to_15')
def check_food_count(env):
    foods = env.get_entities_by_type('food')
    if not (10 <= len(foods) <= 15):
        raise InvariantError(f"Found {len(foods)} food entities, expected 10-15")


@invariant('player_starts_at_0_13')
def check_player_pos(env):
    p = env.get_entities_by_tag('player')[0]
    if (p.x, p.y) != (0, 13):
        raise InvariantError(f"Player at ({p.x}, {p.y}), expected (0, 13)")


@invariant('exit_at_13_0')
def check_exit_pos(env):
    e = env.get_entities_by_tag('exit')[0]
    if (e.x, e.y) != (13, 0):
        raise InvariantError(f"Exit at ({e.x}, {e.y}), expected (13, 0)")


@invariant('no_food_on_player_exit_or_wall')
def check_food_placement(env):
    walls = {(e.x, e.y) for e in env.get_entities_by_tag('solid')}
    player = env.get_entities_by_tag('player')[0]
    exit_ent = env.get_entities_by_tag('exit')[0]
    forbidden = walls | {(player.x, player.y), (exit_ent.x, exit_ent.y)}
    for f in env.get_entities_by_type('food'):
        if (f.x, f.y) in forbidden:
            raise InvariantError(f"Food at ({f.x}, {f.y}) overlaps with forbidden cell")
