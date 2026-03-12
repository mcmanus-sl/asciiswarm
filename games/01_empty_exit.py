"""Game 01: Empty Exit — walk to the exit on an 8x8 grid. No enemies, no obstacles."""

GAME_CONFIG = {
    'grid': (8, 8),
    'max_turns': 200,
    'player_properties': [],
}


def setup(env):
    # Random empty cell for player
    w, h = env.config['grid']
    px = int(env.random() * w)
    py = int(env.random() * h)
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10)

    # Random empty cell for exit (not player's cell)
    while True:
        ex = int(env.random() * w)
        ey = int(env.random() * h)
        if (ex, ey) != (px, py):
            break
    env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)

    # Input handler — move player
    def on_input(event):
        p = env.get_entities_by_tag('player')[0]
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

    # Collision — player reaches exit
    def on_collision(event):
        mover = event.payload['mover']
        if mover.has_tag('player'):
            for occ in event.payload['occupants']:
                if occ.has_tag('exit'):
                    env.end_game('won')

    env.on('collision', on_collision)
