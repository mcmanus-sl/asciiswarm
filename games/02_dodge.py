"""Game 02: Dodge — reach the exit while avoiding a horizontally patrolling wanderer."""

from asciiswarm.kernel.invariants import EXIT_INVARIANTS

INVARIANTS = list(EXIT_INVARIANTS)

GAME_CONFIG = {
    'grid': (10, 10),
    'max_turns': 200,
    'step_penalty': -0.01,
    'player_properties': [],
}


def setup(env):
    w, h = env.config['grid']

    # Player in bottom-left quadrant (x < 5, y >= 5)
    px = int(env.random() * 5)
    py = 5 + int(env.random() * 5)
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10)

    # Exit in top-right quadrant (x >= 5, y < 5)
    ex = 5 + int(env.random() * 5)
    ey = int(env.random() * 5)
    env.create_entity('exit', ex, ey, '>', ['exit'], z_order=5)

    # Wanderer in center row (y=4 or y=5), random x, starts moving east
    wy = 4 if env.random() < 0.5 else 5
    wx = int(env.random() * w)
    # Ensure wanderer doesn't start on player
    while (wx, wy) == (px, py):
        wx = int(env.random() * w)
    wanderer = env.create_entity(
        'wanderer', wx, wy, 'w', ['hazard'], z_order=5,
        properties={'direction': 1},
    )

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

    # Collision handler
    def on_collision(event):
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        # Player walks into exit
        if mover.has_tag('player'):
            for occ in occupants:
                if occ.has_tag('exit'):
                    env.end_game('won')
                    return

        # Player walks into hazard
        if mover.has_tag('player'):
            for occ in occupants:
                if occ.has_tag('hazard'):
                    env.end_game('lost')
                    return

        # Hazard walks into player
        if mover.has_tag('hazard'):
            for occ in occupants:
                if occ.has_tag('player'):
                    env.end_game('lost')
                    return

    env.on('collision', on_collision)

    # Wanderer behavior
    def wanderer_behavior(entity, env):
        d = entity.get('direction', 1)
        env.move_entity(entity.id, entity.x + d, entity.y)
        # Check if next step in same direction would be out of bounds
        if entity.x + d < 0 or entity.x + d >= w:
            entity.set('direction', d * -1)

    env.register_behavior('wanderer', wanderer_behavior)
