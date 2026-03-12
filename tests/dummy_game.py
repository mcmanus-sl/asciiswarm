"""Minimal game module for testing. Player walks to exit on 8x8 grid with one wall."""

GAME_CONFIG = {
    'grid': (8, 8),
    'max_turns': 100,
    'step_penalty': -0.01,
    'player_properties': [
        {'key': 'health', 'max': 10},
    ],
}


def setup(env):
    player = env.create_entity('player', 0, 0, '@', ['player'], z_order=10)
    player.set('health', 10)

    env.create_entity('exit', 7, 7, '>', ['exit'], z_order=5)
    env.create_entity('wall', 3, 3, '#', ['solid'], z_order=1)

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

    def on_before_move(event):
        target_x, target_y = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(target_x, target_y):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    def on_collision(event):
        mover = event.payload['mover']
        if mover.has_tag('player'):
            for occ in event.payload['occupants']:
                if occ.has_tag('exit'):
                    env.end_game('won')

    env.on('collision', on_collision)
