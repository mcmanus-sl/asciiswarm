DEFAULT_ACTIONS = ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait']

DEFAULT_TAGS = ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc']

DEFAULT_GAME_CONFIG = {
    'actions': DEFAULT_ACTIONS,
    'tags': DEFAULT_TAGS,
    'grid': (16, 16),
    'max_turns': 1000,
    'step_penalty': -0.01,
    'player_properties': [],
}
