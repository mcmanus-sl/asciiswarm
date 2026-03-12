import pytest
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


def test_behavior_runs_for_correct_type():
    results = []

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('goblin', 1, 1, 'g', ['hazard'])
        env.register_behavior('goblin', lambda ent, env: results.append(ent.type))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    assert results == ['goblin']


def test_behaviors_run_in_creation_order():
    order = []

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('goblin', 1, 0, 'g', ['hazard'])
        env.create_entity('orc', 2, 0, 'o', ['hazard'])
        env.create_entity('goblin', 3, 0, 'g', ['hazard'])
        env.register_behavior('goblin', lambda ent, env: order.append(('goblin', ent.id)))
        env.register_behavior('orc', lambda ent, env: order.append(('orc', ent.id)))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    assert order == [('goblin', 'e2'), ('orc', 'e3'), ('goblin', 'e4')]


def test_behavior_can_move_entity():
    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('mover', 1, 0, 'm', ['npc'])
        env.register_behavior('mover', lambda ent, env: env.move_entity(ent.id, ent.x + 1, ent.y))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    mover = env.get_entities_by_type('mover')[0]
    assert mover.x == 2


def test_behavior_can_destroy_entity():
    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        target = env.create_entity('target', 1, 0, 't', ['npc'])
        env.register_behavior('target', lambda ent, env: env.destroy_entity(ent.id))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    assert env.get_entities_by_type('target') == []


def test_behavior_can_create_entity():
    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('spawner', 1, 0, 's', ['npc'])
        env.register_behavior('spawner', lambda ent, env: env.create_entity('spawn', 2, 0, 'x', ['npc']))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    spawns = env.get_entities_by_type('spawn')
    assert len(spawns) == 1


def test_destroyed_entity_behavior_skipped():
    """If entity A's behavior destroys entity B, B's behavior does not run."""
    b_ran = [False]

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('killer', 1, 0, 'k', ['npc'])
        env.create_entity('victim', 2, 0, 'v', ['npc'])
        env.register_behavior('killer', lambda ent, env: env.destroy_entity('e3'))
        env.register_behavior('victim', lambda ent, env: b_ran.__setitem__(0, True))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    assert not b_ran[0]


def test_early_termination_stops_behaviors():
    """If end_game() is called during a behavior, remaining behaviors don't execute."""
    orc_ran = [False]

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('goblin', 1, 0, 'g', ['hazard'])
        env.create_entity('orc', 2, 0, 'o', ['hazard'])
        env.register_behavior('goblin', lambda ent, env: env.end_game('lost'))
        env.register_behavior('orc', lambda ent, env: orc_ran.__setitem__(0, True))

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    assert env.status == 'lost'
    assert not orc_ran[0]
