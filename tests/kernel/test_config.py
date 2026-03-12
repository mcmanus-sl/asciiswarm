import pytest
import gymnasium
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.types import DEFAULT_GAME_CONFIG


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


def test_custom_actions_build_correct_action_space():
    config = {
        'actions': ['move_n', 'move_s', 'shoot'],
        'grid': (8, 8),
    }
    mod = _make_module(config=config)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    assert env.action_space == gymnasium.spaces.Discrete(3)


def test_custom_actions_build_correct_action_map():
    config = {
        'actions': ['move_n', 'move_s', 'shoot'],
        'grid': (8, 8),
    }
    mod = _make_module(config=config)
    env = GridGameEnv(mod)
    assert env.ACTION_MAP == {0: 'move_n', 1: 'move_s', 2: 'shoot'}


def test_custom_tags_build_correct_observation_space():
    config = {
        'tags': ['player', 'wall', 'lava'],
        'grid': (8, 8),
    }

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])

    mod = _make_module(config=config, setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    grid_shape = env.observation_space['grid'].shape
    assert grid_shape == (3, 8, 8)  # 3 tags


def test_custom_tags_invalid_tag_raises():
    config = {'tags': ['player', 'custom_tag'], 'grid': (8, 8)}

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])

    mod = _make_module(config=config, setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    with pytest.raises(ValueError, match="Unknown tag"):
        env.create_entity('thing', 1, 0, '?', ['not_declared'])


def test_custom_tags_valid_tag_succeeds():
    config = {'tags': ['player', 'custom_tag'], 'grid': (8, 8)}

    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])

    mod = _make_module(config=config, setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    ent = env.create_entity('thing', 1, 0, '?', ['custom_tag'])
    assert ent.has_tag('custom_tag')


def test_partial_config_merges_with_defaults():
    config = {'grid': (10, 10)}  # only override grid
    mod = _make_module(config=config)
    env = GridGameEnv(mod)
    assert env.config['grid'] == (10, 10)
    assert env.config['actions'] == DEFAULT_GAME_CONFIG['actions']
    assert env.config['tags'] == DEFAULT_GAME_CONFIG['tags']
    assert env.config['max_turns'] == DEFAULT_GAME_CONFIG['max_turns']
    assert env.config['step_penalty'] == DEFAULT_GAME_CONFIG['step_penalty']


def test_missing_game_config_raises():
    mod = SimpleNamespace()
    mod.setup = lambda env: None
    with pytest.raises(ValueError, match="GAME_CONFIG"):
        GridGameEnv(mod)


def test_game_config_not_dict_raises():
    mod = SimpleNamespace()
    mod.GAME_CONFIG = "not a dict"
    mod.setup = lambda env: None
    with pytest.raises(ValueError, match="GAME_CONFIG must be a dict"):
        GridGameEnv(mod)


def test_missing_setup_raises():
    mod = SimpleNamespace()
    mod.GAME_CONFIG = {}
    with pytest.raises(ValueError, match="setup"):
        GridGameEnv(mod)


def test_setup_not_callable_raises():
    mod = SimpleNamespace()
    mod.GAME_CONFIG = {}
    mod.setup = "not callable"
    with pytest.raises(ValueError, match="setup"):
        GridGameEnv(mod)


def test_default_config_works():
    """A game that specifies nothing but setup gets all defaults."""
    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])

    mod = SimpleNamespace()
    mod.GAME_CONFIG = {}  # empty — all defaults
    mod.setup = setup

    env = GridGameEnv(mod)
    env.reset(seed=42)
    assert env.config['grid'] == (16, 16)
    assert env.config['max_turns'] == 1000
