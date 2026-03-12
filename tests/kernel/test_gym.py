import pytest
import numpy as np
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


class TestObservationSpace:
    def test_obs_space_is_dict(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        assert 'grid' in env.observation_space.spaces
        assert 'scalars' in env.observation_space.spaces

    def test_grid_shape(self):
        config = {'tags': ['player', 'solid', 'exit'], 'grid': (10, 8)}
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        assert env.observation_space['grid'].shape == (3, 8, 10)  # (tags, height, width)

    def test_scalars_shape(self):
        config = {
            'grid': (8, 8),
            'player_properties': [
                {'key': 'health', 'max': 10},
                {'key': 'attack', 'max': 5},
            ],
        }
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        # 3 base scalars + 2 player properties
        assert env.observation_space['scalars'].shape == (5,)

    def test_scalars_shape_no_properties(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        assert env.observation_space['scalars'].shape == (3,)

    def test_action_space(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        assert env.action_space.n == 6  # default 6 actions


class TestGetObs:
    def test_returns_dict(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert 'grid' in obs
        assert 'scalars' in obs

    def test_grid_dtype_and_shape(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        assert obs['grid'].dtype == np.float32
        assert obs['grid'].shape == (6, 8, 8)  # 6 default tags, 8x8 grid

    def test_grid_reflects_entity_tags(self):
        def setup(env):
            env.create_entity('player', 3, 4, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        # 'player' is tag index 0 in default tags
        assert obs['grid'][0, 4, 3] == 1.0  # channel, y, x
        # Other cells should be 0
        assert obs['grid'][0, 0, 0] == 0.0

    def test_grid_multi_tag_entity(self):
        config = {'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'], 'grid': (8, 8)}

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('trap', 3, 3, '^', ['solid', 'hazard'])

        mod = _make_module(config=config, setup=setup)
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        # solid=1, hazard=2
        assert obs['grid'][1, 3, 3] == 1.0
        assert obs['grid'][2, 3, 3] == 1.0

    def test_scalars_grid_dims(self):
        config = {'grid': (10, 8)}
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        assert abs(obs['scalars'][0] - 0.10) < 1e-6  # width/100
        assert abs(obs['scalars'][1] - 0.08) < 1e-6  # height/100

    def test_scalars_turn_normalization(self):
        config = {'grid': (8, 8), 'max_turns': 200}
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('wait')
        obs = env._get_obs()
        assert abs(obs['scalars'][2] - 1.0 / 200.0) < 1e-6

    def test_scalars_player_properties(self):
        config = {
            'grid': (8, 8),
            'player_properties': [
                {'key': 'health', 'max': 10},
            ],
        }

        def setup(env):
            p = env.create_entity('player', 0, 0, '@', ['player'])
            p.set('health', 7)

        mod = _make_module(config=config, setup=setup)
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        assert abs(obs['scalars'][3] - 0.7) < 1e-6

    def test_scalars_property_clamped(self):
        """Property exceeding max is clamped to 1.0."""
        config = {
            'grid': (8, 8),
            'player_properties': [
                {'key': 'health', 'max': 10},
            ],
        }

        def setup(env):
            p = env.create_entity('player', 0, 0, '@', ['player'])
            p.set('health', 15)  # exceeds max of 10

        mod = _make_module(config=config, setup=setup)
        env = GridGameEnv(mod)
        obs, _ = env.reset(seed=42)
        assert obs['scalars'][3] == 1.0  # clamped

    def test_obs_deterministic(self):
        """Same state produces same observation."""
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        obs1 = env._get_obs()
        obs2 = env._get_obs()
        np.testing.assert_array_equal(obs1['grid'], obs2['grid'])
        np.testing.assert_array_equal(obs1['scalars'], obs2['scalars'])


class TestStep:
    def test_returns_5_tuple(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        result = env.step(5)  # 'wait'
        assert len(result) == 5

    def test_step_reward_additive(self):
        """step_penalty + intermediate rewards + terminal reward all summed."""
        config = {'grid': (8, 8), 'step_penalty': -0.01}

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

            def on_input(event):
                env.emit('reward', {'amount': 0.5})
                env.end_game('won')

            env.on('input', on_input)

        mod = _make_module(config=config, setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        _, reward, terminated, _, _ = env.step(5)
        # -0.01 + 0.5 + 10.0 = 10.49
        assert abs(reward - 10.49) < 1e-6
        assert terminated

    def test_step_terminated_on_win(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.end_game('won'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(5)
        assert terminated

    def test_step_terminated_on_loss(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.end_game('lost'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        _, reward, terminated, _, _ = env.step(5)
        assert terminated
        assert reward < 0  # step_penalty + -10.0

    def test_step_truncated_at_max_turns(self):
        config = {'grid': (8, 8), 'max_turns': 3}
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        _, _, _, truncated, _ = env.step(5)
        assert not truncated  # turn 1

        _, _, _, truncated, _ = env.step(5)
        assert not truncated  # turn 2

        _, _, _, truncated, _ = env.step(5)
        assert truncated  # turn 3 == max_turns

    def test_step_after_game_over_returns_zero_reward(self):
        """No double terminal reward."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.end_game('won'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.step(5)  # game ends here

        _, reward, terminated, _, _ = env.step(5)
        assert reward == 0.0
        assert terminated

    def test_step_info_contains_turn_and_status(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        _, _, _, _, info = env.step(5)
        assert 'turn' in info
        assert 'status' in info
        assert info['turn'] == 1
        assert info['status'] == 'playing'


class TestReset:
    def test_reset_returns_obs_and_info(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert 'grid' in obs
        assert 'turn' in info

    def test_reset_with_seed_deterministic(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1['grid'], obs2['grid'])

    def test_reset_without_seed_different_layouts(self):
        """Reset without seed should NOT re-seed — different random layouts."""
        positions = []

        def setup(env):
            x = int(env.random() * 8)
            y = int(env.random() * 8)
            env.create_entity('player', x, y, '@', ['player'])
            positions.append((x, y))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.reset()  # no seed — PRNG continues
        env.reset()  # no seed — PRNG continues
        # At least one should differ (extremely likely with 3 random positions)
        assert len(set(positions)) > 1

    def test_reset_clears_event_handlers(self):
        """No duplicate callback accumulation across resets."""
        call_count = [0]

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: call_count.__setitem__(0, call_count[0] + 1))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)

        env.reset(seed=42)
        env.handle_input('wait')
        assert call_count[0] == 1

        env.reset()
        call_count[0] = 0
        env.handle_input('wait')
        assert call_count[0] == 1  # still 1, not 2

    def test_reset_validates_player_singleton_none(self):
        def setup(env):
            pass  # no player created

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        with pytest.raises(ValueError, match="Exactly one.*player"):
            env.reset(seed=42)

    def test_reset_validates_player_singleton_multiple(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('player', 1, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        with pytest.raises(ValueError, match="Exactly one.*player"):
            env.reset(seed=42)

    def test_reset_clears_game_status(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('won')
        assert env.status == 'won'
        env.reset()
        assert env.status == 'playing'


class TestDeterminism:
    def test_two_envs_same_seed_same_random(self):
        mod = _make_module()
        env1 = GridGameEnv(mod)
        env2 = GridGameEnv(mod)
        env1.reset(seed=42)
        env2.reset(seed=42)
        for _ in range(20):
            assert env1.random() == env2.random()

    def test_load_state_deterministic_replay(self):
        """Load state, run N turns, serialize. Repeat. Identical."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        state = env.serialize_state()

        import json
        env.load_state(state)
        for _ in range(10):
            env.handle_input('wait')
        result1 = json.dumps(env.serialize_state(), sort_keys=True)

        env.load_state(state)
        for _ in range(10):
            env.handle_input('wait')
        result2 = json.dumps(env.serialize_state(), sort_keys=True)

        assert result1 == result2


class TestPRNG:
    def test_same_seed_same_sequence(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=99)
        seq1 = [env.random() for _ in range(10)]
        env.reset(seed=99)
        seq2 = [env.random() for _ in range(10)]
        assert seq1 == seq2

    def test_different_seeds_different_sequence(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        seq1 = [env.random() for _ in range(10)]
        env.reset(seed=99)
        seq2 = [env.random() for _ in range(10)]
        assert seq1 != seq2

    def test_prng_round_trip(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)

        for _ in range(10):
            env.random()

        state = env.serialize_state()
        expected = [env.random() for _ in range(5)]

        env.load_state(state)
        actual = [env.random() for _ in range(5)]
        assert expected == actual
