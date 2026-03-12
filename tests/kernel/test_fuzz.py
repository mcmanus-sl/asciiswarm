"""Random agent fuzz tests — the first layer of the RL oracle.

Runs episodes of random actions through the Gymnasium API to verify:
- No exceptions thrown
- Every episode terminates within max_turns
- Game status is never corrupted
- Observation shape is constant
"""

import pytest
import numpy as np
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv


# ---- Fixtures ----

def _load_dummy_game():
    """Load the dummy game module."""
    import tests.dummy_game as dummy
    return dummy


def _run_fuzz_episodes(game_module, num_episodes=1000, seed=42):
    """Run N episodes of random actions via Gym API. Returns stats dict."""
    env = GridGameEnv(game_module, seed=seed)

    stats = {
        'episodes': num_episodes,
        'crashes': 0,
        'wins': 0,
        'losses': 0,
        'truncations': 0,
        'total_steps': 0,
        'episode_lengths': [],
        'obs_grid_shape': None,
        'obs_scalars_shape': None,
        'status_violations': 0,
    }

    obs, info = env.reset()
    stats['obs_grid_shape'] = obs['grid'].shape
    stats['obs_scalars_shape'] = obs['scalars'].shape

    for ep in range(num_episodes):
        obs, info = env.reset()
        assert obs['grid'].shape == stats['obs_grid_shape'], \
            f"Grid shape changed at episode {ep}: {obs['grid'].shape} != {stats['obs_grid_shape']}"
        assert obs['scalars'].shape == stats['obs_scalars_shape'], \
            f"Scalars shape changed at episode {ep}: {obs['scalars'].shape} != {stats['obs_scalars_shape']}"

        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            # Shape consistency every step
            assert obs['grid'].shape == stats['obs_grid_shape']
            assert obs['scalars'].shape == stats['obs_scalars_shape']

            # Status must be valid
            if info['status'] not in ('playing', 'won', 'lost'):
                stats['status_violations'] += 1

        stats['episode_lengths'].append(steps)
        stats['total_steps'] += steps

        if info['status'] == 'won':
            stats['wins'] += 1
        elif info['status'] == 'lost':
            stats['losses'] += 1
        if truncated and not terminated:
            stats['truncations'] += 1

    return stats


# ---- Tests ----

class TestFuzzDummyGame:
    def test_no_crashes_1000_episodes(self):
        """1000 episodes of random actions, no exceptions."""
        game = _load_dummy_game()
        stats = _run_fuzz_episodes(game, num_episodes=1000, seed=42)
        assert stats['crashes'] == 0

    def test_all_episodes_terminate(self):
        """Every episode terminates within max_turns."""
        game = _load_dummy_game()
        max_turns = game.GAME_CONFIG.get('max_turns', 1000)
        stats = _run_fuzz_episodes(game, num_episodes=1000, seed=42)
        for i, length in enumerate(stats['episode_lengths']):
            assert length <= max_turns, \
                f"Episode {i} ran for {length} steps, exceeding max_turns={max_turns}"

    def test_status_never_corrupted(self):
        """Game status is always 'playing', 'won', or 'lost'."""
        game = _load_dummy_game()
        stats = _run_fuzz_episodes(game, num_episodes=1000, seed=42)
        assert stats['status_violations'] == 0

    def test_observation_shape_constant(self):
        """Observation shape never changes across all steps of all episodes."""
        game = _load_dummy_game()
        # If _run_fuzz_episodes completes without AssertionError, shapes are constant
        stats = _run_fuzz_episodes(game, num_episodes=100, seed=42)
        assert stats['obs_grid_shape'] is not None
        assert stats['obs_scalars_shape'] is not None

    def test_fuzz_records_stats(self):
        """Fuzz run produces meaningful statistics."""
        game = _load_dummy_game()
        stats = _run_fuzz_episodes(game, num_episodes=100, seed=42)
        assert stats['episodes'] == 100
        assert len(stats['episode_lengths']) == 100
        assert stats['total_steps'] > 0
        # Some episodes should win (random walk on 8x8 grid)
        # Just check we get a nonzero total of wins + losses + truncations
        total_outcomes = stats['wins'] + stats['losses'] + stats['truncations']
        assert total_outcomes == 100


class TestFuzzMinimalGame:
    """Fuzz a truly minimal game — just player + exit, no walls."""

    def test_minimal_game_1000_episodes(self):
        mod = SimpleNamespace()
        mod.GAME_CONFIG = {'grid': (5, 5), 'max_turns': 50}

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 4, 4, '>', ['exit'])

            def on_input(event):
                p = env.get_entities_by_tag('player')[0]
                action = event.payload['action']
                moves = {
                    'move_n': (0, -1), 'move_s': (0, 1),
                    'move_e': (1, 0), 'move_w': (-1, 0),
                }
                if action in moves:
                    dx, dy = moves[action]
                    env.move_entity(p.id, p.x + dx, p.y + dy)

            env.on('input', on_input)

            def on_collision(event):
                mover = event.payload['mover']
                if mover.has_tag('player'):
                    for occ in event.payload['occupants']:
                        if occ.has_tag('exit'):
                            env.end_game('won')

            env.on('collision', on_collision)

        mod.setup = setup
        stats = _run_fuzz_episodes(mod, num_episodes=1000, seed=42)
        assert stats['crashes'] == 0
        assert stats['status_violations'] == 0
        # On a 5x5 grid with 50 max_turns, random agent should win sometimes
        assert stats['wins'] > 0
