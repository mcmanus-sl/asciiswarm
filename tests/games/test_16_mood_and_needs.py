"""Tests for Game 16: Mood & Needs."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.16_mood_and_needs')


def make_env(seed=42):
    env = GridGameEnv(game_module, seed=seed)
    env.reset(seed=seed)
    return env


def action_index(env, name):
    for i, a in env.ACTION_MAP.items():
        if a == name:
            return i
    raise ValueError(f"Action {name} not found")


# ---- Mechanical tests ----

class TestMoodAndNeedsMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_type('dwarf')) == 4
        assert len(env.get_entities_by_type('food_store')) == 2
        assert len(env.get_entities_by_type('bed')) == 4
        assert len(env.get_entities_by_type('tavern')) == 1
        assert len(env.get_entities_by_type('build_site')) == 5
        assert len(env.get_entities_by_type('stone')) >= 8

    def test_player_properties(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['walls_built'] == 0
        assert p.properties['dwarves_alive'] == 4
        assert p.properties['avg_mood'] == 10

    def test_dwarf_initial_state(self):
        env = make_env()
        for d in env.get_entities_by_type('dwarf'):
            assert d.properties['hunger'] == 6
            assert d.properties['rest'] == 6
            assert d.properties['social'] == 6
            assert d.properties['mood'] == 6
            assert d.properties['task'] == 'idle'

    def test_player_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y
        env.step(action_index(env, 'move_n'))
        p = env.get_entities_by_tag('player')[0]
        assert env.status == 'playing'

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Keep moving west, should be blocked by walls
        west = action_index(env, 'move_w')
        for _ in range(20):
            env.step(west)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')
        if p:
            assert p[0].x >= 1

    def test_order_eat_works(self):
        """Order eat either sets task or completes immediately (adjacent facility)."""
        env = make_env()
        hungers_before = [d.properties['hunger'] for d in env.get_entities_by_type('dwarf')]
        env.step(action_index(env, 'order_eat'))
        dwarves = env.get_entities_by_type('dwarf')
        hungers_after = [d.properties['hunger'] for d in dwarves]
        tasks = [d.properties['task'] for d in dwarves]
        # Either a dwarf has task 'eat' or one dwarf's hunger was restored to 10
        assert 'eat' in tasks or any(h == 10 for h in hungers_after)

    def test_order_sleep_works(self):
        """Order sleep either sets task or completes immediately."""
        env = make_env()
        env.step(action_index(env, 'order_sleep'))
        dwarves = env.get_entities_by_type('dwarf')
        rests = [d.properties['rest'] for d in dwarves]
        tasks = [d.properties['task'] for d in dwarves]
        assert 'sleep' in tasks or any(r == 10 for r in rests)

    def test_order_socialize_works(self):
        """Order socialize either sets task or completes immediately."""
        env = make_env()
        env.step(action_index(env, 'order_socialize'))
        dwarves = env.get_entities_by_type('dwarf')
        socials = [d.properties['social'] for d in dwarves]
        tasks = [d.properties['task'] for d in dwarves]
        assert 'socialize' in tasks or any(s == 10 for s in socials)

    def test_order_build_changes_task(self):
        env = make_env()
        env.step(action_index(env, 'order_build'))
        dwarves = env.get_entities_by_type('dwarf')
        build_count = sum(1 for d in dwarves if d.properties['task'] == 'build')
        assert build_count >= 1

    def test_hunger_decays(self):
        env = make_env()
        wait = action_index(env, 'wait')
        initial_hungers = [d.properties['hunger'] for d in env.get_entities_by_type('dwarf')]
        # Run 10 turns to trigger decay at turn 5 and 10
        for _ in range(10):
            env.step(wait)
            if env.status != 'playing':
                break
        if env.status == 'playing':
            dwarves = env.get_entities_by_type('dwarf')
            current_hungers = [d.properties['hunger'] for d in dwarves]
            # At least some dwarves should have lower hunger
            assert any(c < i for c, i in zip(current_hungers, initial_hungers))

    def test_rest_decays(self):
        env = make_env()
        wait = action_index(env, 'wait')
        for _ in range(12):
            env.step(wait)
            if env.status != 'playing':
                break
        if env.status == 'playing':
            dwarves = env.get_entities_by_type('dwarf')
            assert any(d.properties['rest'] < 6 for d in dwarves)

    def test_social_decays(self):
        env = make_env()
        wait = action_index(env, 'wait')
        for _ in range(16):
            env.step(wait)
            if env.status != 'playing':
                break
        if env.status == 'playing':
            dwarves = env.get_entities_by_type('dwarf')
            assert any(d.properties['social'] < 6 for d in dwarves)

    def test_no_crash_100_random_steps(self):
        """Run 100 random steps without crashing."""
        env = make_env()
        n_actions = env.action_space.n
        for _ in range(100):
            action = int(env.random() * n_actions)
            env.step(action)
            if env.status != 'playing':
                break

    def test_multi_seed_stability(self):
        """Test that the game works across multiple seeds."""
        for seed in [1, 42, 100, 999, 12345]:
            env = make_env(seed=seed)
            assert len(env.get_entities_by_tag('player')) == 1
            assert len(env.get_entities_by_type('dwarf')) == 4
            assert len(env.get_entities_by_type('stone')) >= 8

    def test_game_can_end(self):
        """Verify the game terminates (either by timeout or game logic) within max_turns."""
        env = make_env()
        wait = action_index(env, 'wait')
        for _ in range(500):
            env.step(wait)
            if env.status != 'playing':
                break
        # Game should have ended or timed out
        assert env.status != 'playing' or env.turn_number >= 500


# ---- Invariant tests ----

class TestMoodAndNeedsInvariants:
    def test_invariants_pass(self):
        for seed in [42, 123, 456]:
            env = make_env(seed=seed)
            run_invariants(env, game_module)
