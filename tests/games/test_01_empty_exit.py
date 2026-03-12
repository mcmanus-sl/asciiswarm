"""Tests for Game 01: Empty Exit."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.01_empty_exit')


# ---- Helper ----

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

class TestEmptyExitMechanics:
    def test_player_and_exit_exist(self):
        env = make_env()
        players = env.get_entities_by_tag('player')
        exits = env.get_entities_by_tag('exit')
        assert len(players) == 1
        assert len(exits) == 1

    def test_player_and_exit_different_cells(self):
        for seed in range(10):
            env = make_env(seed=seed)
            p = env.get_entities_by_tag('player')[0]
            e = env.get_entities_by_tag('exit')[0]
            assert (p.x, p.y) != (e.x, e.y)

    def test_move_north(self):
        env = make_env(seed=100)
        p = env.get_entities_by_tag('player')[0]
        old_y = p.y
        if old_y > 0:
            env.step(action_index(env, 'move_n'))
            p = env.get_entities_by_tag('player')[0]
            assert p.y == old_y - 1

    def test_move_south(self):
        env = make_env(seed=100)
        p = env.get_entities_by_tag('player')[0]
        old_y = p.y
        h = env.config['grid'][1]
        if old_y < h - 1:
            env.step(action_index(env, 'move_s'))
            p = env.get_entities_by_tag('player')[0]
            assert p.y == old_y + 1

    def test_move_east(self):
        env = make_env(seed=100)
        p = env.get_entities_by_tag('player')[0]
        old_x = p.x
        w = env.config['grid'][0]
        if old_x < w - 1:
            env.step(action_index(env, 'move_e'))
            p = env.get_entities_by_tag('player')[0]
            assert p.x == old_x + 1

    def test_move_west(self):
        env = make_env(seed=100)
        p = env.get_entities_by_tag('player')[0]
        old_x = p.x
        if old_x > 0:
            env.step(action_index(env, 'move_w'))
            p = env.get_entities_by_tag('player')[0]
            assert p.x == old_x - 1

    def test_out_of_bounds_safe(self):
        """Moving out of bounds does not crash."""
        env = make_env()
        # Move north many times to hit boundary
        n = action_index(env, 'move_n')
        for _ in range(20):
            env.step(n)
            if env.status != 'playing':
                break

    def test_wait_does_nothing(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y
        env.step(action_index(env, 'wait'))
        p = env.get_entities_by_tag('player')[0]
        assert (p.x, p.y) == (old_x, old_y)

    def test_win_on_reaching_exit(self):
        """Player wins by walking onto the exit."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        e = env.get_entities_by_tag('exit')[0]

        # Teleport player next to exit
        env.move_entity(p.id, e.x - 1 if e.x > 0 else e.x + 1, e.y)
        p = env.get_entities_by_tag('player')[0]

        # Move onto exit
        if p.x < e.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        assert env.status == 'won'

    def test_no_lose_condition(self):
        """Game cannot be lost — only won or truncated."""
        env = make_env()
        # Play 200 random actions
        for _ in range(200):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert env.status in ('won', 'playing')


# ---- Fuzz test ----

class TestEmptyExitFuzz:
    def test_fuzz_500_episodes(self):
        env = GridGameEnv(game_module, seed=42)
        wins = 0
        for ep in range(500):
            obs, info = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            if info['status'] == 'won':
                wins += 1
        # On 8x8 with 200 turns, random agent should win sometimes
        assert wins > 0


# ---- Invariant tests ----

class TestEmptyExitInvariants:
    def test_builtin_invariants(self):
        env = make_env()
        results = run_invariants(env, game_module)
        for name, passed, err in results:
            assert passed, f"Invariant {name} failed: {err}"

    def test_invariants_multiple_seeds(self):
        for seed in range(20):
            env = make_env(seed=seed)
            results = run_invariants(env, game_module)
            for name, passed, err in results:
                assert passed, f"Seed {seed}, invariant {name} failed: {err}"
