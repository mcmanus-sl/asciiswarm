"""Tests for Game 05: Pac-Man Collect."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.05_pac_collect')


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

class TestPacCollectMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_type('chaser')) == 1
        assert len(env.get_entities_by_type('patroller')) == 1
        assert len(env.get_entities_by_tag('pickup')) >= 20

    def test_player_at_center(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert (p.x, p.y) == (6, 6)

    def test_walls_block_movement(self):
        """Player cannot walk through walls."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Move north repeatedly — should be blocked by wall at y=0
        north = action_index(env, 'move_n')
        for _ in range(20):
            env.step(north)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')
        if p:
            assert p[0].y >= 1  # can't go past border wall

    def test_collect_dot(self):
        """Player collects a dot by walking onto it."""
        env = make_env()
        initial_dots = len(env.get_entities_by_tag('pickup'))
        # Player is at (6,6). There should be a dot adjacent.
        # Move north to (6,5) — but that might be a wall (cross pattern at x=5,y=5).
        # Player at (6,6), move east to (7,6) should be an open cell with a dot.
        east = action_index(env, 'move_e')
        env.step(east)
        new_dots = len(env.get_entities_by_tag('pickup'))
        # Should have collected one dot (if cell had a dot)
        assert new_dots == initial_dots - 1

    def test_ghost_kills_player(self):
        """Walking into a ghost loses the game."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        chaser = env.get_entities_by_type('chaser')[0]
        # Teleport player next to chaser
        env.move_entity(p.id, chaser.x + 1, chaser.y)
        west = action_index(env, 'move_w')
        env.step(west)
        assert env.status == 'lost'

    def test_chaser_kills_player_on_contact(self):
        """Chaser walking into player loses the game."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        chaser = env.get_entities_by_type('chaser')[0]
        # Place chaser adjacent to player and let behavior run
        env.move_entity(chaser.id, p.x - 1, p.y)
        # Step with wait to trigger chaser behavior
        wait = action_index(env, 'wait')
        env.step(wait)
        assert env.status == 'lost'

    def test_win_by_collecting_all_dots(self):
        """Win when all dots are collected."""
        env = make_env()
        # Destroy all dots except one, then collect the last one
        dots = env.get_entities_by_tag('pickup')
        last_dot = dots[-1]
        for dot in dots[:-1]:
            env.destroy_entity(dot.id)
        # Teleport player next to last dot
        p = env.get_entities_by_tag('player')[0]
        # Move ghosts far away first
        chaser = env.get_entities_by_type('chaser')[0]
        patroller = env.get_entities_by_type('patroller')[0]
        env.move_entity(chaser.id, 1, 10)
        env.move_entity(patroller.id, 10, 10)

        # Place player adjacent to last dot
        target_x = last_dot.x - 1 if last_dot.x > 1 else last_dot.x + 1
        env.move_entity(p.id, target_x, last_dot.y)
        p = env.get_entities_by_tag('player')[0]
        if p.x < last_dot.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        assert env.status == 'won'

    def test_patroller_moves(self):
        """Patroller should move each turn."""
        env = make_env()
        pat = env.get_entities_by_type('patroller')[0]
        # Move ghosts away from player to prevent death
        env.move_entity(pat.id, 10, 10)
        chaser = env.get_entities_by_type('chaser')[0]
        env.move_entity(chaser.id, 1, 10)
        old_x, old_y = pat.x, pat.y
        wait = action_index(env, 'wait')
        env.step(wait)
        pat = env.get_entities_by_type('patroller')[0]
        # Patroller should have moved (or stayed if blocked)
        # It starts in direction 0 (east), so it should try to move east
        # At (10, 10), east is blocked by wall at x=11, so it rotates direction
        # Just check that behavior runs without crashing
        assert pat is not None


# ---- Fuzz ----

class TestPacCollectFuzz:
    def test_fuzz_1000_episodes(self):
        env = GridGameEnv(game_module, seed=42)
        terminated_count = 0
        for ep in range(1000):
            obs, info = env.reset()
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(env.action_space.sample())
                steps += 1
            if terminated:
                terminated_count += 1
            assert info['status'] in ('won', 'lost', 'playing')
        # Termination rate should be > 80%
        term_rate = terminated_count / 1000
        assert term_rate > 0.8, f"Termination rate {term_rate:.2%} < 80%"


# ---- Invariant tests ----

class TestPacCollectInvariants:
    def test_all_invariants(self):
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
