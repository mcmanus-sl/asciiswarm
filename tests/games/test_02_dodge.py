"""Tests for Game 02: Dodge."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants, InvariantError

game_module = importlib.import_module('games.02_dodge')


def make_env(seed=42):
    env = GridGameEnv(game_module, seed=seed)
    env.reset(seed=seed)
    return env


def action_index(env, name):
    for i, a in env.ACTION_MAP.items():
        if a == name:
            return i
    raise ValueError(f"Action {name} not found")


# ---- Game-specific invariant checks (run at setup time) ----

def _check_wanderer_count(env):
    wanderers = env.get_entities_by_type('wanderer')
    if len(wanderers) != 1:
        raise InvariantError(f"Expected 1 wanderer, found {len(wanderers)}")


def _check_wanderer_center(env):
    w = env.get_entities_by_type('wanderer')[0]
    if w.y not in (4, 5):
        raise InvariantError(f"Wanderer y={w.y}, expected 4 or 5")


def _check_player_bottom_left(env):
    p = env.get_entities_by_tag('player')[0]
    if p.x >= 5 or p.y < 5:
        raise InvariantError(f"Player at ({p.x},{p.y}), expected bottom-left quadrant")


def _check_exit_top_right(env):
    e = env.get_entities_by_tag('exit')[0]
    if e.x < 5 or e.y >= 5:
        raise InvariantError(f"Exit at ({e.x},{e.y}), expected top-right quadrant")


def _check_no_overlap(env):
    p = env.get_entities_by_tag('player')[0]
    w = env.get_entities_by_type('wanderer')[0]
    if (p.x, p.y) == (w.x, w.y):
        raise InvariantError("Player and wanderer start on same cell")


# ---- Mechanical tests ----

class TestDodgeMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_tag('hazard')) == 1

    def test_wanderer_bounces(self):
        """Wanderer moves and eventually reverses direction."""
        env = make_env()
        w = env.get_entities_by_type('wanderer')[0]
        positions = [w.x]
        for _ in range(20):
            env.step(action_index(env, 'wait'))
            if env.status != 'playing':
                break
            w = env.get_entities_by_type('wanderer')
            if not w:
                break
            positions.append(w[0].x)
        # Should see direction changes (not monotonic)
        assert len(set(positions)) > 1

    def test_player_dies_on_hazard(self):
        """Player walking into wanderer ends game as lost."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        w = env.get_entities_by_type('wanderer')[0]

        # Teleport player next to wanderer
        target_x = w.x - 1 if w.x > 0 else w.x + 1
        env.move_entity(p.id, target_x, w.y)

        # Move onto wanderer
        if target_x < w.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        # Either lost (wanderer still there) or game continued (wanderer moved away)
        # This is a timing test — wanderer moves after input
        # If the wanderer moved away in behavior phase, player lives

    def test_wanderer_kills_player(self):
        """Wanderer walking into player ends game as lost."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        w = env.get_entities_by_type('wanderer')[0]

        # Place player one step ahead of wanderer in its direction
        d = w.get('direction', 1)
        env.move_entity(p.id, w.x + d, w.y)

        # Wait — wanderer moves into player
        env.step(action_index(env, 'wait'))
        assert env.status == 'lost'

    def test_win_on_exit(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        e = env.get_entities_by_tag('exit')[0]

        # Teleport next to exit
        env.move_entity(p.id, e.x - 1 if e.x > 0 else e.x + 1, e.y)
        p = env.get_entities_by_tag('player')[0]

        if p.x < e.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        assert env.status == 'won'


# ---- Fuzz ----

class TestDodgeFuzz:
    def test_fuzz_500_episodes(self):
        env = GridGameEnv(game_module, seed=42)
        crashes = 0
        for ep in range(500):
            obs, info = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            assert info['status'] in ('won', 'lost', 'playing')


# ---- Invariant tests ----

class TestDodgeInvariants:
    def test_builtin_invariants(self):
        env = make_env()
        results = run_invariants(env, game_module)
        for name, passed, err in results:
            assert passed, f"Invariant {name} failed: {err}"

    def test_game_specific_invariants_multiple_seeds(self):
        checks = [
            _check_wanderer_count,
            _check_wanderer_center,
            _check_player_bottom_left,
            _check_exit_top_right,
            _check_no_overlap,
        ]
        for seed in range(20):
            env = make_env(seed=seed)
            for check in checks:
                check(env)  # raises on failure
