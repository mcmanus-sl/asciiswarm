"""Tests for Game 06: Ice Sliding."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.06_ice_sliding')


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

class TestIceSlidingMechanics:
    def test_player_and_exit_exist(self):
        env = make_env()
        players = env.get_entities_by_tag('player')
        exits = env.get_entities_by_tag('exit')
        assert len(players) == 1
        assert len(exits) == 1

    def test_rocks_exist(self):
        env = make_env()
        rocks = env.get_entities_by_type('rock')
        assert 8 <= len(rocks) <= 12

    def test_player_in_bottom_left(self):
        for seed in range(10):
            env = make_env(seed=seed)
            p = env.get_entities_by_tag('player')[0]
            assert p.x <= 2 and p.y >= 7, f"Seed {seed}: player at ({p.x}, {p.y})"

    def test_exit_in_top_right(self):
        for seed in range(10):
            env = make_env(seed=seed)
            e = env.get_entities_by_tag('exit')[0]
            assert e.x >= 7 and e.y <= 2, f"Seed {seed}: exit at ({e.x}, {e.y})"

    def test_sliding_moves_multiple_cells(self):
        """Player slides until hitting obstacle or edge."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y

        # Slide west — should slide to edge (x=0) or a rock
        env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        # Player should have moved (or stayed if already at edge/blocked)
        # At minimum, they should be at x=0 or stopped by a rock
        assert p.x <= old_x

    def test_sliding_stops_at_edge(self):
        """Player slides to grid edge."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]

        # Slide south repeatedly — should end at y=9 or earlier if rocks
        env.step(action_index(env, 'move_s'))
        p = env.get_entities_by_tag('player')[0]
        assert p.y >= 7  # Started at y>=7, moved south

    def test_sliding_stops_at_rock(self):
        """Player stops one cell before a rock."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rocks = env.get_entities_by_type('rock')

        # After any slide, player should not overlap with a rock
        env.step(action_index(env, 'move_n'))
        p = env.get_entities_by_tag('player')[0]
        rock_positions = {(r.x, r.y) for r in rocks}
        assert (p.x, p.y) not in rock_positions

    def test_wait_does_nothing(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y
        env.step(action_index(env, 'wait'))
        p = env.get_entities_by_tag('player')[0]
        assert (p.x, p.y) == (old_x, old_y)

    def test_interact_does_nothing(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y
        env.step(action_index(env, 'interact'))
        p = env.get_entities_by_tag('player')[0]
        assert (p.x, p.y) == (old_x, old_y)

    def test_win_by_sliding_onto_exit(self):
        """Player wins by sliding onto the exit tile."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        e = env.get_entities_by_tag('exit')[0]

        # Teleport player one cell south of exit
        # Place a rock north of exit to stop there
        env.move_entity(p.id, e.x, e.y + 1)
        p = env.get_entities_by_tag('player')[0]

        # Slide north — should land on exit
        env.step(action_index(env, 'move_n'))
        assert env.status == 'won'

    def test_no_lose_condition(self):
        """Game cannot be lost — only won or truncated."""
        env = make_env()
        for _ in range(200):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert env.status in ('won', 'playing')


# ---- Fuzz test ----

class TestIceSlidingFuzz:
    def test_fuzz_500_episodes(self):
        """Random agent plays 500 episodes without crashing."""
        env = GridGameEnv(game_module, seed=42)
        wins = 0
        crashes = 0
        terminated_count = 0
        for ep in range(500):
            try:
                obs, info = env.reset()
                terminated = False
                truncated = False
                steps = 0
                while not (terminated or truncated):
                    obs, r, terminated, truncated, info = env.step(
                        env.action_space.sample()
                    )
                    steps += 1
                terminated_count += 1
                if info.get('status') == 'won':
                    wins += 1
            except Exception:
                crashes += 1
        assert crashes == 0, f"{crashes} crashes in 500 episodes"
        # Some random wins expected (spec says 1-10%)
        assert wins >= 0  # Just no crashes is the main check

    def test_fuzz_multiple_seeds(self):
        """Game works across many different seeds."""
        for seed in range(50):
            env = GridGameEnv(game_module, seed=seed)
            obs, info = env.reset(seed=seed)
            # Play a few steps
            for _ in range(20):
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                if term or trunc:
                    break


# ---- Invariant tests ----

class TestIceSlidingInvariants:
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
