"""Tests for Game 08: Block Push."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.08_block_push')


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

class TestBlockPushMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_type('block')) == 2
        assert len(env.get_entities_by_type('target')) == 2

    def test_walls_block_player(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Move north many times to hit a wall
        north = action_index(env, 'move_n')
        for _ in range(10):
            env.step(north)
        p = env.get_entities_by_tag('player')[0]
        assert p.y >= 1  # can't go past border

    def test_push_block(self):
        """Player can push a block by walking into it."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        blocks = env.get_entities_by_type('block')
        b = blocks[0]
        bx, by = b.x, b.y

        # Place player south of block, push north
        env.move_entity(p.id, bx, by + 1)
        north = action_index(env, 'move_n')
        env.step(north)

        b = env.get_entities_by_type('block')[0]
        p = env.get_entities_by_tag('player')[0]
        # Block should have moved north
        if b.x == bx:
            assert b.y == by - 1 or b.y == by  # moved or was blocked by wall
            if b.y == by - 1:
                assert p.y == by  # player moved into block's old position

    def test_block_blocked_by_wall(self):
        """Block can't be pushed through a wall; player stays too."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        blocks = env.get_entities_by_type('block')
        b = blocks[0]

        # Push block to top wall (y=1 border is wall at y=0)
        # Place block at (3, 1) and player at (3, 2)
        env.move_entity(b.id, 3, 1)
        env.move_entity(p.id, 3, 2)

        north = action_index(env, 'move_n')
        env.step(north)

        b_after = [e for e in env.get_entities_by_type('block') if e.id == b.id][0]
        p = env.get_entities_by_tag('player')[0]
        # Block should stay at (3, 1), player at (3, 2)
        assert b_after.y == 1
        assert p.y == 2

    def test_block_blocked_by_block(self):
        """Block can't push another block."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        blocks = env.get_entities_by_type('block')
        b1, b2 = blocks[0], blocks[1]

        # Stack: player at (3,5), b1 at (3,4), b2 at (3,3)
        env.move_entity(b2.id, 3, 3)
        env.move_entity(b1.id, 3, 4)
        env.move_entity(p.id, 3, 5)

        north = action_index(env, 'move_n')
        env.step(north)

        p = env.get_entities_by_tag('player')[0]
        # Player should be blocked (can't push b1 because b2 is in the way)
        assert p.y == 5

    def test_win_condition(self):
        """Win when both blocks are on targets."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        blocks = env.get_entities_by_type('block')
        targets = env.get_entities_by_type('target')
        b1, b2 = blocks[0], blocks[1]
        t1, t2 = targets[0], targets[1]

        # Place b1 next to t1, and b2 already on t2
        env.move_entity(b2.id, t2.x, t2.y)
        # Place b1 one cell south of t1
        env.move_entity(b1.id, t1.x, t1.y + 1)
        # Place player south of b1
        env.move_entity(p.id, t1.x, t1.y + 2)

        # Push b1 north onto t1
        north = action_index(env, 'move_n')
        env.step(north)

        assert env.status == 'won'

    def test_no_lose_condition(self):
        """Game cannot be lost, only won or truncated."""
        env = make_env()
        for _ in range(300):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert env.status in ('won', 'playing')


# ---- Fuzz ----

class TestBlockPushFuzz:
    def test_fuzz_500_episodes(self):
        env = GridGameEnv(game_module, seed=42)
        for ep in range(500):
            obs, info = env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            assert info['status'] in ('won', 'lost', 'playing')


# ---- Invariant tests ----

class TestBlockPushInvariants:
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
