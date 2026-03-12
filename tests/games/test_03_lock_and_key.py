"""Tests for Game 03: Lock & Key."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.03_lock_and_key')


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

class TestLockAndKeyMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_type('key')) == 1
        assert len(env.get_entities_by_type('door')) == 1

    def test_walls_block_movement(self):
        """Player cannot walk through walls."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Player is in room 1 (x=1..3). Wall at x=4.
        # Move east until blocked
        east = action_index(env, 'move_e')
        for _ in range(10):
            env.step(east)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')[0]
        # Player should not have crossed into room 2 via the wall
        # (they might have gone through corridor, but x should be bounded)
        assert p.x <= 7  # can't go past room 2 wall

    def test_pick_up_key(self):
        """Player picks up key by walking onto it."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        k = env.get_entities_by_type('key')[0]

        # Teleport player next to key
        target_x = k.x - 1 if k.x > 5 else k.x + 1
        # Need to ensure target isn't a wall — put player in room 2
        env.move_entity(p.id, target_x, k.y)
        p = env.get_entities_by_tag('player')[0]

        assert p.properties['has_key'] == 0

        # Move onto key
        if p.x < k.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_key'] == 1
        assert len(env.get_entities_by_type('key')) == 0

    def test_interact_opens_door_with_key(self):
        """Player with key can interact to open door."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        door = env.get_entities_by_type('door')[0]

        # Give player the key
        p.properties['has_key'] = 1

        # Place player adjacent to door
        env.move_entity(p.id, door.x - 1, door.y)

        env.step(action_index(env, 'interact'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_key'] == 0
        assert len(env.get_entities_by_type('door')) == 0

    def test_interact_without_key_does_nothing(self):
        """Player without key cannot open door."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        door = env.get_entities_by_type('door')[0]

        # Place player adjacent to door (no key)
        env.move_entity(p.id, door.x - 1, door.y)

        env.step(action_index(env, 'interact'))

        assert p.properties['has_key'] == 0
        assert len(env.get_entities_by_type('door')) == 1

    def test_door_blocks_movement(self):
        """Player cannot walk through the door."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        door = env.get_entities_by_type('door')[0]

        # Place player next to door
        env.move_entity(p.id, door.x - 1, door.y)
        env.step(action_index(env, 'move_e'))

        p = env.get_entities_by_tag('player')[0]
        assert p.x == door.x - 1  # blocked

    def test_full_sequence_wins(self):
        """Pick up key, open door, reach exit — wins the game."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        k = env.get_entities_by_type('key')[0]
        door = env.get_entities_by_type('door')[0]
        ex = env.get_entities_by_tag('exit')[0]

        # Step 1: Teleport to key and pick it up
        env.move_entity(p.id, k.x - 1 if k.x > 5 else k.x + 1, k.y)
        p = env.get_entities_by_tag('player')[0]
        if p.x < k.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_key'] == 1

        # Step 2: Go to door and interact
        env.move_entity(p.id, door.x - 1, door.y)
        env.step(action_index(env, 'interact'))
        assert len(env.get_entities_by_type('door')) == 0

        # Step 3: Go to exit
        env.move_entity(p.id, ex.x - 1 if ex.x > 9 else ex.x + 1, ex.y)
        p = env.get_entities_by_tag('player')[0]
        if p.x < ex.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        assert env.status == 'won'

    def test_no_lose_condition(self):
        """Game cannot be lost — only won or truncated."""
        env = make_env()
        for _ in range(300):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert env.status in ('won', 'playing')


# ---- Fuzz ----

class TestLockAndKeyFuzz:
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

class TestLockAndKeyInvariants:
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

    def test_exit_reachable_after_door_destroyed(self):
        """Exit IS reachable once the door is destroyed."""
        from asciiswarm.kernel.invariants import check_exit_reachable
        env = make_env()
        door = env.get_entities_by_type('door')[0]
        env.destroy_entity(door.id)
        check_exit_reachable(env)
