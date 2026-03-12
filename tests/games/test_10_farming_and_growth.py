"""Tests for Game 10: Farming & Growth."""

import importlib
import pytest
from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.10_farming_and_growth')


def make_env(seed=42):
    env = GridGameEnv(game_module, seed=seed)
    env.reset(seed=seed)
    return env


def action_index(env, name):
    for i, a in env.ACTION_MAP.items():
        if a == name:
            return i
    raise ValueError(f"Action {name} not found")


class TestSetup:
    def test_player_exists(self):
        env = make_env()
        players = env.get_entities_by_tag('player')
        assert len(players) == 1

    def test_player_in_farmhouse(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert 1 <= p.x <= 3 and 1 <= p.y <= 3

    def test_player_starts_empty(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['seeds'] == 0
        assert p.properties['crops'] == 0
        assert p.properties['delivered'] == 0

    def test_nine_soil_tiles(self):
        env = make_env()
        soils = env.get_entities_by_type('soil')
        assert len(soils) == 9
        for s in soils:
            assert 5 <= s.x <= 7 and 5 <= s.y <= 7

    def test_one_seedbag(self):
        env = make_env()
        bags = env.get_entities_by_type('seedbag')
        assert len(bags) == 1
        assert bags[0].x == 2 and bags[0].y == 5

    def test_one_bin(self):
        env = make_env()
        bins = env.get_entities_by_type('bin')
        assert len(bins) == 1
        assert bins[0].x == 12 and bins[0].y == 1

    def test_no_sprouts_at_start(self):
        env = make_env()
        assert len(env.get_entities_by_type('sprout')) == 0
        assert len(env.get_entities_by_type('mature')) == 0


class TestMechanics:
    def test_pickup_seedbag(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        bag = env.get_entities_by_type('seedbag')[0]
        # Teleport player next to seedbag and walk into it
        env.move_entity(p.id, bag.x - 1, bag.y)
        move_e = action_index(env, 'move_e')
        env.step(move_e)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['seeds'] == 6
        assert len(env.get_entities_by_type('seedbag')) == 0

    def test_plant_seed_on_soil(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Give player seeds and move to soil
        p.properties['seeds'] = 3
        env.move_entity(p.id, 5, 5)
        interact = action_index(env, 'interact')
        env.step(interact)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['seeds'] == 2
        sprouts = env.get_entities_by_type('sprout')
        assert len(sprouts) == 1
        assert sprouts[0].x == 5 and sprouts[0].y == 5

    def test_cannot_plant_without_seeds(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        env.move_entity(p.id, 5, 5)
        interact = action_index(env, 'interact')
        env.step(interact)
        assert len(env.get_entities_by_type('sprout')) == 0

    def test_cannot_plant_on_existing_sprout(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['seeds'] = 3
        env.move_entity(p.id, 5, 5)
        interact = action_index(env, 'interact')
        env.step(interact)
        assert p.properties['seeds'] == 2
        # Try planting again on same tile
        env.step(interact)
        assert p.properties['seeds'] == 2  # no change
        assert len(env.get_entities_by_type('sprout')) == 1

    def test_sprout_grows_to_mature(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['seeds'] = 1
        env.move_entity(p.id, 5, 5)
        interact = action_index(env, 'interact')
        env.step(interact)
        # Wait 14 more turns (sprout ages once per turn including the plant turn's behavior)
        wait = action_index(env, 'wait')
        for _ in range(14):
            env.step(wait)
        # After 15 total behavior ticks, should be mature
        assert len(env.get_entities_by_type('sprout')) == 0
        mature = env.get_entities_by_type('mature')
        assert len(mature) == 1
        assert mature[0].x == 5 and mature[0].y == 5

    def test_harvest_mature_crop(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Create a mature crop manually
        env.create_entity('mature', 6, 6, '*', ['pickup'], z_order=3)
        env.move_entity(p.id, 5, 6)
        move_e = action_index(env, 'move_e')
        env.step(move_e)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['crops'] == 1
        assert len(env.get_entities_by_type('mature')) == 0

    def test_deliver_crop_to_bin(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['crops'] = 1
        # Move player next to bin (bin is at 12,1)
        env.move_entity(p.id, 11, 1)
        interact = action_index(env, 'interact')
        env.step(interact)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['crops'] == 0
        assert p.properties['delivered'] == 1

    def test_win_by_delivering_five(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['crops'] = 5
        env.move_entity(p.id, 11, 1)
        interact = action_index(env, 'interact')
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(interact)
            if i < 4:
                assert not terminated
            else:
                assert terminated
                assert info['status'] == 'won'

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        env.move_entity(p.id, 1, 1)
        move_n = action_index(env, 'move_n')
        env.step(move_n)
        p = env.get_entities_by_tag('player')[0]
        assert p.y == 1  # blocked by top wall


class TestFuzz:
    def test_fuzz_500_episodes(self):
        env = GridGameEnv(game_module, seed=42)
        for ep in range(500):
            obs, info = env.reset()
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                obs, r, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                steps += 1
                assert steps <= 350  # safety cap
            assert info['status'] in ('won', 'lost', 'playing')


class TestInvariants:
    def test_all_invariants(self):
        env = make_env()
        results = run_invariants(env, game_module)
        for name, passed, err in results:
            assert passed, f"Invariant {name} failed: {err}"

    def test_invariants_multiple_seeds(self):
        for seed in range(10):
            env = make_env(seed=seed)
            results = run_invariants(env, game_module)
            for name, passed, err in results:
                assert passed, f"Invariant {name} failed (seed={seed}): {err}"
