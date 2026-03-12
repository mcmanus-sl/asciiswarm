"""Tests for Game 09: Inventory & Crafting."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.09_inventory_crafting')


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

class TestInventoryCraftingMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_type('workbench')) == 1
        assert len(env.get_entities_by_type('rubble')) == 1
        assert len(env.get_entities_by_type('wood')) >= 4
        assert len(env.get_entities_by_type('ore')) >= 3

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        north = action_index(env, 'move_n')
        start_y = p.y
        for _ in range(20):
            env.step(north)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')[0]
        assert p.y >= 1  # blocked by wall at y=0

    def test_pickup_wood(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wood = env.get_entities_by_type('wood')[0]

        # Teleport player next to wood
        target_x = wood.x - 1 if wood.x > 1 else wood.x + 1
        env.move_entity(p.id, target_x, wood.y)
        p = env.get_entities_by_tag('player')[0]

        assert p.properties['wood'] == 0
        wood_count_before = len(env.get_entities_by_type('wood'))

        if p.x < wood.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['wood'] == 1
        assert len(env.get_entities_by_type('wood')) == wood_count_before - 1

    def test_pickup_ore(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ore = env.get_entities_by_type('ore')[0]

        target_x = ore.x - 1 if ore.x > 1 else ore.x + 1
        env.move_entity(p.id, target_x, ore.y)
        p = env.get_entities_by_tag('player')[0]

        ore_before = p.properties['ore']
        ore_count_before = len(env.get_entities_by_type('ore'))

        if p.x < ore.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['ore'] == ore_before + 1
        assert len(env.get_entities_by_type('ore')) == ore_count_before - 1

    def test_craft_pickaxe(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wb = env.get_entities_by_type('workbench')[0]

        # Give player materials
        p.properties['wood'] = 2
        p.properties['ore'] = 2

        # Place player adjacent to workbench
        env.move_entity(p.id, wb.x - 1, wb.y)

        env.step(action_index(env, 'interact'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_pickaxe'] == 1
        assert p.properties['wood'] == 0
        assert p.properties['ore'] == 0

    def test_craft_without_materials_fails(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wb = env.get_entities_by_type('workbench')[0]

        # No materials
        env.move_entity(p.id, wb.x - 1, wb.y)
        env.step(action_index(env, 'interact'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_pickaxe'] == 0

    def test_mine_rubble(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rubble = env.get_entities_by_type('rubble')[0]

        # Give player pickaxe
        p.properties['has_pickaxe'] = 1

        # Place player adjacent to rubble
        env.move_entity(p.id, rubble.x - 1, rubble.y)

        env.step(action_index(env, 'interact'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_pickaxe'] == 0
        assert len(env.get_entities_by_type('rubble')) == 0

    def test_mine_without_pickaxe_fails(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rubble = env.get_entities_by_type('rubble')[0]

        env.move_entity(p.id, rubble.x - 1, rubble.y)
        env.step(action_index(env, 'interact'))

        assert len(env.get_entities_by_type('rubble')) == 1

    def test_rubble_blocks_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rubble = env.get_entities_by_type('rubble')[0]

        env.move_entity(p.id, rubble.x - 1, rubble.y)
        env.step(action_index(env, 'move_e'))

        p = env.get_entities_by_tag('player')[0]
        assert p.x == rubble.x - 1  # blocked

    def test_full_sequence_wins(self):
        """Craft pickaxe, mine rubble, reach exit — wins the game."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wb = env.get_entities_by_type('workbench')[0]
        rubble = env.get_entities_by_type('rubble')[0]
        exit_ent = env.get_entities_by_tag('exit')[0]

        # Give player materials directly (pickup tested separately)
        p.properties['wood'] = 2
        p.properties['ore'] = 2

        # Craft pickaxe
        env.move_entity(p.id, wb.x - 1, wb.y)
        env.step(action_index(env, 'interact'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_pickaxe'] == 1

        # Mine rubble
        env.move_entity(p.id, rubble.x - 1, rubble.y)
        env.step(action_index(env, 'interact'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_pickaxe'] == 0
        assert len(env.get_entities_by_type('rubble')) == 0

        # Walk to exit
        env.move_entity(p.id, exit_ent.x - 1, exit_ent.y)
        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_wood_cap_at_5(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['wood'] = 5

        # Try picking up more wood
        wood = env.get_entities_by_type('wood')[0]
        target_x = wood.x - 1 if wood.x > 1 else wood.x + 1
        env.move_entity(p.id, target_x, wood.y)
        p = env.get_entities_by_tag('player')[0]
        if p.x < wood.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['wood'] == 5  # capped


# ---- Fuzz ----

class TestInventoryCraftingFuzz:
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

class TestInventoryCraftingInvariants:
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
