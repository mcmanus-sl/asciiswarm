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
        assert len(env.get_entities_by_type('wood')) >= 2
        assert len(env.get_entities_by_type('ore')) >= 2

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        east = action_index(env, 'move_e')
        for _ in range(20):
            env.step(east)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')[0]
        # Should be blocked by rubble wall at x=12
        assert p.x <= 11

    def test_pick_up_wood(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wood = env.get_entities_by_type('wood')[0]
        # Teleport player next to wood
        env.move_entity(p.id, wood.x - 1, wood.y)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['wood'] == 0
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['wood'] == 1

    def test_pick_up_ore(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Create a fresh ore at a known empty spot
        ore_ent = env.create_entity('ore', 5, 5, 'o', ['pickup'], z_order=3)
        env.move_entity(p.id, 4, 5)
        assert p.properties['ore'] == 0
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['ore'] == 1

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
        p.properties['wood'] = 1
        p.properties['ore'] = 2
        env.move_entity(p.id, wb.x - 1, wb.y)
        env.step(action_index(env, 'interact'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['has_pickaxe'] == 0

    def test_mine_rubble(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rubble = env.get_entities_by_type('rubble')[0]
        p.properties['has_pickaxe'] = 1
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

    def test_full_sequence_wins(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wb = env.get_entities_by_type('workbench')[0]
        rubble = env.get_entities_by_type('rubble')[0]
        ex = env.get_entities_by_tag('exit')[0]

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

        # Reach exit
        env.move_entity(p.id, ex.x - 1, ex.y)
        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_no_lose_condition(self):
        env = make_env()
        for _ in range(400):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert env.status in ('won', 'playing')

    def test_wood_cap_at_5(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['wood'] = 5
        # Create a wood entity and walk onto it
        wood_ent = env.create_entity('wood', 5, 5, 't', ['pickup'], z_order=3)
        env.move_entity(p.id, 4, 5)
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['wood'] == 5

    def test_ore_cap_at_5(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['ore'] = 5
        ore_ent = env.create_entity('ore', 5, 5, 'o', ['pickup'], z_order=3)
        env.move_entity(p.id, 4, 5)
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['ore'] == 5


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

    def test_exit_reachable_after_rubble_destroyed(self):
        from asciiswarm.kernel.invariants import check_exit_reachable
        env = make_env()
        rubble = env.get_entities_by_type('rubble')[0]
        env.destroy_entity(rubble.id)
        check_exit_reachable(env)
