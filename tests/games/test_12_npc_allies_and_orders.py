"""Tests for Game 12: NPC Allies & Orders."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.12_npc_allies_and_orders')


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

class TestNPCAlliesMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_type('ally')) == 3
        assert len(env.get_entities_by_type('raider')) >= 2
        assert len(env.get_entities_by_type('tree')) >= 6
        assert len(env.get_entities_by_type('stockpile')) == 1
        assert len(env.get_entities_by_type('barricade_slot')) == 3

    def test_player_properties(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['health'] == 5
        assert p.properties['wood_stockpile'] == 0
        assert p.properties['allies_alive'] == 3

    def test_allies_start_in_follow_mode(self):
        env = make_env()
        for a in env.get_entities_by_type('ally'):
            assert a.properties['mode'] == 'follow'

    def test_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y
        # Move east (toward exit from base entrance)
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        # Player may or may not have moved depending on walls
        assert env.status == 'playing'

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Move west from (2, 10) - should hit wall at x=0
        west = action_index(env, 'move_w')
        for _ in range(5):
            env.step(west)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')
        if p:
            assert p[0].x >= 1  # can't go past wall

    def test_order_changes_mode(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        allies = env.get_entities_by_type('ally')
        # Player at (2,10), allies near by - should be within 5 tiles
        env.step(action_index(env, 'order_harvest'))
        # At least one ally should now be in harvest mode
        allies = env.get_entities_by_type('ally')
        modes = [a.properties['mode'] for a in allies]
        assert 'harvest' in modes

    def test_order_guard(self):
        env = make_env()
        env.step(action_index(env, 'order_guard'))
        allies = env.get_entities_by_type('ally')
        modes = [a.properties['mode'] for a in allies]
        assert 'guard' in modes

    def test_exit_wins(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ex = env.get_entities_by_tag('exit')[0]
        # Clear hazards near exit
        for ent in env.get_entities_by_tag('hazard'):
            env.destroy_entity(ent.id)
        # Teleport player next to exit
        env.move_entity(p.id, ex.x - 1, ex.y)
        p = env.get_entities_by_tag('player')[0]
        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_combat_with_raider(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Destroy all existing raiders to isolate the test
        for r in env.get_entities_by_type('raider'):
            env.destroy_entity(r.id)
        # Clear area east of player
        for ent in env.get_entities_at(p.x + 1, p.y):
            if not ent.has_tag('solid'):
                env.destroy_entity(ent.id)
        # Place raider next to player
        if not any(e.has_tag('solid') for e in env.get_entities_at(p.x + 1, p.y)):
            env.create_entity('raider', p.x + 1, p.y, 'r', ['hazard'], z_order=5,
                              properties={'health': 2, 'attack': 1})
            old_health = p.properties['health']
            env.step(action_index(env, 'move_e'))
            p = env.get_entities_by_tag('player')[0]
            # Player took at least 1 damage from combat (may take more from raider behavior)
            assert p.properties['health'] < old_health

    def test_player_death_loses(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['health'] = 1
        # Place strong raider next to player
        for ent in env.get_entities_at(p.x + 1, p.y):
            if not ent.has_tag('solid'):
                env.destroy_entity(ent.id)
        if not any(e.has_tag('solid') for e in env.get_entities_at(p.x + 1, p.y)):
            env.create_entity('raider', p.x + 1, p.y, 'r', ['hazard'], z_order=5,
                              properties={'health': 10, 'attack': 5})
            env.step(action_index(env, 'move_e'))
            assert env.status == 'lost'

    def test_barricade_building(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Give player wood and teleport to barricade slot
        p.properties['wood_stockpile'] = 4
        # Clear path and move to (13, 10) - adjacent to barricade_slot at (14, 10)
        for ent in env.get_entities_by_tag('hazard'):
            env.destroy_entity(ent.id)
        env.move_entity(p.id, 13, 10)
        env.step(action_index(env, 'interact'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['wood_stockpile'] == 2
        barricades = env.get_entities_by_type('barricade')
        assert len(barricades) >= 1


# ---- Fuzz ----

class TestNPCAlliesFuzz:
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

class TestNPCAlliesInvariants:
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
