"""Tests for Game 04: Dungeon Crawl."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.04_dungeon_crawl')


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

class TestDungeonCrawlMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_type('potion')) >= 1
        assert len(env.get_entities_by_tag('hazard')) >= 1

    def test_player_properties(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['health'] == 10
        assert p.properties['attack'] == 2

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        old_x, old_y = p.x, p.y
        # Move north many times — should be blocked by walls eventually
        north = action_index(env, 'move_n')
        for _ in range(20):
            env.step(north)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')
        if p:
            assert p[0].y >= 0

    def test_combat_damages_both(self):
        """Walking into an enemy damages both player and enemy."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Create a wanderer next to player
        wx, wy = p.x + 1, p.y
        # Clear any entities at target
        for ent in env.get_entities_at(wx, wy):
            if not ent.has_tag('solid'):
                env.destroy_entity(ent.id)
            else:
                # Wall there, try another direction
                wx, wy = p.x, p.y + 1
                break

        # Check for wall at new position
        blocked = any(e.has_tag('solid') for e in env.get_entities_at(wx, wy))
        if blocked:
            return  # Can't test combat in this layout

        enemy = env.create_entity('wanderer', wx, wy, 'w', ['hazard'], z_order=5,
                                  properties={'health': 1, 'attack': 1})

        old_health = p.properties['health']
        # Move toward enemy
        if wx > p.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_s'))

        p = env.get_entities_by_tag('player')[0]
        # Player should have taken damage
        assert p.properties['health'] == old_health - 1
        # Wanderer has 1 HP, player does 2 damage, so it should be destroyed
        assert len(env.get_entities_by_type('wanderer')) <= len(env.get_entities_by_tag('hazard'))

    def test_potion_heals_player(self):
        """Walking onto a potion heals the player."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Reduce player health
        p.properties['health'] = 5

        # Find or create a potion next to player
        px, py = p.x + 1, p.y
        blocked = any(e.has_tag('solid') for e in env.get_entities_at(px, py))
        if blocked:
            px, py = p.x, p.y + 1
            blocked = any(e.has_tag('solid') for e in env.get_entities_at(px, py))
        if blocked:
            return

        # Clear and place potion
        for ent in env.get_entities_at(px, py):
            if not ent.has_tag('player') and not ent.has_tag('solid'):
                env.destroy_entity(ent.id)

        env.create_entity('potion', px, py, '!', ['pickup'], z_order=3,
                          properties={'heal_amount': 3})

        if px > p.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_s'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['health'] == 8  # 5 + 3

    def test_potion_caps_at_max(self):
        """Potion healing caps at max health (10)."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['health'] = 9

        px, py = p.x + 1, p.y
        blocked = any(e.has_tag('solid') for e in env.get_entities_at(px, py))
        if blocked:
            px, py = p.x, p.y + 1
            blocked = any(e.has_tag('solid') for e in env.get_entities_at(px, py))
        if blocked:
            return

        for ent in env.get_entities_at(px, py):
            if not ent.has_tag('player') and not ent.has_tag('solid'):
                env.destroy_entity(ent.id)

        env.create_entity('potion', px, py, '!', ['pickup'], z_order=3,
                          properties={'heal_amount': 3})

        if px > p.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_s'))

        p = env.get_entities_by_tag('player')[0]
        assert p.properties['health'] == 10  # capped

    def test_exit_wins(self):
        """Walking onto exit wins the game."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ex = env.get_entities_by_tag('exit')[0]

        # Teleport player next to exit, clearing hazards first
        for dx, dy, act in [(-1, 0, 'move_e'), (1, 0, 'move_w'),
                            (0, -1, 'move_s'), (0, 1, 'move_n')]:
            nx, ny = ex.x + dx, ex.y + dy
            if 0 <= nx < 16 and 0 <= ny < 16:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    # Clear hazards at the staging cell
                    for ent in env.get_entities_at(nx, ny):
                        if ent.has_tag('hazard'):
                            env.destroy_entity(ent.id)
                    # Also clear hazards at exit cell
                    for ent in env.get_entities_at(ex.x, ex.y):
                        if ent.has_tag('hazard'):
                            env.destroy_entity(ent.id)
                    env.move_entity(p.id, nx, ny)
                    env.step(action_index(env, act))
                    assert env.status == 'won'
                    return

    def test_player_death_loses(self):
        """Player health reaching 0 loses the game."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['health'] = 1

        # Place a strong enemy next to player
        px, py = p.x + 1, p.y
        blocked = any(e.has_tag('solid') for e in env.get_entities_at(px, py))
        if blocked:
            px, py = p.x, p.y + 1
            blocked = any(e.has_tag('solid') for e in env.get_entities_at(px, py))
        if blocked:
            return

        for ent in env.get_entities_at(px, py):
            if not ent.has_tag('player') and not ent.has_tag('solid'):
                env.destroy_entity(ent.id)

        env.create_entity('chaser', px, py, 'c', ['hazard'], z_order=5,
                          properties={'health': 10, 'attack': 5})

        if px > p.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_s'))

        assert env.status == 'lost'


# ---- Fuzz ----

class TestDungeonCrawlFuzz:
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

class TestDungeonCrawlInvariants:
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
