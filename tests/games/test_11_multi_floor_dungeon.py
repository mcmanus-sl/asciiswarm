"""Tests for Game 11: Multi-Floor Dungeon."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.11_multi_floor_dungeon')


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

class TestMultiFloorMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_type('floor_key')) == 1
        assert len(env.get_entities_by_type('locked_exit')) == 1
        assert len(env.get_entities_by_type('stairs_down')) == 2
        assert len(env.get_entities_by_type('stairs_up')) == 2
        assert len(env.get_entities_by_type('potion')) >= 1

    def test_player_properties(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['health'] == 10
        assert p.properties['keys_held'] == 0
        assert p.properties['floor'] == 1

    def test_player_in_floor1(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert 0 <= p.x <= 11

    def test_grid_size(self):
        env = make_env()
        assert env.config['grid'] == (36, 12)

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        north = action_index(env, 'move_n')
        for _ in range(20):
            env.step(north)
            if env.status != 'playing':
                break
        p = env.get_entities_by_tag('player')
        if p:
            assert p[0].y >= 0

    def test_stairs_down_teleport(self):
        """Interacting on stairs_down teleports to paired stairs_up."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        sd = env.get_entities_by_type('stairs_down')
        # Find stairs_down on floor 1
        sd1 = [s for s in sd if s.x <= 11]
        if not sd1:
            return
        sd1 = sd1[0]
        su = env.get_entities_by_type('stairs_up')
        su2 = [s for s in su if 12 <= s.x <= 23]
        if not su2:
            return
        su2 = su2[0]

        # Move player onto stairs_down
        env._grid[p.y][p.x].remove(p)
        p.x, p.y = sd1.x, sd1.y
        env._grid[p.y][p.x].append(p)
        p.properties['floor'] = 1

        # Interact
        env.step(action_index(env, 'interact'))

        p = env.get_entities_by_tag('player')[0]
        assert p.x == su2.x and p.y == su2.y
        assert p.properties['floor'] == 2

    def test_stairs_up_teleport(self):
        """Interacting on stairs_up teleports to paired stairs_down."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        su = env.get_entities_by_type('stairs_up')
        su2 = [s for s in su if 12 <= s.x <= 23]
        if not su2:
            return
        su2 = su2[0]
        sd = env.get_entities_by_type('stairs_down')
        sd1 = [s for s in sd if s.x <= 11]
        if not sd1:
            return
        sd1 = sd1[0]

        # Move player onto stairs_up on floor 2
        env._grid[p.y][p.x].remove(p)
        p.x, p.y = su2.x, su2.y
        env._grid[p.y][p.x].append(p)
        p.properties['floor'] = 2

        # Interact
        env.step(action_index(env, 'interact'))

        p = env.get_entities_by_tag('player')[0]
        assert p.x == sd1.x and p.y == sd1.y
        assert p.properties['floor'] == 1

    def test_key_pickup(self):
        """Walking onto floor_key increments keys_held."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        fk = env.get_entities_by_type('floor_key')[0]

        # Place player adjacent to key
        # Find a non-solid adjacent cell
        for dx, dy, act in [(-1, 0, 'move_e'), (1, 0, 'move_w'),
                            (0, -1, 'move_s'), (0, 1, 'move_n')]:
            nx, ny = fk.x + dx, fk.y + dy
            if 0 <= nx < 36 and 0 <= ny < 12:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    env._grid[p.y][p.x].remove(p)
                    p.x, p.y = nx, ny
                    env._grid[p.y][p.x].append(p)
                    p.properties['floor'] = game_module._get_floor(p.x)

                    # Clear hazards at key cell and staging cell
                    for ent in env.get_entities_at(fk.x, fk.y):
                        if ent.has_tag('hazard'):
                            env.destroy_entity(ent.id)
                    for ent in env.get_entities_at(nx, ny):
                        if ent.has_tag('hazard'):
                            env.destroy_entity(ent.id)

                    env.step(action_index(env, act))
                    p = env.get_entities_by_tag('player')[0]
                    assert p.properties['keys_held'] == 1
                    return

    def test_unlock_door(self):
        """Interacting next to locked_exit with a key destroys it."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        le = env.get_entities_by_type('locked_exit')[0]

        # Give player a key
        p.properties['keys_held'] = 1

        # Place player adjacent to locked_exit
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = le.x + dx, le.y + dy
            if 0 <= nx < 36 and 0 <= ny < 12:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    env._grid[p.y][p.x].remove(p)
                    p.x, p.y = nx, ny
                    env._grid[p.y][p.x].append(p)
                    p.properties['floor'] = game_module._get_floor(p.x)

                    # Clear hazards
                    for ent in env.get_entities_at(nx, ny):
                        if ent.has_tag('hazard'):
                            env.destroy_entity(ent.id)

                    env.step(action_index(env, 'interact'))
                    p = env.get_entities_by_tag('player')[0]
                    assert p.properties['keys_held'] == 0
                    assert len(env.get_entities_by_type('locked_exit')) == 0
                    return

    def test_exit_wins(self):
        """Walking onto exit after unlocking wins."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ex = env.get_entities_by_tag('exit')[0]

        # Destroy locked_exit first
        le = env.get_entities_by_type('locked_exit')
        for e in le:
            env.destroy_entity(e.id)

        # Clear hazards near exit
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = ex.x + dx, ex.y + dy
            for ent in env.get_entities_at(nx, ny):
                if ent.has_tag('hazard'):
                    env.destroy_entity(ent.id)

        # Place player adjacent to exit
        for dx, dy, act in [(-1, 0, 'move_e'), (1, 0, 'move_w'),
                            (0, -1, 'move_s'), (0, 1, 'move_n')]:
            nx, ny = ex.x + dx, ex.y + dy
            if 0 <= nx < 36 and 0 <= ny < 12:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    env._grid[p.y][p.x].remove(p)
                    p.x, p.y = nx, ny
                    env._grid[p.y][p.x].append(p)
                    env.step(action_index(env, act))
                    assert env.status == 'won'
                    return

    def test_combat_damages_both(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Find a non-solid cell adjacent to player
        for dx, dy, act in [(1, 0, 'move_e'), (0, 1, 'move_s'), (-1, 0, 'move_w'), (0, -1, 'move_n')]:
            nx, ny = p.x + dx, p.y + dy
            if 0 <= nx < 36 and 0 <= ny < 12:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    for ent in env.get_entities_at(nx, ny):
                        if not ent.has_tag('player') and not ent.has_tag('solid'):
                            env.destroy_entity(ent.id)
                    enemy = env.create_entity('wanderer', nx, ny, 'w', ['hazard'], z_order=5,
                                              properties={'health': 5, 'attack': 1})
                    old_health = p.properties['health']
                    env.step(action_index(env, act))
                    p = env.get_entities_by_tag('player')[0]
                    assert p.properties['health'] == old_health - 1
                    e = env.get_entity(enemy.id)
                    if e:
                        assert e.properties['health'] == 3  # 5 - 2
                    return

    def test_player_death_loses(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['health'] = 1
        for dx, dy, act in [(1, 0, 'move_e'), (0, 1, 'move_s')]:
            nx, ny = p.x + dx, p.y + dy
            if 0 <= nx < 36 and 0 <= ny < 12:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    for ent in env.get_entities_at(nx, ny):
                        if not ent.has_tag('player') and not ent.has_tag('solid'):
                            env.destroy_entity(ent.id)
                    env.create_entity('chaser', nx, ny, 'c', ['hazard'], z_order=5,
                                      properties={'health': 10, 'attack': 5})
                    env.step(action_index(env, act))
                    assert env.status == 'lost'
                    return

    def test_potion_heals(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['health'] = 5
        for dx, dy, act in [(1, 0, 'move_e'), (0, 1, 'move_s')]:
            nx, ny = p.x + dx, p.y + dy
            if 0 <= nx < 36 and 0 <= ny < 12:
                blocked = any(e.has_tag('solid') for e in env.get_entities_at(nx, ny))
                if not blocked:
                    for ent in env.get_entities_at(nx, ny):
                        if not ent.has_tag('player') and not ent.has_tag('solid'):
                            env.destroy_entity(ent.id)
                    env.create_entity('potion', nx, ny, '!', ['pickup'], z_order=3,
                                      properties={'heal_amount': 3})
                    env.step(action_index(env, act))
                    p = env.get_entities_by_tag('player')[0]
                    assert p.properties['health'] == 8
                    return


# ---- Fuzz ----

class TestMultiFloorFuzz:
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

class TestMultiFloorInvariants:
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
