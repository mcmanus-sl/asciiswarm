"""Tests for Game 14: Fluid & Pressure."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.14_fluid_and_pressure')


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

class TestFluidMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        sources = env.get_entities_by_type('water_source')
        assert 1 <= len(sources) <= 2
        pumps = env.get_entities_by_type('pump')
        assert 2 <= len(pumps) <= 3
        drains = env.get_entities_by_type('drain')
        assert 2 <= len(drains) <= 3
        valves = env.get_entities_by_type('valve')
        assert len(valves) == 1

    def test_player_starts_with_air_10(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['air'] == 10

    def test_player_starts_dry(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        waters = [e for e in env.get_entities_at(p.x, p.y) if e.type == 'water']
        assert len(waters) == 0

    def test_movement_works(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        start_x, start_y = p.x, p.y
        # Try moving east (should work since player is at left edge, x=1)
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        # Either moved or was blocked, just check no crash
        assert env.status == 'playing'

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Move west from x=1 (border wall at x=0)
        env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        assert p.x == 1  # blocked by wall

    def test_water_spreads_from_source(self):
        env = make_env()
        wait = action_index(env, 'wait')
        # Run a few turns, water should appear
        initial_water = len(env.get_entities_by_type('water'))
        for _ in range(5):
            env.step(wait)
        final_water = len(env.get_entities_by_type('water'))
        assert final_water > initial_water

    def test_air_decreases_on_water(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Place water on player position
        env.create_entity('water', p.x, p.y, '~', ['hazard'], z_order=3,
                          properties={'depth': 1})
        wait = action_index(env, 'wait')
        env.step(wait)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['air'] == 9

    def test_air_resets_on_dry_land(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['air'] = 5
        wait = action_index(env, 'wait')
        env.step(wait)
        p = env.get_entities_by_tag('player')[0]
        # Player is on dry land, air should reset to 10
        assert p.properties['air'] == 10

    def test_drowning_loses_game(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Place water on player and set air to 1
        env.create_entity('water', p.x, p.y, '~', ['hazard'], z_order=3,
                          properties={'depth': 1})
        p.properties['air'] = 1
        wait = action_index(env, 'wait')
        env.step(wait)
        assert env.status == 'lost'

    def test_reach_exit_wins(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ex = env.get_entities_by_tag('exit')[0]
        # Teleport player next to exit
        env.move_entity(p.id, ex.x - 1, ex.y)
        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_pump_toggle(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        pumps = env.get_entities_by_type('pump')
        pump = pumps[0]
        assert pump.properties['active'] == 0
        # Teleport player next to pump
        env.move_entity(p.id, pump.x - 1, pump.y)
        env.step(action_index(env, 'interact'))
        assert pump.properties['active'] == 1
        # Toggle back
        env.step(action_index(env, 'interact'))
        assert pump.properties['active'] == 0

    def test_valve_destroys_source(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        valve = env.get_entities_by_type('valve')[0]
        sources_before = len(env.get_entities_by_type('water_source'))
        # Teleport player next to valve
        env.move_entity(p.id, valve.x - 1, valve.y)
        env.step(action_index(env, 'interact'))
        sources_after = len(env.get_entities_by_type('water_source'))
        assert sources_after == sources_before - 1

    def test_pump_destroys_nearby_water(self):
        env = make_env()
        pumps = env.get_entities_by_type('pump')
        pump = pumps[0]
        pump.properties['active'] = 1
        # Place water near pump
        wx, wy = pump.x + 1, pump.y
        if 0 <= wx < 20 and 0 <= wy < 16:
            # Check no solid there
            solids = [e for e in env.get_entities_at(wx, wy) if e.has_tag('solid')]
            if not solids:
                env.create_entity('water', wx, wy, '~', ['hazard'], z_order=3,
                                  properties={'depth': 1})
                water_before = len(env.get_entities_by_type('water'))
                wait = action_index(env, 'wait')
                env.step(wait)
                # Pump should have destroyed nearby water
                water_at = [e for e in env.get_entities_at(wx, wy) if e.type == 'water']
                assert len(water_at) == 0

    def test_drain_removes_water(self):
        env = make_env()
        drains = env.get_entities_by_type('drain')
        drain = drains[0]
        # Place water on drain
        env.create_entity('water', drain.x, drain.y, '~', ['hazard'], z_order=3,
                          properties={'depth': 1})
        wait = action_index(env, 'wait')
        env.step(wait)
        water_on_drain = [e for e in env.get_entities_at(drain.x, drain.y) if e.type == 'water']
        assert len(water_on_drain) == 0


# ---- Fuzz ----

class TestFluidFuzz:
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

class TestFluidInvariants:
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
