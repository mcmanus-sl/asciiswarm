"""Tests for Game 07: Hunger Clock."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.07_hunger_clock')


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

class TestHungerClockMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        foods = env.get_entities_by_type('food')
        assert 10 <= len(foods) <= 15

    def test_player_starts_at_correct_position(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert (p.x, p.y) == (0, 13)

    def test_exit_at_correct_position(self):
        env = make_env()
        e = env.get_entities_by_tag('exit')[0]
        assert (e.x, e.y) == (13, 0)

    def test_player_starts_with_food_20(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['food'] == 20

    def test_walls_block_movement(self):
        """Player cannot walk through walls."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Move west from (0, 13) — out of bounds, should stay
        west = action_index(env, 'move_w')
        env.step(west)
        p = env.get_entities_by_tag('player')[0]
        assert p.x == 0

    def test_food_decreases_each_turn(self):
        """Player food decreases by 1 each turn."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        initial_food = p.properties['food']
        wait = action_index(env, 'wait')
        env.step(wait)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['food'] == initial_food - 1

    def test_starvation_loses_game(self):
        """Player loses when food reaches 0."""
        env = make_env()
        wait = action_index(env, 'wait')
        # Wait 20 turns to starve
        for _ in range(20):
            obs, r, term, trunc, info = env.step(wait)
            if term or trunc:
                break
        assert env.status == 'lost'

    def test_collect_food_restores(self):
        """Collecting food increases food property."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        foods = env.get_entities_by_type('food')
        if not foods:
            return
        # Teleport player next to a food
        f = foods[0]
        target_x = f.x - 1 if f.x > 0 else f.x + 1
        env.move_entity(p.id, target_x, f.y)
        p = env.get_entities_by_tag('player')[0]

        # Reduce food to test healing
        p.properties['food'] = 10

        # Move onto the food
        if p.x < f.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        # Food should have increased by 5 (to 15), then decreased by 1 (turn_end) = 14
        assert p.properties['food'] == 14

    def test_food_capped_at_20(self):
        """Food cannot exceed 20."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        foods = env.get_entities_by_type('food')
        if not foods:
            return
        f = foods[0]
        target_x = f.x - 1 if f.x > 0 else f.x + 1
        env.move_entity(p.id, target_x, f.y)
        p = env.get_entities_by_tag('player')[0]

        # Set food to max
        p.properties['food'] = 20

        if p.x < f.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        # Capped at 20, minus 1 for turn_end = 19
        assert p.properties['food'] == 19

    def test_reach_exit_wins(self):
        """Player wins by walking onto exit."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ex = env.get_entities_by_tag('exit')[0]

        # Teleport player next to exit
        env.move_entity(p.id, ex.x - 1, ex.y)
        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_movement_works(self):
        """Player can move in all directions."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Player starts at (0, 13). Move north.
        env.step(action_index(env, 'move_n'))
        p = env.get_entities_by_tag('player')[0]
        assert p.y == 12

    def test_food_entity_destroyed_on_pickup(self):
        """Food entity is destroyed when collected."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        foods = env.get_entities_by_type('food')
        initial_count = len(foods)
        if not foods:
            return
        f = foods[0]
        target_x = f.x - 1 if f.x > 0 else f.x + 1
        env.move_entity(p.id, target_x, f.y)
        if p.x < f.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        assert len(env.get_entities_by_type('food')) == initial_count - 1


# ---- Fuzz ----

class TestHungerClockFuzz:
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

class TestHungerClockInvariants:
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
