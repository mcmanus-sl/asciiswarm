"""Tests for Game 13: Ecology & Spawning."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.13_ecology_and_spawning')


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

class TestEcologyMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        rabbits = env.get_entities_by_type('rabbit')
        assert 8 <= len(rabbits) <= 10
        wolves = env.get_entities_by_type('wolf')
        assert 2 <= len(wolves) <= 3

    def test_player_starts_with_correct_properties(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['food'] == 15
        assert p.properties['health'] == 10

    def test_bushes_exist(self):
        env = make_env()
        bushes = env.get_entities_by_type('bush')
        assert 5 <= len(bushes) <= 7

    def test_wolves_far_from_player(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wolves = env.get_entities_by_type('wolf')
        for wolf in wolves:
            dist = abs(wolf.x - p.x) + abs(wolf.y - p.y)
            assert dist >= 8, f"Wolf at ({wolf.x}, {wolf.y}) too close: {dist}"

    def test_movement_works(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        start_y = p.y
        env.step(action_index(env, 'move_n'))
        p = env.get_entities_by_tag('player')[0]
        assert p.y == start_y - 1

    def test_walls_block_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        # Move west into boundary wall repeatedly
        west = action_index(env, 'move_w')
        for _ in range(5):
            env.step(west)
        p = env.get_entities_by_tag('player')[0]
        # Should be blocked by boundary wall at x=0
        assert p.x >= 1

    def test_player_hunts_rabbit(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rabbits = env.get_entities_by_type('rabbit')
        if not rabbits:
            return
        r = rabbits[0]
        initial_count = len(rabbits)

        # Teleport player next to rabbit
        target_x = r.x - 1 if r.x > 1 else r.x + 1
        env.move_entity(p.id, target_x, r.y)
        p = env.get_entities_by_tag('player')[0]
        p.properties['food'] = 10  # lower food for testing

        # Move onto rabbit
        if p.x < r.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        p = env.get_entities_by_tag('player')[0]
        # Food should be 10 + 3 = 13, possibly minus hunger tick
        assert p.properties['food'] >= 12
        # Rabbit count should decrease
        assert len(env.get_entities_by_type('rabbit')) < initial_count

    def test_hunger_clock(self):
        """Food decreases every 3 turns."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        initial_food = p.properties['food']
        wait = action_index(env, 'wait')
        # Wait 3 turns
        for _ in range(3):
            env.step(wait)
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['food'] == initial_food - 1

    def test_starvation_loses_game(self):
        """Player loses when food reaches 0."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['food'] = 1
        wait = action_index(env, 'wait')
        # Wait until food drops
        for _ in range(3):
            obs, r, term, trunc, info = env.step(wait)
            if term or trunc:
                break
        assert env.status == 'lost'

    def test_wolf_combat(self):
        """Player takes damage from wolf collision."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        wolves = env.get_entities_by_type('wolf')
        if not wolves:
            return
        wolf = wolves[0]
        # Teleport player next to wolf
        target_x = wolf.x - 1 if wolf.x > 1 else wolf.x + 1
        env.move_entity(p.id, target_x, wolf.y)
        p = env.get_entities_by_tag('player')[0]
        initial_health = p.properties['health']

        # Move onto wolf
        if p.x < wolf.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))

        p = env.get_entities_by_tag('player')[0]
        # Player should have taken at least 2 damage (wolf behavior may add more)
        assert p.properties['health'] <= initial_health - 2

    def test_reach_exit_wins(self):
        """Player wins by walking onto exit."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        ex = env.get_entities_by_tag('exit')[0]
        env.move_entity(p.id, ex.x - 1, ex.y)
        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_player_death_from_wolf(self):
        """Player loses when health reaches 0 from wolf attacks."""
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        p.properties['health'] = 2
        wolves = env.get_entities_by_type('wolf')
        if not wolves:
            return
        wolf = wolves[0]
        wolf.properties['health'] = 10  # keep wolf alive
        target_x = wolf.x - 1 if wolf.x > 1 else wolf.x + 1
        env.move_entity(p.id, target_x, wolf.y)

        if p.x < wolf.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        assert env.status == 'lost'


# ---- Fuzz ----

class TestEcologyFuzz:
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

class TestEcologyInvariants:
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
