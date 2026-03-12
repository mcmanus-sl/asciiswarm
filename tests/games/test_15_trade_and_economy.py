"""Tests for Game 15: Trade & Economy."""

import importlib
import pytest

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants

game_module = importlib.import_module('games.15_trade_and_economy')


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

class TestTradeEconomyMechanics:
    def test_entities_exist(self):
        env = make_env()
        assert len(env.get_entities_by_tag('player')) == 1
        assert len(env.get_entities_by_tag('exit')) == 1
        assert len(env.get_entities_by_type('merchant_a')) == 1
        assert len(env.get_entities_by_type('merchant_b')) == 1
        bandits = env.get_entities_by_type('bandit')
        assert 3 <= len(bandits) <= 4

    def test_player_starts_with_correct_properties(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['gold'] == 15
        assert p.properties['goods_a'] == 0
        assert p.properties['goods_b'] == 0
        assert p.properties['health'] == 5

    def test_player_position(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        assert p.x == 2
        assert p.y == 5

    def test_movement(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        orig_x = p.x
        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        assert p.x == orig_x + 1

    def test_walls_block(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        west = action_index(env, 'move_w')
        for _ in range(10):
            env.step(west)
        p = env.get_entities_by_tag('player')[0]
        assert p.x >= 1

    def test_buy_from_merchant_a(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        m = env.get_entities_by_type('merchant_a')[0]
        env.move_entity(p.id, m.x + 1, m.y)
        p = env.get_entities_by_tag('player')[0]
        initial_gold = p.properties['gold']

        env.step(action_index(env, 'buy'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['goods_a'] == 1
        assert p.properties['gold'] == initial_gold - 5

    def test_sell_to_merchant_b(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        m = env.get_entities_by_type('merchant_b')[0]
        p.properties['goods_a'] = 3
        env.move_entity(p.id, m.x - 1, m.y)

        env.step(action_index(env, 'sell'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['goods_a'] == 2
        assert p.properties['gold'] == 15 + 12

    def test_price_increases_on_buy(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        m = env.get_entities_by_type('merchant_a')[0]
        env.move_entity(p.id, m.x + 1, m.y)
        p.properties['gold'] = 100

        buy = action_index(env, 'buy')
        env.step(buy)
        env.step(buy)
        p = env.get_entities_by_tag('player')[0]
        # First buy: 5 gold, second buy: 6 gold = 11 total
        assert p.properties['gold'] == 100 - 11
        assert p.properties['goods_a'] == 2

    def test_cant_buy_without_gold(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        m = env.get_entities_by_type('merchant_a')[0]
        env.move_entity(p.id, m.x + 1, m.y)
        p.properties['gold'] = 0

        env.step(action_index(env, 'buy'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['goods_a'] == 0

    def test_cant_sell_without_goods(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        m = env.get_entities_by_type('merchant_b')[0]
        env.move_entity(p.id, m.x - 1, m.y)

        initial_gold = p.properties['gold']
        env.step(action_index(env, 'sell'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['gold'] == initial_gold

    def test_harbor_requires_50_gold(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        harbor = env.get_entities_by_type('harbor')[0]
        env.move_entity(p.id, harbor.x - 1, harbor.y)
        p.properties['gold'] = 30

        env.step(action_index(env, 'move_e'))
        assert env.status == 'playing'

    def test_harbor_win_with_50_gold(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        harbor = env.get_entities_by_type('harbor')[0]
        env.move_entity(p.id, harbor.x - 1, harbor.y)
        p.properties['gold'] = 50

        env.step(action_index(env, 'move_e'))
        assert env.status == 'won'

    def test_bandit_combat(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        bandits = env.get_entities_by_type('bandit')
        if not bandits:
            return
        b = bandits[0]
        b_id = b.id
        # Place player adjacent without triggering collisions (use direct position)
        # Find an empty cell adjacent to bandit
        for dx in [-1, 1]:
            nx = b.x + dx
            if 1 <= nx < 19:
                ents = env.get_entities_at(nx, b.y)
                if not any(e.has_tag('hazard') or e.has_tag('solid') for e in ents):
                    env.move_entity(p.id, nx, b.y)
                    p = env.get_entities_by_tag('player')[0]
                    break
        p.properties['health'] = 5
        initial_gold = p.properties['gold']

        # Move toward bandit
        if p.x < b.x:
            env.step(action_index(env, 'move_e'))
        else:
            env.step(action_index(env, 'move_w'))
        p = env.get_entities_by_tag('player')[0]
        # Bandit should be dead (hp=2, player atk=2)
        assert env.get_entity(b_id) is None
        assert p.properties['health'] < 5  # took damage from combat
        assert p.properties['gold'] >= initial_gold + 5

    def test_player_dies_at_0_health(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        bandits = env.get_entities_by_type('bandit')
        if not bandits:
            return
        b = bandits[0]
        env.move_entity(p.id, b.x - 1, b.y)
        p.properties['health'] = 1

        env.step(action_index(env, 'move_e'))
        assert env.status == 'lost'

    def test_gold_pile_pickup(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        piles = env.get_entities_by_type('gold_pile')
        if not piles:
            return
        g = piles[0]
        initial_count = len(piles)
        env.move_entity(p.id, g.x - 1, g.y)
        initial_gold = p.properties['gold']

        env.step(action_index(env, 'move_e'))
        p = env.get_entities_by_tag('player')[0]
        assert len(env.get_entities_by_type('gold_pile')) == initial_count - 1
        assert p.properties['gold'] == initial_gold + 5

    def test_rest_stop_heals(self):
        env = make_env()
        p = env.get_entities_by_tag('player')[0]
        rests = env.get_entities_by_type('rest_stop')
        if not rests:
            return
        r = rests[0]
        env.move_entity(p.id, r.x + 1, r.y)
        p.properties['health'] = 2

        env.step(action_index(env, 'interact'))
        p = env.get_entities_by_tag('player')[0]
        assert p.properties['health'] == 4


# ---- Fuzz ----

class TestTradeEconomyFuzz:
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

class TestTradeEconomyInvariants:
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
