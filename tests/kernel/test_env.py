import json
import pytest
import numpy as np
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


# ---- Entity management tests ----

class TestEntityManagement:
    def test_create_entity(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        wall = env.create_entity('wall', 3, 4, '#', ['solid'])
        assert wall.x == 3
        assert wall.y == 4
        assert wall.has_tag('solid')
        assert env.get_entity(wall.id) is wall

    def test_entity_at_correct_position(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        wall = env.create_entity('wall', 5, 5, '#', ['solid'])
        entities = env.get_entities_at(5, 5)
        assert wall in entities

    def test_invalid_tag_raises(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        with pytest.raises(ValueError, match="Unknown tag"):
            env.create_entity('thing', 0, 0, '?', ['nonexistent_tag'])

    def test_empty_tags_raises(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        with pytest.raises(ValueError, match="at least one tag"):
            env.create_entity('thing', 0, 0, '?', [])

    def test_move_entity(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]
        result = env.move_entity(player.id, 1, 0)
        assert result is True
        assert player.x == 1
        assert player.y == 0
        assert env.get_entities_at(0, 0) == []
        assert player in env.get_entities_at(1, 0)

    def test_move_out_of_bounds(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]
        result = env.move_entity(player.id, -1, 0)
        assert result is False
        assert player.x == 0

    def test_destroy_entity(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        wall = env.create_entity('wall', 3, 3, '#', ['solid'])
        env.destroy_entity(wall.id)
        assert env.get_entity(wall.id) is None
        assert env.get_entities_at(3, 3) == []
        assert wall not in env.get_entities_by_type('wall')

    def test_multiple_entities_same_cell(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        a = env.create_entity('npc', 3, 3, 'a', ['npc'], z_order=1)
        b = env.create_entity('npc', 3, 3, 'b', ['npc'], z_order=5)
        entities = env.get_entities_at(3, 3)
        assert len(entities) == 2
        assert a in entities
        assert b in entities

    def test_get_entities_by_tag(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('wall', 1, 0, '#', ['solid'])
            env.create_entity('wall', 2, 0, '#', ['solid'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        solids = env.get_entities_by_tag('solid')
        assert len(solids) == 2

    def test_get_all_entities_creation_order(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('wall', 1, 0, '#', ['solid'])
            env.create_entity('exit', 2, 0, '>', ['exit'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        all_ents = env.get_all_entities()
        assert [e.id for e in all_ents] == ['e1', 'e2', 'e3']

    def test_sequential_entity_ids(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('a', 1, 0, 'a', ['npc'])
            env.create_entity('b', 2, 0, 'b', ['npc'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        ids = [e.id for e in env.get_all_entities()]
        assert ids == ['e1', 'e2', 'e3']


# ---- Event integration tests ----

class TestEvents:
    def test_before_move_cancels(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.on('before_move', lambda e: e.cancel())
        player = env.get_entities_by_tag('player')[0]
        result = env.move_entity(player.id, 1, 0)
        assert result is False
        assert player.x == 0

    def test_collision_fires_on_occupied_cell(self):
        collisions = []

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 1, 0, '>', ['exit'])
            env.on('collision', lambda e: collisions.append(e.payload))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]
        env.move_entity(player.id, 1, 0)
        assert len(collisions) == 1
        assert collisions[0]['mover'] is player

    def test_collision_cancel_prevents_move(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('wall', 1, 0, '#', ['solid'])
            env.on('collision', lambda e: e.cancel())

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]
        result = env.move_entity(player.id, 1, 0)
        assert result is False
        assert player.x == 0

    def test_reward_events_accumulate(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.emit('reward', {'amount': 0.5}))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(5)  # 'wait'
        # step_penalty (-0.01) + reward (0.5)
        assert abs(reward - 0.49) < 1e-6

    def test_entity_created_event(self):
        events = []

        def setup(env):
            env.on('entity_created', lambda e: events.append(e.payload['entity'].type))
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        assert 'player' in events

    def test_entity_destroyed_event(self):
        events = []

        def setup(env):
            env.on('entity_destroyed', lambda e: events.append(e.payload['entity'].type))
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('target', 1, 0, 't', ['npc'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.destroy_entity('e2')
        assert events == ['target']


# ---- Turn loop tests ----

class TestTurnLoop:
    def test_turn_sequence(self):
        order = []

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('turn_start', lambda e: order.append('turn_start'))
            env.on('input', lambda e: order.append('input'))
            env.on('turn_end', lambda e: order.append('turn_end'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('wait')
        assert order == ['turn_start', 'input', 'turn_end']

    def test_behaviors_between_input_and_turn_end(self):
        order = []

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('mob', 1, 0, 'm', ['npc'])
            env.on('input', lambda e: order.append('input'))
            env.on('turn_end', lambda e: order.append('turn_end'))
            env.register_behavior('mob', lambda ent, env: order.append('behavior'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('wait')
        assert order == ['input', 'behavior', 'turn_end']

    def test_invalid_action_is_noop(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        turn_before = env.turn_number
        env.handle_input('nonexistent_action')
        assert env.turn_number == turn_before  # no turn was taken

    def test_handle_input_after_end_game_is_noop(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('won')
        turn_before = env.turn_number
        env.handle_input('wait')
        assert env.turn_number == turn_before

    def test_cascade_depth_guard(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.emit('input', {'action': 'wait', 'payload': None}))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.events.max_cascade_depth = 10  # low enough to hit before Python recursion limit
        with pytest.raises(RuntimeError, match="cascade depth"):
            env.handle_input('wait')

    def test_turn_number_increments(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        assert env.turn_number == 0
        env.handle_input('wait')
        assert env.turn_number == 1
        env.handle_input('wait')
        assert env.turn_number == 2


# ---- Game control tests ----

class TestGameControl:
    def test_end_game_won(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('won')
        assert env.status == 'won'

    def test_end_game_lost(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('lost')
        assert env.status == 'lost'

    def test_end_game_invalid_status(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        with pytest.raises(ValueError, match="must be 'won' or 'lost'"):
            env.end_game('draw')
