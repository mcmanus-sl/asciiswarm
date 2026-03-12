"""Kernel stress tests — edge cases beyond basic Step 1 coverage."""

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


# ---- Mass create/destroy ----

class TestMassEntityOps:
    def test_create_destroy_hundreds(self):
        """Create and destroy hundreds of entities. No stale references."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        ids = []
        for i in range(300):
            x, y = i % 8, (i // 8) % 8
            e = env.create_entity('npc', x, y, 'n', ['npc'])
            ids.append(e.id)

        assert len(env.get_all_entities()) == 301  # 1 player + 300 npcs

        for eid in ids:
            env.destroy_entity(eid)

        assert len(env.get_all_entities()) == 1  # only player remains
        for eid in ids:
            assert env.get_entity(eid) is None

    def test_create_destroy_interleaved(self):
        """Interleaved creates and destroys don't corrupt state."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        alive = []
        for i in range(100):
            e = env.create_entity('npc', i % 8, (i // 8) % 8, 'n', ['npc'])
            alive.append(e.id)
            if len(alive) > 10:
                env.destroy_entity(alive.pop(0))

        # Verify remaining entities are queryable
        for eid in alive:
            assert env.get_entity(eid) is not None

    def test_entity_ids_never_reused(self):
        """After destroy, new entities get fresh IDs."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        e2 = env.create_entity('npc', 1, 0, 'n', ['npc'])
        env.destroy_entity(e2.id)
        e3 = env.create_entity('npc', 1, 0, 'n', ['npc'])
        assert e3.id != e2.id
        assert int(e3.id[1:]) > int(e2.id[1:])


# ---- Boundary movement ----

class TestBoundaryMovement:
    def test_move_to_every_boundary_cell(self):
        """Move entity to all 4 edges and corners."""
        config = {'grid': (10, 10)}
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]

        # Top-left corner (already there at 0,0)
        assert env.move_entity(player.id, 0, 0)

        # All corners
        for x, y in [(9, 0), (0, 9), (9, 9)]:
            assert env.move_entity(player.id, x, y)
            assert player.x == x and player.y == y

        # All edge cells
        for x in range(10):
            assert env.move_entity(player.id, x, 0)
            assert env.move_entity(player.id, x, 9)
        for y in range(10):
            assert env.move_entity(player.id, 0, y)
            assert env.move_entity(player.id, 9, y)

    def test_off_grid_every_direction(self):
        """Off-grid moves fail in all 4 directions and diagonals."""
        config = {'grid': (5, 5)}
        mod = _make_module(config=config)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]

        # From center
        env.move_entity(player.id, 2, 2)

        # Off north
        assert not env.move_entity(player.id, 2, -1)
        # Off south
        assert not env.move_entity(player.id, 2, 5)
        # Off west
        assert not env.move_entity(player.id, -1, 2)
        # Off east
        assert not env.move_entity(player.id, 5, 2)
        # Off diagonals
        assert not env.move_entity(player.id, -1, -1)
        assert not env.move_entity(player.id, 5, 5)
        assert not env.move_entity(player.id, -1, 5)
        assert not env.move_entity(player.id, 5, -1)

        # Large off-grid
        assert not env.move_entity(player.id, 100, 100)
        assert not env.move_entity(player.id, -100, -100)


# ---- Cell stacking ----

class TestCellStacking:
    def test_20_plus_entities_same_cell(self):
        """Fill a cell with 20+ entities. Observation and rendering still correct."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'], z_order=100)
            for i in range(25):
                env.create_entity('npc', 3, 3, 'n', ['npc'], z_order=i)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        entities_at = env.get_entities_at(3, 3)
        assert len(entities_at) == 25

        # Observation: npc channel at (3,3) should be 1.0
        obs = env._get_obs()
        npc_idx = env.config['tags'].index('npc')
        assert obs['grid'][npc_idx, 3, 3] == 1.0

        # Rendering: highest z-order entity glyph shown
        ascii_out = env.render_ascii()
        lines = ascii_out.split('\n')
        assert lines[3][3] == 'n'  # z_order 24 is highest among npcs

    def test_stacked_entities_multi_tag_observation(self):
        """Multiple entities on same cell with different tags all visible in obs."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('hazard_thing', 4, 4, 'h', ['hazard'])
            env.create_entity('pickup_thing', 4, 4, 'p', ['pickup'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        obs = env._get_obs()
        tags = env.config['tags']
        assert obs['grid'][tags.index('hazard'), 4, 4] == 1.0
        assert obs['grid'][tags.index('pickup'), 4, 4] == 1.0


# ---- Event handler stress ----

class TestEventHandlerStress:
    def test_50_plus_handlers_fire_in_order(self):
        """Register 50+ handlers. All fire in registration order."""
        order = []

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            for i in range(55):
                env.on('input', lambda e, idx=i: order.append(idx))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('wait')
        assert order == list(range(55))

    def test_unsubscribe_mid_iteration_50_handlers(self):
        """Unsubscribe handler 25 during execution of handler 10. Others still fire."""
        results = []
        unsubs = {}

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

            def make_handler(idx):
                def handler(e):
                    results.append(idx)
                    if idx == 10 and 25 in unsubs:
                        unsubs[25]()
                return handler

            for i in range(50):
                unsub = env.on('input', make_handler(i))
                unsubs[i] = unsub

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('wait')

        # All 50 should have fired (handler list was snapshotted at emit time)
        assert len(results) == 50
        assert results == list(range(50))

        # Second emit: handler 25 should be gone
        results.clear()
        env.handle_input('wait')
        assert 25 not in results
        assert len(results) == 49

    def test_collision_cancellation_chain(self):
        """Handler A cancels. Handler B still fires but cancel is already set."""
        handler_b_ran = [False]
        handler_b_saw_cancelled = [None]

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('target', 1, 0, 't', ['npc'])

            def handler_a(event):
                event.cancel()

            def handler_b(event):
                handler_b_ran[0] = True
                handler_b_saw_cancelled[0] = event.cancelled

            env.on('collision', handler_a)
            env.on('collision', handler_b)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        player = env.get_entities_by_tag('player')[0]
        result = env.move_entity(player.id, 1, 0)

        assert result is False  # cancelled
        assert handler_b_ran[0]  # B still fired
        assert handler_b_saw_cancelled[0] is True  # B saw it was already cancelled


# ---- Serialization stress ----

class TestSerializationStress:
    def test_serialize_1000_entities(self):
        """Serialize a world with 1000+ entities. Round-trip byte-identical."""
        config = {'grid': (50, 50)}

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            for i in range(1000):
                x, y = (i + 1) % 50, ((i + 1) // 50) % 50
                e = env.create_entity('npc', x, y, 'n', ['npc'])
                e.set('value', i)

        mod = _make_module(config=config, setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        state1 = env.serialize_state()
        json1 = json.dumps(state1, sort_keys=True)

        env.load_state(state1)
        state2 = env.serialize_state()
        json2 = json.dumps(state2, sort_keys=True)

        assert json1 == json2
        assert len(state1['entities']) == 1001

    def test_determinism_100_turns_from_loaded_state(self):
        """Load state, run 100 random inputs, serialize. Repeat. Identical."""
        import random as stdlib_random

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

            def on_input(event):
                p = env.get_entities_by_tag('player')[0]
                action = event.payload['action']
                moves = {
                    'move_n': (0, -1), 'move_s': (0, 1),
                    'move_e': (1, 0), 'move_w': (-1, 0),
                }
                if action in moves:
                    dx, dy = moves[action]
                    env.move_entity(p.id, p.x + dx, p.y + dy)

            env.on('input', on_input)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        state = env.serialize_state()

        actions = env.config['actions']
        # Generate fixed action sequence
        action_rng = stdlib_random.Random(999)
        action_seq = [action_rng.choice(actions) for _ in range(100)]

        # Run 1
        env.load_state(state)
        for a in action_seq:
            if env.status != 'playing':
                break
            env.handle_input(a)
        result1 = json.dumps(env.serialize_state(), sort_keys=True)

        # Run 2
        env.load_state(state)
        for a in action_seq:
            if env.status != 'playing':
                break
            env.handle_input(a)
        result2 = json.dumps(env.serialize_state(), sort_keys=True)

        assert result1 == result2


# ---- End game / Gym integration stress ----

class TestEndGameStress:
    def test_end_game_prevents_further_turns(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('won')

        for _ in range(10):
            env.handle_input('wait')
        assert env.turn_number == 0  # no turns were executed

    def test_step_terminated_after_end_game(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('lost')

        obs, reward, terminated, truncated, info = env.step(5)
        assert terminated is True
        assert reward == 0.0  # early exit guard

    def test_reset_fully_restores_after_end_game(self):
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.end_game('won')
        assert env.status == 'won'

        env.reset()
        assert env.status == 'playing'
        assert env.turn_number == 0
        assert len(env.get_entities_by_tag('player')) == 1

    def test_prng_same_sequence_after_serialize_load(self):
        """env.random() produces same sequence after serialize → load."""
        mod = _make_module()
        env = GridGameEnv(mod)
        env.reset(seed=42)

        for _ in range(50):
            env.random()

        state = env.serialize_state()
        expected = [env.random() for _ in range(20)]

        env.load_state(state)
        actual = [env.random() for _ in range(20)]
        assert expected == actual

    def test_cascade_guard_configurable(self):
        """Cascade guard fires at configured depth."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

            def recursive(e):
                env.emit('custom_loop')

            env.on('custom_loop', recursive)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.events.max_cascade_depth = 5

        with pytest.raises(RuntimeError, match="cascade depth"):
            env.emit('custom_loop')
