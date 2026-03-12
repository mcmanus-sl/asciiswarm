"""Userland simulation tests — simulate what AGENT SWARM will actually do.

These test patterns that game authors will use against the kernel API.
"""

import pytest
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


class TestBehaviorPatterns:
    def test_move_toward_target(self):
        """Register a behavior that moves an entity toward a target every turn."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            chaser = env.create_entity('chaser', 7, 7, 'c', ['hazard'])
            chaser.set('target_x', 0)
            chaser.set('target_y', 0)

            def chase_behavior(entity, env):
                tx, ty = entity.get('target_x'), entity.get('target_y')
                dx = 1 if tx > entity.x else (-1 if tx < entity.x else 0)
                dy = 1 if ty > entity.y else (-1 if ty < entity.y else 0)
                # Prefer horizontal movement
                if dx != 0:
                    env.move_entity(entity.id, entity.x + dx, entity.y)
                elif dy != 0:
                    env.move_entity(entity.id, entity.x, entity.y + dy)

            env.register_behavior('chaser', chase_behavior)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        for _ in range(10):
            env.handle_input('wait')

        chaser = env.get_entities_by_type('chaser')[0]
        # After 10 turns moving toward (0,0) from (7,7), should be closer
        assert chaser.x < 7 or chaser.y < 7
        # Should be at roughly (0, 4) — moved 7 left first, then 3 down
        # Actually: moves left each turn (dx=-1) until x=0 (7 turns), then down (3 turns)
        assert chaser.x == 0
        assert chaser.y == 4  # 7 horizontal + 3 vertical = 10 turns


class TestCollisionPatterns:
    def test_trap_destroys_mover(self):
        """Collision handler destroys the mover (simulating a trap)."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('victim', 1, 0, 'v', ['npc'])
            env.create_entity('trap', 2, 0, '^', ['hazard'])

            def on_collision(event):
                mover = event.payload['mover']
                for occ in event.payload['occupants']:
                    if occ.has_tag('hazard') and mover.type == 'victim':
                        env.destroy_entity(mover.id)
                        event.cancel()

            env.on('collision', on_collision)

            def victim_behavior(entity, env):
                env.move_entity(entity.id, entity.x + 1, entity.y)

            env.register_behavior('victim', victim_behavior)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('wait')  # victim moves into trap

        assert env.get_entities_by_type('victim') == []

    def test_wall_cancels_move(self):
        """Collision handler cancels the move (simulating a wall via before_move)."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('wall', 1, 0, '#', ['solid'])

            def on_before_move(event):
                tx, ty = event.payload['to_x'], event.payload['to_y']
                for ent in env.get_entities_at(tx, ty):
                    if ent.has_tag('solid'):
                        event.cancel()
                        return

            env.on('before_move', on_before_move)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        player = env.get_entities_by_tag('player')[0]
        result = env.move_entity(player.id, 1, 0)
        assert result is False
        assert player.x == 0


class TestChainReaction:
    def test_create_emit_destroy_one_turn(self):
        """Chain: A's behavior creates B, B's behavior emits custom event,
        that event's handler destroys A. All resolves in one turn."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            a = env.create_entity('spawner', 1, 0, 'a', ['npc'])

            def spawner_behavior(entity, env):
                b = env.create_entity('emitter', 2, 0, 'b', ['npc'])

            def emitter_behavior(entity, env):
                env.emit('destroy_spawner', {'target': 'spawner'})

            def on_destroy_spawner(event):
                spawners = env.get_entities_by_type('spawner')
                for s in spawners:
                    env.destroy_entity(s.id)

            env.register_behavior('spawner', spawner_behavior)
            env.register_behavior('emitter', emitter_behavior)
            env.on('destroy_spawner', on_destroy_spawner)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        assert len(env.get_entities_by_type('spawner')) == 1
        env.handle_input('wait')  # one turn

        # Spawner (A) created emitter (B), but B was created after the snapshot
        # was taken for this turn, so B's behavior does NOT run this turn.
        # A is still alive after turn 1.
        assert len(env.get_entities_by_type('spawner')) == 1
        assert len(env.get_entities_by_type('emitter')) == 1

        # Turn 2: now B's behavior runs, emitting event that destroys A
        env.handle_input('wait')
        assert len(env.get_entities_by_type('spawner')) == 0
        # A also spawned another B this turn, plus existing B ran
        assert len(env.get_entities_by_type('emitter')) >= 1


class TestInteractAction:
    def test_interact_handler_fires(self):
        """Register an input handler that responds to 'interact'. Verify it ran."""
        interact_data = []

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

            def on_input(event):
                if event.payload['action'] == 'interact':
                    interact_data.append('interacted')

            env.on('input', on_input)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('interact')
        assert interact_data == ['interacted']

    def test_interact_near_entity(self):
        """Interact while adjacent to an NPC triggers a response."""
        npc_talked = [False]

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('npc', 1, 0, 'N', ['npc'])

            def on_input(event):
                if event.payload['action'] == 'interact':
                    player = env.get_entities_by_tag('player')[0]
                    # Check cardinal neighbors for NPC
                    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                        nx, ny = player.x + dx, player.y + dy
                        for ent in env.get_entities_at(nx, ny):
                            if ent.has_tag('npc'):
                                npc_talked[0] = True

            env.on('input', on_input)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        env.handle_input('interact')
        assert npc_talked[0]


class TestEndGameGymIntegration:
    def test_end_game_won_step_returns_positive_reward(self):
        """end_game('won') → step() returns terminated=True and positive reward."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.end_game('won'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(5)
        assert terminated is True
        assert reward > 0  # step_penalty + 10.0

    def test_end_game_lost_step_returns_negative_reward(self):
        """end_game('lost') → step() returns terminated=True and negative reward."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.on('input', lambda e: env.end_game('lost'))

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(5)
        assert terminated is True
        assert reward < 0  # step_penalty + -10.0

    def test_reward_event_on_winning_turn(self):
        """Intermediate reward emitted on the same turn as end_game('won') is included."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

            def on_input(e):
                env.emit('reward', {'amount': 0.5})
                env.end_game('won')

            env.on('input', on_input)

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        _, reward, _, _, _ = env.step(5)
        # step_penalty (-0.01) + intermediate (0.5) + terminal (10.0) = 10.49
        assert abs(reward - 10.49) < 1e-6


class TestPickupPattern:
    def test_walk_onto_pickup_destroy_it(self):
        """Common pattern: player walks onto pickup, collision handler destroys it."""
        def setup(env):
            player = env.create_entity('player', 0, 0, '@', ['player'], z_order=10)
            player.set('health', 5)
            env.create_entity('potion', 1, 0, '!', ['pickup'], z_order=3)

            def on_input(event):
                p = env.get_entities_by_tag('player')[0]
                if event.payload['action'] == 'move_e':
                    env.move_entity(p.id, p.x + 1, p.y)

            env.on('input', on_input)

            def on_collision(event):
                mover = event.payload['mover']
                if mover.has_tag('player'):
                    for occ in list(event.payload['occupants']):
                        if occ.has_tag('pickup'):
                            mover.set('health', min(10, mover.get('health', 0) + 3))
                            env.destroy_entity(occ.id)
                            env.emit('reward', {'amount': 0.1})

            env.on('collision', on_collision)

        config = {
            'grid': (8, 8),
            'player_properties': [{'key': 'health', 'max': 10}],
        }
        mod = _make_module(config=config, setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        # Walk east onto the potion
        env.handle_input('move_e')

        player = env.get_entities_by_tag('player')[0]
        assert player.get('health') == 8  # 5 + 3
        assert player.x == 1
        assert env.get_entities_by_type('potion') == []
