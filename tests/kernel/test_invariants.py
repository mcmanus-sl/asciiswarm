"""Tests for the invariant test framework."""

import pytest
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import (
    invariant, run_invariants, Invariant, InvariantError,
    get_game_invariants, BUILTIN_INVARIANTS,
    check_player_singleton, check_exit_exists, check_no_empty_tags,
    check_exit_reachable,
)


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


# ---- Built-in invariant tests ----

class TestBuiltinInvariants:
    def test_player_singleton_pass(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        check_player_singleton(env)  # should not raise

    def test_player_singleton_fail_zero(self):
        """Fails when no player exists (tested post-setup via direct call)."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        # Destroy the player to simulate broken state
        player = env.get_entities_by_tag('player')[0]
        env.destroy_entity(player.id)
        with pytest.raises(InvariantError, match="player"):
            check_player_singleton(env)

    def test_player_singleton_fail_multiple(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        # Add a second player-tagged entity
        env.create_entity('fake', 1, 0, 'f', ['player'])
        with pytest.raises(InvariantError):
            check_player_singleton(env)

    def test_exit_exists_pass(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        check_exit_exists(env)

    def test_exit_exists_fail(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        with pytest.raises(InvariantError, match="exit"):
            check_exit_exists(env)

    def test_no_empty_tags_pass(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        check_no_empty_tags(env)

    def test_exit_reachable_pass(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        check_exit_reachable(env)

    def test_exit_reachable_fail(self):
        """Exit blocked by walls on all sides."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 4, 4, '>', ['exit'])
            # Surround exit with walls
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                env.create_entity('wall', 4 + dx, 4 + dy, '#', ['solid'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        with pytest.raises(InvariantError, match="not reachable"):
            check_exit_reachable(env)

    def test_exit_reachable_through_gap(self):
        """Exit reachable through a single gap in a wall line."""
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 0, '>', ['exit'])
            # Wall line with gap at y=0 x=3
            for y in range(1, 8):
                env.create_entity('wall', 3, y, '#', ['solid'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)
        check_exit_reachable(env)  # should pass — gap at top


# ---- Game-specific invariant registration ----

class TestGameInvariants:
    def test_decorator_registration(self):
        """@invariant decorator registers game-specific invariants."""
        mod = SimpleNamespace()
        mod.GAME_CONFIG = {'grid': (8, 8)}

        @invariant('custom_check')
        def my_check(env):
            pass

        mod.check = my_check
        mod.setup = lambda env: env.create_entity('player', 0, 0, '@', ['player'])

        invs = get_game_invariants(mod)
        assert len(invs) == 1
        assert invs[0].name == 'custom_check'

    def test_invariants_list_registration(self):
        """INVARIANTS list on module registers invariants."""
        mod = SimpleNamespace()
        mod.GAME_CONFIG = {'grid': (8, 8)}
        mod.setup = lambda env: env.create_entity('player', 0, 0, '@', ['player'])
        mod.INVARIANTS = [
            Invariant('check_a', lambda env: None),
            Invariant('check_b', lambda env: None),
        ]

        invs = get_game_invariants(mod)
        assert len(invs) == 2

    def test_run_invariants_all_pass(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        results = run_invariants(env, mod)
        for name, passed, err in results:
            assert passed, f"Invariant {name} failed: {err}"

    def test_run_invariants_reports_failure(self):
        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            # No exit — exit_exists will fail

        mod = _make_module(setup=setup)
        env = GridGameEnv(mod)
        env.reset(seed=42)

        results = run_invariants(env, mod)
        result_dict = {name: (passed, err) for name, passed, err in results}
        assert not result_dict['exit_exists'][0]  # should fail
        assert result_dict['player_singleton'][0]  # should pass

    def test_game_specific_invariant_runs(self):
        """Game-specific invariant runs alongside built-ins."""
        game_check_ran = [False]

        @invariant('game_custom')
        def check_game(env):
            game_check_ran[0] = True

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

        mod = _make_module(setup=setup)
        mod.check_game = check_game

        env = GridGameEnv(mod)
        env.reset(seed=42)

        results = run_invariants(env, mod)
        assert game_check_ran[0]
        names = [r[0] for r in results]
        assert 'game_custom' in names

    def test_game_invariant_failure_reported(self):
        @invariant('always_fails')
        def bad_check(env):
            raise InvariantError("intentional failure")

        def setup(env):
            env.create_entity('player', 0, 0, '@', ['player'])
            env.create_entity('exit', 7, 7, '>', ['exit'])

        mod = _make_module(setup=setup)
        mod.bad_check = bad_check

        env = GridGameEnv(mod)
        env.reset(seed=42)

        results = run_invariants(env, mod)
        result_dict = {name: (passed, err) for name, passed, err in results}
        assert not result_dict['always_fails'][0]
        assert 'intentional failure' in result_dict['always_fails'][1]
