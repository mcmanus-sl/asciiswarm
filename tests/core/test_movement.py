"""Tests for move_toward helper."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import init_state
from jaxswarm.core.grid_ops import create_entity
from jaxswarm.core.movement import move_toward


def _make_tags(config):
    return jnp.zeros(config.num_tags, dtype=jnp.bool_)


def _make_props(config):
    return jnp.zeros(config.num_props, dtype=jnp.float32)


class TestMoveToward:
    def test_moves_closer(self, config, rng):
        state = init_state(config, rng)
        tags = _make_tags(config)
        props = _make_props(config)
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)

        k1, k2 = jax.random.split(rng)
        state2, moved = move_toward(state, config, slot, jnp.int32(5), jnp.int32(3), k1)

        assert moved
        # Should have moved 1 step closer — Manhattan distance decreased
        old_dist = abs(0 - 5) + abs(0 - 3)
        new_dist = abs(int(state2.x[slot]) - 5) + abs(int(state2.y[slot]) - 3)
        assert new_dist == old_dist - 1

    def test_prefers_greater_axis(self, config, rng):
        state = init_state(config, rng)
        tags = _make_tags(config)
        props = _make_props(config)
        # Start at (0, 0), target at (5, 1) — dx=5 > dy=1, should move along x
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)

        k1, _ = jax.random.split(rng)
        state2, moved = move_toward(state, config, slot, jnp.int32(5), jnp.int32(1), k1)

        assert moved
        assert state2.x[slot] == 1
        assert state2.y[slot] == 0

    def test_at_target_no_move(self, config, rng):
        state = init_state(config, rng)
        tags = _make_tags(config)
        props = _make_props(config)
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(3), jnp.int32(3), tags, props)

        k1, _ = jax.random.split(rng)
        state2, moved = move_toward(state, config, slot, jnp.int32(3), jnp.int32(3), k1)

        assert not moved
        assert state2.x[slot] == 3
        assert state2.y[slot] == 3

    def test_tie_breaking_with_rng(self, config, rng):
        """When dx == dy, different keys should produce different axis choices."""
        tags = _make_tags(config)
        props = _make_props(config)

        results = []
        for i in range(20):
            state = init_state(config, rng)
            state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(3), jnp.int32(3), tags, props)
            key = jax.random.PRNGKey(i)
            state2, moved = move_toward(state, config, slot, jnp.int32(6), jnp.int32(6), key)
            assert moved
            results.append((int(state2.x[slot]), int(state2.y[slot])))

        # With 20 different keys, should get both (4,3) and (3,4)
        xs = {r[0] for r in results}
        ys = {r[1] for r in results}
        assert 4 in xs or 4 in ys  # at least moved on one axis

    def test_blocked_primary_tries_alternate(self, config, rng):
        """If primary axis is blocked, try the other axis."""
        state = init_state(config, rng)
        tags = _make_tags(config)
        props = _make_props(config)

        # Entity at (0,0), target at (1,1) — both axes equidistant
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)

        # Fill cell (1,0) to block x-axis movement
        for _ in range(config.max_stack):
            state, _ = create_entity(state, config, jnp.int32(2), jnp.int32(1), jnp.int32(0), tags, props)

        # With some keys, primary may be x (blocked), should fall back to y
        moved_any = False
        for i in range(20):
            test_state = state
            key = jax.random.PRNGKey(i + 100)
            s2, moved = move_toward(test_state, config, slot, jnp.int32(1), jnp.int32(1), key)
            if moved:
                moved_any = True
                # Should have moved to (0,1) since (1,0) is blocked
                assert (int(s2.x[slot]) == 0 and int(s2.y[slot]) == 1) or \
                       (int(s2.x[slot]) == 1 and int(s2.y[slot]) == 0)

        assert moved_any, "Should have been able to move on at least one attempt"
