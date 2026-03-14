"""Tests for determinism — same key = same results."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import init_state
from jaxswarm.core.grid_ops import create_entity, destroy_entity, move_entity


def _make_tags(config):
    return jnp.zeros(config.num_tags, dtype=jnp.bool_)


def _make_props(config):
    return jnp.zeros(config.num_props, dtype=jnp.float32)


def _states_equal(s1, s2):
    leaves1 = jax.tree.leaves(s1)
    leaves2 = jax.tree.leaves(s2)
    return all(jnp.array_equal(a, b) for a, b in zip(leaves1, leaves2))


class TestDeterminism:
    def test_init_same_key(self, config):
        key = jax.random.PRNGKey(123)
        s1 = init_state(config, key)
        s2 = init_state(config, key)
        assert _states_equal(s1, s2)

    def test_crud_sequence_same_key(self, config):
        key = jax.random.PRNGKey(99)
        tags = _make_tags(config)
        props = _make_props(config)

        def run_sequence(k):
            s = init_state(config, k)
            s, slot0 = create_entity(s, config, jnp.int32(1), jnp.int32(2), jnp.int32(3), tags, props)
            s, moved = move_entity(s, config, slot0, jnp.int32(3), jnp.int32(3))
            s, slot1 = create_entity(s, config, jnp.int32(2), jnp.int32(0), jnp.int32(0), tags, props)
            s = destroy_entity(s, config, slot0)
            return s

        s1 = run_sequence(key)
        s2 = run_sequence(key)
        assert _states_equal(s1, s2)

    def test_different_keys_different(self, config):
        k1 = jax.random.PRNGKey(0)
        k2 = jax.random.PRNGKey(1)
        s1 = init_state(config, k1)
        s2 = init_state(config, k2)
        # rng_keys should differ
        assert not jnp.array_equal(s1.rng_key, s2.rng_key)
