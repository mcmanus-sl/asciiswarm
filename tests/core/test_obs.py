"""Tests for observation builder."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import init_state
from jaxswarm.core.grid_ops import create_entity
from jaxswarm.core.obs import get_obs


def _make_tags(config, *indices):
    t = jnp.zeros(config.num_tags, dtype=jnp.bool_)
    for i in indices:
        t = t.at[i].set(True)
    return t


def _make_props(config, *values):
    p = jnp.zeros(config.num_props, dtype=jnp.float32)
    for i, v in enumerate(values):
        p = p.at[i].set(v)
    return p


class TestGetObs:
    def test_keys(self, config, state):
        obs = get_obs(state, config)
        assert 'grid' in obs
        assert 'scalars' in obs

    def test_grid_shape(self, config, state):
        obs = get_obs(state, config)
        assert obs['grid'].shape == (config.num_tags, config.grid_h, config.grid_w)

    def test_scalar_shape(self, config, state):
        obs = get_obs(state, config)
        assert obs['scalars'].shape == (3 + config.num_props,)

    def test_grid_tag_at_position(self, config, state):
        tags = _make_tags(config, 2)
        props = _make_props(config)
        state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(3), jnp.int32(5), tags, props)
        obs = get_obs(state, config)
        assert obs['grid'][2, 5, 3] == 1.0

    def test_grid_empty_cells_zero(self, config, state):
        obs = get_obs(state, config)
        assert (obs['grid'] == 0.0).all()

    def test_grid_multiple_entities_same_tag_clamped(self, config, state):
        tags = _make_tags(config, 0)
        props = _make_props(config)
        state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        obs = get_obs(state, config)
        assert obs['grid'][0, 0, 0] == 1.0  # clamped, not 2.0

    def test_scalar_grid_dims(self, config, state):
        obs = get_obs(state, config)
        assert obs['scalars'][0] == config.grid_w / 100.0
        assert obs['scalars'][1] == config.grid_h / 100.0

    def test_scalar_turn_number(self, config, state):
        state = state.replace(turn_number=jnp.int32(50))
        obs = get_obs(state, config)
        assert jnp.isclose(obs['scalars'][2], 50.0 / config.max_turns)

    def test_scalar_player_props_normalized(self, config, state):
        # Set player properties and create player entity
        tags = _make_tags(config)
        props = _make_props(config, 5.0, 2.5, 0.5, 50.0)
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state = state.replace(player_idx=slot)
        obs = get_obs(state, config)
        # prop_maxes = (10.0, 5.0, 1.0, 100.0)
        assert jnp.isclose(obs['scalars'][3], 0.5)   # 5/10
        assert jnp.isclose(obs['scalars'][4], 0.5)   # 2.5/5
        assert jnp.isclose(obs['scalars'][5], 0.5)   # 0.5/1
        assert jnp.isclose(obs['scalars'][6], 0.5)   # 50/100

    def test_scalar_props_clamped(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config, 11.0)  # exceeds max of 10.0
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state = state.replace(player_idx=slot)
        obs = get_obs(state, config)
        assert obs['scalars'][3] == 1.0  # clamped

    def test_deterministic(self, config, state):
        tags = _make_tags(config, 1, 3)
        props = _make_props(config, 3.0, 1.0)
        state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(4), jnp.int32(2), tags, props)
        obs1 = get_obs(state, config)
        obs2 = get_obs(state, config)
        assert jnp.array_equal(obs1['grid'], obs2['grid'])
        assert jnp.array_equal(obs1['scalars'], obs2['scalars'])
