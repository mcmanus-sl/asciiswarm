"""Tests for EnvState creation and init_state."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import init_state


class TestInitState:
    def test_shapes(self, config, rng):
        state = init_state(config, rng)
        assert state.alive.shape == (config.max_entities,)
        assert state.entity_type.shape == (config.max_entities,)
        assert state.x.shape == (config.max_entities,)
        assert state.y.shape == (config.max_entities,)
        assert state.tags.shape == (config.max_entities, config.num_tags)
        assert state.properties.shape == (config.max_entities, config.num_props)
        assert state.grid.shape == (config.grid_h, config.grid_w, config.max_stack)
        assert state.grid_count.shape == (config.grid_h, config.grid_w)
        assert state.game_state.shape == (config.game_state_size,)

    def test_all_dead(self, config, rng):
        state = init_state(config, rng)
        assert not state.alive.any()

    def test_grid_empty(self, config, rng):
        state = init_state(config, rng)
        assert (state.grid == -1).all()
        assert (state.grid_count == 0).all()

    def test_rng_stored(self, config, rng):
        state = init_state(config, rng)
        assert jnp.array_equal(state.rng_key, rng)

    def test_scalars_zero(self, config, rng):
        state = init_state(config, rng)
        assert state.turn_number == 0
        assert state.status == 0
        assert state.reward_acc == 0.0
        assert state.player_idx == 0

    def test_pytree_round_trip(self, config, rng):
        """Flatten to pytree leaves, reconstruct, assert element-wise equality."""
        state = init_state(config, rng)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = treedef.unflatten(leaves)
        orig_leaves = jax.tree_util.tree_leaves(state)
        recon_leaves = jax.tree_util.tree_leaves(reconstructed)
        for a, b in zip(orig_leaves, recon_leaves):
            assert jnp.array_equal(a, b)
