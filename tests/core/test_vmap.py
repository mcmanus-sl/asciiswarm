"""vmap smoke tests — batched operations."""

import jax
import jax.numpy as jnp
import functools
from jaxswarm.core.state import EnvConfig, init_state
from jaxswarm.core.grid_ops import create_entity, move_entity
from jaxswarm.core.obs import get_obs


CONFIG = EnvConfig(
    grid_w=8,
    grid_h=8,
    max_entities=8,
    max_stack=2,
    num_entity_types=3,
    num_tags=6,
    num_props=4,
    num_actions=6,
    max_turns=200,
    step_penalty=-0.01,
    game_state_size=4,
    prop_maxes=(10.0, 5.0, 1.0, 100.0),
)

N_ENVS = 64


def _create(state, entity_type, x, y, tags, props):
    return create_entity(state, CONFIG, entity_type, x, y, tags, props)


def _move(state, slot, new_x, new_y):
    return move_entity(state, CONFIG, slot, new_x, new_y)


def _obs(state):
    return get_obs(state, CONFIG)


class TestVmap:
    def test_vmap_init_state(self):
        keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
        batched_init = jax.vmap(functools.partial(init_state, CONFIG))
        states = batched_init(keys)
        assert states.alive.shape == (N_ENVS, CONFIG.max_entities)
        assert states.grid.shape == (N_ENVS, CONFIG.grid_h, CONFIG.grid_w, CONFIG.max_stack)

    def test_vmap_create_entity(self):
        keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
        batched_init = jax.vmap(functools.partial(init_state, CONFIG))
        states = batched_init(keys)

        tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_)
        props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)

        batch_tags = jnp.broadcast_to(tags, (N_ENVS, CONFIG.num_tags))
        batch_props = jnp.broadcast_to(props, (N_ENVS, CONFIG.num_props))
        batch_types = jnp.ones(N_ENVS, dtype=jnp.int32)
        batch_x = jnp.zeros(N_ENVS, dtype=jnp.int32)
        batch_y = jnp.zeros(N_ENVS, dtype=jnp.int32)

        batched_create = jax.vmap(_create)
        new_states, slots = batched_create(states, batch_types, batch_x, batch_y, batch_tags, batch_props)

        assert new_states.alive.shape == (N_ENVS, CONFIG.max_entities)
        assert (slots == 0).all()
        assert new_states.alive[:, 0].all()

    def test_vmap_move_entity(self):
        keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
        batched_init = jax.vmap(functools.partial(init_state, CONFIG))
        states = batched_init(keys)

        tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_)
        props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
        batch_tags = jnp.broadcast_to(tags, (N_ENVS, CONFIG.num_tags))
        batch_props = jnp.broadcast_to(props, (N_ENVS, CONFIG.num_props))

        batched_create = jax.vmap(_create)
        states, slots = batched_create(
            states, jnp.ones(N_ENVS, dtype=jnp.int32),
            jnp.zeros(N_ENVS, dtype=jnp.int32),
            jnp.zeros(N_ENVS, dtype=jnp.int32),
            batch_tags, batch_props
        )

        batched_move = jax.vmap(_move)
        new_states, moved = batched_move(
            states, slots,
            jnp.ones(N_ENVS, dtype=jnp.int32),
            jnp.zeros(N_ENVS, dtype=jnp.int32),
        )

        assert moved.all()
        assert (new_states.x[:, 0] == 1).all()

    def test_vmap_get_obs(self):
        keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
        batched_init = jax.vmap(functools.partial(init_state, CONFIG))
        states = batched_init(keys)

        batched_obs = jax.vmap(_obs)
        obs = batched_obs(states)

        assert obs['grid'].shape == (N_ENVS, CONFIG.num_tags, CONFIG.grid_h, CONFIG.grid_w)
        assert obs['scalars'].shape == (N_ENVS, 3 + CONFIG.num_props)

    def test_no_cross_contamination(self):
        keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
        batched_init = jax.vmap(functools.partial(init_state, CONFIG))
        states = batched_init(keys)

        tags = jnp.zeros(CONFIG.num_tags, dtype=jnp.bool_).at[0].set(True)
        props = jnp.zeros(CONFIG.num_props, dtype=jnp.float32)
        batch_tags = jnp.broadcast_to(tags, (N_ENVS, CONFIG.num_tags))
        batch_props = jnp.broadcast_to(props, (N_ENVS, CONFIG.num_props))

        batch_types = jnp.ones(N_ENVS, dtype=jnp.int32)
        batch_x = jnp.zeros(N_ENVS, dtype=jnp.int32)
        batch_y = jnp.zeros(N_ENVS, dtype=jnp.int32)

        batched_create = jax.vmap(_create)
        new_states, slots = batched_create(states, batch_types, batch_x, batch_y, batch_tags, batch_props)

        batched_obs = jax.vmap(_obs)
        obs = batched_obs(new_states)

        # All envs should have entity at (0,0) with tag 0
        assert (obs['grid'][:, 0, 0, 0] == 1.0).all()

        # No cross-contamination: tag 0 should not appear at (1, 0)
        assert (obs['grid'][:, 0, 0, 1] == 0.0).all()
