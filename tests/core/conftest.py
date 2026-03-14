"""Shared test fixtures."""

import pytest
import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvConfig, init_state


@pytest.fixture
def config():
    return EnvConfig(
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


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def state(config, rng):
    return init_state(config, rng)
