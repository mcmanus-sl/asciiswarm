"""Hunger system — decrements food each turn, death on starvation."""

import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def hunger_system(state: EnvState, config: EnvConfig, food_prop_idx: int = 0, drain: float = 1.0) -> EnvState:
    """
    Subtract `drain` from player's food property each turn.
    If food <= 0, set status = -1.
    """
    pidx = state.player_idx
    food = state.properties[pidx, food_prop_idx]
    new_food = food - drain
    state = state.replace(
        properties=state.properties.at[pidx, food_prop_idx].set(new_food),
    )
    starved = (new_food <= 0) & (state.status == 0)
    state = state.replace(
        status=jnp.where(starved, jnp.int32(-1), state.status),
    )
    return state
