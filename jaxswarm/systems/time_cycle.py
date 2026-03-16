"""Time cycle system — day/night with multi-meter depletion (food, stamina, thirst)."""

import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def time_cycle_system(
    state: EnvState, config: EnvConfig,
    day_length: int = 50,
    food_prop: int = 0,
    stamina_prop: int = 1,
    thirst_prop: int = 2,
    food_drain: float = 0.5,
    stamina_drain_day: float = 0.2,
    stamina_drain_night: float = 1.0,
    thirst_drain: float = 0.5,
) -> EnvState:
    """
    Day/night cycle with multi-meter depletion.
    - Day: moderate stamina drain, normal food/thirst drain.
    - Night: heavy stamina drain (unless near warmth source).

    Returns updated state. Does NOT check death — caller handles that.

    Note: warmth protection is applied externally by the game (reducing stamina drain
    when player is near fire). This system just applies the base drain.
    """
    pidx = state.player_idx
    turn = state.turn_number
    cycle_pos = turn % (day_length * 2)  # 0..2*day_length-1
    is_night = cycle_pos >= day_length

    # Drain rates
    stamina_drain = jnp.where(is_night, stamina_drain_night, stamina_drain_day)

    food = state.properties[pidx, food_prop] - food_drain
    stamina = state.properties[pidx, stamina_prop] - stamina_drain
    thirst = state.properties[pidx, thirst_prop] - thirst_drain

    # Clamp to 0
    food = jnp.maximum(food, 0.0)
    stamina = jnp.maximum(stamina, 0.0)
    thirst = jnp.maximum(thirst, 0.0)

    state = state.replace(
        properties=state.properties
            .at[pidx, food_prop].set(food)
            .at[pidx, stamina_prop].set(stamina)
            .at[pidx, thirst_prop].set(thirst),
    )
    return state


def is_night(turn_number: jnp.int32, day_length: int = 50) -> jnp.bool_:
    """Check if it's currently night."""
    cycle_pos = turn_number % (day_length * 2)
    return cycle_pos >= day_length
