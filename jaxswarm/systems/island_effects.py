"""Island effects — global multipliers from unlocked chunks."""

import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def compute_island_effects(
    state: EnvState,
    chunks_unlocked: jnp.ndarray,
    effect_multipliers: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute global multipliers based on which chunks are unlocked.

    Args:
        state: Current game state
        chunks_unlocked: bool[num_chunks] — which chunks are active
        effect_multipliers: float32[num_chunks, num_effects] — multipliers per chunk
            Each row: [fishing_mult, water_mult, food_mult, stamina_mult, ...]

    Returns:
        float32[num_effects] — combined multipliers (product of all active chunk effects)
    """
    # Start with 1.0 base multipliers
    num_effects = effect_multipliers.shape[1]
    base = jnp.ones(num_effects, dtype=jnp.float32)

    # For each unlocked chunk, multiply in its effects
    active = chunks_unlocked[:, None] * effect_multipliers + \
             (~chunks_unlocked[:, None]) * jnp.ones_like(effect_multipliers)
    combined = active.prod(axis=0)

    return combined
