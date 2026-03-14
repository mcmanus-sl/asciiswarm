"""Observation builder — grid tensor + scalar vector."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def get_obs(state: EnvState, config: EnvConfig) -> dict:
    """
    Returns {'grid': float32[num_tags, H, W], 'scalars': float32[3 + num_props]}.

    Grid observation: for each alive entity, scatter its tags onto the spatial grid.
    Fully vectorized — no lax.fori_loop or lax.scan over entities.
    """
    # Grid observation — vectorized scatter
    # alive_mask: [max_entities] -> broadcast to [max_entities, num_tags]
    alive_mask = state.alive  # [E]
    tag_vals = state.tags.astype(jnp.float32)  # [E, T]
    active_tags = tag_vals * alive_mask[:, None]  # [E, T] — zero out dead entities

    # Build [E, T, H, W] one-hot grid positions, then reduce
    # More efficient: use scatter-add
    obs_grid = jnp.zeros((config.num_tags, config.grid_h, config.grid_w), dtype=jnp.float32)

    # For each entity, add its tags to the grid at its position
    # Use advanced indexing: obs_grid[:, y[e], x[e]] += active_tags[e, :]
    # Vectorized via .at[].add with entity indices
    ys = state.y  # [E]
    xs = state.x  # [E]
    tag_indices = jnp.arange(config.num_tags)  # [T]

    # Expand: for each (entity, tag) pair, add to obs_grid[tag, y, x]
    # Shape: [E, T]
    entity_idx = jnp.arange(config.max_entities)

    # Use scatter: for each entity e, for each tag t:
    #   obs_grid[t, y[e], x[e]] += active_tags[e, t]
    # Expand indices for all (entity, tag) combinations
    tag_broadcast = jnp.broadcast_to(tag_indices[None, :], (config.max_entities, config.num_tags))  # [E, T]
    y_broadcast = jnp.broadcast_to(ys[:, None], (config.max_entities, config.num_tags))  # [E, T]
    x_broadcast = jnp.broadcast_to(xs[:, None], (config.max_entities, config.num_tags))  # [E, T]

    obs_grid = obs_grid.at[
        tag_broadcast.ravel(),
        y_broadcast.ravel(),
        x_broadcast.ravel(),
    ].add(active_tags.ravel())

    # Clamp to [0, 1] — multiple entities with same tag at same cell should be 1.0
    obs_grid = jnp.clip(obs_grid, 0.0, 1.0)

    # Scalar observation
    prop_maxes = jnp.array(config.prop_maxes, dtype=jnp.float32)
    player_props = state.properties[state.player_idx]  # [num_props]
    # Avoid division by zero
    safe_maxes = jnp.where(prop_maxes > 0, prop_maxes, 1.0)
    normalized_props = jnp.clip(player_props / safe_maxes, 0.0, 1.0)

    scalars = jnp.concatenate([
        jnp.array([config.grid_w / 100.0, config.grid_h / 100.0], dtype=jnp.float32),
        jnp.array([state.turn_number / config.max_turns], dtype=jnp.float32),
        normalized_props,
    ])

    return {'grid': obs_grid, 'scalars': scalars}
