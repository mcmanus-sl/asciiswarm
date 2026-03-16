"""Emergence system — submerged masking. Flip water to terrain on fish action."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def reveal_chunk(
    state: EnvState, config: EnvConfig,
    chunk_x: int, chunk_y: int,
    chunk_w: int, chunk_h: int,
    water_type: int = 2,
    terrain_types: jnp.ndarray = None,
    terrain_tags: jnp.ndarray = None,
    npc_slots: jnp.ndarray = None,
) -> EnvState:
    """
    Reveal a pre-configured terrain chunk by flipping water entities to terrain.

    The chunk region (chunk_x..chunk_x+chunk_w, chunk_y..chunk_y+chunk_h) is pre-allocated
    with water entities. This function:
    1. Finds water entities in the bounding box
    2. Replaces their entity_type with terrain_types (pre-configured in base_map)
    3. Updates their tags accordingly
    4. Flips alive=True for any NPC slots associated with this chunk

    To XLA: a boolean mask update. To the player: an island rose from the sea.

    Args:
        state: Current game state
        config: Game config
        chunk_x, chunk_y: Top-left corner of chunk
        chunk_w, chunk_h: Dimensions of chunk
        water_type: Entity type ID for water tiles
        terrain_types: int32[max_entities] — target entity types for revealed tiles
                       (only applied where water exists in the chunk)
        terrain_tags: bool[max_entities, num_tags] — target tags for revealed tiles
        npc_slots: int32[N] — entity slots to flip alive=True (-1 = skip)
    """
    # Find water entities in the chunk bounding box
    in_chunk = (
        state.alive &
        (state.entity_type == water_type) &
        (state.x >= chunk_x) & (state.x < chunk_x + chunk_w) &
        (state.y >= chunk_y) & (state.y < chunk_y + chunk_h)
    )

    # Replace entity types and tags for water tiles in chunk
    if terrain_types is not None:
        new_etype = jnp.where(in_chunk, terrain_types, state.entity_type)
        state = state.replace(entity_type=new_etype)

    if terrain_tags is not None:
        # For each entity in chunk, replace tags
        expanded_mask = in_chunk[:, None]  # [max_entities, 1]
        new_tags = jnp.where(expanded_mask, terrain_tags, state.tags)
        state = state.replace(tags=new_tags)

    # Remove solid tag from revealed terrain (make it passable)
    # The caller should set up terrain_tags correctly

    # Flip NPC slots alive
    if npc_slots is not None:
        def flip_npc(state, slot):
            is_valid = slot >= 0
            new_alive = state.alive.at[slot].set(True)
            state = state.replace(
                alive=jnp.where(is_valid, new_alive, state.alive),
            )
            return state, None

        state, _ = jax.lax.scan(flip_npc, state, npc_slots)

    return state
