"""Entity CRUD, grid queries, and rebuild_grid — all pure functions."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig


def create_entity(
    state: EnvState,
    config: EnvConfig,
    entity_type: jnp.int32,
    x: jnp.int32,
    y: jnp.int32,
    tags: jax.Array,
    props: jax.Array,
) -> tuple[EnvState, jnp.int32]:
    """Find first free slot, fill it, update grid. Returns (new_state, slot) or (state, -1)."""
    has_free = ~state.alive.all()
    slot = jnp.argmin(state.alive)  # first False = first free

    # Build the would-be new state
    new_alive = state.alive.at[slot].set(True)
    new_entity_type = state.entity_type.at[slot].set(entity_type)
    new_x = state.x.at[slot].set(x)
    new_y = state.y.at[slot].set(y)
    new_tags = state.tags.at[slot].set(tags)
    new_props = state.properties.at[slot].set(props)

    # Add slot to grid cell
    count = state.grid_count[y, x]
    safe_count = jnp.minimum(count, config.max_stack - 1)
    new_grid = state.grid.at[y, x, safe_count].set(slot)
    new_grid_count = state.grid_count.at[y, x].set(count + 1)

    new_state = state.replace(
        alive=new_alive,
        entity_type=new_entity_type,
        x=new_x,
        y=new_y,
        tags=new_tags,
        properties=new_props,
        grid=new_grid,
        grid_count=new_grid_count,
    )

    # Select based on whether a free slot exists
    final_state = jax.tree.map(
        lambda n, o: jnp.where(has_free, n, o), new_state, state
    )
    final_slot = jnp.where(has_free, slot, jnp.int32(-1))

    return final_state, final_slot


def destroy_entity(state: EnvState, config: EnvConfig, slot: jnp.int32) -> EnvState:
    """Set alive[slot]=False, remove from grid, zero out slot data."""
    ey = state.y[slot]
    ex = state.x[slot]

    # Remove slot from grid cell via branchless compaction
    cell = state.grid[ey, ex]  # [max_stack]
    mask = cell == slot
    cleared = jnp.where(mask, jnp.int32(-1), cell)
    # Sort so -1s go to end: argsort on (cleared == -1) pushes True to back
    order = jnp.argsort(cleared == -1)
    compacted = cleared[order]

    new_grid = state.grid.at[ey, ex].set(compacted)
    new_grid_count = state.grid_count.at[ey, ex].set(
        jnp.maximum(state.grid_count[ey, ex] - 1, 0)
    )

    # Zero out entity data
    new_alive = state.alive.at[slot].set(False)
    new_entity_type = state.entity_type.at[slot].set(0)
    new_tags = state.tags.at[slot].set(jnp.zeros(state.tags.shape[1], dtype=jnp.bool_))
    new_props = state.properties.at[slot].set(jnp.zeros(state.properties.shape[1], dtype=jnp.float32))
    new_x = state.x.at[slot].set(0)
    new_y = state.y.at[slot].set(0)

    return state.replace(
        alive=new_alive,
        entity_type=new_entity_type,
        x=new_x,
        y=new_y,
        tags=new_tags,
        properties=new_props,
        grid=new_grid,
        grid_count=new_grid_count,
    )


def move_entity(
    state: EnvState,
    config: EnvConfig,
    slot: jnp.int32,
    new_x: jnp.int32,
    new_y: jnp.int32,
) -> tuple[EnvState, jnp.bool_]:
    """Move entity to (new_x, new_y). Returns (new_state, moved)."""
    old_x = state.x[slot]
    old_y = state.y[slot]

    in_bounds = (new_x >= 0) & (new_x < config.grid_w) & (new_y >= 0) & (new_y < config.grid_h)

    # Clamp for safe indexing even on false branch
    safe_nx = jnp.clip(new_x, 0, config.grid_w - 1)
    safe_ny = jnp.clip(new_y, 0, config.grid_h - 1)

    has_room = state.grid_count[safe_ny, safe_nx] < config.max_stack
    can_move = in_bounds & has_room

    # Remove from old cell (branchless compaction)
    old_cell = state.grid[old_y, old_x]
    mask = old_cell == slot
    cleared = jnp.where(mask, jnp.int32(-1), old_cell)
    order = jnp.argsort(cleared == -1)
    compacted = cleared[order]

    grid_after_remove = state.grid.at[old_y, old_x].set(compacted)
    count_after_remove = state.grid_count.at[old_y, old_x].set(
        jnp.maximum(state.grid_count[old_y, old_x] - 1, 0)
    )

    # Add to new cell
    new_count = count_after_remove[safe_ny, safe_nx]
    safe_idx = jnp.minimum(new_count, config.max_stack - 1)
    grid_after_add = grid_after_remove.at[safe_ny, safe_nx, safe_idx].set(slot)
    count_after_add = count_after_remove.at[safe_ny, safe_nx].set(new_count + 1)

    new_x_arr = state.x.at[slot].set(safe_nx)
    new_y_arr = state.y.at[slot].set(safe_ny)

    moved_state = state.replace(
        x=new_x_arr,
        y=new_y_arr,
        grid=grid_after_add,
        grid_count=count_after_add,
    )

    final_state = jax.tree.map(
        lambda m, o: jnp.where(can_move, m, o), moved_state, state
    )

    return final_state, can_move


def get_entities_at(
    state: EnvState, x: jnp.int32, y: jnp.int32
) -> tuple[jax.Array, jnp.int32]:
    """Return (slot_indices, count) for entities at (x, y)."""
    return state.grid[y, x], state.grid_count[y, x]


def find_by_tag(state: EnvState, tag_idx: int) -> jax.Array:
    """Return boolean mask of alive entities with the given tag."""
    return state.alive & state.tags[:, tag_idx]


def find_by_type(state: EnvState, type_id: int) -> jax.Array:
    """Return boolean mask of alive entities of the given type."""
    return state.alive & (state.entity_type == type_id)


def rebuild_grid(state: EnvState, config: EnvConfig) -> EnvState:
    """Rebuild grid tensor from scratch using alive, x, y arrays."""
    grid = jnp.full((config.grid_h, config.grid_w, config.max_stack), -1, dtype=jnp.int32)
    grid_count = jnp.zeros((config.grid_h, config.grid_w), dtype=jnp.int32)

    def place_entity(carry, slot_idx):
        g, gc = carry
        is_alive = state.alive[slot_idx]
        ey = state.y[slot_idx]
        ex = state.x[slot_idx]
        count = gc[ey, ex]
        safe_count = jnp.minimum(count, config.max_stack - 1)

        new_g = g.at[ey, ex, safe_count].set(
            jnp.where(is_alive, slot_idx, g[ey, ex, safe_count])
        )
        new_gc = gc.at[ey, ex].set(
            jnp.where(is_alive, count + 1, count)
        )
        return (new_g, new_gc), None

    (grid, grid_count), _ = jax.lax.scan(
        place_entity, (grid, grid_count), jnp.arange(config.max_entities)
    )

    return state.replace(grid=grid, grid_count=grid_count)
