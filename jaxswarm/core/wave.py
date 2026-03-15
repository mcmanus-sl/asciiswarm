"""Cellular Automata distance field — XLA-friendly Dijkstra map via min-pooling."""

import jax
import jax.numpy as jnp


def compute_distance_field(wall_mask: jnp.bool_, target_mask: jnp.bool_) -> jnp.float32:
    """
    Compute shortest-path distances from target cells to all reachable cells.
    Pure tensor math — no queues, no dynamic branching.

    Args:
        wall_mask: bool[H, W] — True where impassable
        target_mask: bool[H, W] — True at goal cells (distance = 0)

    Returns:
        float32[H, W] — distance from nearest target. 999.0 = unreachable/wall.
    """
    h, w = wall_mask.shape
    max_steps = h + w

    grid = jnp.where(target_mask, 0.0, 999.0)
    grid = jnp.where(wall_mask, 999.0, grid)

    def step_fn(g, _):
        n_up = jnp.pad(g[:-1, :], ((1, 0), (0, 0)), constant_values=999.0)
        n_down = jnp.pad(g[1:, :], ((0, 1), (0, 0)), constant_values=999.0)
        n_left = jnp.pad(g[:, :-1], ((0, 0), (1, 0)), constant_values=999.0)
        n_right = jnp.pad(g[:, 1:], ((0, 0), (0, 1)), constant_values=999.0)

        min_neighbors = jnp.minimum(jnp.minimum(n_up, n_down), jnp.minimum(n_left, n_right))
        new_g = jnp.minimum(g, min_neighbors + 1.0)

        new_g = jnp.where(target_mask, 0.0, new_g)
        new_g = jnp.where(wall_mask, 999.0, new_g)
        return new_g, None

    final_grid, _ = jax.lax.scan(step_fn, grid, None, length=max_steps)
    return final_grid
