"""EnvState, EnvConfig, and init_state — the foundation of the engine."""

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class EnvConfig:
    grid_w: int
    grid_h: int
    max_entities: int
    max_stack: int
    num_entity_types: int
    num_tags: int
    num_props: int
    num_actions: int
    max_turns: int
    step_penalty: float
    game_state_size: int
    prop_maxes: tuple


@chex.dataclass
class EnvState:
    # Entity storage (struct-of-arrays)
    alive: jax.Array        # bool [max_entities]
    entity_type: jax.Array  # int32 [max_entities]
    x: jax.Array            # int32 [max_entities]
    y: jax.Array            # int32 [max_entities]
    tags: jax.Array         # bool [max_entities, num_tags]
    properties: jax.Array   # float32 [max_entities, num_props]

    # Grid (spatial index)
    grid: jax.Array         # int32 [H, W, max_stack]
    grid_count: jax.Array   # int32 [H, W]

    # Game state
    turn_number: jax.Array  # int32 scalar
    status: jax.Array       # int32 scalar (0=playing, 1=won, -1=lost)
    reward_acc: jax.Array   # float32 scalar
    player_idx: jax.Array   # int32 scalar
    game_state: jax.Array   # float32 [game_state_size]

    # PRNG
    rng_key: jax.Array      # PRNGKey


def init_state(config: EnvConfig, rng_key: jax.Array) -> EnvState:
    """Create a blank state with all slots empty."""
    return EnvState(
        alive=jnp.zeros(config.max_entities, dtype=jnp.bool_),
        entity_type=jnp.zeros(config.max_entities, dtype=jnp.int32),
        x=jnp.zeros(config.max_entities, dtype=jnp.int32),
        y=jnp.zeros(config.max_entities, dtype=jnp.int32),
        tags=jnp.zeros((config.max_entities, config.num_tags), dtype=jnp.bool_),
        properties=jnp.zeros((config.max_entities, config.num_props), dtype=jnp.float32),
        grid=jnp.full((config.grid_h, config.grid_w, config.max_stack), -1, dtype=jnp.int32),
        grid_count=jnp.zeros((config.grid_h, config.grid_w), dtype=jnp.int32),
        turn_number=jnp.int32(0),
        status=jnp.int32(0),
        reward_acc=jnp.float32(0.0),
        player_idx=jnp.int32(0),
        game_state=jnp.zeros(config.game_state_size, dtype=jnp.float32),
        rng_key=rng_key,
    )
