# First Claude Code Prompt — JAX Core Bootstrap

## Roles

- **HUMAN**: You, the developer. You run Claude Code, review output, make design decisions, tune game balance using RL diagnostics.
- **CLAUDE CODE**: The AI pair-programmer. It receives the prompt below and builds the JAX core.
- **RL ORACLE**: A GPU-native RL pipeline that serves as a behavioral compiler — it doesn't just pass/fail, it tells you *how* the agent won and what it exploited. Built in Step 2.

## Instructions for HUMAN

1. Start from the root of your project (which should contain `DEVELOPMENT_PLAN.md` and game specs in `game-specs/`).
2. Copy everything inside the code fence below and paste it as your first Claude Code prompt.
3. While CLAUDE CODE works, watch for these five things:
   - **Fixed shapes**: All arrays must have shapes known at JIT trace time. No Python-side conditionals that change array shapes. If CLAUDE CODE uses `if` on a traced value, ask it to use `jax.lax.cond` or `jnp.where` instead.
   - **No Python side effects**: All state mutation must go through `state.replace()` or `.at[].set()`. No in-place mutation, no global state, no `print` inside JIT'd functions.
   - **Grid consistency**: After every `create_entity`, `destroy_entity`, `move_entity`, the `grid` tensor must stay in sync with `x`, `y`, `alive` arrays. If CLAUDE CODE rebuilds the grid from scratch each step, that's fine — just verify correctness.
   - **vmap compatibility**: `reset` and `step` must work under `jax.vmap`. No global state, no Python randomness, no host callbacks.
   - **Determinism**: Same `rng_key` → same trajectory. Test by running 100 steps, comparing against a second run from the same key.
4. Once all tests pass and HUMAN is satisfied, proceed to Step 2 by telling CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step 2."

---

## Prompt for CLAUDE CODE (copy the code fence below)

~~~
Read DEVELOPMENT_PLAN.md for full project context, the step plan, and the Engine Contract. Pay special attention to the Engine Contract section — it defines the JAX-native state representation, composable ECS architecture, immutable structural rules, and universal reward signal.

# Roles in this project

- HUMAN: The developer sitting at the keyboard. That's who you're talking to.
- CLAUDE CODE: You. The AI pair-programmer building the JAX core right now.
- RL ORACLE: A GPU-native RL pipeline that acts as a behavioral compiler for game mechanics. Built in Step 2.

# Your role

You are pair-programming with HUMAN. You are building the CORE of a turn-based ASCII game engine in JAX. This core provides pure-function primitives that game modules compose into complete environments via reusable ECS Systems. Everything must be JIT-compatible and vmap-safe. This is Step 1 of DEVELOPMENT_PLAN.md. You are NOT autonomous — ask HUMAN questions when you hit design forks.

# Design philosophy

The core provides low-level primitives (entity CRUD, grid ops, observation). Games compose these primitives via **ECS Systems** — reusable pure functions like `combat_system(state)`, `hunger_system(state)`, `fluid_dynamics_system(state)`. Each system is `EnvState -> EnvState`. Games wire systems together in their `step()` function. Game 14 (INF FORTRESS) imports all prior systems and runs them concurrently — it writes no new logic, just composition.

In Step 1, you build only the core primitives. The ECS systems are built incrementally starting in Step 3 as each game introduces new mechanics.

# What to build

Initialize a Python project (pytest for testing; jax[cuda12], equinox, optax, chex as dependencies) and implement the core with the architecture below.

## Core: EnvState and EnvConfig

### EnvConfig — Per-game configuration

```python
@chex.dataclass
class EnvConfig:
    grid_w:           int
    grid_h:           int
    max_entities:     int
    max_stack:        int       # max entities per grid cell
    num_entity_types: int       # integer type IDs: 0 = empty/unused slot
    num_tags:         int       # boolean tag channels
    num_props:        int       # float32 properties per entity
    num_actions:      int       # discrete action count
    max_turns:        int
    step_penalty:     float
    game_state_size:  int       # extra float32 slots for game-specific state
    prop_maxes:       tuple[float, ...]  # per-property normalization maxes for observation clamping
```

This is a static config — it is NOT part of the JIT-traced state. It parameterizes array shapes. Game modules export this as `CONFIG`.

### EnvState — All mutable game state

```python
@chex.dataclass
class EnvState:
    # Entity storage (struct-of-arrays)
    alive:        jnp.bool_[MAX_ENTITIES]           # which slots are occupied
    entity_type:  jnp.int32[MAX_ENTITIES]           # integer type ID per slot
    x:            jnp.int32[MAX_ENTITIES]            # x position
    y:            jnp.int32[MAX_ENTITIES]            # y position
    tags:         jnp.bool_[MAX_ENTITIES, NUM_TAGS]  # boolean tag matrix
    properties:   jnp.float32[MAX_ENTITIES, NUM_PROPS]  # property vector per entity

    # Grid (spatial index — derived from entity positions)
    grid:         jnp.int32[H, W, MAX_STACK]        # entity slot indices at each cell
    grid_count:   jnp.int32[H, W]                   # count of entities per cell

    # Game state
    turn_number:  jnp.int32
    status:       jnp.int32          # 0 = playing, 1 = won, -1 = lost
    reward_acc:   jnp.float32        # accumulated intermediate reward this step
    player_idx:   jnp.int32          # slot index of the player entity
    game_state:   jnp.float32[GAME_STATE_SIZE]  # game-specific counters/milestones

    # PRNG
    rng_key:      jax.random.PRNGKey
```

**All shapes are fixed at JIT trace time.** Entity "creation" sets `alive[slot] = True` and fills the slot's arrays. Entity "destruction" sets `alive[slot] = False`. No dynamic allocation.

### Initialization

```python
def init_state(config: EnvConfig, rng_key: PRNGKey) -> EnvState:
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
```

## Core: Entity CRUD (grid_ops.py)

All functions are pure — they take state and return new state. No in-place mutation.

### create_entity

```python
def create_entity(
    state: EnvState,
    entity_type: int,
    x: int, y: int,
    tags: jnp.bool_[NUM_TAGS],
    props: jnp.float32[NUM_PROPS],
) -> tuple[EnvState, jnp.int32]:
    """
    Find the first free slot (alive == False), fill it, update the grid.
    Returns (new_state, slot_index). If no free slot, returns (state, -1).
    """
```

Implementation notes:
- Use `jnp.argmin(state.alive)` to find the first free slot. If `state.alive.all()`, no free slot — return -1 and leave state unchanged.
- Set `alive[slot] = True`, fill `entity_type`, `x`, `y`, `tags`, `properties`.
- Add the slot index to `grid[y, x]` at position `grid_count[y, x]`, then increment `grid_count[y, x]`.
- Use `jnp.where` / `jax.lax.cond` for the "no free slot" branch — no Python `if` on traced values.

### destroy_entity

```python
def destroy_entity(state: EnvState, slot: jnp.int32) -> EnvState:
    """
    Set alive[slot] = False. Remove slot from grid[y, x]. Compact the grid stack.
    """
```

Implementation notes:
- Set `alive[slot] = False`.
- Remove the slot index from `grid[state.y[slot], state.x[slot]]`. Compact remaining entries (shift left to fill the gap). Decrement `grid_count`.
- **Branchless compaction**: To remove an element from a 1D JAX array and shift the rest left, do not use loops. Replace the target element with -1, then use `argsort`: `arr = jnp.where(arr == target_slot, -1, arr); compacted = arr[jnp.argsort(arr == -1)]`. Since `False < True`, this pushes all -1s to the back in a single vectorized step.
- Zero out the slot's type, tags, properties (prevents stale data from leaking into future reuse of the slot).

### move_entity

```python
def move_entity(
    state: EnvState, slot: jnp.int32,
    new_x: jnp.int32, new_y: jnp.int32,
) -> tuple[EnvState, jnp.bool_]:
    """
    Move entity from current position to (new_x, new_y).
    Returns (new_state, moved). moved=False if out of bounds or target cell full.
    """
```

Implementation notes:
- Check bounds: `0 <= new_x < grid_w` and `0 <= new_y < grid_h`.
- Check target cell capacity: `grid_count[new_y, new_x] < max_stack`.
- If valid: remove from old cell's grid stack, add to new cell's grid stack, update `x[slot]` and `y[slot]`.
- Return `(new_state, True)` on success, `(state, False)` on failure.
- All branching via `jnp.where`, not Python `if`.
- **Warning on JAX branching**: JAX evaluates **all branches** of `jnp.where` before selecting one. If you use an index like `grid_count` to update an array, you must clamp it first (e.g., `safe_idx = jnp.minimum(grid_count, max_stack - 1)`) so the false branch doesn't trace an out-of-bounds index, even if that branch's result is ultimately discarded. Out-of-bounds `.at[].set()` calls are silently dropped in JAX, which hides bugs.

### Query functions

```python
def get_entities_at(state: EnvState, x: int, y: int) -> tuple[jnp.int32[MAX_STACK], jnp.int32]:
    """Return (slot_indices, count) for entities at (x, y)."""
    return state.grid[y, x], state.grid_count[y, x]

def find_by_tag(state: EnvState, tag_idx: int) -> jnp.bool_[MAX_ENTITIES]:
    """Return a boolean mask of alive entities that have the given tag."""
    return state.alive & state.tags[:, tag_idx]

def find_by_type(state: EnvState, type_id: int) -> jnp.bool_[MAX_ENTITIES]:
    """Return a boolean mask of alive entities of the given type."""
    return state.alive & (state.entity_type == type_id)
```

### rebuild_grid

```python
def rebuild_grid(state: EnvState, config: EnvConfig) -> EnvState:
    """
    Rebuild the grid tensor from scratch using alive, x, y arrays.
    Use this after bulk state modifications to guarantee consistency.
    """
```

This is the "nuclear option" for grid consistency. Iterate all alive entities, place them in `grid[y, x]`. Games can call this after complex multi-entity operations. The core may also call it internally if simpler than maintaining incremental updates.

## Core: Observation Builder (obs.py)

```python
def get_obs(state: EnvState, config: EnvConfig) -> dict:
    """
    Returns {'grid': jnp.float32[NUM_TAGS, H, W], 'scalars': jnp.float32[N]}.
    """
```

### Grid observation

- Shape: `(num_tags, grid_h, grid_w)`, dtype float32.
- For each alive entity, for each of its tags, set `obs_grid[tag_idx, y, x] = 1.0`.
- Build efficiently using fully vectorized scatter updates. DO NOT use `jax.lax.fori_loop` or `jax.lax.scan` over the entity array, as this forces XLA to sequence operations and destroys GPU parallelism.

### Scalar observation

- Shape: `(3 + num_props,)`, dtype float32.
- Index 0: `grid_w / 100.0`
- Index 1: `grid_h / 100.0`
- Index 2: `turn_number / max_turns`
- Index 3..3+num_props: player's properties, normalized by dividing by `config.prop_maxes[i]`, clamped to [0, 1] via `jnp.clip`.

## Core: Movement Helper (movement.py)

```python
def move_toward(
    state: EnvState, slot: jnp.int32,
    target_x: jnp.int32, target_y: jnp.int32,
    rng_key: PRNGKey,
) -> tuple[EnvState, jnp.bool_]:
    """
    Move entity one step toward (target_x, target_y) using greedy Manhattan distance.
    Prefer the axis with greater distance. Break ties using rng_key.
    Returns (new_state, moved).
    """
```

This is a convenience function for NPC behaviors. It does NOT use BFS/pathfinding — just greedy one-step movement. BFS requires dynamic queues incompatible with JIT. Greedy Manhattan suffices for NPC movement in these game specs.

## Core: Turn Loop Pattern

The core does NOT implement a generic turn loop. Instead, it provides the primitives above, and each game module composes them — along with ECS Systems — into its own `step` function. The canonical pattern is:

```python
def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    # Reset per-step accumulator
    state = state.replace(reward_acc=jnp.float32(0.0))

    # Increment turn
    state = state.replace(turn_number=state.turn_number + 1)

    # Phase 1: Process player input (uses movement_system, interact_system, etc.)
    state = process_input(state, action)

    # Phase 2: Run entity behaviors (uses behavior_system with jax.lax.switch dispatch)
    state = run_behaviors(state)

    # Phase 3: Turn-end effects (uses hunger_system, farming_system, stress_system, etc.)
    state = turn_end(state)

    # Compute observation and reward
    obs = get_obs(state, CONFIG)
    reward = CONFIG.step_penalty + state.reward_acc
    reward = reward + jnp.where(state.status == 1, 10.0, 0.0)
    reward = reward + jnp.where(state.status == -1, -10.0, 0.0)
    done = (state.status != 0) | (state.turn_number >= CONFIG.max_turns)

    return state, obs, reward, done
```

Each ECS system is a pure function `EnvState -> EnvState`. Games compose systems in their phases. Game 14 imports all prior systems without rewriting any logic.

## Game Module Interface

Every game module exports:

```python
CONFIG = EnvConfig(
    grid_w=8, grid_h=8,
    max_entities=8, max_stack=2,
    num_entity_types=3, num_tags=6, num_props=4,
    num_actions=6, max_turns=200,
    step_penalty=-0.01,
    game_state_size=4,
)

def reset(rng_key: PRNGKey) -> tuple[EnvState, dict]:
    """Initialize game state and return (state, initial_obs)."""
    ...

def step(state: EnvState, action: jnp.int32) -> tuple[EnvState, dict, jnp.float32, jnp.bool_]:
    """Advance one turn. Returns (new_state, obs, reward, done)."""
    ...
```

Both `reset` and `step` must be pure functions compatible with `jax.jit` and `jax.vmap`.

## What CLAUDE CODE builds in Step 1

### Source files

```
jaxswarm/
  core/
    __init__.py      — public API exports
    state.py         — EnvState, EnvConfig, init_state
    grid_ops.py      — create_entity, destroy_entity, move_entity, queries, rebuild_grid
    obs.py           — get_obs (grid + scalars observation builder)
    movement.py      — move_toward helper
  systems/           — empty, populated starting Step 3
    __init__.py
  games/             — empty, populated starting Step 3
    __init__.py
  __init__.py
tests/
  core/
    test_state.py    — EnvState creation, init_state
    test_grid_ops.py — entity CRUD, movement, grid consistency
    test_obs.py      — observation shapes, values, normalization
    test_movement.py — move_toward correctness
    test_determinism.py — same key = same results
    test_vmap.py     — vmap smoke tests (reset, step primitives)
pyproject.toml
```

### Tests to write immediately

Write thorough tests for every core primitive. At minimum:

**State tests (test_state.py):**
- `init_state` creates state with correct shapes for given config
- All `alive` slots are False in initial state
- All grid cells are empty (-1) in initial state
- `rng_key` is stored correctly

**Grid ops tests (test_grid_ops.py):**
- Create entity: slot is alive, position correct, tags correct, properties correct
- Create entity: grid[y, x] contains the slot index, grid_count incremented
- Create entity when all slots full: returns (state, -1), state unchanged
- Destroy entity: slot is not alive, grid updated, properties zeroed
- Destroy entity: grid_count decremented, stack compacted (no gaps)
- Move entity: old cell updated, new cell updated, x/y updated
- Move entity out of bounds: returns (state, False), entity stays
- Move entity to full cell: returns (state, False), entity stays
- Multiple entities on same cell: grid stack correct, grid_count correct
- Create → destroy → create reuses slot
- `get_entities_at` returns correct slots and count
- `find_by_tag` returns correct mask (only alive entities with tag)
- `find_by_type` returns correct mask
- `rebuild_grid` produces same grid as incremental operations
- Grid consistency after create, destroy, move sequences

**Observation tests (test_obs.py):**
- `get_obs` returns dict with 'grid' and 'scalars' keys
- Grid observation shape: `(num_tags, grid_h, grid_w)`
- Grid observation: entity with tag X at (x,y) → obs_grid[X, y, x] == 1.0
- Grid observation: empty cells are 0.0
- Grid observation: multiple entities with same tag at same cell → still 1.0 (not 2.0)
- Scalar observation shape: `(3 + num_props,)`
- Scalar observation: grid dimensions normalized by 100
- Scalar observation: turn number normalized by max_turns
- Scalar observation: player properties normalized and clamped to [0, 1]
- Scalar observation: property exceeding max is clamped (e.g., 11/10 → 1.0)
- Observation is deterministic: same state → same output

**Movement tests (test_movement.py):**
- `move_toward` moves entity one step closer to target
- `move_toward` prefers axis with greater distance
- `move_toward` breaks ties using rng_key
- `move_toward` does not move if already at target
- `move_toward` handles blocked movement (returns state unchanged, moved=False)

**Determinism tests (test_determinism.py):**
- Two calls to `init_state` with same key produce identical states
- Sequence of create → move → destroy with same key produces identical results
- Different keys produce different results

**vmap tests (test_vmap.py):**
- `jax.vmap(init_state, in_axes=(None, 0))(config, keys)` works with 64 keys
- `jax.vmap(create_entity)` works across batched states
- `jax.vmap(move_entity)` works across batched states
- `jax.vmap(get_obs)` works across batched states
- No cross-contamination: modifying env 0 doesn't affect env 1

## What NOT to build

- No game logic. No enemies, items, combat, farming. That's built in Steps 3–5 as ECS Systems.
- No training loop. That's Step 2.
- No evaluator. That's Step 2.
- No CLI or interactive mode.
- No Gymnasium dependency. No SB3. No PyTorch.

## Watchlist for HUMAN

These are the most common failure modes when writing JAX game engines. Check for them during review:

1. **Fixed shapes**: All arrays must have shapes known at JIT trace time. No Python-side conditionals that change array shapes. `jnp.where` and `jax.lax.cond` are the JAX equivalents of `if/else`.

2. **No Python side effects**: All state mutation via `state.replace()` / `.at[].set()`. No in-place mutation. No `print` inside JIT'd functions (use `jax.debug.print` if needed during development).

3. **vmap compatibility**: `reset` and `step` must work under `jax.vmap`. No global state, no Python randomness, no host callbacks, no `io.StringIO`.

4. **Determinism**: Same `rng_key` → same trajectory. Test by running operations twice from the same key and comparing element-wise.

5. **Grid consistency**: `grid` tensor must stay in sync with `x`, `y`, `alive` arrays after every create/destroy/move. The simplest approach is to call `rebuild_grid` after every operation during development, then optimize later.

6. **PRNG discipline**: Every use of randomness must consume a key via `jax.random.split`. Never reuse a key. The `rng_key` in `EnvState` must be split before use and the new key stored back.

After building all of this, run all tests and make sure they pass. Then ask HUMAN what they think before moving on.

When HUMAN is satisfied, HUMAN will proceed to Step 2 as described in DEVELOPMENT_PLAN.md.
~~~
