# First Claude Code Prompt — Kernel Bootstrap

## Roles

- **HUMAN**: You, the developer. You run Claude Code, review output, make design decisions.
- **CLAUDE CODE**: The AI pair-programmer. It receives the prompt below and builds the kernel.
- **AGENT SWARM**: Future autonomous Claude instances that will write game modules against the kernel. They don't exist yet. The kernel is being built FOR them.
- **RL EVALUATOR**: A future RL-based evaluation pipeline (random agent + trained PPO agent + invariant tests) that validates games quantitatively. Doesn't exist yet.

## Instructions for HUMAN

1. Start from the root of your project (which should contain `DEVELOPMENT_PLAN.md` and game specs in `game-specs/`).
2. Copy everything inside the code fence below and paste it as your first Claude Code prompt.
3. While CLAUDE CODE works, watch for these five things:
   - **Collision cancellation**: Make sure collision events fire BEFORE the move completes, so userland handlers can cancel a move (e.g., walking into a wall). If CLAUDE CODE doesn't implement `event.cancel()`, ask it to.
   - **Behavior execution order**: Entities should process in deterministic order (by creation order). If CLAUDE CODE uses dict iteration, verify it's insertion-ordered (Python 3.7+ guarantees this, but verify the code relies on it intentionally).
   - **Input event payload**: `action` should be a string from the game's declared action list (in `GAME_CONFIG`). The kernel validates the action is in the game's declared set and ignores invalid actions.
   - **Serializer round-trip**: `serialize → load → serialize` must produce byte-identical JSON output. This is the single most important test. The entire mechanical test harness depends on it.
   - **Gym interface**: `env.step()` must return the standard Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)`. `env.reset()` must return `(obs, info)`. Observation and action spaces must be correctly typed and built dynamically from `GAME_CONFIG`.
4. Once all tests pass and HUMAN is satisfied, proceed to Step 2 by telling CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step 2."

---

## Prompt for CLAUDE CODE (copy the code fence below)

~~~
Read DEVELOPMENT_PLAN.md for full project context, the 7-step plan, and the Engine Contract. Pay special attention to the Engine Contract section — it defines the per-game GAME_CONFIG system, immutable structural rules, and universal reward signal that every game must obey.

# Roles in this project

- HUMAN: The developer sitting at the keyboard. That's who you're talking to.
- CLAUDE CODE: You. The AI pair-programmer building the kernel right now.
- AGENT SWARM: Future autonomous Claude instances that will write game modules against this kernel. They don't exist yet. You are building the kernel FOR them.
- RL EVALUATOR: A future RL pipeline that trains PPO agents against games to evaluate them. Doesn't exist yet.

# Your role

You are pair-programming with HUMAN. You are building the KERNEL of a turn-based ASCII game engine in Python. This kernel IS a Gymnasium environment — games built on it are automatically RL-evaluable. This is Step 1 of DEVELOPMENT_PLAN.md. You are NOT autonomous — ask HUMAN questions when you hit design forks.

# What to build

Initialize a Python project (pytest for testing, gymnasium and numpy as dependencies) and implement the kernel with the architecture below.

## Core: Grid World (also a Gymnasium Env)

- `GridGameEnv` class, inheriting from `gymnasium.Env`.
- Constructor takes a `game_module` (a Python object/module with `GAME_CONFIG` dict and `setup(env)` function) and an optional `seed` (int, default 42).
- The kernel reads `game_module.GAME_CONFIG` and merges it with defaults (see types.py below) to produce the resolved config.
- Grid dimensions come from `config['grid']` — a `(width, height)` tuple.
- Each cell holds an ordered list of entities (multiple entities per cell).
- The env is the single source of truth for all game state.
- `env.random()` → returns a deterministic float in [0, 1) from a seeded PRNG. Use Python's `random.Random` with the seed. The PRNG state is included in `serialize_state()` and restored by `load_state()`.
- **Native `random.random()` and `numpy.random` are strictly forbidden in engine and userland code.** All randomness must go through `env.random()` to preserve determinism.
- The env tracks a `turn_number` (int, starts at 0, incremented each `handle_input` call).
- The env tracks a `status` field: `'playing'` | `'won'` | `'lost'`, default `'playing'`.
- `env.end_game(status: str)` — sets the status. Must be `'won'` or `'lost'`. Once called, further `handle_input()` calls are no-ops (the game is over).

### Gymnasium interface

- `action_space`: `gymnasium.spaces.Discrete(len(config['actions']))` — built dynamically from the game's declared action list.
- `ACTION_MAP`: a dict mapping int → string, built from `config['actions']`. E.g., if `actions = ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait']`, then `ACTION_MAP = {0: 'move_n', 1: 'move_s', ..., 5: 'wait'}`.
- `observation_space`: `gymnasium.spaces.Dict` containing two keys:
  - `'grid'`: `gymnasium.spaces.Box(low=0.0, high=1.0, shape=(len(config['tags']), height, width), dtype=np.float32)` — channel-first spatial tag map.
  - `'scalars'`: `gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(3 + len(config['player_properties']),), dtype=np.float32)` — normalized grid dimensions, turn number, and player properties. Uses infinite bounds because even clamped values can hit floating-point edge cases (e.g., turn 1001/1000 = 1.001) that crash SB3's strict bounds checking. The neural net doesn't use the Space bounds — it only cares about the actual numbers.
  This Dict structure lets SB3's `MultiInputPolicy` automatically apply a CNN to the grid (preserving spatial locality) and an MLP to the scalars. A flat 1D observation would destroy all spatial structure and make pathfinding unlearnable.
- `step(action: int)` → `(obs, reward, terminated, truncated, info)`
  - **Early exit guard:** If `status != 'playing'` at the very start of `step()` (game was already over before this call), immediately return `(_get_obs(), 0.0, True, False, info)` without calling `handle_input` or computing rewards. The terminal reward is a one-time payout on the exact step the game ends — calling `step()` again must not award another +10.0.
  - Resets `self._current_step_reward = 0.0`.
  - The kernel has a built-in listener for `reward` events that adds `payload['amount']` to `self._current_step_reward`.
  - Translates the int action to a string via `ACTION_MAP`.
  - Calls `handle_input(action_string)`.
  - Computes reward additively: start with `config['step_penalty']` + `self._current_step_reward`. Then, if `status == 'won'`, add +10.0. If `status == 'lost'`, add -10.0. Terminal rewards must massively outweigh cumulative step penalties — otherwise PPO learns to die instantly because `-10.01` (1-step death) beats `-3.0` (400-step win with step_penalty=-0.01). This ensures intermediate rewards emitted during the winning/losing turn are never silently discarded.
  - `terminated = (status != 'playing')`.
  - `truncated = (turn_number >= config['max_turns'])`. The environment MUST natively enforce its own turn limit. Without this, evaluation scripts that loop `while not (terminated or truncated)` will hang forever on games where the random agent never triggers a win/loss.
  - `info = {'turn': turn_number, 'status': status}`.
  - Returns the observation via `_get_obs()`.
- `reset(seed=None, options=None)` → `(obs, info)`
  - If `seed` is provided, re-seeds the PRNG. **CRITICAL: If `seed` is `None`, do NOT reset the PRNG.** Allow it to continue its sequence so every episode generates a different random layout. If you re-seed to the stored instance seed every reset, every episode produces the identical dungeon — the agent memorizes one path and fails to generalize.
  - Clears all state (entities, grid, behaviors, turn number, status). **MEMORY LEAK GUARD: You MUST explicitly clear all registered event handlers/callbacks in the Event System.** Because `setup(env)` is called every reset and re-registers handlers, failing to wipe them means by episode 500 every collision fires 500 duplicate callbacks, grinding the CPU to a halt.
  - Calls the game module's `setup(env)` to rebuild the world from scratch.
  - Validates that exactly one entity tagged `player` exists (the player singleton rule). Raises an error if violated.
  - Returns the initial observation.
- `_get_obs()` → `dict` with `'grid'` and `'scalars'` numpy arrays (see Observation Vector below).

### Game module interface

Every game module must export two things:

1. **`GAME_CONFIG`** — a dict declaring the game's configuration. All keys are optional; the kernel merges with defaults. See the Engine Contract in DEVELOPMENT_PLAN.md for the full key list and defaults.

2. **`setup(env)`** — a function that receives the `GridGameEnv` instance and initializes the game: creates entities, registers behaviors and event handlers, sets up spawning logic and termination conditions. Called by `reset()`.

The kernel validates the game module on construction:
- `GAME_CONFIG` must exist and be a dict.
- `setup` must exist and be callable.
- Missing either raises `ValueError`.

The kernel merges the game's `GAME_CONFIG` with `DEFAULT_GAME_CONFIG` from types.py. Game values override defaults. The resolved config is stored as `env.config`.

## Core: Entity System

- Entities have:
  - A unique string ID, deterministically generated via internal counter: `'e1'`, `'e2'`, etc. The counter is serialized and restored by `load_state()`. Do NOT use `uuid` or any random source for IDs.
  - A string `type` (e.g., `'player'`, `'wall'`, `'goblin'`).
  - A list of string `tags` validated against the game's declared tag list (`config['tags']`). Entities may have multiple tags. The tag list must not be empty.
  - A position `(x, y)`.
  - A `glyph`: single character for ASCII rendering.
  - A `z_order`: int for rendering (highest z on top).
  - A `properties` dict (`dict[str, Any]`). **Serialization guardrail:** Properties MUST be strictly JSON-serializable primitives (`int`, `float`, `str`, `bool`, `list`, `dict`). Never store Entity instances or functions in properties. If an entity needs to reference another entity, store its string ID (e.g., `entity.set('target', player.id)`, not `entity.set('target', player)`).
- `env.create_entity(type, x, y, glyph, tags, z_order=0, properties=None)` → Entity
  - `tags` is required. The kernel validates that every tag is in the game's declared tag list (`config['tags']`). Raises `ValueError` for unknown tags.
- `env.destroy_entity(id)` → None
- `env.move_entity(id, x, y)` → bool (False if out of bounds or cancelled by event handler)
- `env.get_entities_at(x, y)` → list[Entity]
- `env.get_entities_by_type(type)` → list[Entity]
- `env.get_entities_by_tag(tag)` → list[Entity]
- `env.get_entity(id)` → Entity | None
- `env.get_all_entities()` → list[Entity] (in creation order)
- Entities can set/get arbitrary properties: `entity.set(key, value)`, `entity.get(key, default=None)`

## Core: Event System

- Simple pub/sub. The kernel emits built-in events. Userland code can emit custom events.
- Built-in events: `turn_start`, `input`, `before_move`, `collision`, `entity_created`, `entity_destroyed`, `turn_end`, `reward`
- Event payloads:
  - `input`: `{ 'action': str, 'payload': Any }`
  - `before_move`: `{ 'entity': Entity, 'from_x': int, 'from_y': int, 'to_x': int, 'to_y': int }` — fires before ANY move, even to empty cells. Supports cancellation.
  - `collision`: `{ 'mover': Entity, 'occupants': list[Entity], 'x': int, 'y': int }` — fires when moving to an occupied cell. Supports cancellation.
  - `entity_created`: `{ 'entity': Entity }`
  - `entity_destroyed`: `{ 'entity': Entity }`
  - `reward`: `{ 'amount': float }` — emitted by userland for intermediate reward shaping. The Gym `step()` method sums these.
- Cancellable events: `before_move` and `collision`. State-changing operations (move) emit the event BEFORE mutating state. Handlers receive an event object with a `cancel()` method. If cancelled, the operation is aborted and returns False. Do NOT implement post-mutation rollback.
- `env.on(event_name, handler)` → unsubscribe callable
- `env.emit(event_name, payload)` — userland can emit custom events.
- Handlers execute in registration order.

## Core: Turn Loop

- `env.handle_input(action: str, payload=None)` — entry point for a turn.
- The kernel validates that `action` is in the game's declared action list (`config['actions']`). If not, the call is a no-op and returns.
- If `env.status != 'playing'`, the call is a no-op and returns.
- Sequence: increment `turn_number` → emit `turn_start` → emit `input` → run all registered behavior handlers for each entity → emit `turn_end`.
- **Snapshot iteration guardrail:** When iterating entities for behaviors, the kernel MUST iterate over a shallow copy/snapshot of the entity list (e.g., `list(env.get_all_entities())`). Before calling each entity's behavior, verify the entity still exists (`env.get_entity(id) is not None`). If an earlier behavior destroyed it this turn, skip it.
- **Early termination guardrail:** After each behavior execution and after each event emission, check `env.status != 'playing'`. If the game has ended mid-turn (e.g., a goblin killed the player), immediately stop processing remaining behaviors and skip to `turn_end`. This prevents zombie entities from acting on dead players or corrupted state.
- The turn is fully synchronous. No async. State is deterministic given the same inputs.
- **Cascade depth guardrail:** The kernel tracks event emission depth. If a single turn exceeds a configurable `max_cascade_depth` (default: 1000), raise an error. This prevents userland code from creating infinite event loops.

## Core: Entity Behaviors

- `env.register_behavior(entity_type: str, handler: Callable[[Entity, GridGameEnv], None])`
- During the turn loop, after input is processed, the kernel iterates all entities in creation order and calls their registered behavior handler if one exists for that entity's type.
- This is how AGENT SWARM writes game logic. A behavior for type `'goblin'` runs every turn for every goblin.
- Behaviors can call any kernel primitive (move, destroy, create, emit, query).

## Core: ASCII Renderer

- `env.render_ascii()` → str — returns the grid as a string. Each cell shows the glyph of the highest z-order entity. Empty cells show `'.'`.
- Output is exactly `height` lines of `width` characters. No padding, no border, no decoration. Clean grid.
- Also serves as the Gymnasium `render()` method when `render_mode='ansi'`.

## Core: State Serializer

- `env.serialize_state()` → a plain dict (JSON-serializable) containing: grid dimensions, all entities with their full state (including tags), all registered behavior type→handler mappings are NOT serialized (they are re-registered by the setup function on load), the PRNG state, the entity ID counter, the turn number, and the game status.
- **PRNG serialization guardrail:** `random.getstate()` returns a tuple containing an inner tuple of integers. `json.dumps()` converts all tuples to lists. Inside `load_state`, you MUST manually convert the parsed PRNG state back to a nested tuple before calling `random.setstate()`, otherwise Python throws `TypeError: argument must be a tuple`. LLMs miss this 100% of the time.
- `env.load_state(state: dict)` → restores from a serialized state. Must re-instantiate actual Entity class instances with working methods. Raw dicts are not sufficient.
- **Hydration guardrail:** Do NOT call `setup(env)` inside `load_state`. Assume the environment has already been initialized (e.g., via a previous `reset()`). `load_state` must simply clear the current entity list, re-instantiate Entity class instances from the dict data, and overwrite the turn number, ID counter, status, and PRNG state. If you call `setup()` again, it will re-spawn all entities on top of the loaded ones — doubling the player, walls, and enemies.

### Observation (`_get_obs`)

The observation is a `dict` with two keys, matching the `Dict` observation space. The kernel reads from the resolved config:
- `obs_tags`: `config['tags']` — the list of tag strings to include in the spatial observation.
- `player_properties`: `config['player_properties']` — list of `{'key': str, 'max': number}` dicts.

**`'grid'`** — `np.ndarray` of shape `(len(obs_tags), height, width)`, dtype float32, channel-first:
- For each tag (channel), for each cell `(y, x)`: `1.0` if any entity in that cell has that tag, `0.0` otherwise.
- This preserves spatial locality so a CNN can learn "wall is north of player" from adjacency in the tensor, rather than requiring the MLP to memorize that index 10 and index 40 are neighbors.
- **Performance guardrail:** Do NOT build this array by iterating over every cell and every tag (O(width×height×tags)). Instead, initialize `np.zeros(shape)`, then iterate over `env.get_all_entities()` exactly once. For each entity, look up its tag indices and set `grid[tag_index, ent.y, ent.x] = 1.0`. This O(entities) approach is critical for RL training speed — the naive nested loop runs 300M iterations over 100k steps on a 30×20 grid, turning a 2-minute evaluation into a 2-hour one.

**`'scalars'`** — `np.ndarray` of shape `(3 + len(player_properties),)`, dtype float32:
1. **Grid dimensions (normalized):** `width / 100.0`, `height / 100.0` — 2 values.
2. **Turn number (normalized):** `turn_number / config['max_turns']` — 1 value.
3. **Player properties:** Find the single entity tagged `'player'`. For each entry in `player_properties`, read `entry['key']` from the player entity, normalize by dividing by `entry['max']`, and **clamp to [0.0, 1.0]** via `np.clip`. This is critical — if userland code sets health to 11/10, the raw normalized value of 1.1 would exceed the `Box(high=1.0)` bound and crash SB3. Default `0.0` if property is missing. If no player entity exists, output `0.0` for each.

The shape of both arrays is fixed for a given game configuration and does not change between steps, regardless of how many entities the swarm creates or destroys.

## types.py — Defaults and Constants

```python
DEFAULT_ACTIONS = ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait']

DEFAULT_TAGS = ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc']

DEFAULT_GAME_CONFIG = {
    'actions': DEFAULT_ACTIONS,
    'tags': DEFAULT_TAGS,
    'grid': (16, 16),
    'max_turns': 1000,
    'step_penalty': -0.01,
    'player_properties': [],
}
```

The kernel merges `game_module.GAME_CONFIG` over `DEFAULT_GAME_CONFIG`. Game values win.

## What NOT to build

- No game logic. No enemies, items, doors, keys, health, combat. That is all userland — AGENT SWARM builds it later.
- No CLI or interactive mode yet.
- No AI, pathfinding, or FOV. Those are userland concerns.

## Project structure

```
asciiswarm/
  kernel/
    __init__.py
    env.py            — GridGameEnv class (gymnasium.Env subclass)
    entity.py         — Entity class
    events.py         — pub/sub event system
    renderer.py       — ASCII output
    serializer.py     — state save/load helpers
    types.py          — DEFAULT_ACTIONS, DEFAULT_TAGS, DEFAULT_GAME_CONFIG
  __init__.py         — public API exports
tests/
  kernel/
    test_env.py       — GridGameEnv + Gymnasium interface tests
    test_entity.py
    test_events.py
    test_renderer.py
    test_serializer.py
    test_behaviors.py
    test_config.py    — GAME_CONFIG tests
games/                — empty, populated in Step 5
game-specs/           — populated in Step 0
pyproject.toml
```

## Tests to write immediately

Write thorough tests for every kernel primitive. These tests are the foundation that AGENT SWARM will rely on. At minimum:

**Entity tests:**
- Create entity, verify it exists at correct position with correct tags
- Entity with tag not in the game's declared tag list raises ValueError
- Entity with empty tag list raises ValueError
- entity.set / entity.get work correctly
- entity.get with default returns default for missing keys

**World tests:**
- Create entity, verify exists at correct position
- Move entity, verify old cell empty, new cell occupied
- Move out of bounds returns False, entity stays
- Destroy entity, verify removed from grid and all query methods
- Multiple entities on same cell, z-order rendering
- get_entities_by_tag returns correct entities
- get_all_entities returns in creation order
- Entity IDs are sequential: e1, e2, e3...

**Event tests:**
- Handlers fire in registration order, carry correct payloads
- before_move fires before every move attempt, cancellation prevents move
- Collision event fires when moving into occupied cell
- Collision cancellation prevents the move
- Unsubscribe works, including mid-iteration
- Custom events can be emitted and handled
- Reward events accumulate correctly

**Behavior tests:**
- Behavior handlers run for correct entity types, in creation order
- Behaviors can create/destroy/move entities

**Turn loop tests:**
- Full turn sequence: turn_start → input → behaviors → turn_end
- handle_input with action not in game's declared action list is a no-op
- handle_input after end_game is a no-op
- Event cascade guard throws at max depth
- Snapshot iteration: destroying an entity mid-behavior-loop does not crash or skip other entities
- Destroyed entity's behavior is skipped (not called after destruction)
- Early termination: if end_game() called during a behavior, remaining behaviors do not execute
- Early termination: zombie entities don't act on dead players (goblin kills player → orc doesn't pathfind to corpse)

**Serializer tests:**
- Serialize → load → serialize produces identical JSON (round-trip)
- Load state, run input, assert deterministic output
- Entity instances after load have working set/get methods
- PRNG state survives round-trip: random() sequence continues correctly
- Entity ID counter survives round-trip
- Turn number survives round-trip
- Game status survives round-trip

**Gym interface tests:**
- observation_space is a Dict with 'grid' and 'scalars' keys
- observation_space['grid'].shape == (len(tags), height, width)
- observation_space['scalars'].shape == (3 + len(player_properties),)
- action_space is Discrete(len(config['actions']))
- step() returns correct 5-tuple types
- step() returns terminated=True when game has ended
- step() reward is additive: step_penalty + intermediate rewards + terminal reward (all summed, not exclusive)
- step() on winning turn includes both intermediate reward events AND +10.0 terminal reward
- step() accumulates reward events via built-in listener
- reset() restores to initial state
- reset() with seed produces deterministic initial state
- reset() without seed produces different random layouts across episodes (PRNG continues, not re-seeded)
- reset() clears all event handlers — no duplicate callback accumulation across episodes
- reset() validates player singleton — raises if no player or multiple players after setup
- step() after game already ended returns reward=0.0 (no double terminal reward)
- step() sets truncated=True when turn_number >= max_turns
- Serializer round-trip: PRNG tuple/list conversion handled correctly (no TypeError on setstate)
- load_state does not call setup() — no duplicate entity spawning
- Entity properties containing non-JSON-serializable values (e.g., Entity references) are caught
- _get_obs() returns dict with 'grid' and 'scalars' numpy arrays
- _get_obs() is deterministic: same state = same output
- _get_obs() grid channels correctly reflect entity tag positions with spatial locality
- _get_obs() player properties are correct and clamped to [0.0, 1.0]
- _get_obs() property exceeding max is clamped (e.g., health 11/10 → 1.0, not 1.1)
- _get_obs() turn normalization uses config['max_turns']

**Config tests:**
- Game with custom actions builds correct action_space (Discrete of custom size)
- Game with custom tags builds correct observation_space
- Game with custom actions — ACTION_MAP maps to correct strings
- Game with custom tags — entity with tag not in custom list raises ValueError
- Game with custom tags — entity with tag in custom list succeeds
- Partial GAME_CONFIG merges correctly with defaults (e.g., only override 'grid', rest stays default)
- Game module missing GAME_CONFIG raises ValueError
- Game module missing setup() raises ValueError
- Default config works (game that specifies nothing but setup gets all defaults)

**Determinism tests:**
- Two envs with same seed produce identical random() sequences
- Load state, run N turns of fixed input, serialize. Repeat from same loaded state. Output identical.

**PRNG tests:**
- env.random() returns same sequence from same seed
- Serialize → load → env.random() continues sequence correctly

After building all of this, run all tests and make sure they pass. Then ask HUMAN what they think before moving on.

When HUMAN is satisfied, HUMAN will proceed to Step 2 as described in DEVELOPMENT_PLAN.md.
~~~
