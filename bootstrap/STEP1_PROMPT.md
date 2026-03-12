# First Claude Code Prompt — Kernel Bootstrap

## Roles

- **HUMAN**: You, the developer. You run Claude Code, review output, make design decisions.
- **CLAUDE CODE**: The AI pair-programmer. It receives the prompt below and builds the kernel.
- **AGENT SWARM**: Future autonomous Claude instances that will write game modules against the kernel. They don't exist yet. The kernel is being built FOR them.
- **RL EVALUATOR**: A future RL-based evaluation pipeline (random agent + trained PPO agent + invariant tests) that validates games quantitatively. Doesn't exist yet.

## Instructions for HUMAN

1. Start from the root of your project (which should contain DEVELOPMENT_PLAN.md and game specs in `game_specs/`).
2. Copy everything inside the code fence below and paste it as your first Claude Code prompt.
3. While CLAUDE CODE works, watch for these five things:
   - **Collision cancellation**: Make sure collision events fire BEFORE the move completes, so userland handlers can cancel a move (e.g., walking into a wall). If CLAUDE CODE doesn't implement `event.cancel()`, ask it to.
   - **Behavior execution order**: Entities should process in deterministic order (by creation order or ID sort). If CLAUDE CODE uses dict iteration, verify it's stable (Python 3.7+ dicts are insertion-ordered, but be explicit).
   - **Input event payload**: `action` should be a string, `payload` optional. The action vocabulary (e.g., `"move_north"`, `"use"`) is defined by userland, not kernel. The kernel doesn't know what actions mean.
   - **Serializer round-trip**: `serialize → load → serialize` must produce byte-identical output. This is the single most important test. The entire mechanical test harness depends on it.
   - **Gymnasium contract**: `env.reset()` and `env.step()` must return properly shaped tuples. Run `check_env()` from Stable-Baselines3 to verify.
4. Once all tests pass and HUMAN is satisfied, proceed to Step 2 by telling CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step 2."

---

## Prompt for CLAUDE CODE (copy the code fence below)

~~~
Read DEVELOPMENT_PLAN.md for full project context and the 7-step plan.

# Roles in this project

- HUMAN: The developer sitting at the keyboard. That's who you're talking to.
- CLAUDE CODE: You. The AI pair-programmer building the kernel right now.
- AGENT SWARM: Future autonomous Claude instances that will write game modules against this kernel. They don't exist yet. You are building the kernel FOR them.
- RL EVALUATOR: A future RL-based evaluation pipeline (random agent + trained PPO agent + invariant tests) that validates games quantitatively. Doesn't exist yet.

# Your role

You are pair-programming with HUMAN. You are building the KERNEL of a turn-based ASCII game engine in Python. The kernel is a `gymnasium.Env` subclass — the engine IS the Gymnasium environment, no separate wrapper needed. This kernel is a stable foundation that AGENT SWARM will later write game modules against. This is Step 1 of DEVELOPMENT_PLAN.md. You are NOT autonomous — ask HUMAN questions when you hit design forks.

# Key architecture decision

The kernel subclasses `gymnasium.Env` directly. This means `reset()`, `step()`, `observation_space`, and `action_space` are native to the engine. Games are Python modules with a `setup(env)` function that registers entity types, behaviors, spawning rules, and termination conditions. Each game is one file. The swarm writes these files.

There is NO separate Gymnasium wrapper. There is NO TypeScript-Python bridge. There is NO FFI boundary. Everything is Python. `PPO("MlpPolicy", YourGameEnv()).learn(100_000)` just works.

# What to build

Initialize a Python project (pytest for testing, gymnasium as dependency) and implement the kernel with the architecture below. Target ~500 lines of kernel code. This should be small, trusted, and boring.

## Core: Grid World
- `GridGameEnv` class: subclasses `gymnasium.Env`. Constructor takes `width`, `height`, a `game_module` (the Python module with a `setup()` function), and an optional `seed` (int, default: 42).
- Internally maintains a 2D grid of cells. Each cell holds an ordered stack of entities (multiple entities per cell).
- The env is the single source of truth for all game state.
- `env.random()` → returns a deterministic float in [0, 1) from a seeded PRNG (use Python's `random.Random` with the seed). The PRNG state is serialized and restored by `serialize_state()`/`load_state()`.
- **`random.random()` and `numpy.random` are strictly forbidden in engine and userland code.** All randomness must go through `env.random()` to preserve determinism.

## Core: Entity System
- Entities have: a unique string ID, deterministically generated via internal counter (e.g., `"e1"`, `"e2"`, ...). The counter is serialized and restored by `load_state()`. Do NOT use `uuid4()` or any random source for IDs. Entities also have: a string `type`, a position `(x, y)`, a glyph (single character for ASCII render), a z-order (for rendering — highest z on top), and a `properties` dict (`dict[str, Any]`).
- `env.create_entity(type, x, y, glyph, z_order=0, properties=None)` → Entity
- `env.destroy_entity(id)` → None
- `env.move_entity(id, x, y)` → bool (False if out of bounds)
- `env.get_entities_at(x, y)` → list[Entity]
- `env.get_entities_by_type(type)` → list[Entity]
- `env.get_entity(id)` → Entity | None
- Entities can set/get arbitrary properties: `entity.set(key, value)`, `entity.get(key)`

## Core: Event System
- Simple pub/sub. The kernel emits built-in events. Userland code (written later by AGENT SWARM) can emit custom events.
- Built-in events: `turn_start`, `input`, `collision` (when an entity moves into an occupied cell), `entity_created`, `entity_destroyed`, `turn_end`
- All events carry a payload dict. `collision` carries `{"mover": Entity, "occupants": list[Entity], "x": int, "y": int}`.
- Events must support cancellation: operations that change state (e.g., `move_entity`) must emit events BEFORE mutating state. Handlers receive an event object with a `cancel()` method. If a handler calls `cancel()`, the operation is aborted and returns `False`. Do NOT implement post-mutation rollback.
- `env.on(event_name, handler)` → unsubscribe callable
- `env.emit(event_name, payload)` — userland can emit custom events too
- Handlers execute in registration order.

## Core: Turn Loop
- `env.step(action)` — this is the Gymnasium entry point. `action` is an int index into the action space. The game module's `setup()` defines the mapping from action indices to action strings.
- Internal sequence: emit `turn_start` → emit `input` with `{"action": action_string, "payload": None}` → run all registered behavior handlers for each entity (see below) → emit `turn_end` → compute observation, reward, terminated, truncated.
- The turn is fully synchronous. No async. State is deterministic given the same inputs.
- Guardrail: The kernel must track event emission depth during a turn. If a single turn exceeds a configurable max cascade depth (default: 1000), raise an error. This prevents autonomous userland code from creating infinite event loops.

## Core: Entity Behaviors (the hook point for AGENT SWARM code)
- `env.register_behavior(entity_type: str, handler: Callable[[Entity, GridGameEnv], None])`
- During the turn loop, after input is processed, the kernel iterates all entities in deterministic order (by creation order) and calls their registered behavior handler if one exists.
- This is how AGENT SWARM will write game logic. A behavior for type "goblin" runs every turn for every goblin. A behavior for type "fire" might spread each turn.
- Behaviors can call any kernel primitive (move, destroy, create, emit, query).

## Core: ASCII Renderer
- `env.render()` → returns the grid as a string (Gymnasium's `render_mode="ansi"`). Each cell shows the glyph of the highest z-order entity. Empty cells show `"."` (configurable).
- Output is exactly `height` lines of `width` characters. No padding, no border, no decoration. Clean grid.

## Core: State Serializer
- `env.serialize_state()` → a plain JSON-serializable dict containing: grid dimensions, all entities with their full properties, the current turn number, the PRNG state, and the entity ID counter.
- `env.load_state(state)` → restores from a serialized state. `load_state()` must re-instantiate actual `Entity` class instances with working methods (`set()`, `get()`, etc.). Raw parsed dicts are not sufficient. This enables deterministic testing: set up state, run input, assert new state.
- Entity IDs: deterministically generated via internal counter (e.g., `"e1"`, `"e2"`, ...). Counter is serialized and restored by `load_state()` to prevent collisions after load.
- Serialization format: entities sorted by numeric ID, property keys sorted alphabetically. This guarantees `json.dumps(serialize_state()) == json.dumps(serialize_state())` for identical states.

## Core: Gymnasium Integration
- `observation_space`: a `gymnasium.spaces.Box` or `gymnasium.spaces.Dict` that encodes the grid state. The game module's `setup()` declares which entity types and property keys matter, and the kernel builds the observation space from that.
- `action_space`: a `gymnasium.spaces.Discrete(n)` where n is the number of valid actions declared by the game module.
- `reset(seed=None)` → (observation, info). Calls the game module's `setup()` to initialize entities and behaviors, returns initial observation.
- `step(action)` → (observation, reward, terminated, truncated, info). Runs one turn, returns results.
- The game module's `setup()` also registers a `compute_reward` function and a `check_terminated` function that the kernel calls after each turn.

## What NOT to build
- No game logic. No enemies, items, doors, keys, health, combat. That is all userland — AGENT SWARM builds it later.
- No CLI or interactive mode yet.
- No config files.
- No AI, pathfinding, or FOV. Those are userland concerns.
- No separate Gymnasium wrapper — the kernel IS the env.

## Project structure

```
asciiswarm/
  kernel/
    __init__.py
    env.py            — GridGameEnv class (gymnasium.Env subclass)
    entity.py         — Entity class
    events.py         — pub/sub event system
    renderer.py       — ASCII output
    serializer.py     — state save/load
    types.py          — shared types
  __init__.py         — public API exports
tests/
  kernel/
    test_world.py
    test_entity.py
    test_events.py
    test_renderer.py
    test_serializer.py
    test_behaviors.py
    test_gymnasium.py — Gymnasium contract tests
pyproject.toml
```

## Tests to write immediately
Write thorough tests for every kernel primitive. These tests are the foundation that AGENT SWARM will rely on. At minimum:
- Create entity, verify it exists at correct position
- Move entity, verify old cell empty, new cell occupied
- Move out of bounds returns False, entity stays
- Destroy entity, verify removed from grid and queries
- Multiple entities on same cell, z-order rendering
- Event handlers fire in order, carry correct payloads
- Collision event fires when moving into occupied cell
- Collision cancellation prevents the move
- Behavior handlers run for correct entity types, in creation order
- Full turn sequence: input → behaviors → events all fire correctly
- Serialize → load → serialize produces identical JSON (round-trip)
- Load state, run input, assert deterministic output
- `env.random()` returns same sequence from same seed
- Serialize → load → `env.random()` continues the sequence correctly
- Two envs with same seed produce identical random sequences
- Event cascade guard raises when max depth exceeded
- `env.reset()` returns valid observation matching observation_space
- `env.step(action)` returns correctly shaped tuple
- `check_env()` from Stable-Baselines3 passes

After building this, run all tests and make sure they pass. Then ask HUMAN what they think before moving on.

When HUMAN is satisfied, HUMAN will proceed to Step 2 as described in DEVELOPMENT_PLAN.md.
~~~
