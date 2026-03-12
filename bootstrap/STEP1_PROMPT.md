# First Claude Code Prompt — Kernel Bootstrap

## Roles

- **HUMAN**: You, the developer. You run Claude Code, review output, make design decisions.
- **CLAUDE CODE**: The AI pair-programmer. It receives the prompt below and builds the kernel.
- **AGENT SWARM**: Future autonomous Claude instances that will write game logic against the kernel. They don't exist yet. The kernel is being built FOR them.
- **RL EVALUATOR**: A future RL-based evaluation pipeline (random agent + trained PPO agent + invariant tests) that validates games quantitatively. Doesn't exist yet.

## Instructions for HUMAN

1. Start from the root of your empty project (which should contain ORIGINALDESIGN.md and DEVELOPMENT_PLAN.md).
2. Copy everything inside the code fence below and paste it as your first Claude Code prompt.
3. While CLAUDE CODE works, watch for these four things:
   - **Collision cancellation**: Make sure collision events fire BEFORE the move completes, so userland handlers can cancel a move (e.g., walking into a wall). If CLAUDE CODE doesn't implement `event.cancel()`, ask it to.
   - **Behavior execution order**: Entities should process in deterministic order (by creation order or ID sort). If CLAUDE CODE uses Map iteration, verify it's stable.
   - **Input event payload**: `action` should be a string, `payload` optional. The action vocabulary (e.g., `"move_north"`, `"use"`) is defined by userland, not kernel. The kernel doesn't know what actions mean.
   - **Serializer round-trip**: `serialize → load → serialize` must produce byte-identical output. This is the single most important test. The entire mechanical test harness depends on it.
4. Once all tests pass and HUMAN is satisfied, proceed to Step 2 by telling CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step 2."

---

## Prompt for CLAUDE CODE (copy the code fence below)

~~~
Read DEVELOPMENT_PLAN.md for full project context and the 7-step plan.

# Roles in this project

- HUMAN: The developer sitting at the keyboard. That's who you're talking to.
- CLAUDE CODE: You. The AI pair-programmer building the kernel right now.
- AGENT SWARM: Future autonomous Claude instances that will write game logic against this kernel. They don't exist yet. You are building the kernel FOR them.
- RL EVALUATOR: A future RL-based evaluation pipeline (random agent + trained PPO agent + invariant tests) that validates games quantitatively. Doesn't exist yet.

# Your role

You are pair-programming with HUMAN. You are building the KERNEL of a turn-based ASCII game engine in TypeScript. This kernel is a stable foundation that AGENT SWARM will later write game logic against. This is Step 1 of DEVELOPMENT_PLAN.md. You are NOT autonomous — ask HUMAN questions when you hit design forks.

# What to build

Initialize a TypeScript project (Node, strict mode, vitest for testing) and implement the kernel with the architecture below.

## Core: Grid World
- `World` class: a 2D grid of cells. Constructor takes width, height, and an optional `seed` (number, default: 42).
- Each cell holds an ordered stack of entities (multiple entities per cell).
- The world is the single source of truth for all game state.
- `world.random()` → returns a deterministic float in [0, 1) from a seeded PRNG (e.g., a simple mulberry32 or xoshiro implementation). The PRNG state is serialized and restored by `serializeState()`/`loadState()`.
- **Native `Math.random()` is strictly forbidden in engine and userland code.** All randomness must go through `world.random()` to preserve determinism.

## Core: Entity System
- Entities have: a unique string ID, deterministically generated via internal counter (e.g., `e1`, `e2`, ...). The counter is serialized and restored by `loadState()`. Do NOT use `crypto.randomUUID()` or any random source for IDs. Entities also have: a string `type`, a position `{x, y}`, a glyph (single character for ASCII render), a z-order (for rendering — highest z on top), and a `properties` map (`Record<string, any>`).
- `world.createEntity(type, x, y, glyph, zOrder?, properties?)` → Entity
- `world.destroyEntity(id)` → void
- `world.moveEntity(id, x, y)` → boolean (false if out of bounds)
- `world.getEntitiesAt(x, y)` → Entity[]
- `world.getEntitiesByType(type)` → Entity[]
- `world.getEntity(id)` → Entity | undefined
- Entities can set/get arbitrary properties: `entity.set(key, value)`, `entity.get(key)`

## Core: Event System
- Simple pub/sub. The kernel emits built-in events. Userland code (written later by AGENT SWARM) can emit custom events.
- Built-in events: `turn_start`, `input`, `collision` (when an entity moves into an occupied cell), `entity_created`, `entity_destroyed`, `turn_end`
- All events carry a payload object. `collision` carries `{ mover: Entity, occupants: Entity[], x: number, y: number }`.
- Events must support cancellation: operations that change state (e.g., `moveEntity`) must emit events BEFORE mutating state. Handlers receive an event object with a `cancel()` method. If a handler calls `cancel()`, the operation is aborted and returns `false`. Do NOT implement post-mutation rollback.
- `world.on(eventName, handler)` → unsubscribe function
- `world.emit(eventName, payload)` — userland can emit custom events too
- Handlers execute in registration order.

## Core: Turn Loop
- `world.handleInput(action: string, payload?: any)` — this is the entry point for a turn.
- Sequence: emit `turn_start` → emit `input` with `{ action, payload }` → run all registered behavior handlers for each entity (see below) → emit `turn_end`
- The turn is fully synchronous. No async. State is deterministic given the same inputs.
- Guardrail: The kernel must track event emission depth during a turn. If a single turn exceeds a configurable max cascade depth (default: 1000), throw an error. This prevents autonomous userland code from creating infinite event loops.

## Core: Entity Behaviors (the hook point for AGENT SWARM code)
- `world.registerBehavior(entityType: string, handler: (entity: Entity, world: World) => void)`
- During the turn loop, after input is processed, the kernel iterates all entities in deterministic order (by creation order) and calls their registered behavior handler if one exists.
- This is how AGENT SWARM will write game logic. A behavior for type "goblin" runs every turn for every goblin. A behavior for type "fire" might spread each turn.
- Behaviors can call any kernel primitive (move, destroy, create, emit, query).

## Core: ASCII Renderer
- `world.renderASCII()` → string — returns the grid as a string. Each cell shows the glyph of the highest z-order entity. Empty cells show `.` (configurable).
- Output is exactly `height` lines of `width` characters. No padding, no border, no decoration. Clean grid.

## Core: State Serializer
- `world.serializeState()` → a plain JSON object containing: grid dimensions, all entities with their full properties, and the current turn number.
- `world.loadState(state)` → restores from a serialized state. `loadState()` must re-instantiate actual `Entity` class instances with working methods (`set()`, `get()`, etc.). Raw parsed JSON objects are not sufficient. This enables deterministic testing: set up state, run input, assert new state.
- `world.toGymObservation(entityTypes: string[], propertyKeys: string[])` → a flat `number[]` suitable for RL agent consumption. The caller specifies which entity types and property keys matter (this is game-specific, not kernel-specific). The encoding:
  - Grid dimensions: `[width, height]`
  - Turn number: `[turn]`
  - For each cell `(x, y)` in row-major order: for each entity type in `entityTypes`, a `1` if an entity of that type is present, `0` otherwise. This produces a `width * height * entityTypes.length` block.
  - For each entity type in `entityTypes`: for the first entity of that type found (by creation order), output each property key's numeric value from `propertyKeys` (default to `0` if missing). This produces an `entityTypes.length * propertyKeys.length` block.
  - Total length is deterministic given the inputs: `3 + (width * height * entityTypes.length) + (entityTypes.length * propertyKeys.length)`.
- This is CRITICAL. The mechanical test harness and RL EVALUATOR (both built later) depend on this.

## What NOT to build
- No game logic. No enemies, items, doors, keys, health, combat. That is all userland — AGENT SWARM builds it later.
- No CLI or interactive mode yet.
- No config files.
- No AI, pathfinding, or FOV. Those are userland concerns.

## Project structure

src/
  kernel/
    World.ts          — the main class, grid, entity management
    Entity.ts         — entity class
    EventSystem.ts    — pub/sub
    Renderer.ts       — ASCII output
    Serializer.ts     — state save/load
    types.ts          — shared interfaces and types
  index.ts            — public API exports
tests/
  kernel/
    world.test.ts
    entity.test.ts
    events.test.ts
    renderer.test.ts
    serializer.test.ts
    behaviors.test.ts

## Tests to write immediately
Write thorough tests for every kernel primitive. These tests are the foundation that AGENT SWARM will rely on. At minimum:
- Create entity, verify it exists at correct position
- Move entity, verify old cell empty, new cell occupied
- Move out of bounds returns false, entity stays
- Destroy entity, verify removed from grid and queries
- Multiple entities on same cell, z-order rendering
- Event handlers fire in order, carry correct payloads
- Collision event fires when moving into occupied cell
- Collision cancellation prevents the move
- Behavior handlers run for correct entity types, in creation order
- Full turn sequence: input → behaviors → events all fire correctly
- Serialize → load → serialize produces identical JSON (round-trip)
- Load state, run input, assert deterministic output
- `toGymObservation` returns correct length, correct entity presence flags, correct property values
- `toGymObservation` is deterministic: same state + same args = same output
- `world.random()` returns same sequence from same seed
- Serialize → load → `world.random()` continues the sequence correctly
- Two worlds with same seed produce identical random sequences
- Event cascade guard throws when max depth exceeded

After building this, run all tests and make sure they pass. Then ask HUMAN what they think before moving on.

When HUMAN is satisfied, HUMAN will proceed to Step 2 as described in DEVELOPMENT_PLAN.md.
~~~
