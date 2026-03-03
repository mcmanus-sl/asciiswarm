# First Claude Code Prompt — Kernel Bootstrap

## Roles

- **HUMAN**: You, the developer. You run Claude Code, review output, make design decisions.
- **CLAUDE CODE**: The AI pair-programmer. It receives the prompt below and builds the kernel.
- **AGENT SWARM**: Future autonomous Claude instances that will write game logic against the kernel. They don't exist yet. The kernel is being built FOR them.
- **PLAYTEST AGENT**: A future specialized Claude instance that plays the game and evaluates it. Doesn't exist yet.

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
- PLAYTEST AGENT: A future specialized Claude instance that plays and evaluates games. Doesn't exist yet.

# Your role

You are pair-programming with HUMAN. You are building the KERNEL of a turn-based ASCII game engine in TypeScript. This kernel is a stable foundation that AGENT SWARM will later write game logic against. This is Step 1 of DEVELOPMENT_PLAN.md. You are NOT autonomous — ask HUMAN questions when you hit design forks.

# What to build

Initialize a TypeScript project (Node, strict mode, vitest for testing) and implement the kernel with the architecture below.

## Core: Grid World
- `World` class: a 2D grid of cells. Constructor takes width, height.
- Each cell holds an ordered stack of entities (multiple entities per cell).
- The world is the single source of truth for all game state.

## Core: Entity System
- Entities have: a unique string ID (auto-generated), a string `type`, a position `{x, y}`, a glyph (single character for ASCII render), a z-order (for rendering — highest z on top), and a `properties` map (`Record<string, any>`).
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
- Events must support cancellation: handlers receive an event object with a `cancel()` method. If cancelled, the operation that triggered the event (e.g., a move) is rolled back.
- `world.on(eventName, handler)` → unsubscribe function
- `world.emit(eventName, payload)` — userland can emit custom events too
- Handlers execute in registration order.

## Core: Turn Loop
- `world.handleInput(action: string, payload?: any)` — this is the entry point for a turn.
- Sequence: emit `turn_start` → emit `input` with `{ action, payload }` → run all registered behavior handlers for each entity (see below) → emit `turn_end`
- The turn is fully synchronous. No async. State is deterministic given the same inputs.

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
- `world.loadState(state)` → restores from a serialized state. This enables deterministic testing: set up state, run input, assert new state.
- This is CRITICAL. The mechanical test harness and PLAYTEST AGENT (both built later) depend on this.

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

After building this, run all tests and make sure they pass. Then ask HUMAN what they think before moving on.

When HUMAN is satisfied, HUMAN will proceed to Step 2 as described in DEVELOPMENT_PLAN.md.
~~~
