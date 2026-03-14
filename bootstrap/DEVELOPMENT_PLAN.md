# INF FORTRESS — Development Plan

## The Experiment

This project tests whether **hardware-accelerated RL can serve as a real-time compiler for interactive software** — specifically, game mechanics.

Traditional game development validates mechanics through playtesting: humans play, report bugs, and argue about balance. Unit tests catch crashes but say nothing about whether a game is fun, solvable, or exploitable. We replace both with a **JAX-native RL oracle** that compiles an entire training run (environment stepping + policy + gradients) into a single XLA graph on a 96 GB GPU. The feedback loop collapses from weeks to seconds.

The product is a **game engine**. The test suite is **14 games of escalating complexity**. The oracle is **a PPO agent that learns to play them at a million frames per second**.

When you change a rule — make spike traps cost 5 wood instead of 10 — you hit run. Three minutes later, the oracle tells you whether the agent exploited the change, failed to learn it, or discovered an emergent strategy you didn't design. You are using RL as a behavioral compiler. The error messages aren't "segfault" — they're "PPO found an infinite item duplication exploit" or "reward gradient too sparse, agent never discovers the crafting chain."

**Hardware**: RTX PRO 6000 Blackwell (96 GB VRAM) + RTX 5070 Ti (16 GB). Local GPU, test immediately.

---

## Roles

- **HUMAN**: The developer. Runs Claude Code, reviews output, makes design decisions, tunes game balance using RL diagnostics.
- **CLAUDE CODE**: AI pair-programmer. Builds the engine, writes games, interprets RL diagnostics, fixes exploits.
- **RL ORACLE**: The behavioral compiler. Trains PPO against any game and produces diagnostic reports — not just pass/fail, but *how* the agent won, what it exploited, and where reward gradients are sparse.

There is no agent swarm. No Docker. No git-based task locking. HUMAN and CLAUDE CODE build all 14 games sequentially, using the RL oracle as a real-time feedback loop.

## How to use this document

This file and STEP1_PROMPT.md are the complete bootstrap for the project. After Step 1, HUMAN tells CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step N." Each step below has enough detail for CLAUDE CODE to execute.

---

## The Engine Contract

These are the structural rules that make the RL oracle possible. They are baked into the core. Every game built on the engine must obey them.

### JAX-Native Design

The engine is written in pure JAX. State is fixed-shape JAX tensors, not Python objects. There is no event system — the turn loop uses explicit phase-based dispatch. Entity storage uses struct-of-arrays with an alive mask, not dicts of objects.

**Dependencies**: `jax[cuda12]`, `equinox`, `optax`, `chex`. NOT `gymnasium`, `stable-baselines3`, `torch`.

### Composable ECS Architecture

Games are NOT monolithic `step()` functions. The engine provides vectorized **Systems** — reusable JAX functions that operate on `EnvState`:

```python
# Core systems (built in Step 1)
movement_system(state, actions)     # player + NPC movement
collision_system(state)             # detect/resolve entity overlaps

# Game systems (built incrementally, Steps 3–5)
combat_system(state)                # damage exchange, death
fluid_dynamics_system(state)        # CA magma spread
hunger_system(state)                # food depletion
stress_system(state)                # psychology, tantrum cascades
farming_system(state)               # growth ticks, harvest
crafting_system(state)              # workbench recipes
push_system(state)                  # sokoban-style block pushing
trap_system(state)                  # trap placement and triggering
```

Each system is a pure function `EnvState -> EnvState`. Games compose systems via `jax.lax.switch` dispatch. Game 14 (INF FORTRESS) doesn't rewrite anything — it imports all prior systems and runs them concurrently in a single `step()`.

This prevents XLA compilation times from exploding. Monolithic step functions with deep branching cause OOM during tracing. Small, tested, composable systems keep the trace graph manageable.

### Per-Game Configuration via `EnvConfig`

Every game module exports a `CONFIG` dataclass that declares the game's dimensions, entity budget, action count, and tuning parameters.

```python
CONFIG = EnvConfig(
    grid_w=8, grid_h=8,
    max_entities=8, max_stack=2,
    num_entity_types=3, num_tags=6, num_props=4,
    num_actions=6, max_turns=200,
    step_penalty=-0.01,
    game_state_size=4,
    prop_maxes=(1.0, 1.0, 1.0, 1.0),
)
```

| Field | Purpose |
|-------|---------|
| `max_entities` | Fixed entity slot budget — all arrays are this size |
| `max_stack` | Max entities per grid cell (for the grid tensor) |
| `num_entity_types` | Integer entity type count (0 = unused slot) |
| `num_tags` | Number of boolean tag channels |
| `num_props` | Properties per entity (float32 vector) |
| `num_actions` | Discrete action count |
| `game_state_size` | Extra float32 slots for milestones, counters, etc. |
| `prop_maxes` | Per-property normalization maxes for observation clamping |

PPO trains from scratch per game (except Game 14, which uses curriculum learning). The shape only needs to be constant within a single game across timesteps.

### Immutable Structural Rules

1. **Player singleton.** Exactly one entity with type equal to the player type ID must exist after `reset()`. `player_idx` in `EnvState` points to its slot.

2. **Termination via status.** Games set `state.status` to a nonzero value (1 = won, -1 = lost) to signal termination. There is no other way to end a game.

3. **Config declares truth.** Entity types, tags, and actions are bounded by the game's `EnvConfig`. Creating an entity with type >= `num_entity_types` is undefined.

4. **Game module interface.** Every game module must export `CONFIG` (EnvConfig) and implement `reset(rng_key) -> (EnvState, obs)` and `step(state, action) -> (EnvState, obs, reward, done)`.

5. **Determinism.** Given the same `rng_key`, `reset()` must produce the same initial state. Given the same state and action, `step()` must produce the same outcome. All randomness goes through `jax.random`.

### State Representation

```python
@chex.dataclass
class EnvState:
    alive:        jnp.bool_[MAX_ENTITIES]
    entity_type:  jnp.int32[MAX_ENTITIES]
    x, y:         jnp.int32[MAX_ENTITIES]
    tags:         jnp.bool_[MAX_ENTITIES, NUM_TAGS]
    properties:   jnp.float32[MAX_ENTITIES, NUM_PROPS]
    grid:         jnp.int32[H, W, MAX_STACK]
    grid_count:   jnp.int32[H, W]
    turn_number:  jnp.int32
    status:       jnp.int32          # 0=playing, 1=won, -1=lost
    reward_acc:   jnp.float32
    player_idx:   jnp.int32
    game_state:   jnp.float32[GAME_STATE_SIZE]
    rng_key:      PRNGKey
```

All arrays have shapes known at JIT trace time. No Python-side conditionals that change array shapes.

### Universal Reward Signal

Reward is computed **additively** each step:
- Start with `config.step_penalty` (default `-0.01`)
- Add `state.reward_acc` (intermediate rewards accumulated during the step)
- If `status == 1` (won), add `+10.0`
- If `status == -1` (lost), add `-10.0`

### Observation

Observation is a dict with two keys:
- `'grid'`: `jnp.float32[C, H, W]` — channel-first spatial tag map. Channel per tag. `1.0` if any alive entity in that cell has that tag.
- `'scalars'`: `jnp.float32[N]` — normalized grid dimensions, turn number, and player properties.

### Training Architecture

PureJaxRL pattern: `jax.jit` + `jax.vmap` + `jax.lax.scan`. The entire training loop (env stepping, observation, policy forward pass, gradient computation) runs as a single compiled XLA program. No Python-side loops, no host-device transfers during training.

- **Policy**: Equinox `ActorCritic` — small CNN for grid + MLP for scalars, combined into shared value/policy heads.
- **Optimizer**: `optax.adam` with linear LR schedule.
- **Parallelism**: `jax.vmap` over 4096 environments.

### PRNG

All randomness uses `jax.random` with key splitting. `random.random()` and `numpy.random` are strictly forbidden. The RNG key is part of `EnvState` and split on every use.

---

## The RL-TDD Loop

This is the core workflow for every game, starting at Step 3:

```
1. HUMAN describes the mechanic ("spike traps cost 10 wood")
2. CLAUDE CODE writes the JAX system function
3. Deterministic trace: hardcoded action sequence proves the game is solvable
4. RL Oracle runs: random agent fuzz + PPO training + diagnostics
5. HUMAN reads the diagnostic report:
   - "PPO wins 99% of games by turn 15 — trap is too cheap"
   - "PPO never discovers crafting — reward gradient too sparse"
   - "In top 5% of wins, action sequence is [pick, drop, pick, drop, win] — item dupe bug"
6. HUMAN and CLAUDE CODE adjust the mechanic, go to step 2
```

This loop replaces playtesting. The GPU is the playtester.

---

## Step Plan

| Step | What | Who |
|------|------|-----|
| 0 | Game specs (14 specs, HUMAN-authored) | HUMAN |
| 1 | JAX core: state, grid_ops, obs, movement | HUMAN + CLAUDE CODE |
| 2 | Core tests + training loop + evaluator with diagnostics | HUMAN + CLAUDE CODE |
| 3 | Games 01–03 (reference) + ECS systems: movement, collision | HUMAN + CLAUDE CODE |
| 4 | Games 04–10 (primitives) + ECS systems: combat, hunger, push, farming, crafting | HUMAN + CLAUDE CODE |
| 5 | Games 11–14 (macro-specs) + ECS systems: fluid dynamics, psychology, siege | HUMAN + CLAUDE CODE |

No Step 3 (agent prompt), no Step 4 (Docker infrastructure), no Step 7 (swarm). Zero tokens and zero compute hours fighting containers and git conflicts.

---

## Step 0: HUMAN writes game specs

*Already done.* 14 specs in `game-specs/`, escalating from trivial (Empty Exit) to extreme (INF FORTRESS).

| # | Game | New ECS System Introduced |
|---|------|--------------------------|
| 01 | Empty Exit | (pipeline validation only) |
| 02 | Dodge | `behavior_system` (NPC dispatch) |
| 03 | Lock & Key | `interact_system` |
| 04 | Dungeon Crawl | `combat_system` |
| 05 | Pac-Man Collect | (collection win condition — no new system) |
| 06 | Ice Sliding | `slide_system` (momentum physics) |
| 07 | Hunger Clock | `hunger_system` |
| 08 | Block Push | `push_system` |
| 09 | Inventory & Crafting | `crafting_system` |
| 10 | Farming & Growth | `farming_system` |
| 11 | Digging Deep | `fluid_dynamics_system` |
| 12 | Tantrum Spiral | `stress_system` |
| 13 | Siege Architecture | `trap_system`, `spawn_system` |
| 14 | INF FORTRESS | All systems composed |

**Output**: `game-specs/01-empty-exit.md` through `game-specs/14-inf-fortress.md`

---

## Step 1: HUMAN + CLAUDE CODE build the JAX core

**HUMAN tells CLAUDE CODE**: Paste the contents of STEP1_PROMPT.md.

The core provides pure-function primitives for entity management, grid operations, observation building, and movement — all JIT-compatible and vmap-safe.

What gets built:
- `jaxswarm/core/state.py` — `EnvState`, `EnvConfig`, `init_state`
- `jaxswarm/core/grid_ops.py` — `create_entity`, `destroy_entity`, `move_entity`, queries, `rebuild_grid`
- `jaxswarm/core/obs.py` — `get_obs` (grid + scalars observation builder)
- `jaxswarm/core/movement.py` — `move_toward` helper
- `tests/core/` — determinism, CRUD, movement, observation shapes, vmap smoke tests

Full API spec is in STEP1_PROMPT.md.

**Done when**: All core tests pass. A trivial game (place player, place exit, win on collision) can be `jax.jit`'d and `jax.vmap`'d over 64 environments without error.

---

## Step 2: HUMAN + CLAUDE CODE build the test harness, training loop, and evaluator

This step combines what was previously three separate steps. Everything needed to run the RL-TDD loop gets built here.

### 2A: Core test harness

- **Determinism**: Same `rng_key` → identical trajectory. Different key → different trajectory.
- **vmap**: `jax.vmap(reset)(keys)` with 64 keys, no cross-contamination.
- **State round-trip**: Pytree flatten → reconstruct → element-wise identical.
- **Grid consistency**: `grid` tensor matches `x`, `y`, `alive` after every CRUD operation.
- **Entity lifecycle**: Create to `max_entities`, destroy, reuse slot.
- **Observation shapes**: Match `EnvConfig`. Normalized, clamped.
- **Random agent fuzz**: 1000 episodes, no NaN, all terminate within `max_turns`.

### 2B: PureJaxRL training loop

- `jaxswarm/train.py` — PPO with `jax.lax.scan` + `jax.vmap`.
- Equinox `ActorCritic`: small CNN (kernel_size=3, stride=1, padding=1) for grid + MLP for scalars.
- `optax.adam`, configurable LR, 4096 parallel envs.
- Checkpoint saving via `eqx.tree_serialise_leaves`.

### 2C: Evaluator with diagnostics

`evaluate_game.py` — the behavioral compiler. Not just pass/fail — it outputs **LLM-readable forensics** that CLAUDE CODE can act on.

**Layer 1 — Deterministic Trace:**
Before running PPO, execute a hardcoded action sequence (provided per game spec) via `jax.lax.scan`. Assert `status == 1`. This mathematically proves the game code is solvable. If PPO later fails to win, you know it's an RL exploration / reward shaping issue, not a game logic bug.

**Layer 2 — Random Agent:**
JIT-compiled random action loop over 1000 episodes (vmap'd). Records: NaN count, termination rate, win rate, average episode length.

**Layer 3 — PPO Training:**
Trains PPO for 100k+ steps. Records: win rate at 10k, 50k, 100k. Learning delta.

**Layer 4 — Trajectory Forensics:**
This is what makes the evaluator a *compiler*, not just a test runner:

- **Trajectory summaries**: For the top 5% and bottom 5% of episodes (by reward), log the action sequence and key state transitions. Output is LLM-readable, e.g.: `"Top trajectory: [move_e x12, interact, move_n x5, interact, move_e x3 → WON at turn 33]"`. This instantly reveals exploits (e.g., item duplication) or degenerate strategies (e.g., hiding in a corner).
- **Behavioral statistics**: What % of winning episodes used `interact`? What was the average inventory at win time? Did the agent ever build a barricade near magma (panic wall)?
- **Visual replays**: For HUMAN, render the highest-reward trajectory as an ASCII frame sequence (or .mp4). HUMAN needs to *see* the exploit to understand it.

**Layer 5 — Invariant Tests:**
Game-specific structural checks on initial state (exit reachable, entity budget not exceeded, etc.).

**Output format**: JSON + trajectory log + optional replay file.

```json
{
  "game": "08_block_push",
  "deterministic_trace": { "action_sequence": [2,2,0,0,4,...], "status": 1, "turns": 28, "pass": true },
  "random_agent": { "nan_count": 0, "termination_rate": 0.98, "win_rate": 0.03, "avg_length": 142, "pass": true },
  "ppo": {
    "win_rate_10k": 0.01, "win_rate_50k": 0.08, "win_rate_100k": 0.15,
    "learning_delta": 0.14, "pass": true
  },
  "forensics": {
    "top_trajectory": "move_e, move_e, move_n, push_block_n, move_w, push_block_w → WON turn 22",
    "interact_usage_in_wins": 0.0,
    "avg_turns_to_win": 31.4
  },
  "invariants": { "total": 9, "passed": 9, "failed": [], "pass": true },
  "overall_pass": true
}
```

**Done when**: The evaluator runs end-to-end on a trivial game. Training loop produces non-NaN gradients. Trajectory forensics output is parseable.

---

## Step 3: HUMAN + CLAUDE CODE build reference games (01–03)

The RL-TDD loop begins here. For each game:

1. CLAUDE CODE writes the game module (`jaxswarm/games/game_NN.py`) composing core primitives.
2. Deterministic trace proves solvability.
3. RL oracle runs. HUMAN and CLAUDE CODE read the diagnostics.
4. Tune mechanics until RL results match expected ranges from the game spec.

### Games 01–03

| Game | New ECS System | Expected PPO 100k Win Rate |
|------|---------------|---------------------------|
| 01 Empty Exit | — | >90% |
| 02 Dodge | `behavior_system` (NPC dispatch via `jax.lax.switch`) | >50% |
| 03 Lock & Key | `interact_system` (adjacent entity interaction) | >20% |

These systems become reusable building blocks for all subsequent games.

**Done when**: All three games pass the evaluator. Win rates within expected ranges. Systems are factored into `jaxswarm/systems/`.

**Output**: `jaxswarm/games/game_{01,02,03}.py`, `jaxswarm/systems/{movement,behavior,interact}.py`

---

## Step 4: HUMAN + CLAUDE CODE build primitive games (04–10)

Each game introduces one new ECS system. The RL-TDD loop catches balance issues, exploits, and reward shaping problems in real time.

| Game | New System | Key RL-TDD Question |
|------|-----------|---------------------|
| 04 Dungeon Crawl | `combat_system` | Is combat balanced? Does PPO learn to use potions? |
| 05 Pac-Man Collect | — | Does PPO learn to avoid ghosts while collecting? |
| 06 Ice Sliding | `slide_system` | Does PPO learn indirect movement (choose direction, can't choose distance)? |
| 07 Hunger Clock | `hunger_system` | Does PPO learn to detour for food? |
| 08 Block Push | `push_system` | Does PPO solve spatial puzzles? |
| 09 Inventory & Crafting | `crafting_system` | Can PPO discover a 6-step dependency chain? |
| 10 Farming & Growth | `farming_system` | Can PPO learn delayed payoffs (plant now, harvest in 15 turns)? |

### Systemic Regression Testing

Since all games share the same core and ECS systems, **every change to a system is tested against all prior games**. When adding `combat_system` for Game 04, CLAUDE CODE re-runs the evaluator on Games 01–03. If changing `max_stack` drops Game 08's win rate from 95% to 12%, you catch it immediately.

The Blackwell GPU can evaluate all 10 primitive games in parallel (`jax.vmap` over game configs, or sequential eval in ~10 minutes total). This is the CI/CD pipeline for gameplay mechanics.

**Done when**: Games 04–10 all pass the evaluator. No regressions on games 01–03.

**Output**: 7 game modules, 5+ new ECS systems in `jaxswarm/systems/`.

---

## Step 5: HUMAN + CLAUDE CODE build macro-specs (11–14)

The macro-specs test what happens when primitives overlap. This is where "Dwarf Fortress" behavior emerges.

| Game | Systems Composed | Emergent Behavior to Validate |
|------|-----------------|------------------------------|
| 11 Digging Deep | mining + `fluid_dynamics_system` | Agent builds barricades near approaching magma ("panic wall") |
| 12 Tantrum Spiral | push + `stress_system` | Agent discovers "work-life balance" — pacing work to avoid tantrums |
| 13 Siege Architecture | mining + `trap_system` + `spawn_system` | Agent mines chokepoints and funnels goblins through kill zones |
| 14 INF FORTRESS | ALL systems concurrent | Agent demonstrates "weaponized fluids" — mining magma onto goblins |

### Solving the Exploration Chasm

Games 11–14 have dependency chains too long for random exploration to discover. Two mitigations:

1. **Deterministic traces**: Hardcoded action sequences prove solvability before PPO ever runs. If the trace fails, it's a code bug. If PPO fails but the trace passes, it's a reward shaping problem.

2. **Curriculum learning** (Game 14 only): Pre-train on Games 01–03, fine-tune on 07+10 (hunger+farming), fine-tune on 11+13 (mining+siege), then tackle 14.

### The "Dwarf Fortress" Metric

Game 14's CI pipeline passes ONLY if the agent demonstrates at least one instance of "weaponized fluids" — intentionally mining a path to release magma onto approaching goblins. This validates that overlapping systems create emergent strategic options that the agent discovers without being explicitly programmed to.

**Done when**: Games 11–14 pass the evaluator. Emergent behaviors observed in trajectory forensics.

**Output**: 4 game modules, remaining ECS systems, curriculum training pipeline.

---

## Project Structure

```
jaxswarm/
  core/
    __init__.py
    state.py          — EnvState, EnvConfig, init_state
    grid_ops.py       — entity CRUD, movement, grid queries, rebuild_grid
    obs.py            — observation builder
    movement.py       — move_toward helper
  systems/
    __init__.py
    movement.py       — player + NPC movement dispatch
    behavior.py       — NPC behavior dispatch (jax.lax.switch)
    interact.py       — adjacent entity interaction
    combat.py         — damage exchange, death
    hunger.py         — food depletion
    push.py           — sokoban block pushing
    farming.py        — growth ticks, harvest
    crafting.py       — workbench recipes
    slide.py          — ice momentum physics
    fluid_dynamics.py — CA magma spread
    stress.py         — psychology, tantrum cascades
    trap.py           — trap placement and triggering
    spawn.py          — timed entity spawning (goblin waves)
  games/
    game_01_empty_exit.py
    game_02_dodge.py
    ...
    game_14_inf_fortress.py
  train.py            — PureJaxRL PPO training loop
  network.py          — Equinox ActorCritic
  __init__.py
evaluate_game.py      — RL oracle with diagnostics
tests/
  core/               — core primitive tests
  systems/            — per-system unit tests
  games/              — per-game invariant tests
pyproject.toml
bootstrap/            — this document + STEP1_PROMPT.md + game-specs/
```
