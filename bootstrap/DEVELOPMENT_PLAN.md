# ASCII Game Engine — Development Plan

## The Experiment

This project tests whether the Carlini agent-swarm methodology (parallel autonomous Claude instances, task locking via git, mechanical test convergence) works for creative software — specifically, game development.

The product is a **game engine**. The test suite is **multiple games**. The oracle is **an RL agent that learns to play them**.

In Carlini's C compiler experiment, GCC defined "correct." The compiler's job was to produce binaries that behaved identically to GCC's output. Here, the RL agent defines "functional." A game's job is to be learnable by PPO given a fixed engine contract. If the RL agent's win rate climbs during training, the game works. If it doesn't, the game is broken.

---

## Roles

- **HUMAN**: The developer. Runs Claude Code, reviews output, makes design decisions, builds infrastructure.
- **CLAUDE CODE**: AI pair-programmer. Works under HUMAN supervision during Steps 1–6.
- **AGENT SWARM**: Future autonomous Claude instances. They write games (userland) against the kernel. They do not exist until Step 7.
- **RL EVALUATOR**: The oracle. A pipeline that trains a PPO agent against any game built on the engine and produces quantitative pass/fail verdicts.

## Supervision Model

| Step | Who does the work | Autonomous? |
|------|-------------------|-------------|
| 0. Game specs | HUMAN | No — HUMAN writes these |
| 1. Kernel | HUMAN + CLAUDE CODE | No — pair programming |
| 2. Mechanical tests | HUMAN + CLAUDE CODE | No — pair programming |
| 3. Agent prompt | HUMAN | No — HUMAN writes this |
| 4. Infrastructure | HUMAN | No — HUMAN builds this |
| 5. Reference games | HUMAN + CLAUDE CODE | No — pair programming |
| 6. RL evaluation pipeline | HUMAN + CLAUDE CODE | No — pair programming |
| 7. Swarm on userland | AGENT SWARM | Yes — autonomous |

Steps 1 and 2 are where HUMAN spends disproportionate time. They are not autonomous. Everything after Step 4 benefits from autonomy. Everything before it needs HUMAN's hands on it.

## How to use this document

This file and STEP1_PROMPT.md are the complete bootstrap for the project. After Step 1, HUMAN tells CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step N." Each step below has enough detail for CLAUDE CODE to execute.

---

## The Engine Contract

These are the structural rules that make the RL oracle possible. They are baked into the kernel. Every game built on the engine must obey them. The swarm cannot change them.

### Per-Game Configuration via `GAME_CONFIG`

Every game module exports a `GAME_CONFIG` dict that declares the game's action space, observation tags, player properties, grid dimensions, and tuning parameters. The kernel builds Gymnasium spaces dynamically from this config. There is no global fixed tensor shape — each game defines its own.

```python
# Example: a game module declares its config
GAME_CONFIG = {
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait'],
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'grid': (8, 8),
    'max_turns': 200,
    'step_penalty': -0.01,
    'player_properties': [
        {'key': 'health', 'max': 10},
    ],
}

def setup(env):
    # Register entities, behaviors, event handlers
    ...
```

The kernel provides **standard defaults** for any keys omitted from `GAME_CONFIG`:

| Key | Default | Description |
|-----|---------|-------------|
| `actions` | `['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait']` | Valid action strings |
| `tags` | `['player', 'solid', 'hazard', 'pickup', 'exit', 'npc']` | Valid entity tags |
| `grid` | `(16, 16)` | `(width, height)` |
| `max_turns` | `1000` | Turn limit (for observation normalization) |
| `step_penalty` | `-0.01` | Per-step reward penalty |
| `player_properties` | `[]` | Properties included in observation vector |

Games can use all defaults, override some, or replace everything. A simple game might only set `grid` and `max_turns`. A complex game might declare custom actions (`'shoot_n'`, `'craft'`, `'stealth'`) and custom tags (`'ammo'`, `'craftable'`, `'hidden'`).

The kernel:
- Builds `action_space = Discrete(len(config['actions']))` and `ACTION_MAP` from the config.
- Builds `observation_space` as a `Dict` with a channel-first `'grid'` tensor (for CNN spatial reasoning) and a `'scalars'` vector (for MLP). SB3's `MultiInputPolicy` handles this natively.
- Validates entity tags against the game's declared tag list (not a global constant).
- Validates actions against the game's declared action list.

PPO trains from scratch per game — there is no shared agent, no transfer learning, no reason for identical tensor shapes across games. The shape only needs to be constant within a single game across timesteps.

### Immutable Structural Rules

These five rules cannot be overridden by any game:

1. **Player singleton.** Exactly one entity tagged `player` must exist after `setup()` completes. The kernel validates this on `reset()`.

2. **Termination via `end_game()`.** Games must call `env.end_game('won')` or `env.end_game('lost')` to signal termination. The Gym `step()` method reads this. There is no other way to end a game.

3. **Config declares truth.** The kernel validates entities and actions against the game's own `GAME_CONFIG` declaration. If a game creates an entity with a tag not in its declared tag list, or receives an action not in its declared action list, the kernel raises an error. The game's config is its contract with the kernel.

4. **Game module interface.** Every game module must export `GAME_CONFIG` (dict) and `setup(env)` (callable). Missing either raises an error on load.

5. **Determinism.** Given the same seed, `setup()` must produce the same initial state. Given the same state and action sequence, the game must produce the same outcome. All randomness goes through `env.random()`.

### Universal Reward Signal

Reward is computed **additively** each step — all components are summed, never exclusive:
- Start with `config['step_penalty']` (default `-0.01`)
- Add any intermediate `reward` events emitted during the turn (e.g., +0.1 for picking up an item)
- If `env.status == 'won'`, add `+10.0`
- If `env.status == 'lost'`, add `-10.0`

This ensures intermediate rewards emitted on the same turn as a terminal event are never silently discarded.

---

### Why Python, not TypeScript

The RL pipeline is Python (Stable-Baselines3, Gymnasium). If the kernel is TypeScript, every game needs a Python↔TypeScript bridge (subprocess + JSON, or WASM). This adds latency, debugging pain, and serialization bugs at the boundary — all for zero benefit. The kernel should be Python so that the engine IS the Gymnasium environment. No wrapper, no FFI, no bridge. `PPO("MultiInputPolicy", YourGameEnv()).learn(100_000)` just works.

### Why RL instead of Claude-as-judge

The original plan used a Claude playtest agent as the oracle — Claude plays the game, then Claude judges whether the game is good. This has the "oracle problem": the judge is as fallible as the builder. An RL agent solves this by making the evaluation objective and quantitative. If PPO can't learn to win after N episodes, the game is broken. If it wins in 5 episodes, there's no depth. If win rate climbs from 0% to 60% over training, you have real mechanical substance. No subjective judgment needed.

### The engine IS the Gymnasium environment

The kernel subclasses `gymnasium.Env` directly. There is no separate wrapper. The core class looks like:

```python
class GridGameEnv(gymnasium.Env):
    # The engine IS the env. No wrapper needed.
    # reset(), step(), observation_space, action_space are native.
```

Games are Python modules — a `GAME_CONFIG` dict plus a `setup(env)` function that registers entity types, behaviors, spawning rules, and termination conditions. Each game is one file. The swarm writes these files.

---

## Step 0: HUMAN writes game specs

**HUMAN does this alone** before any code is written.

HUMAN writes 5–8 one-page game specifications at escalating complexity. These are the "test programs" — like Carlini's progression from test suites to SQLite to Redis to Linux. Each spec defines:

- A title and one-paragraph description.
- Grid dimensions.
- `GAME_CONFIG` block (actions, tags, player_properties, grid, max_turns, step_penalty).
- Entity types and their tags.
- Player properties (the ones that appear in the observation vector).
- Behavior descriptions for each entity type (what it does each turn).
- The `interact` mapping (what happens when the player uses `interact` near each entity type).
- Win condition (what triggers `env.end_game('won')`).
- Lose condition (what triggers `env.end_game('lost')`).
- RL evaluation criteria: expected random-agent win rate range, minimum PPO learning delta (win rate at 100k steps minus win rate at 10k steps).

### Suggested progression

| # | Game | Complexity | New Engine Capability | Built By |
|---|------|-----------|----------------------|----------|
| 1 | **Empty Exit** | Trivial | Pipeline validation | Reference |
| 2 | **Dodge** | Low | Enemy behaviors, hazard avoidance | Reference |
| 3 | **Lock & Key** | Medium | `interact` action, entity state mutation, multi-step dependency | Reference |
| 4 | **Dungeon Crawl** | Medium-High | Combat, health, procgen rooms | Swarm |
| 5 | **Pac-Man Collect** | Medium | Collection-based win condition (all pickups), deterministic enemy AI | Swarm |
| 6 | **Ice Sliding** | Medium | `before_move` chaining / momentum physics (slide until hitting solid) | Swarm |
| 7 | **Hunger Clock** | Medium-High | Ticking resource depletion (`food` decreases each turn) | Swarm |
| 8 | **Block Push** | Medium-High | Push mechanics via collision chain (player→block→wall) | Swarm |

Games 1–3 are built by HUMAN + CLAUDE CODE as reference implementations in Step 5. Games 4–8 are swarm tasks in Step 7.

**Done when**: Game spec files exist in `game-specs/` and each spec is complete enough that a developer could implement it without asking questions. Each spec includes a `GAME_CONFIG` block.

**Output**: `game-specs/01-empty-exit.md` through `game-specs/08-block-push.md`

---

## Step 1: HUMAN + CLAUDE CODE build the kernel

**HUMAN tells CLAUDE CODE**: Paste the contents of STEP1_PROMPT.md.

HUMAN supervises CLAUDE CODE to build the kernel. This is the stable foundation that AGENT SWARM will later write games against. The kernel IS a Gymnasium environment — games built on it are automatically RL-evaluable.

The kernel includes: the entity system (with semantic tags validated against `GAME_CONFIG`), the grid, the turn loop, the behavior registration system, the event system, the ASCII renderer, the state serializer, the Gym interface (`reset`, `step`, `observation_space`, `action_space` — all built dynamically from `GAME_CONFIG`), the seeded PRNG, and the game status primitive.

Full API spec is in STEP1_PROMPT.md.

HUMAN tests it, reviews it, owns it. AGENT SWARM does not touch this.

**Done when**: All kernel tests pass. HUMAN has reviewed the code and is satisfied with the API surface. A trivial game (place player, place exit, `end_game('won')` on collision) can be loaded and trained against with PPO for 10 episodes without crashing.

**Output**: A tested, stable Python kernel with full test coverage.

---

## Step 2: HUMAN + CLAUDE CODE build the mechanical test harness

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Step 1 — the kernel is built and tested. Proceed to Step 2: build the expanded mechanical test harness."

Step 1 includes basic kernel unit tests. Step 2 expands those into a comprehensive regression suite that AGENT SWARM will run before every commit.

### What CLAUDE CODE builds

**Kernel stress tests** — edge cases the Step 1 tests didn't cover:
- Create and destroy hundreds of entities rapidly. No stale references.
- Move entities to every boundary cell. Off-grid in every direction.
- Fill a cell with 20+ entities. Tag-based observation still correct. Z-order rendering still correct.
- Register 50+ event handlers. All fire in order. Unsubscribe works mid-iteration.
- Collision cancellation chains: handler A cancels, handler B would have fired — does it?
- Serialize a world with 1000+ entities. Round-trip still byte-identical.
- Load state, run 100 turns of random input, serialize. Run the same 100 inputs from the same loaded state. Output must be identical (determinism proof).
- Verify `env.random()` produces same sequence after serialize → load round-trip.
- Event cascade guard throws at configured max depth.
- `end_game()` sets status correctly. Further `handle_input()` calls are no-ops after game ends.
- Gym `step()` returns `terminated=True` when game has ended. `reset()` restores to initial state.

**Random agent fuzz tests** — the first layer of the oracle:
- Load any game module, run 1000 episodes of random valid actions.
- Assert: no exceptions thrown during any episode.
- Assert: every episode terminates within `max_turns` (from `GAME_CONFIG`, default 1000).
- Assert: game status is always one of `'playing'`, `'won'`, `'lost'` — never corrupted.
- Assert: observation shape is constant across all steps of all episodes.
- Record: win rate, average episode length, distribution of terminal states.

**Invariant test framework** — extensible by agents:
- A base class / decorator that agents can use to register new invariant tests for their games.
- Built-in invariants:
  - Exactly one entity tagged `player` exists at game start.
  - At least one entity tagged `exit` exists at game start.
  - Every entity type used in the game has a registered behavior (or is inert by design, marked with an `inert` flag).
  - The exit is reachable from the player's starting position (BFS over non-`solid` tiles).
  - No entity has an empty tag list.
- Game-specific invariants are added by the game module itself (e.g., "every room has at least one exit," "total enemies equals wave_count * enemies_per_wave").

**Userland simulation tests** — tests that simulate what AGENT SWARM will actually do:
- Register a behavior that moves an entity toward a target every turn. Run 10 turns. Verify positions.
- Register a collision handler that destroys the mover (simulating a trap). Verify the entity is gone.
- Register a collision handler that cancels the move (simulating a wall). Verify the entity stays.
- Create a chain reaction: entity A's behavior creates entity B, B's behavior emits a custom event, that event's handler destroys A. Verify all resolves in one turn.
- Register an input handler for `interact`. Fire it. Verify the handler ran.
- Call `end_game('won')`. Verify Gym `step()` returns `terminated=True` and reward `+1.0`.

**Test runner output format** — designed for AGENT SWARM consumption:
- On success: one line per test file, e.g. `PASS kernel/world_test.py (23 tests)`
- On failure: `FAIL kernel/world_test.py` followed by `ERROR: [test name] — [one-line reason]`
- Summary line: `TOTAL: 247/250 passed`
- All verbose output goes to a log file, not stdout. AGENT SWARM's context window must not be polluted.
- Include a `--fast` flag that runs a deterministic 10% sample. Seed is passed via `--seed N` CLI argument (default: hash of container hostname). AGENT SWARM uses `--fast` during development, full suite before pushing.

**Done when**: The test suite has 200+ assertions, all pass, and the runner output is clean and parseable. The random agent fuzz test can load a trivial game and run 1000 episodes without error.

**Output**: Expanded test suite in `tests/`. A test runner script with `--fast` and `--seed` flags. The invariant test framework.

---

## Step 3: HUMAN writes AGENT_PROMPT.md

**HUMAN does this alone** — no CLAUDE CODE needed.

HUMAN writes the prompt document that the agent loop feeds to every fresh AGENT SWARM instance. It tells each agent:

- What the project is and what the kernel API looks like (or where to find the docs).
- **The Engine Contract** (copied from this document): per-game `GAME_CONFIG` declarations, immutable structural rules, universal reward signal. Violating these is equivalent to writing invalid C — the oracle will reject it.
- How to read a game spec: find your assigned game in `game-specs/`, implement it as a single Python module in `games/`.
- How to pick a task: check `current_tasks/` for what's claimed, check which games in `game-specs/` don't have a passing implementation yet, pick one.
- How to claim a task: write a file to `current_tasks/` (e.g., `current_tasks/implement_sokoban.txt`) and push. If git rejects the push: run `git pull --rebase`. If the only conflict is in `current_tasks/` — the task was claimed by another agent; abort the rebase (`git rebase --abort`) and pick a different task. If the rebase is clean or conflicts are only in userland code — resolve conflicts, re-run `--fast` tests, and push again.
- How to work: implement the game module, run mechanical tests with `--fast`, run the random agent fuzz test on your game, run invariant tests, fix issues, run the full suite before pushing.
- How to finish: push to upstream, remove the task lock file, update `PROGRESS.md`.
- How to handle merge conflicts: pull, resolve, verify tests still pass, push.
- How to leave notes: maintain `PROGRESS.md` — what was done, what was tried, what failed, known issues.

**Done when**: `AGENT_PROMPT.md` is in the repo root and contains all of the above.

**Output**: `AGENT_PROMPT.md`

---

## Step 4: HUMAN sets up the infrastructure

**HUMAN does this alone** — no CLAUDE CODE needed for infrastructure, though HUMAN may use CLAUDE CODE to help write scripts.

### The agent loop (per container)
```bash
#!/bin/bash
while true; do
    COMMIT=$(git rev-parse --short=6 HEAD)
    LOGFILE="agent_logs/agent_${COMMIT}_$(date +%s).log"

    claude --dangerously-skip-permissions \
           -p "$(cat AGENT_PROMPT.md)" \
           --model claude-opus-4-6 &> "$LOGFILE"
done
```

### Container setup
- One Docker container per agent.
- A bare git repo mounted to `/upstream` in each container.
- Each agent clones `/upstream` to `/workspace` on startup.
- When done with a task, agent pushes from `/workspace` to `/upstream`.
- Python environment with: `gymnasium`, `stable-baselines3`, `pytest`, `numpy` pre-installed.

### Task locking
- Agent claims a task by writing a file to `current_tasks/` (e.g., `current_tasks/implement_combat_roguelike.txt`) containing a short description of the approach.
- Agent pushes this file to upstream. If git rejects the push: agent runs `git pull --rebase`. If the only conflict is in `current_tasks/` — the task was claimed by another agent; abort the rebase and pick a different task. If the rebase is clean or conflicts are only in userland code — resolve conflicts, re-run `--fast` tests, and push again.
- When done, agent removes the lock file and pushes.

### How many agents
- Start with 4. Scale to 8–16 once HUMAN is confident the test harness catches regressions.
- More agents only help when tasks are independent. If agents duplicate work, reduce count or add more game specs.

**Done when**: HUMAN can spin up N containers and each one runs the loop, claims tasks, pushes code.

**Output**: Dockerfiles, shell scripts, bare git repo setup.

---

## Step 5: HUMAN + CLAUDE CODE write reference games

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–4. Proceed to Step 5: build the reference games."

HUMAN and CLAUDE CODE hand-write 3 games from the game specs (games 1–3: Empty Exit, Dodge, Lock & Key). These serve three purposes:

1. **Validate the kernel API.** If these are painful to write, fix the kernel before the swarm hits the same friction at scale.
2. **Calibrate the RL pipeline.** These are known-good baselines with expected difficulty curves.
3. **Prove the Gym interface works end-to-end.** Each reference game must train PPO successfully.

### What CLAUDE CODE builds

For each reference game (games 1–3 from the specs):

- A game module: `games/01_empty_exit.py`, `games/02_dodge.py`, `games/03_lock_and_key.py`
- Each module exports `GAME_CONFIG` and a `setup(env)` function that registers entity types, behaviors, event handlers, spawning logic, and termination conditions.
- Each module has its own mechanical tests: `tests/games/test_01_empty_exit.py`, etc.
- Each module declares its game-specific invariant tests using the framework from Step 2.

### Where it lives

```
games/
  01_empty_exit.py
  02_dodge.py
  03_lock_and_key.py
tests/
  games/
    test_01_empty_exit.py
    test_02_dodge.py
    test_03_lock_and_key.py
```

**Done when**: All three reference games are playable, all mechanical tests pass, all invariant tests pass, and the random agent fuzz test completes 1000 episodes on each without error.

**Output**: Three complete reference games with tests.

---

## Step 6: HUMAN + CLAUDE CODE build the RL evaluation pipeline

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–5. Proceed to Step 6: build the RL evaluation pipeline."

This is the oracle. It takes any game module and produces a quantitative pass/fail verdict.

### What CLAUDE CODE builds

A script `evaluate_game.py` that:

1. **Layer 1 — Random Agent.** Loads the game, runs 1000 episodes of random actions. Records:
   - Crash count (must be 0).
   - Termination rate (% of episodes that end before `max_turns`).
   - Win rate.
   - Average episode length.
   - Pass criteria: zero crashes, termination rate > 80%, win rate between the game spec's declared bounds.

2. **Layer 2 — PPO Training.** Trains a Stable-Baselines3 PPO agent (`MultiInputPolicy` — automatically applies CNN to the grid observation and MLP to scalars) for 100k timesteps. **CRITICAL: Because game grids are small (e.g., 8×8 to 30×20), SB3's default NatureCNN will crash with negative dimension errors (it hardcodes kernel_size=8, stride=4 designed for 84×84 Atari frames). CLAUDE CODE must implement a custom `BaseFeaturesExtractor` (e.g., 2-layer CNN with kernel_size=3, stride=1, padding=1) and pass it via `policy_kwargs`.** Records:
   - Win rate at 10k steps.
   - Win rate at 100k steps.
   - Learning delta (win rate at 100k minus win rate at 10k).
   - Average episode length over training.
   - Pass criteria: learning delta > 0 (the agent is actually learning), final win rate exceeds the game spec's declared minimum.

3. **Layer 3 — Invariant Tests.** Runs all registered invariant tests for the game module. All must pass.

4. **Output.** A single JSON report:
   ```json
   {
     "game": "04_dungeon_crawl",
     "random_agent": { "crashes": 0, "termination_rate": 0.95, "win_rate": 0.03, "avg_length": 87.2, "pass": true },
     "ppo": { "win_rate_10k": 0.05, "win_rate_100k": 0.38, "learning_delta": 0.33, "pass": true },
     "invariants": { "total": 12, "passed": 12, "failed": [], "pass": true },
     "overall_pass": true
   }
   ```

### Calibration

HUMAN runs the evaluator against all three reference games. Expected results:

| Game | Random Win Rate | PPO 100k Win Rate | Learning Delta |
|------|----------------|-------------------|----------------|
| Empty Exit | 5–30% | >90% | >30% |
| Dodge | 1–10% | >50% | >15% |
| Lock & Key | <2% | >20% | >10% |

If these don't hold, HUMAN adjusts the game difficulty or the PPO hyperparameters until they do. Only then is the oracle trusted to evaluate swarm output.

**Done when**: The evaluator produces correct verdicts for all three reference games.

**Output**: `evaluate_game.py`. Calibration results. PPO hyperparameter config.

---

## Step 7: AGENT SWARM goes autonomous on userland

**HUMAN starts the containers.** AGENT SWARM runs autonomously from here.

Each agent's cycle:
1. Pull from upstream.
2. Read `AGENT_PROMPT.md` and `PROGRESS.md`.
3. Check `current_tasks/` — see what's claimed. Check `game-specs/` for games that don't have a passing implementation in `games/` yet.
4. Claim a game (write lock file, push).
5. Implement the game as a single Python module in `games/`, exporting `GAME_CONFIG` and `setup(env)`.
6. Run mechanical tests with `--fast`. Fix regressions.
7. Run random agent fuzz test on their game. Fix crashes.
8. Run invariant tests. Fix violations.
9. Run full mechanical test suite. All must pass.
10. Push to upstream. Remove lock file. Update `PROGRESS.md`.

HUMAN runs `evaluate_game.py` periodically (not every commit — PPO training is expensive) on new game implementations. If the RL evaluation fails, HUMAN can either intervene or add new mechanical tests / invariant checks that encode the problem, which AGENT SWARM will then fix autonomously.

### Scaling

The number of game specs is the scaling knob. Start with 5 swarm games (specs 4–8). If agents converge fast, write more specs. If they struggle, simplify existing specs or decompose them into sub-tasks (e.g., "implement enemy behaviors for dungeon crawl" and "implement item system for dungeon crawl" as separate tasks).

Consider specialized agent roles: one for code quality / deduplication across game modules, one for documentation, one for writing additional invariant tests.

**Done when**: HUMAN decides the games meet their quality bar, informed by RL evaluation verdicts, invariant test results, and their own judgment.

**Output**: Multiple complete games, built autonomously by AGENT SWARM, validated by both mechanical tests and the RL oracle.
