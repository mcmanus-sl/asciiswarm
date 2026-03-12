# AsciiSwarm — Development Plan

## What this project is actually testing

You're not testing whether Claude can build a game engine. You're testing whether the Carlini loop — swarm of agents, task locking, mechanical tests, autonomous convergence — works for creative software (games) instead of well-specified software (compilers). The engine is infrastructure. Every hour spent building a custom kernel is an hour not spent on the actual experiment.

Carlini didn't build Rust. He used Rust as the existing, trusted platform and spent his effort on the test harness and the task decomposition. The kernel here plays the same role — it must be small, trusted, and boring.

## Roles

- **HUMAN**: The developer. Runs Claude Code, reviews output, makes design decisions, builds infrastructure.
- **CLAUDE CODE**: AI pair-programmer. Works under HUMAN supervision during Steps 0–6.
- **AGENT SWARM**: Future autonomous Claude instances. They write game modules (userland) against the kernel. They do not exist until Step 7.
- **RL EVALUATOR**: A layered evaluation system (random agent, trained RL agent, invariant checks) that validates games quantitatively. Built in Steps 2, 5, and 6.

## Supervision Model

| Step | Who does the work | Autonomous? |
|------|-------------------|-------------|
| 0. Game specs | HUMAN | No — HUMAN writes these |
| 1. Kernel | HUMAN + CLAUDE CODE | No — pair programming |
| 2. Test harness + random agent | HUMAN + CLAUDE CODE | No — pair programming |
| 3. Agent prompt | HUMAN | No — HUMAN writes this |
| 4. Infrastructure | HUMAN | No — HUMAN builds this |
| 5. Reference games | HUMAN + CLAUDE CODE | No — pair programming |
| 6. RL evaluation pipeline | HUMAN + CLAUDE CODE | No — pair programming |
| 7. Swarm on userland | AGENT SWARM | Yes — autonomous |

The critical lesson from Carlini's C compiler project: Steps 0–2 are where HUMAN spends disproportionate time. They are not autonomous. Everything after Step 4 benefits from autonomy. Everything before it needs HUMAN's hands on it.

### Why Python, not TypeScript

The RL pipeline is Python (Stable-Baselines3, Gymnasium). If the kernel is TypeScript, every game needs a Python↔TypeScript bridge (subprocess + JSON, or WASM). This adds latency, debugging pain, and serialization bugs at the boundary — all for zero benefit. The kernel should be Python so that the engine IS the Gymnasium environment. No wrapper, no FFI, no bridge. `PPO("MlpPolicy", YourGameEnv()).learn(100_000)` just works.

### Why RL instead of Claude-as-judge

The original plan used a Claude playtest agent as the oracle — Claude plays the game, then Claude judges whether the game is good. This has the "oracle problem": the judge is as fallible as the builder. An RL agent solves this by making the evaluation objective and quantitative. If PPO can't learn to win after N episodes, the game is broken. If it wins in 5 episodes, there's no depth. If win rate climbs from 0% to 60% over training, you have real mechanical substance. No subjective judgment needed.

### The engine IS the Gymnasium environment

The kernel subclasses `gymnasium.Env` directly. There is no separate wrapper. The core class looks like:

```python
class GridGameEnv(gymnasium.Env):
    # The engine IS the env. No wrapper needed.
    # reset(), step(), observation_space, action_space are native.
```

Games are Python modules — a `setup(env)` function that registers entity types, behaviors, spawning rules, and termination conditions. Each game is one file. The swarm writes these files.

## How to use this document

This file and STEP1_PROMPT.md are the complete bootstrap for the project. After Step 1, HUMAN tells CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step N." Each step below has enough detail for CLAUDE CODE to execute.

---

## Step 0: HUMAN writes game specs

**HUMAN does this alone** — no CLAUDE CODE needed.

HUMAN writes 5–8 one-page game specs at escalating complexity. These are the "test programs" — like Carlini's progression from test suites to SQLite to Redis to Linux. The number of games is the scaling knob for the experiment.

### Suggested progression

1. **Walk to exit** — empty grid, player walks to goal tile. Simplest possible game.
2. **Avoid patrol** — walk to exit, one patrolling enemy to dodge.
3. **Keys and doors** — multi-room dungeon, locked doors, keys to find.
4. **Combat roguelike** — health, potions, three enemy types with different behaviors.
5. **Sokoban puzzle** — push boxes onto targets.
6. **Tower defense** — enemies spawn in waves, player places turrets.
7. **Stealth game** — enemies with sight cones, player must avoid detection.
8. **Survival** — hunger, crafting, resource gathering, day/night cycle.

Games 1–3 are reference games that HUMAN builds in Step 5. Games 4–8 are swarm tasks for Step 7.

### What each spec must include

- Grid dimensions and layout description.
- Entity types with glyphs and behaviors.
- Valid player actions.
- Win condition and lose condition.
- RL evaluation criteria: expected random-agent crash rate (should be 0), approximate PPO win rate range after 500 episodes (e.g., "40–70%"), and any degenerate strategies to watch for.

**Done when**: `game_specs/` directory contains 5–8 numbered spec files.

**Output**: `game_specs/01_walk_to_exit.md`, `game_specs/02_avoid_patrol.md`, etc.

---

## Step 1: HUMAN + CLAUDE CODE build the kernel

**HUMAN tells CLAUDE CODE**: Paste the contents of STEP1_PROMPT.md.

HUMAN supervises CLAUDE CODE to build the kernel — a Python class that subclasses `gymnasium.Env` and provides the grid, entity system, turn loop, event system, behavior hooks, ASCII renderer, and state serializer. This is the stable foundation that AGENT SWARM will later write game logic against — equivalent to Rust in Carlini's compiler project. Everything else trusts this layer.

The kernel should be ~500 lines. It is deliberately small. Full API spec is in STEP1_PROMPT.md.

HUMAN tests it, reviews it, owns it. This is a day or two of focused work, not a week. AGENT SWARM does not touch this unsupervised.

**Done when**: All kernel tests pass. HUMAN has reviewed the code and is satisfied with the API surface. The kernel works as a Gymnasium environment (reset/step/observation_space/action_space all functional).

**Output**: A tested, stable Python kernel with full test coverage. `pip install -e .` works.

---

## Step 2: HUMAN + CLAUDE CODE build the mechanical test harness

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Step 1 — the kernel is built and tested. Proceed to Step 2: build the expanded mechanical test harness."

This is where Carlini spent most of his effort and said so explicitly. Step 1 includes basic kernel unit tests. Step 2 expands those into a comprehensive regression suite that AGENT SWARM will run before every commit.

### What CLAUDE CODE builds

**Kernel stress tests** — edge cases the Step 1 tests didn't cover:
- Create and destroy hundreds of entities rapidly. No memory leaks, no stale references.
- Move entities to every boundary cell. Off-grid in every direction.
- Fill a cell with 20+ entities. Z-order rendering still correct. Queries still correct.
- Register 50+ event handlers. All fire in order. Unsubscribe works mid-iteration.
- Collision cancellation chains: handler A cancels, handler B would have fired — does it?
- Serialize a world with 1000+ entities. Round-trip still byte-identical.
- Load state, run 100 turns of random input with registered behaviors, serialize. Run the same 100 inputs again from the same loaded state. Output must be identical (determinism proof).

**Userland simulation tests** — tests that simulate what AGENT SWARM will actually do:
- Register a behavior that moves an entity toward a target every turn. Run 10 turns. Verify positions.
- Register a collision handler that destroys the mover (simulating a trap). Verify the entity is gone after the move.
- Register a collision handler that cancels the move (simulating a wall). Verify the entity stays.
- Create a chain reaction: entity A's behavior creates entity B, entity B's behavior emits a custom event, that event's handler destroys entity A. Verify all of it resolves in one turn.
- Register an input handler that creates an entity on action "place_bomb". Fire input. Verify entity exists.

**Gymnasium contract tests** — verify the kernel works as a proper Gym env:
- `env.reset()` returns valid observation and info dict.
- `env.step(action)` returns (obs, reward, terminated, truncated, info).
- `observation_space.contains(obs)` is True for all observations.
- `action_space.contains(action)` is True for all valid actions.
- `check_env(env)` from Stable-Baselines3 passes.

**Random agent fuzz tests** — Layer 1 of the RL evaluation stack, built early because it's cheap and catches crashes:
- A test that feeds 1000 random valid actions into a game world with registered behaviors. Asserts: no exceptions thrown, world state valid after every turn, serializer round-trip still works after every turn.
- Parameterized by seed for reproducibility. Different seeds on different runs.
- This is the first gate on every commit. If random play crashes the game, nothing else matters.
- Lives in `tests/test_random_agent.py`.

**Test runner output format** — designed for AGENT SWARM consumption, not human consumption:
- On success: one line per test file, e.g. `PASS tests/test_world.py (23 tests)`
- On failure: `FAIL tests/test_world.py` followed by `ERROR: [test name] — [one-line reason]`
- Summary line: `TOTAL: 247/250 passed`
- All verbose output goes to a log file, not stdout. AGENT SWARM's context window must not be polluted.
- Include a `--fast` flag that runs a deterministic 10% sample (different per agent via seed). AGENT SWARM uses `--fast` during development, full suite before pushing.

**Done when**: The test suite has 200+ assertions, all pass, and the runner output is clean and parseable.

**Output**: Expanded test suite in `tests/`. A test runner script with `--fast` and `--seed` flags.

---

## Step 3: HUMAN writes AGENT_PROMPT.md

**HUMAN does this alone** — no CLAUDE CODE needed.

HUMAN writes the prompt document that the agent loop feeds to every fresh AGENT SWARM instance. This is the equivalent of Carlini's agent prompt. It tells each agent:

- What the project is and what the kernel API looks like (or where to find the docs).
- What the current game spec is (or where to find it in `game_specs/`).
- How to pick a task: check `current_tasks/` for what's already claimed, read progress docs, pick the next most obvious unclaimed task.
- How to claim a task: write a file to `current_tasks/` (e.g., `current_tasks/implement_combat_roguelike.txt`) and push. If git rejects the push: run `git pull --rebase`. If the only conflict is in `current_tasks/` — the task was claimed by another agent; abort the rebase (`git rebase --abort`) and pick a different task. If the rebase is clean or conflicts are only in userland code — resolve conflicts, re-run `--fast` tests, and push again.
- How to work: implement the game as a Python module in `games/`, run the mechanical tests with `--fast`, fix regressions, run the full suite before pushing.
- How to finish: push to upstream, remove the task lock file, update progress docs with what was done and any known issues.
- How to leave notes: maintain a `PROGRESS.md` and update it frequently. If stuck, document what was tried and what failed so the next agent in this container doesn't repeat the work.
- What a game module looks like: a single Python file with a `setup(env)` function that registers entity types, behaviors, spawning rules, and termination conditions. Reference games in `games/reference/` are the examples to follow.

**Done when**: AGENT_PROMPT.md is in the repo root and contains all of the above.

**Output**: AGENT_PROMPT.md

---

## Step 4: HUMAN sets up the infrastructure

**HUMAN does this alone** — no CLAUDE CODE needed for infrastructure, though HUMAN may use CLAUDE CODE to help write scripts.

HUMAN builds the agent loop and parallelism infrastructure. This is nearly identical to what Carlini did:

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

### Task locking
- Agent claims a task by writing a file to `current_tasks/` (e.g., `current_tasks/implement_combat_roguelike.txt`) containing a short description of the approach.
- Agent pushes this file to upstream. If git rejects the push: agent runs `git pull --rebase`. If the only conflict is in `current_tasks/` — the task was claimed by another agent; abort the rebase and pick a different task. If the rebase is clean or conflicts are only in userland code — resolve conflicts, re-run `--fast` tests, and push again.
- When done, agent removes the lock file and pushes.

### How many agents
- Start with 4. Scale to 8–16 once HUMAN is confident the test harness catches regressions.
- More agents are only useful when there are many independent tasks. If agents start duplicating work, reduce count or decompose tasks further.
- Each game spec is a naturally independent task — perfect parallelism. 8 game specs = 8 agents can work simultaneously with zero coordination overhead.

**Done when**: HUMAN can spin up N containers and each one runs the loop, claims tasks, pushes code.

**Output**: Dockerfiles, shell scripts, bare git repo setup.

---

## Step 5: HUMAN + CLAUDE CODE write reference games

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–4. Proceed to Step 5: build the reference games."

Carlini didn't need a reference implementation because GCC defined "correct output." This project has no equivalent oracle. So before AGENT SWARM runs, HUMAN and CLAUDE CODE hand-write the first 2–3 games from the game specs in `game_specs/`.

### What CLAUDE CODE builds

Reference implementations for games 1–3 (or however many HUMAN has specced at the easy end). Each game is a single Python module with a `setup(env)` function. For example:

**Game 1: Walk to exit** — trivial, validates that the kernel works end-to-end.
**Game 2: Avoid patrol** — validates enemy behaviors, collision events.
**Game 3: Keys and doors** — validates multi-entity interaction, inventory-like properties, state complexity.

### What this validates

1. The kernel API is ergonomic enough for real game logic. If this is painful to write, HUMAN and CLAUDE CODE fix the kernel API before AGENT SWARM hits the same friction at scale.
2. The RL evaluation pipeline (Step 6) has known-good baselines to calibrate against.
3. The agent swarm has concrete examples of what a game module looks like.

### RL calibration

After each reference game works, HUMAN trains a PPO agent against it:

```python
from stable_baselines3 import PPO
env = GridGameEnv(game_module=walk_to_exit)
model = PPO("MlpPolicy", env).learn(100_000)
```

No wrapper needed — the kernel IS the Gym env. Verify that PPO win rates match the expected ranges from the game specs. If they don't, fix the game or the spec.

### Where it lives

```
games/
  reference/
    walk_to_exit.py       — game 1
    avoid_patrol.py       — game 2
    keys_and_doors.py     — game 3
tests/
  games/
    test_walk_to_exit.py  — deterministic tests for game 1
    test_avoid_patrol.py  — deterministic tests for game 2
    test_keys_and_doors.py — deterministic tests for game 3
```

Each reference game gets its own mechanical tests: "player moves north, verify position changed." "Player walks into wall, verify position unchanged." "Player walks into enemy, verify health decreased."

**Done when**: Reference games are playable, all game tests pass, PPO achieves expected win rates from the game specs.

**Output**: 2–3 complete game modules in `games/reference/` with their own tests. PPO training logs confirming calibration.

---

## Step 6: HUMAN + CLAUDE CODE build the RL evaluation pipeline

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–5. Proceed to Step 6: build the RL evaluation pipeline."

This is the oracle. It takes any game module and produces a pass/fail verdict. This is the equivalent of Carlini's "compile the Linux kernel" integration test — an expensive but high-signal check that runs periodically on the integrated build.

### The four evaluation layers

**Layer 1: Random agent** (built in Step 2, runs on every commit)
- Feeds random valid actions for 1000 turns. No training, no cost.
- Pure fuzz testing. Catches crashes, invalid states, exceptions.
- First gate. If this fails, nothing else runs.

**Layer 2: RL agent** (the core of this step)
- Trains a Stable-Baselines3 PPO agent directly against the game's `GridGameEnv` (no wrapper — the engine IS the env).
- Collects metrics: win rate over training, average episode length, convergence speed, state coverage (% of grid cells visited), degenerate strategy detection.
- Outputs a structured JSON report.

**Layer 3: Invariant tests** (structural checks, run alongside RL)
- BFS reachability: player can always reach the exit from spawn.
- Entity coverage: every entity type has a registered behavior.
- No orphaned rooms: all rooms are connected.
- State bounds: no property values outside expected ranges after N turns of play.
- These are deterministic and catch structural problems the RL agent might work around without surfacing.
- Game specs can declare additional game-specific invariants.

**Layer 4: Claude playtest** (optional, human-triggered only)
- NOT part of the automated pipeline.
- HUMAN can trigger this occasionally for subjective "does this feel coherent" evaluation.
- The one thing RL metrics genuinely can't tell you.

### What CLAUDE CODE builds

A generic evaluation function that takes any game module and produces a verdict:

```python
def evaluate_game(game_module, spec) -> EvalReport:
    env = GridGameEnv(game_module=game_module)
    # Layer 1: random agent
    # Layer 2: PPO training
    # Layer 3: invariant checks
    # Compare metrics against spec's expected ranges
    return report
```

### RL evaluation report format

```json
{
  "game": "combat_roguelike",
  "random_agent": {
    "turns_played": 1000,
    "crashes": 0,
    "invalid_states": 0,
    "pass": true
  },
  "rl_agent": {
    "episodes_trained": 500,
    "final_win_rate": 0.62,
    "win_rate_curve": [0.0, 0.05, 0.12, 0.31, 0.48, 0.62],
    "avg_episode_length": 47,
    "state_coverage": 0.78,
    "convergence_episode": 320,
    "degenerate_strategies": ["agent camps corner at (0,0) in 12% of wins"]
  },
  "invariants": {
    "exit_reachable": true,
    "all_behaviors_registered": true,
    "all_rooms_connected": true,
    "property_bounds_valid": true
  },
  "spec_compliance": {
    "win_rate_in_expected_range": true,
    "expected_range": [0.4, 0.7]
  },
  "verdict": "healthy"
}
```

### Interpreting results

- **Broken/unwinnable**: RL win rate stays at 0% after full training. Random agent may also crash.
- **Too easy / no depth**: RL solves it in <20 episodes. Win rate jumps to 90%+ immediately.
- **Healthy**: Win rate climbs gradually (0% → 40–70%) over hundreds of episodes. Agent visits most of the grid. No degenerate strategies dominate.
- **Degenerate design**: RL finds a dominant strategy (e.g., camp a safe corner, ignore items). Win rate is high but state coverage is low.
- **Balance problems**: Agent never picks up certain items, or dies to the same enemy placement consistently.

### Framework

- **Stable-Baselines3** for PPO (pip-installable, works out of the box for small discrete envs).
- **Gymnasium** — native, no wrapper needed. The kernel IS the env.
- No FFI boundary. No subprocess communication. No WASM. Everything is Python.

**Done when**: The evaluation pipeline runs against all reference games and produces healthy reports. PPO achieves expected win rates from the game specs. Invariant tests all pass. Report format is stable and parseable.

**Output**: Generic evaluation function. Evaluation reports for all reference games. Trained baseline agents.

---

## Step 7: AGENT SWARM goes autonomous on userland

**HUMAN starts the containers**. AGENT SWARM runs autonomously from here.

Each game spec in `game_specs/` that doesn't have a completed implementation in `games/` is a task. This is naturally parallel — each agent picks a different game spec, just like Carlini's agents compiling different C programs.

Each agent's cycle:
1. Pull from upstream.
2. Read AGENT_PROMPT.md and PROGRESS.md.
3. Check `current_tasks/` — see what's claimed, pick an unclaimed game spec.
4. Claim the task (write lock file, push).
5. Implement the game as a Python module in `games/` following the pattern of reference games.
6. Run mechanical tests with `--fast`. Fix regressions.
7. Run full mechanical test suite. All must pass.
8. Push to upstream. Remove lock file. Update PROGRESS.md.

The RL evaluation pipeline runs periodically (not every commit — training takes minutes) on the integrated build. Layer 1 (random agent) runs on every commit as part of the test suite. If the RL evaluation surfaces problems (win rate regression, degenerate strategies, broken invariants), HUMAN can either intervene directly or add new mechanical tests that encode the problem, which AGENT SWARM will then fix autonomously.

### The scaling knob

The number of game specs is how you scale the experiment:
- Start with 5 specs. If the swarm converges fast, write 10 more.
- If agents struggle, simplify the specs or break complex games into subtasks.
- Carlini scaled from test suites to SQLite to Redis to Linux. You scale from "walk to exit" to "full roguelike" to "tower defense."

### Parallelism guidance (from Carlini)

- Each game spec is independent — perfect parallel task. 8 specs = 8 agents with zero coordination.
- When agents converge on the same problem (e.g., a kernel bug that affects multiple games), reduce agent count or decompose the problem further.
- Consider specialized agent roles: one agent for code quality/deduplication, one for writing new game specs based on what's been learned.

**Done when**: HUMAN decides the games meet their quality bar, informed by RL evaluation metrics (win rate, convergence, coverage, degenerate strategy detection) and their own judgment.

**Output**: Multiple complete games, built autonomously by AGENT SWARM, validated by mechanical tests, random agent fuzzing, RL evaluation pipeline, and invariant checks.
