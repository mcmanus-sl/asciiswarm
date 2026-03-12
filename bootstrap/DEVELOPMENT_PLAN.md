# ASCII Game Engine — Development Plan

## Roles

- **HUMAN**: The developer. Runs Claude Code, reviews output, makes design decisions, builds infrastructure.
- **CLAUDE CODE**: AI pair-programmer. Works under HUMAN supervision during Steps 1–6.
- **AGENT SWARM**: Future autonomous Claude instances. They write game logic (userland) against the kernel. They do not exist until Step 7.
- **RL EVALUATOR**: A layered evaluation system (random agent, trained RL agent, invariant checks) that replaces the need for a Claude-as-judge playtest agent. Built in Steps 2, 5, and 6.

## Supervision Model

| Step | Who does the work | Autonomous? |
|------|-------------------|-------------|
| 1. Kernel | HUMAN + CLAUDE CODE | No — pair programming |
| 2. Mechanical tests + random agent | HUMAN + CLAUDE CODE | No — pair programming |
| 3. Agent prompt | HUMAN | No — HUMAN writes this |
| 4. Infrastructure | HUMAN | No — HUMAN builds this |
| 5. Reference game + Gym wrapper | HUMAN + CLAUDE CODE | No — pair programming |
| 6. RL evaluation pipeline | HUMAN + CLAUDE CODE | No — pair programming |
| 7. Swarm on userland | AGENT SWARM | Yes — autonomous |

The critical lesson from Carlini's C compiler project: Steps 1 and 2 are where HUMAN spends disproportionate time. They are not autonomous. Everything after Step 4 benefits from autonomy. Everything before it needs HUMAN's hands on it.

### Why RL instead of Claude-as-judge

The original plan used a Claude playtest agent as the oracle — Claude plays the game, then Claude judges whether the game is good. This has the "oracle problem": the judge is as fallible as the builder. An RL agent solves this by making the evaluation objective and quantitative. If PPO can't learn to win after N episodes, the game is broken. If it wins in 5 episodes, there's no depth. If win rate climbs from 0% to 60% over training, you have real mechanical substance. No subjective judgment needed.

## How to use this document

This file and STEP1_CLAUDE_CODE_PROMPT.md are the complete bootstrap for the project. After Step 1, HUMAN tells CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step N." Each step below has enough detail for CLAUDE CODE to execute.

---

## Step 1: HUMAN + CLAUDE CODE build the kernel

**HUMAN tells CLAUDE CODE**: Paste the contents of STEP1_CLAUDE_CODE_PROMPT.md.

HUMAN supervises CLAUDE CODE to build the kernel. This is the stable foundation that AGENT SWARM will later write game logic against — equivalent to the Rust standard library in Carlini's compiler project. Everything else trusts this layer.

The kernel includes: the entity system, the grid, the turn loop, the hook/behavior registration system, the ASCII renderer, the structured state serializer, and a `toGymObservation()` method on the serializer that produces a flat numeric representation of game state for RL agent consumption. Full API spec is in STEP1_CLAUDE_CODE_PROMPT.md.

HUMAN tests it, reviews it, owns it. This is probably a few days of focused work. AGENT SWARM does not touch this unsupervised.

**Done when**: All kernel tests pass. HUMAN has reviewed the code and is satisfied with the API surface.

**Output**: A tested, stable TypeScript kernel with full test coverage.

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

**Random agent fuzz tests** — Layer 1 of the RL evaluation stack, built early because it's cheap and catches crashes:
- A test that feeds 1000 random valid actions into a game world with registered behaviors. Asserts: no exceptions thrown, world state valid after every turn, serializer round-trip still works after every turn.
- Parameterized by seed for reproducibility. Different seeds on different runs.
- This is the first gate on every commit. If random play crashes the game, nothing else matters.
- Lives in `tests/kernel/random-agent.test.ts`.

**Test runner output format** — designed for AGENT SWARM consumption, not human consumption:
- On success: one line per test file, e.g. `PASS kernel/world.test.ts (23 tests)`
- On failure: `FAIL kernel/world.test.ts` followed by `ERROR: [test name] — [one-line reason]`
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
- What the current game design spec is (or where to find it).
- How to pick a task: check `current_tasks/` for what's already claimed, read progress docs, pick the next most obvious unclaimed task.
- How to claim a task: write a file to `current_tasks/` (e.g., `current_tasks/implement_goblin_ai.txt`) and push. If git rejects the push: run `git pull --rebase`. If the only conflict is in `current_tasks/` — the task was claimed by another agent; abort the rebase (`git rebase --abort`) and pick a different task. If the rebase is clean or conflicts are only in userland code — resolve conflicts, re-run `--fast` tests, and push again.
- How to work: implement the feature in userland, run the mechanical tests with `--fast`, fix regressions, run the full suite before pushing.
- How to finish: push to upstream, remove the task lock file, update progress docs with what was done and any known issues.
- How to leave notes: maintain a `PROGRESS.md` and update it frequently. If stuck, document what was tried and what failed so the next agent in this container doesn't repeat the work.

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
- Agent claims a task by writing a file to `current_tasks/` (e.g., `current_tasks/implement_goblin_ai.txt`) containing a short description of the approach.
- Agent pushes this file to upstream. If git rejects the push: agent runs `git pull --rebase`. If the only conflict is in `current_tasks/` — the task was claimed by another agent; abort the rebase and pick a different task. If the rebase is clean or conflicts are only in userland code — resolve conflicts, re-run `--fast` tests, and push again.
- When done, agent removes the lock file and pushes.

### How many agents
- Start with 4. Scale to 8–16 once HUMAN is confident the test harness catches regressions.
- More agents are only useful when there are many independent tasks. If agents start duplicating work, reduce count or decompose tasks further.

**Done when**: HUMAN can spin up N containers and each one runs the loop, claims tasks, pushes code.

**Output**: Dockerfiles, shell scripts, bare git repo setup.

---

## Step 5: HUMAN + CLAUDE CODE write a reference game

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–4. Proceed to Step 5: build the reference game."

Carlini didn't need a reference implementation because GCC defined "correct output." This project has no equivalent oracle. So before AGENT SWARM runs, HUMAN and CLAUDE CODE hand-write one minimal complete game in userland.

### What CLAUDE CODE builds

A small, complete, playable game using only the kernel API. Suggested scope:

- A 5-room dungeon (rooms connected by doors).
- One player entity controlled by input actions: `move_north`, `move_south`, `move_east`, `move_west`, `use`, `wait`.
- Wall entities that block movement (via collision cancellation).
- One enemy type with simple behavior (e.g., moves toward player if within 3 tiles, otherwise wanders randomly).
- One item type (e.g., a health potion) that the player can pick up by walking into it.
- A health property on the player. Enemy collision reduces health. Potion restores health.
- A win condition (reach a specific tile) and a lose condition (health reaches 0).
- A game-over state where further input does nothing.

### What this validates

1. The kernel API is ergonomic enough for real game logic. If this is painful to write, HUMAN and CLAUDE CODE fix the kernel API before AGENT SWARM hits the same friction at scale.
2. The RL evaluation pipeline (Step 6) has a known-good baseline to calibrate against.

### Gymnasium wrapper

After the reference game works, CLAUDE CODE builds a Gymnasium-compatible wrapper for it. This is the calibration step — equivalent to what the old plan did with Claude playtest calibration, but now it's "can PPO learn to beat the reference dungeon in under 500 episodes?"

The wrapper:
- Translates kernel state into a Gymnasium observation (via `toGymObservation()`).
- Maps discrete action indices to the game's valid action strings (`move_north`, `move_south`, etc.).
- Returns reward: `+1` win, `-1` lose, `-0.01` per turn (encourages efficiency).
- Returns `done=True` on win, lose, or max turns exceeded.
- Exposes `reset()` and `step(action)` per the Gymnasium API.

HUMAN trains a baseline Stable-Baselines3 PPO agent against this wrapper. If PPO can learn to beat the reference dungeon (win rate > 50% after 500 episodes), the wrapper and reward signal work. If not, fix them before the swarm runs.

### Where it lives

```
userland/
  reference-game/
    game.ts         — all handlers, behaviors, and setup
    game.test.ts    — deterministic tests for this specific game
  gym-wrapper/
    env.py          — Gymnasium environment wrapping the kernel (via subprocess or WASM)
    train_baseline.py — PPO training script for calibration
    test_wrapper.py — tests that the wrapper satisfies Gymnasium API contract
```

The reference game also gets its own mechanical tests: "player moves north, verify position changed." "Player walks into wall, verify position unchanged." "Player walks into enemy, verify health decreased." "Player picks up potion, verify potion removed from grid and health increased."

**Done when**: The reference game is playable, all reference game tests pass, the Gymnasium wrapper passes API contract tests, and a PPO agent achieves >50% win rate on the reference game within 500 episodes.

**Output**: A complete mini-game in `userland/reference-game/` with its own tests. A Gymnasium wrapper in `userland/gym-wrapper/` with a trained baseline agent.

---

## Step 6: HUMAN + CLAUDE CODE build the RL evaluation pipeline

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–5. Proceed to Step 6: build the RL evaluation pipeline."

This replaces the original Claude-as-judge playtest agent. Instead of asking Claude "is this game good?", we train an RL agent and read the metrics. This is the equivalent of Carlini's "compile the Linux kernel" integration test — an expensive but high-signal check that runs periodically on the integrated build.

### The four evaluation layers

**Layer 1: Random agent** (built in Step 2, runs on every commit)
- Feeds random valid actions for 1000 turns. No training, no cost.
- Pure fuzz testing. Catches crashes, invalid states, exceptions.
- First gate. If this fails, nothing else runs.

**Layer 2: RL agent** (the core of this step)
- Uses the Gymnasium wrapper from Step 5.
- Trains a Stable-Baselines3 PPO agent (two-layer MLP policy, sufficient for small discrete environments).
- Collects metrics: win rate over training, average episode length, convergence speed, state coverage (% of grid cells visited), degenerate strategy detection.
- Outputs a structured JSON report.

**Layer 3: Invariant tests** (structural checks, run alongside RL)
- BFS reachability: player can always reach the exit from spawn.
- Entity coverage: every entity type has a registered behavior.
- No orphaned rooms: all rooms are connected.
- State bounds: no property values outside expected ranges after N turns of play.
- These are deterministic and catch structural problems the RL agent might work around without surfacing.

**Layer 4: Claude playtest** (optional, human-triggered only)
- NOT part of the automated pipeline.
- HUMAN can trigger this occasionally for subjective "does this feel coherent" evaluation.
- The one thing RL metrics genuinely can't tell you.

### What CLAUDE CODE builds

An evaluation pipeline script that:

1. Runs Layer 1 (random agent) — fast, first gate.
2. Trains a PPO agent against the game's Gymnasium wrapper for N episodes (configurable, default 500).
3. Runs Layer 3 invariant tests against the game's starting state.
4. Collects all metrics and produces a structured JSON evaluation report.

### RL evaluation report format

```json
{
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
- **Gymnasium** for the environment API.
- Communication between Python (RL) and TypeScript (kernel) via subprocess + JSON over stdin/stdout, or a compiled WASM module.

**Done when**: The evaluation pipeline runs against the reference game and produces a healthy report. PPO achieves >50% win rate. Invariant tests all pass. Report format is stable and parseable.

**Output**: Evaluation pipeline script. Trained baseline agent. Structured evaluation report for the reference game.

---

## Step 7: AGENT SWARM goes autonomous on userland

**HUMAN starts the containers**. AGENT SWARM runs autonomously from here.

With the design spec decomposed into tasks, AGENT SWARM runs the loop from Step 4. Agents grab task locks on work items like "implement ranged combat handler" or "build level 3 enemy spawning" or "add item pickup interaction" and write userland code against the stable kernel API.

Each agent's cycle:
1. Pull from upstream.
2. Read AGENT_PROMPT.md and PROGRESS.md.
3. Check `current_tasks/` — see what's claimed, pick something unclaimed.
4. Claim the task (write lock file, push).
5. Implement the feature in userland.
6. Run mechanical tests with `--fast`. Fix regressions.
7. Run full mechanical test suite. All must pass.
8. Push to upstream. Remove lock file. Update PROGRESS.md.

The RL evaluation pipeline runs periodically (not every commit — training takes minutes) on the integrated build. Layer 1 (random agent) runs on every commit as part of the test suite. If the RL evaluation surfaces problems (win rate regression, degenerate strategies, broken invariants), HUMAN can either intervene directly or add new mechanical tests that encode the problem, which AGENT SWARM will then fix autonomously.

### Parallelism guidance (from Carlini)

- When there are many distinct failing tests or independent features, parallelism is trivial: each agent picks a different one.
- When agents converge on the same problem (e.g., a single integration bug), reduce agent count or decompose the problem further.
- Consider specialized agent roles: one agent for code quality/deduplication, one for documentation, one for performance.

**Done when**: HUMAN decides the game meets their quality bar, informed by RL evaluation metrics (win rate, convergence, coverage, degenerate strategy detection) and their own judgment.

**Output**: A complete game, built autonomously by AGENT SWARM, validated by mechanical tests, random agent fuzzing, RL evaluation pipeline, and invariant checks.
