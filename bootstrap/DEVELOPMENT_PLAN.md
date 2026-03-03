# ASCII Game Engine — Development Plan

## Roles

- **HUMAN**: The developer. Runs Claude Code, reviews output, makes design decisions, builds infrastructure.
- **CLAUDE CODE**: AI pair-programmer. Works under HUMAN supervision during Steps 1–6.
- **AGENT SWARM**: Future autonomous Claude instances. They write game logic (userland) against the kernel. They do not exist until Step 7.
- **PLAYTEST AGENT**: A specialized Claude instance (not part of AGENT SWARM) that plays the game through the ASCII interface and evaluates it. Built in Step 6.

## Supervision Model

| Step | Who does the work | Autonomous? |
|------|-------------------|-------------|
| 1. Kernel | HUMAN + CLAUDE CODE | No — pair programming |
| 2. Mechanical tests | HUMAN + CLAUDE CODE | No — pair programming |
| 3. Agent prompt | HUMAN | No — HUMAN writes this |
| 4. Infrastructure | HUMAN | No — HUMAN builds this |
| 5. Reference game | HUMAN + CLAUDE CODE | No — pair programming |
| 6. Playtest agent | HUMAN + CLAUDE CODE | No — pair programming |
| 7. Swarm on userland | AGENT SWARM | Yes — autonomous |

The critical lesson from Carlini's C compiler project: Steps 1 and 2 are where HUMAN spends disproportionate time. They are not autonomous. Everything after Step 4 benefits from autonomy. Everything before it needs HUMAN's hands on it.

## How to use this document

This file and STEP1_CLAUDE_CODE_PROMPT.md are the complete bootstrap for the project. After Step 1, HUMAN tells CLAUDE CODE: "Read DEVELOPMENT_PLAN.md, proceed to Step N." Each step below has enough detail for CLAUDE CODE to execute.

---

## Step 1: HUMAN + CLAUDE CODE build the kernel

**HUMAN tells CLAUDE CODE**: Paste the contents of STEP1_CLAUDE_CODE_PROMPT.md.

HUMAN supervises CLAUDE CODE to build the kernel. This is the stable foundation that AGENT SWARM will later write game logic against — equivalent to the Rust standard library in Carlini's compiler project. Everything else trusts this layer.

The kernel includes: the entity system, the grid, the turn loop, the hook/behavior registration system, the ASCII renderer, and the structured state serializer. Full API spec is in STEP1_CLAUDE_CODE_PROMPT.md.

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
- How to claim a task: write a file to `current_tasks/` (e.g., `current_tasks/implement_goblin_ai.txt`) and push. If git rejects because another agent claimed it first, pick a different task.
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
- Agent pushes this file to upstream. If git rejects (another agent claimed it), agent picks a different task.
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
2. PLAYTEST AGENT has a known-good baseline to calibrate against in Step 6.

### Where it lives

```
userland/
  reference-game/
    game.ts         — all handlers, behaviors, and setup
    game.test.ts    — deterministic tests for this specific game
```

The reference game also gets its own mechanical tests: "player moves north, verify position changed." "Player walks into wall, verify position unchanged." "Player walks into enemy, verify health decreased." "Player picks up potion, verify potion removed from grid and health increased."

**Done when**: The reference game is playable via a simple test script that feeds input actions and prints ASCII output. All reference game tests pass.

**Output**: A complete mini-game in `userland/reference-game/` with its own tests.

---

## Step 6: HUMAN + CLAUDE CODE build the playtest agent

**HUMAN tells CLAUDE CODE**: "Read DEVELOPMENT_PLAN.md. We've completed Steps 1–5. Proceed to Step 6: build the playtest agent."

PLAYTEST AGENT is a specialized Claude instance. It is NOT part of AGENT SWARM. It does not write code. It pulls the latest build, plays the game through the ASCII interface plus the structured state serializer, and produces a structured evaluation.

This is the equivalent of Carlini's "compile the Linux kernel" integration test — the expensive, slow, high-signal check that runs after agents think a feature is done.

### What CLAUDE CODE builds

A playtest harness script that:

1. Boots the game from a known starting state.
2. Feeds the ASCII render + structured state to a Claude API call.
3. Claude decides an action (from the valid action set).
4. The harness applies the action, advances the turn, captures the new state.
5. Repeats for N turns (configurable, default 50).
6. After the play session, feeds the full game transcript (sequence of states and actions) to a Claude evaluation call.

### PLAYTEST AGENT evaluation criteria

The evaluation call produces a structured JSON report, not prose. Fields include:

- `playable`: boolean — could PLAYTEST AGENT complete actions and observe results?
- `winnable`: boolean — is there a path to the win condition?
- `losable`: boolean — is there a fail state?
- `meaningful_choices`: boolean — were there turns where the best action wasn't obvious?
- `enemy_threat`: boolean — did enemies actually affect play?
- `progression`: boolean — did game state change meaningfully over the session?
- `stuck`: boolean — was PLAYTEST AGENT ever unable to make progress?
- `notes`: string[] — specific observations, e.g. "enemy never moved", "couldn't find exit"
- `turn_count`: number — how many turns the session lasted
- `outcome`: "win" | "lose" | "incomplete"

### Calibration

HUMAN runs PLAYTEST AGENT against the reference game from Step 5 first. The reference game should score `true` on all boolean fields. If it doesn't, HUMAN adjusts PLAYTEST AGENT's prompts until it does. Only then is PLAYTEST AGENT trusted to evaluate AGENT SWARM output.

**Done when**: PLAYTEST AGENT plays the reference game and produces a correct structured evaluation.

**Output**: Playtest harness script. Evaluation prompt. Calibration results against the reference game.

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

HUMAN runs PLAYTEST AGENT periodically (not every commit — it's expensive) on the integrated build. If PLAYTEST AGENT surfaces problems, HUMAN can either intervene directly or add new mechanical tests that encode the problem, which AGENT SWARM will then fix autonomously.

### Parallelism guidance (from Carlini)

- When there are many distinct failing tests or independent features, parallelism is trivial: each agent picks a different one.
- When agents converge on the same problem (e.g., a single integration bug), reduce agent count or decompose the problem further.
- Consider specialized agent roles: one agent for code quality/deduplication, one for documentation, one for performance.

**Done when**: HUMAN decides the game meets their quality bar, informed by PLAYTEST AGENT evaluations and their own judgment.

**Output**: A complete game, built autonomously by AGENT SWARM, validated by both mechanical tests and PLAYTEST AGENT.
