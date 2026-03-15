---
name: experiment
description: GPU-aware RL experiment runner — batch train, eval, and replay games with live monitoring
user_invocable: true
---

# /experiment — GPU-Aware RL Experiment Runner

You are an autonomous experiment runner for the asciiswarm RL-TDD project. You orchestrate the full train → eval → replay pipeline with GPU health monitoring.

## Phase 0: GPU Health Check

Run this BEFORE asking any questions:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
```

Report to the user:
- GPU 0 name, VRAM free/total
- If VRAM free < 4096 MB or GPU is unavailable → warn and ask if they want to proceed
- If healthy → report briefly and move to Phase 1

## Phase 1: Interactive Questions (MANDATORY — ask before ANY training)

### Q1: Which games?

Discover available games:
```bash
ls jaxswarm/games/game_*.py
```

Also check which already have trained weights:
```bash
ls weights/game_*.eqx 2>/dev/null
```

Present a numbered list like:
```
Available games:
  1. game_01_empty_exit      [weights: ✓]
  2. game_02_dodge            [weights: ✓]
  3. game_03_lock_and_key     [weights: ✓]
  4. game_04_dungeon_crawl    [weights: ✗]
  5. game_05_pac_man_collect  [weights: ✗]
  6. game_06_ice_sliding      [weights: ✗]

Which games? (e.g. "1,3,5", "all", "4-6")
```

### Q2: What pipeline mode?

Ask:
```
Pipeline mode:
  1. Full (default) — train + 5-layer eval + save weights + replay
  2. Train + eval — skip replay generation
  3. Eval only — re-evaluate existing weights (no training)
  4. Replay only — regenerate replay.mp4 from existing weights

Which mode? (1-4, default: 1)
```

For "Eval only" or "Replay only", the selected games MUST have existing weights — warn if not.

### Q3 (optional — ask based on context):

- If multiple games selected: "Stop on first failure, or push through all games?"
- If retraining games that already have weights: "Any notes on what changed that prompted retraining?" (include in summary)

## Phase 2: Autonomous Execution

### For each selected game, sequentially:

#### Pre-flight GPU check
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0
```
If free VRAM < 4096 MB → report to user and abort this game.

#### Run training + evaluation (Full / Train+eval modes)
```bash
python evaluate_game.py jaxswarm.games.game_XX_name
```

- This runs all 5 layers: deterministic trace, random agent, PPO training, trajectory forensics, invariant checks
- It saves weights automatically on PPO pass
- **Timeout**: Set a 10-minute timeout per game. Training typically takes 10-60 seconds.
- Capture ALL stdout/stderr

#### Parse output for each game

Extract from the JSON report printed between the `====` lines:
- `overall_pass`: true/false
- `ppo.win_rate`: the PPO win rate
- `ppo.training_time_s`: training wall time
- `forensics.eval_win_rate`: eval win rate with trained policy
- `forensics.top_trajectory`: best trajectory description
- `invariants.passed` / `invariants.total`

Also look for SPS (steps per second) in the training output — it appears as `SPS: X.XXM` in train.py output.

#### Report inline after each game

Print a concise result:
```
✓ game_01_empty_exit | win=0.9812 | SPS=1.2M/s | 12.3s | PASS
✗ game_04_dungeon_crawl | win=0.0312 | SPS=1.1M/s | 45.2s | FAIL
```

#### Stop-on-failure behavior
If user chose stop-on-failure and a game fails → halt immediately, report what happened.

### Eval-only mode

For eval-only, there is no standalone eval script — you still run `evaluate_game.py` which does all 5 layers including training. Inform the user that eval-only will retrain (it's fast). Alternatively, if the user truly wants no retraining, suggest they inspect the existing weights manually.

### Replay generation (Full / Replay-only modes)

After all games complete (or in replay-only mode):
```bash
python replay_games.py
```

Note: `replay_games.py` replays ALL games in its hardcoded GAMES list that have weights. It produces `replay.mp4`.

### Final Summary

Print a summary table:
```
┌─────────────────────────┬──────────┬──────────┬────────┬────────┐
│ Game                    │ Win Rate │ SPS      │ Time   │ Status │
├─────────────────────────┼──────────┼──────────┼────────┼────────┤
│ game_01_empty_exit      │ 0.9812   │ 1.2M/s   │ 12.3s  │ PASS   │
│ game_02_dodge           │ 0.4521   │ 1.1M/s   │ 15.1s  │ PASS   │
│ ...                     │          │          │        │        │
└─────────────────────────┴──────────┴──────────┴────────┴────────┘
```

Then report:
- Total wall time for the entire experiment
- Final GPU state (run `nvidia-smi` one more time)
- Whether replay.mp4 was generated
- Count: X/Y games passed

## Important Notes

- Working directory is always the project root (`/home/mcmanus/asciiswarm`)
- GPU 0 = RTX PRO 6000 Blackwell 96GB — this is the training GPU
- `CUDA_VISIBLE_DEVICES=0` is already set in evaluate_game.py and replay_games.py
- Each `evaluate_game.py` run is independent — no shared state between games
- Training is FAST on this hardware (typically 10-60 seconds per game) — don't be afraid to run all games
- The game module path format is `jaxswarm.games.game_XX_name` (dots, not slashes)
