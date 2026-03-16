# Plan: Training Infrastructure for Games 11-14

## Context
Games 11-14 scale dramatically: Game 14 is 32×32 / 512 entities / 10 tags / 8 props / 1000 turns — composing ALL prior systems. Current training (hardcoded 4096 envs × 128 steps, single GPU, no progress output) won't scale. Game 04 already uses 73GB VRAM at 16×16/128 entities. We need adaptive configs, multi-GPU parallelism, a scalable network, progress monitoring, and curriculum transfer.

**Games 01-10 are untouched.** Existing weights, game code, and the current `ActorCritic` network all stay as-is. All changes are additive.

## Changes (ordered by dependency)

### 1. Scaled Network Architecture — New Class Alongside Existing
**File:** `jaxswarm/network.py`

Current `ActorCritic` stays for games 01-10. Add a new `ScaledActorCritic` for games 11-14 that uses strided convolutions + adaptive resize to a fixed spatial output:

```
Conv(tags→32, k=3, s=1, p=1) → ReLU
Conv(32→64,  k=3, s=2, p=1) → ReLU    # halves spatial
Conv(64→64,  k=3, s=2, p=1) → ReLU    # halves again
jax.image.resize to (4,4)              # fixed output regardless of grid
→ 64×4×4 = 1024 features (always)

Linear(1024+32 → 128) → ReLU          # wider trunk
Actor: Linear(128 → num_actions)
Critic: Linear(128 → 1)
```

- 16×16 → 8→4 → 1024. 32×32 → 16→8 → resize(4,4) → 1024. Works at any grid size.
- Fixed feature size enables curriculum transfer across different grid sizes
- Evaluator/trainer picks network class based on game config (e.g. `config.grid_h > 16 or config.max_entities > 128`)
- **Games 01-10 weights untouched** — they keep using the original `ActorCritic`

### 2. Training Progress Monitoring — Chunked Training Loop
**File:** `jaxswarm/train.py`

Refactor `_train_loop` from one monolithic `jax.lax.scan(length=num_updates)` into chunks:

- Python for-loop over chunks of ~50 updates each
- Each chunk still a single `@eqx.filter_jit` scan (no perf loss — same traced function reused)
- Between chunks: print SPS, win rate, elapsed time
- Optional callback for intermediate weight saves

```
  [1/16] SPS: 210,000  Win: 0.12  (25.3s)
  [2/16] SPS: 215,000  Win: 0.34  (50.1s)
```

Backward compatible — existing games benefit from progress output too.

### 3. Adaptive TrainConfig — VRAM-Aware Batch Sizing
**Files:** `jaxswarm/train.py` (add `auto_train_config()`), `evaluate_game.py` (use it with fallback)

Auto-compute `num_envs` and `num_steps` from game CONFIG + available VRAM:

- Obs grid dominates VRAM: `num_tags × grid_h × grid_w × 4 bytes` per env per step
- State arrays: `max_entities × (~num_tags + num_props*4 + 16)` per env
- Budget = 60% of available VRAM (leave 40% for XLA workspace + optimizer + intermediates)
- `num_envs` rounded to power of 2, clamped [256, 4096]
- Compensate fewer envs with more updates to maintain total env steps ~400M
- **Backward compat:** for games 01-10, the formula produces the same 4096 envs / 128 steps

| Game | Grid | Entities | Est. num_envs | num_steps |
|------|------|----------|---------------|-----------|
| 01-03 | 8-12 | 8-64 | 4096 | 128 |
| 04,09 | 16×16 | 28-128 | 4096 | 128 |
| 11 | 16×16 | 256 | 2048 | 128 |
| 14 | 32×32 | 512 | 512-1024 | 64 |

### 4. Multi-GPU Training — Process-Level Parallelism
**Files:** `train_parallel.py` (new), remove hardcoded `CUDA_VISIBLE_DEVICES` from `train.py` and `evaluate_game.py` (accept as CLI arg / env var instead)

Process isolation via `CUDA_VISIBLE_DEVICES` (not JAX pmap — heterogeneous GPUs make that impractical):

```bash
# Train game 11 on Blackwell while regression-testing 01-10 on 5070 Ti
python train_parallel.py \
  --train game_11:gpu0 \
  --eval game_01,game_02,game_03:gpu1
```

- Spawns subprocesses with appropriate `CUDA_VISIBLE_DEVICES`
- `auto_train_config` accepts `vram_gb` param (90 for GPU:0, 14 for GPU:1)
- Collects JSON reports from each process
- 5070 Ti handles games 01-10 regression (all fit at reduced num_envs)

### 5. Curriculum Training for Game 14
**File:** `curriculum.py` (new)

Game 14 spec says standard PPO <1% win rate — must transfer from prior games.

**4-phase curriculum:**
1. **Navigation:** 01 → 02 → 03 (spatial skills)
2. **Survival:** 07 + 10 (hunger + farming)
3. **Mining+Siege:** 11 + 13 (digging + defense)
4. **Game 14:** lr=1e-4, ent_coef=0.02 for wider exploration

**Weight transfer between different-shaped games (`transfer_weights()`):**
- `conv2`, `conv3`, `shared_fc`, `critic`: always same shape → copy directly
- `conv1`: copy min(src_tags, tgt_tags) input channels, zero-init rest
- `scalar_fc`: copy min(src_props, tgt_props) columns, zero-init rest
- `actor`: copy min(src_actions, tgt_actions) rows, zero-init rest

Zero-init for new channels = no opinion on new features, but spatial skills preserved.

## Implementation Order

1. **Progress monitoring** (Part 2) — immediate value, zero risk to existing games
2. **Adaptive configs** (Part 3) — needed before training larger games
3. **Scaled network** (Part 1) — needed when building games 11-14
4. **Multi-GPU** (Part 4) — enables parallel dev workflow
5. **Curriculum** (Part 5) — only needed once games 11-13 exist

Parts 1-3 first, parts 4-5 built alongside game development.

## Verification

1. Run existing evaluator on games 01-10 — all pass unchanged (no regression)
2. `auto_train_config` for game 04 → ~4096 envs (backward compat confirmed)
3. `auto_train_config` for 32×32/512 entity → 512-1024 envs (doesn't OOM)
4. Multi-GPU: game_01 eval on GPU:1 concurrently with game_04 train on GPU:0
5. Chunked training: SPS matches monolithic (±5%)
6. New `ScaledActorCritic` trains successfully on a test game with 32×32 grid
