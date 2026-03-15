# JAXBIBLE: JAX Performance Rules for asciiswarm

Hard-won lessons from building a JAX game engine + PureJaxRL training pipeline. Violating these rules causes 3-10x slowdowns or silent JIT hell.

---

## Rule 1: Never `rebuild_grid` in `step()`

**The mistake:** Calling `rebuild_grid(state, CONFIG)` at the end of every `step()`. This runs a `jax.lax.scan` over `max_entities` to reconstruct the grid tensor from scratch — every single environment step, across all 4096 parallel envs.

**The fix:** Remove it. `get_obs()` builds observations directly from entity arrays (`state.x`, `state.y`, `state.tags`, `state.alive`) via a single vectorized scatter-add. It never reads the grid tensor.

**Impact:** Game 08 went from **287K SPS to 1.09M SPS** — a **3.8x speedup** — just by deleting one line.

**When `rebuild_grid` IS needed:** Only in `reset()`, where you've bulk-initialized entity arrays and need the grid tensor consistent for any `get_entities_at` / `move_entity` calls during the first step.

---

## Rule 2: Zero Python loops inside JIT

Everything inside `@eqx.filter_jit` (or called from it) must use JAX control flow:

| Python pattern | JAX replacement | When to use |
|---|---|---|
| `for i in range(n):` | `jax.lax.scan(f, init, xs, length=n)` | Accumulating state over iterations |
| `for i in range(n):` | `jax.lax.fori_loop(0, n, body, init)` | Mutating a carry without collecting outputs |
| `if x == 0: ... elif x == 1:` | `jax.lax.switch(x, [fn0, fn1, ...], args)` | Dispatch on a traced integer |
| `if cond:` | `jnp.where(cond, true_val, false_val)` | Branchless conditional assignment |
| `while not done:` | `jax.lax.scan` with a flag in carry | Bounded iteration with early-exit semantics |

A Python `for` loop unrolls at trace time — 781 iterations becomes 781x the graph. `jax.lax.scan` compiles to a single loop node.

**Our training loop:** 4 nested scans — updates → rollout steps → PPO epochs → minibatches. One `@eqx.filter_jit` compilation, ~15s compile, then pure execution.

---

## Rule 3: Vectorize entity queries — avoid per-entity loops where possible

**The slow way (per-entity fori_loop):**
```python
# Check each stack slot at target cell
has_solid = jax.lax.fori_loop(0, config.max_stack, check_solid, False)
```

**The fast way (vectorized boolean mask):**
```python
at_target = state.alive & (state.x == tx) & (state.y == ty)
has_solid = (at_target & state.tags[:, 1]).any()
```

The vectorized version broadcasts over all entities in one op. No loop, no scan.

**When fori_loop IS correct:** When you need sequential semantics (entity A's move affects entity B's move), or when iterating over grid stack slots where `max_stack` is small (2-3).

---

## Rule 4: `get_obs()` is fully vectorized — don't rebuild state for it

The observation builder scatters entity tags onto a spatial grid in one operation:

```python
obs_grid = obs_grid.at[tags, ys, xs].add(active_tags)  # single scatter-add
```

It reads `state.x`, `state.y`, `state.tags`, `state.alive` directly. It does NOT read `state.grid`. So any step function that ends with `rebuild_grid` just to feed `get_obs` is doing pure waste.

---

## Rule 5: Branchless conditionals via `jnp.where` + `jax.tree.map`

JAX traces both branches. Use `jnp.where` for scalars/arrays, `jax.tree.map` for full state updates:

```python
# Apply new_state only if condition is true
state = jax.tree.map(
    lambda n, o: jnp.where(condition, n, o), new_state, state
)
```

This is how `move_entity`, `slide_player`, and behavior dispatch all work — compute the result unconditionally, then gate it.

---

## Rule 6: Bounded loops with carry flags for early exit

JAX has no `while` with dynamic termination inside scan. Simulate it:

```python
def slide_step(carry, _):
    state, still_sliding = carry
    can_move = still_sliding & in_bounds & ~blocked
    # ... do move ...
    still_sliding = can_move & ~on_exit  # stop condition
    return (state, still_sliding), None

jax.lax.scan(slide_step, (state, True), None, length=max_grid_dim)
```

The scan always runs `length` iterations, but the flag gates all effects after termination. Cost: wasted compute on no-op iterations, but no recompilation.

---

## Rule 7: `PYTHONUNBUFFERED=1` for training output

JAX training with `jax.lax.scan` produces no Python-side output during execution. Without unbuffered stdout, you'll think it's stuck compiling when it's actually training:

```bash
PYTHONUNBUFFERED=1 python evaluate_game.py jaxswarm.games.game_07_hunger_clock
```

**Diagnosis trick:** Run 2-update compile test vs 10-update test. If the time difference is linear in updates, it's execution time, not compile time.

---

## Rule 8: Compile time is fixed, execution time scales linearly

Our pipeline: ~15s compile (one-time) + ~1.5s per update cycle (execution).

- 2 updates: 15s + 3s = **18s**
- 10 updates: 15s + 15s = **30s**
- 781 updates: 15s + 1170s = **~20 min**

If your full run takes 20 min, that's not JIT hell — that's 409M environment steps at real throughput. Measure SPS to know.

---

## Performance Reference

| Game | SPS (before fix) | SPS (after fix) | What changed |
|---|---|---|---|
| Game 07 | 287K | 287K | (no rebuild_grid in step — already clean) |
| Game 08 | ~287K | 1,085K | Removed `rebuild_grid` from step() |
| Games 01-06 | ~900K-1.1M | same | Were already clean |

**Target SPS:** >500K on RTX PRO 6000 Blackwell for simple games. Complex games (many entities, farming growth) may be lower.

---

## Rule 9: Per-step wave/distance fields are expensive but sometimes necessary

`compute_distance_field` runs a `jax.lax.scan` of `H + W` iterations, each doing 4 pad+min ops on the full grid. On a 16x16 grid that's 32 iterations of tensor ops **per step per env**.

This is acceptable for reward shaping (Game 09) but understand the cost:
- Simple games (no wave): ~1M SPS
- Wave-per-step games: expect ~200-400K SPS

If the wave target doesn't change every step, consider caching the distance field in `game_state` and only recomputing when the quest phase changes. But for games where the topology changes (rubble mined, doors opened), you must recompute.

---

## Rule 10: Python for-loops in `reset()` are fine

Small Python for-loops in `reset()` (e.g., `for i in range(5): x = x.at[i].set(...)`) unroll at trace time into a flat sequence of `.at[].set()` ops. This is fine because:
- `reset()` is called once per episode reset, not per step
- The unrolled ops are simple index assignments
- The loop count is a small constant (5-10), not `max_entities`

Don't waste time converting these to `jax.lax.fori_loop` — the trace overhead is negligible.
