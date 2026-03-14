"""Behavioral Compiler — RL-TDD evaluator with 5-layer diagnostics."""

import sys
import json
import importlib
import time

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxswarm.train import train, TrainConfig


def run_deterministic_trace(game_module):
    """Layer 1: Execute hardcoded action sequence, assert win."""
    trace = game_module.DETERMINISTIC_TRACE
    key = jax.random.PRNGKey(0)
    state, obs = game_module.reset(key)

    def scan_step(state, action):
        state, obs, reward, done = game_module.step(state, jnp.int32(action))
        return state, (reward, done, state.status)

    scan_step_jit = jax.jit(scan_step)
    actions = jnp.array(trace, dtype=jnp.int32)

    final_state, (rewards, dones, statuses) = jax.lax.scan(
        scan_step_jit, state, actions
    )

    # Find first done
    done_mask = dones.astype(jnp.int32)
    any_done = done_mask.any()
    first_done_idx = jnp.argmax(done_mask)
    final_status = jnp.where(any_done, statuses[first_done_idx], final_state.status)

    passed = bool(final_status == 1)
    turns = int(jnp.where(any_done, first_done_idx + 1, len(trace)))

    return {
        "action_sequence_length": len(trace),
        "status": int(final_status),
        "turns": turns,
        "pass": passed,
    }


def run_random_agent(game_module, num_episodes=1000):
    """Layer 2: Random agent fuzz test."""
    config = game_module.CONFIG

    def run_episode(key):
        k_reset, k_actions = jax.random.split(key)
        state, obs = game_module.reset(k_reset)

        def step_fn(carry, action_key):
            state, obs, cumulative_reward, done_flag = carry
            action = jax.random.randint(action_key, (), 0, config.num_actions)
            new_state, new_obs, reward, done = game_module.step(state, action)
            # If already done, don't update (done_flag is scalar per episode)
            state = jax.tree.map(
                lambda n, o: jnp.where(done_flag, o, n), new_state, state
            )
            obs = jax.tree.map(
                lambda n, o: jnp.where(done_flag, o, n), new_obs, obs
            )
            cumulative_reward = jnp.where(done_flag, cumulative_reward, cumulative_reward + reward)
            done_flag = done_flag | done
            return (state, obs, cumulative_reward, done_flag), None

        action_keys = jax.random.split(k_actions, config.max_turns)
        init_carry = (state, obs, jnp.float32(0.0), jnp.bool_(False))
        (final_state, _, cumulative_reward, terminated), _ = jax.lax.scan(
            step_fn, init_carry, action_keys
        )

        won = final_state.status == 1
        has_nan = jnp.isnan(cumulative_reward)
        ep_length = final_state.turn_number

        return terminated, won, has_nan, ep_length, cumulative_reward

    keys = jax.random.split(jax.random.PRNGKey(42), num_episodes)
    run_batch = jax.jit(jax.vmap(run_episode))
    terminated, won, has_nan, ep_lengths, rewards = run_batch(keys)

    nan_count = int(has_nan.sum())
    term_rate = float(terminated.mean())
    win_rate = float(won.mean())
    avg_length = float(ep_lengths.mean())

    return {
        "num_episodes": num_episodes,
        "nan_count": nan_count,
        "termination_rate": round(term_rate, 4),
        "win_rate": round(win_rate, 4),
        "avg_length": round(avg_length, 1),
        "pass": nan_count == 0 and term_rate > 0.95,
    }


def run_ppo_training(game_module, key):
    """Layer 3: PPO training with win rate tracking."""
    # We'll train in phases to get win rates at different checkpoints
    # Phase 1: ~10k steps
    steps_per_update = 128 * 4096  # num_steps * num_envs
    updates_10k = max(1, 10_000 // (128))  # ~78 updates for 10k env-steps per env
    updates_50k = max(1, 50_000 // (128))
    updates_100k = max(1, 100_000 // (128))

    # Single training run with enough updates
    tc = TrainConfig(
        num_envs=4096,
        num_steps=128,
        num_updates=updates_100k,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        num_minibatches=4,
        update_epochs=4,
    )

    print(f"  Training PPO: {tc.num_updates} updates × {tc.num_steps} steps × {tc.num_envs} envs")
    print(f"  Total env steps: {tc.num_updates * tc.num_steps * tc.num_envs:,}")

    t0 = time.time()
    network, metrics = train(game_module, tc, key=key)
    elapsed = time.time() - t0

    win_rate = float(metrics["win_rate"])
    total_episodes = int(metrics["total_episodes"])
    total_wins = int(metrics["total_wins"])

    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Win rate: {win_rate:.4f} ({total_wins}/{total_episodes} episodes)")

    return network, {
        "win_rate": round(win_rate, 4),
        "total_episodes": total_episodes,
        "total_wins": total_wins,
        "training_time_s": round(elapsed, 1),
        "pass": win_rate > 0.1,
    }


def run_forensics(game_module, network, num_episodes=100):
    """Layer 4: Trajectory forensics — run trained policy, extract best trajectory."""
    config = game_module.CONFIG

    def run_eval_episode(key):
        k_reset, k_act = jax.random.split(key)
        state, obs = game_module.reset(k_reset)

        def step_fn(carry, rng):
            state, obs, done_flag, cumulative_reward = carry
            logits, _ = network(obs)
            action = jax.random.categorical(rng, logits)

            new_state, new_obs, reward, done = game_module.step(state, action)
            state = jax.tree.map(
                lambda n, o: jnp.where(done_flag, o, n), new_state, state
            )
            obs = jax.tree.map(
                lambda n, o: jnp.where(done_flag, o, n), new_obs, obs
            )
            cumulative_reward = jnp.where(done_flag, cumulative_reward, cumulative_reward + reward)
            saved_action = jnp.where(done_flag, jnp.int32(-1), action)
            done_flag = done_flag | done
            return (state, obs, done_flag, cumulative_reward), saved_action

        action_keys = jax.random.split(k_act, config.max_turns)
        init_carry = (state, obs, jnp.bool_(False), jnp.float32(0.0))
        (final_state, _, _, cumulative_reward), actions = jax.lax.scan(
            step_fn, init_carry, action_keys
        )

        return cumulative_reward, actions, final_state.status, final_state.turn_number

    keys = jax.random.split(jax.random.PRNGKey(99), num_episodes)
    run_batch = jax.jit(jax.vmap(run_eval_episode))
    rewards, all_actions, statuses, turn_numbers = run_batch(keys)

    # Find best episode
    best_idx = int(jnp.argmax(rewards))
    best_actions = all_actions[best_idx].tolist()
    best_turns = int(turn_numbers[best_idx])
    best_status = int(statuses[best_idx])

    # Trim to actual length (remove -1 padding)
    trimmed = [a for a in best_actions if a >= 0]

    # Map to action names
    action_names = game_module.ACTION_NAMES
    named_actions = [action_names[a] if a < len(action_names) else f"action_{a}" for a in trimmed]

    # Compress repeated actions
    compressed = []
    if named_actions:
        current = named_actions[0]
        count = 1
        for a in named_actions[1:]:
            if a == current:
                count += 1
            else:
                compressed.append(f"{current} x{count}" if count > 1 else current)
                current = a
                count = 1
        compressed.append(f"{current} x{count}" if count > 1 else current)

    status_str = {1: "WON", -1: "LOST", 0: "TIMEOUT"}.get(best_status, f"status={best_status}")
    trajectory_str = ", ".join(compressed) + f" → {status_str} at turn {best_turns}"

    # Win stats
    wins = statuses == 1
    eval_win_rate = float(wins.mean())
    avg_turns_to_win = float(jnp.where(wins, turn_numbers, 0).sum() / jnp.maximum(wins.sum(), 1))

    return {
        "eval_win_rate": round(eval_win_rate, 4),
        "top_trajectory": trajectory_str,
        "avg_turns_to_win": round(avg_turns_to_win, 1),
        "best_reward": round(float(rewards[best_idx]), 2),
        "num_eval_episodes": num_episodes,
    }


def run_invariants(game_module):
    """Layer 5: Game-specific structural checks."""
    key = jax.random.PRNGKey(0)
    state, _ = game_module.reset(key)
    checks = game_module.invariant_checks(state)

    passed = sum(1 for _, v in checks if v)
    failed = [name for name, v in checks if not v]

    return {
        "total": len(checks),
        "passed": passed,
        "failed": failed,
        "pass": len(failed) == 0,
    }


def evaluate(module_path: str):
    """Run all 5 evaluation layers on a game module."""
    print(f"=== Evaluating: {module_path} ===\n")

    game_module = importlib.import_module(module_path)

    report = {"game": module_path}

    # Layer 1: Deterministic Trace
    print("[Layer 1] Deterministic Trace...")
    t0 = time.time()
    report["deterministic_trace"] = run_deterministic_trace(game_module)
    print(f"  Status: {report['deterministic_trace']['status']}, "
          f"Turns: {report['deterministic_trace']['turns']}, "
          f"Pass: {report['deterministic_trace']['pass']} "
          f"({time.time()-t0:.1f}s)\n")

    # Layer 2: Random Agent
    print("[Layer 2] Random Agent (1000 episodes)...")
    t0 = time.time()
    report["random_agent"] = run_random_agent(game_module)
    print(f"  NaN: {report['random_agent']['nan_count']}, "
          f"Term: {report['random_agent']['termination_rate']}, "
          f"Win: {report['random_agent']['win_rate']}, "
          f"Avg Length: {report['random_agent']['avg_length']} "
          f"({time.time()-t0:.1f}s)\n")

    # Layer 3: PPO Training
    print("[Layer 3] PPO Training...")
    t0 = time.time()
    key = jax.random.PRNGKey(0)
    network, ppo_results = run_ppo_training(game_module, key)
    report["ppo"] = ppo_results
    print(f"  ({time.time()-t0:.1f}s)\n")

    # Layer 4: Trajectory Forensics
    print("[Layer 4] Trajectory Forensics (100 episodes)...")
    t0 = time.time()
    report["forensics"] = run_forensics(game_module, network)
    print(f"  Top: {report['forensics']['top_trajectory']}")
    print(f"  Eval Win Rate: {report['forensics']['eval_win_rate']} "
          f"({time.time()-t0:.1f}s)\n")

    # Layer 5: Invariant Checks
    print("[Layer 5] Invariant Checks...")
    t0 = time.time()
    report["invariants"] = run_invariants(game_module)
    print(f"  {report['invariants']['passed']}/{report['invariants']['total']} passed "
          f"({time.time()-t0:.1f}s)\n")

    # Overall
    report["overall_pass"] = all([
        report["deterministic_trace"]["pass"],
        report["random_agent"]["pass"],
        report["ppo"]["pass"],
        report["invariants"]["pass"],
    ])

    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60)

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_game.py <module_path>")
        print("Example: python evaluate_game.py jaxswarm.games.game_01_empty_exit")
        sys.exit(1)

    report = evaluate(sys.argv[1])
    sys.exit(0 if report["overall_pass"] else 1)
