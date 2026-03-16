"""Multi-GPU parallel training — train on Blackwell, regression on 5070 Ti."""

import os
import sys
import argparse
import importlib
import time

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxswarm.train import train, auto_train_config, TrainConfig
from jaxswarm.network import select_network


def train_on_device(game_module, device_idx=0, **train_overrides):
    """Train a game on a specific GPU device."""
    device = jax.devices()[device_idx]
    game_name = game_module.__name__.split('.')[-1]

    print(f"[GPU {device_idx}] Training {game_name} on {device}")

    config = game_module.CONFIG

    # Get VRAM for this device
    try:
        props = device.client.device_memory_stats(device)
        vram_gb = props['bytes_limit'] / (1024 ** 3)
    except Exception:
        vram_gb = 24.0

    tc = auto_train_config(config, vram_gb, **train_overrides)
    print(f"[GPU {device_idx}] Config: {tc.num_envs} envs, {tc.num_updates} updates")

    key = jax.random.PRNGKey(0)

    with jax.default_device(device):
        network, metrics = train(game_module, tc, key=key)

    win_rate = float(metrics["win_rate"])
    print(f"[GPU {device_idx}] {game_name} done — win rate: {win_rate:.4f}")

    return network, metrics


def regression_test(game_module, network, device_idx=0, num_episodes=100):
    """Run evaluation episodes on a specific GPU (e.g., 5070 Ti for regression)."""
    device = jax.devices()[device_idx]
    config = game_module.CONFIG

    print(f"[GPU {device_idx}] Regression: {num_episodes} episodes")

    with jax.default_device(device):
        def run_episode(key):
            k_reset, k_act = jax.random.split(key)
            state, obs = game_module.reset(k_reset)

            def step_fn(carry, rng):
                state, done_flag, reward_acc = carry
                from jaxswarm.core.obs import get_obs
                obs = get_obs(state, config)
                logits, _ = network(obs)
                action = jax.random.categorical(rng, logits)
                new_state, _, reward, done = game_module.step(state, action)
                state = jax.tree.map(
                    lambda n, o: jnp.where(done_flag, o, n), new_state, state
                )
                reward_acc = jnp.where(done_flag, reward_acc, reward_acc + reward)
                done_flag = done_flag | done
                return (state, done_flag, reward_acc), None

            action_keys = jax.random.split(k_act, config.max_turns)
            (final_state, _, total_reward), _ = jax.lax.scan(
                step_fn, (state, jnp.bool_(False), jnp.float32(0.0)), action_keys
            )
            return final_state.status == 1

        keys = jax.random.split(jax.random.PRNGKey(42), num_episodes)
        wins = jax.jit(jax.vmap(run_episode))(keys)

    win_rate = float(wins.mean())
    print(f"[GPU {device_idx}] Regression win rate: {win_rate:.4f}")
    return win_rate


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU training")
    parser.add_argument("game", help="Game module path (e.g., jaxswarm.games.game_14_inf_fortress)")
    parser.add_argument("--train-gpu", type=int, default=0, help="GPU index for training")
    parser.add_argument("--test-gpu", type=int, default=None, help="GPU index for regression testing")
    parser.add_argument("--num-updates", type=int, default=None, help="Override num_updates")
    parser.add_argument("--save-weights", action="store_true", help="Save weights after training")
    args = parser.parse_args()

    game_module = importlib.import_module(args.game)
    game_name = args.game.split('.')[-1]

    # Training
    overrides = {}
    if args.num_updates:
        overrides['num_updates'] = args.num_updates

    network, metrics = train_on_device(game_module, args.train_gpu, **overrides)

    # Optional regression on second GPU
    if args.test_gpu is not None:
        regression_test(game_module, network, args.test_gpu)

    # Save weights
    if args.save_weights:
        os.makedirs("weights", exist_ok=True)
        path = f"weights/{game_name}.eqx"
        eqx.tree_serialise_leaves(path, network)
        print(f"Weights saved: {path}")


if __name__ == "__main__":
    main()
