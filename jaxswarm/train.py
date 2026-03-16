"""PureJaxRL PPO training loop — entire graph compiled to XLA."""

import os
import math
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial
from typing import NamedTuple

from jaxswarm.network import ActorCritic, select_network


class Transition(NamedTuple):
    obs: dict
    action: jnp.int32
    reward: jnp.float32
    done: jnp.bool_
    log_prob: jnp.float32
    value: jnp.float32


class TrainConfig(NamedTuple):
    num_envs: int = 4096
    num_steps: int = 128        # rollout length per update
    num_updates: int = 100      # total PPO updates (total steps = num_updates * num_steps * num_envs)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4


def auto_train_config(config, available_vram_gb=None, **overrides):
    """Compute VRAM-aware TrainConfig from a game's EnvConfig.

    Estimates per-env memory footprint from grid size, entity count, tags, and props,
    then picks the largest power-of-2 num_envs that fits in 60% of available VRAM.
    Compensates fewer envs with more updates to maintain ~400M total env steps.
    """
    if available_vram_gb is None:
        try:
            props = jax.devices()[0].client.device_memory_stats(jax.devices()[0])
            available_vram_gb = props['bytes_limit'] / (1024 ** 3)
        except Exception:
            available_vram_gb = 24.0  # conservative fallback

    # Per-env-per-step memory estimate (bytes)
    # Observation grid: num_tags * grid_h * grid_w * 4 bytes (float32)
    obs_bytes = config.num_tags * config.grid_h * config.grid_w * 4
    # Scalar obs: (3 + num_props) * 4
    scalar_bytes = (3 + config.num_props) * 4
    # State arrays: alive + entity_type + x + y = 4 * max_entities * 4
    #   + tags: max_entities * num_tags * 1 (bool)
    #   + properties: max_entities * num_props * 4
    #   + grid: grid_h * grid_w * max_stack * 4
    #   + misc scalars ~64 bytes
    state_bytes = (
        config.max_entities * (16 + config.num_tags + config.num_props * 4)
        + config.grid_h * config.grid_w * getattr(config, 'max_stack', 3) * 4
        + 64
    )
    # Total per env: state + rollout buffer (obs + action + reward + done + log_prob + value per step)
    rollout_per_step = obs_bytes + scalar_bytes + 4 + 4 + 1 + 4 + 4  # per step per env

    num_steps = 128
    per_env_bytes = state_bytes + rollout_per_step * num_steps
    # Network + optimizer roughly 2x network params; estimate network size from config
    # For small games this is <10MB, for large games ~50MB — negligible vs env memory

    budget_bytes = available_vram_gb * (1024 ** 3) * 0.60
    max_envs = int(budget_bytes / per_env_bytes)

    # Round down to power of 2, clamp [256, 4096]
    num_envs = 2 ** int(math.log2(max(max_envs, 1)))
    num_envs = max(256, min(4096, num_envs))

    # Reduce num_steps for very large games if num_envs is already at minimum
    if num_envs == 256 and per_env_bytes > budget_bytes / 256:
        num_steps = 64

    # Compensate fewer envs with more updates to maintain ~400M total env steps
    target_total_steps = 400_000_000
    num_updates = max(50, target_total_steps // (num_envs * num_steps))

    tc_kwargs = dict(
        num_envs=num_envs,
        num_steps=num_steps,
        num_updates=num_updates,
    )
    tc_kwargs.update(overrides)
    return TrainConfig(**tc_kwargs)


def _linear_schedule(train_config: TrainConfig):
    def schedule(count):
        total = train_config.num_updates * train_config.update_epochs * train_config.num_minibatches
        frac = 1.0 - count / total
        return train_config.lr * frac
    return schedule


def train(game_module, train_config: TrainConfig = TrainConfig(), *, key,
          network_cls=None, win_rate_checkpoints=None, chunk_size=50):
    """
    Full PureJaxRL PPO training with chunked progress output.
    Returns (trained_network, metrics_dict).

    chunk_size: number of PPO updates per compiled chunk (progress printed between chunks).
    """
    config = game_module.CONFIG

    if win_rate_checkpoints is None:
        win_rate_checkpoints = []

    # Select network class
    if network_cls is None:
        network_cls = select_network(config)

    # Initialize network
    k1, k2, k3 = jax.random.split(key, 3)
    network = network_cls(config, key=k1)

    # Optimizer with linear LR schedule
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adam(learning_rate=_linear_schedule(train_config)),
    )
    opt_state = tx.init(eqx.filter(network, eqx.is_array))

    # Initialize environments
    env_keys = jax.random.split(k2, train_config.num_envs)
    reset_fn = jax.vmap(game_module.reset)
    env_states, env_obs = reset_fn(env_keys)

    # Step function (vmapped)
    step_fn = jax.vmap(game_module.step)

    def _expand_done(done, target):
        """Expand done (num_envs,) to broadcast with target shape."""
        shape = (done.shape[0],) + (1,) * (target.ndim - 1)
        return done.reshape(shape)

    def _get_action(network, obs, rng):
        logits, value = jax.vmap(lambda o: network(o))(obs)
        action = jax.random.categorical(rng, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), action]
        return action, log_prob, value

    def _compute_gae(rewards, values, dones, last_value):
        """GAE advantage computation via reverse scan."""
        def _body(carry, t):
            gae, next_value = carry
            done, value, reward = t
            delta = reward + train_config.gamma * next_value * (1 - done) - value
            gae = delta + train_config.gamma * train_config.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _body,
            (jnp.zeros(rewards.shape[1]), last_value),
            (dones, values, rewards),
            reverse=True,
        )
        returns = advantages + values
        return advantages, returns

    def _ppo_loss(network, batch, rng):
        obs, actions, old_log_probs, advantages, returns = batch

        logits, values = jax.vmap(lambda o: network(o))(obs)
        log_probs_all = jax.nn.log_softmax(logits)
        new_log_probs = log_probs_all[jnp.arange(logits.shape[0]), actions]

        # Policy loss (clipped)
        ratio = jnp.exp(new_log_probs - old_log_probs)
        adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss1 = -ratio * adv_normalized
        loss2 = -jnp.clip(ratio, 1 - train_config.clip_eps, 1 + train_config.clip_eps) * adv_normalized
        policy_loss = jnp.maximum(loss1, loss2).mean()

        # Value loss
        value_loss = ((values - returns) ** 2).mean()

        # Entropy bonus
        probs = jax.nn.softmax(logits)
        entropy = -(probs * log_probs_all).sum(-1).mean()

        total_loss = policy_loss + train_config.vf_coef * value_loss - train_config.ent_coef * entropy
        return total_loss, (policy_loss, value_loss, entropy)

    def _make_train_chunk(chunk_len):
        """Create a jitted chunk function with static scan length."""
        @eqx.filter_jit
        def _train_chunk(network, opt_state, env_states, env_obs, rng,
                         total_wins, total_episodes):

            def _update_step(carry, _):
                network, opt_state, env_states, env_obs, rng, total_wins, total_episodes = carry

                # === Rollout phase ===
                def _env_step(carry, _):
                    env_states, env_obs, rng = carry
                    rng, k_act, k_reset = jax.random.split(rng, 3)

                    action, log_prob, value = _get_action(network, env_obs, k_act)
                    env_states, env_obs_new, reward, done = step_fn(env_states, action)

                    reset_keys = jax.random.split(k_reset, train_config.num_envs)
                    fresh_states, fresh_obs = reset_fn(reset_keys)
                    env_states = jax.tree.map(
                        lambda f, o: jnp.where(_expand_done(done, f), f, o),
                        fresh_states, env_states,
                    )
                    env_obs_next = jax.tree.map(
                        lambda f, o: jnp.where(_expand_done(done, f), f, o),
                        fresh_obs, env_obs_new,
                    )

                    transition = Transition(
                        obs=env_obs, action=action, reward=reward,
                        done=done, log_prob=log_prob, value=value,
                    )
                    return (env_states, env_obs_next, rng), transition

                (env_states, env_obs, rng), transitions = jax.lax.scan(
                    _env_step, (env_states, env_obs, rng), None, length=train_config.num_steps
                )

                step_wins = (transitions.done & (transitions.reward > 5.0)).sum()
                step_episodes = transitions.done.sum()
                total_wins = total_wins + step_wins
                total_episodes = total_episodes + step_episodes

                _, last_value = jax.vmap(lambda o: network(o))(env_obs)
                advantages, returns = _compute_gae(
                    transitions.reward, transitions.value,
                    transitions.done.astype(jnp.float32), last_value
                )

                # === PPO update phase ===
                batch_size = train_config.num_steps * train_config.num_envs
                flat_obs = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[2:]), transitions.obs)
                flat_actions = transitions.action.reshape(batch_size)
                flat_log_probs = transitions.log_prob.reshape(batch_size)
                flat_advantages = advantages.reshape(batch_size)
                flat_returns = returns.reshape(batch_size)

                def _epoch(carry, _):
                    network, opt_state, rng = carry
                    rng, k_perm = jax.random.split(rng)
                    perm = jax.random.permutation(k_perm, batch_size)
                    minibatch_size = batch_size // train_config.num_minibatches

                    def _minibatch(carry, start_idx):
                        network, opt_state = carry
                        idx = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))
                        mb_obs = jax.tree.map(lambda x: x[idx], flat_obs)
                        mb_actions = flat_actions[idx]
                        mb_log_probs = flat_log_probs[idx]
                        mb_advantages = flat_advantages[idx]
                        mb_returns = flat_returns[idx]
                        batch = (mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_returns)

                        grads, _ = eqx.filter_grad(_ppo_loss, has_aux=True)(network, batch, rng)
                        updates, opt_state_new = tx.update(
                            eqx.filter(grads, eqx.is_array),
                            opt_state,
                            eqx.filter(network, eqx.is_array),
                        )
                        network = eqx.apply_updates(network, updates)
                        return (network, opt_state_new), None

                    starts = jnp.arange(train_config.num_minibatches) * minibatch_size
                    (network, opt_state), _ = jax.lax.scan(_minibatch, (network, opt_state), starts)
                    return (network, opt_state, rng), None

                (network, opt_state, rng), _ = jax.lax.scan(
                    _epoch, (network, opt_state, rng), None, length=train_config.update_epochs
                )

                return (network, opt_state, env_states, env_obs, rng, total_wins, total_episodes), None

            init_carry = (network, opt_state, env_states, env_obs, rng,
                          total_wins, total_episodes)
            (network, opt_state, env_states, env_obs, rng, total_wins, total_episodes), _ = jax.lax.scan(
                _update_step, init_carry, None, length=chunk_len
            )

            win_rate = jnp.where(total_episodes > 0, total_wins / total_episodes, 0.0)
            return network, opt_state, env_states, env_obs, rng, total_wins, total_episodes, win_rate

        return _train_chunk

    # === Chunked training with progress output ===
    num_updates = train_config.num_updates
    # Clamp chunk_size so we don't exceed num_updates
    chunk_size = min(chunk_size, num_updates)
    num_chunks = (num_updates + chunk_size - 1) // chunk_size

    # Pre-build jitted chunk functions (at most 2: full chunk + remainder)
    chunk_fns = {}

    total_wins = jnp.int32(0)
    total_episodes = jnp.int32(0)
    rng = k3
    t0 = time.time()
    steps_so_far = 0

    for chunk_idx in range(num_chunks):
        updates_remaining = num_updates - chunk_idx * chunk_size
        this_chunk = min(chunk_size, updates_remaining)

        if this_chunk not in chunk_fns:
            chunk_fns[this_chunk] = _make_train_chunk(this_chunk)

        chunk_t0 = time.time()
        (network, opt_state, env_states, env_obs, rng,
         total_wins, total_episodes, win_rate) = chunk_fns[this_chunk](
            network, opt_state, env_states, env_obs, rng,
            total_wins, total_episodes
        )
        jax.block_until_ready((network, total_wins))
        chunk_elapsed = time.time() - chunk_t0

        chunk_steps = this_chunk * train_config.num_steps * train_config.num_envs
        steps_so_far += chunk_steps
        chunk_sps = chunk_steps / max(chunk_elapsed, 1e-6)
        elapsed_total = time.time() - t0

        print(f"  [{chunk_idx+1}/{num_chunks}] SPS: {chunk_sps:,.0f}  "
              f"Win: {float(win_rate):.4f}  ({elapsed_total:.1f}s)")

    elapsed = time.time() - t0
    total_steps = num_updates * train_config.num_steps * train_config.num_envs
    sps = total_steps / max(elapsed, 1e-6)
    print(f"  Global SPS: {sps:,.0f} steps/sec ({total_steps:,} steps in {elapsed:.1f}s)")

    metrics = {
        "total_wins": total_wins,
        "total_episodes": total_episodes,
        "win_rate": win_rate,
    }
    return network, metrics
