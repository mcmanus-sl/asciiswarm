"""PureJaxRL PPO training loop — entire graph compiled to XLA."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial
from typing import NamedTuple

from jaxswarm.network import ActorCritic


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


def _linear_schedule(train_config: TrainConfig):
    def schedule(count):
        total = train_config.num_updates * train_config.update_epochs * train_config.num_minibatches
        frac = 1.0 - count / total
        return train_config.lr * frac
    return schedule


def train(game_module, train_config: TrainConfig = TrainConfig(), *, key, win_rate_checkpoints=None):
    """
    Full PureJaxRL PPO training. Returns (trained_network, metrics_dict).
    win_rate_checkpoints: list of update indices at which to record win rate.
    """
    config = game_module.CONFIG

    if win_rate_checkpoints is None:
        win_rate_checkpoints = []

    # Initialize network
    k1, k2, k3 = jax.random.split(key, 3)
    network = ActorCritic(config, key=k1)

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

    @eqx.filter_jit
    def _train_loop(network, opt_state, env_states, env_obs, rng):
        """Single compiled training loop."""

        def _update_step(carry, _):
            network, opt_state, env_states, env_obs, rng, total_wins, total_episodes = carry

            # === Rollout phase ===
            def _env_step(carry, _):
                env_states, env_obs, rng = carry
                rng, k_act, k_reset = jax.random.split(rng, 3)

                # Get actions
                action, log_prob, value = _get_action(network, env_obs, k_act)

                # Step environments
                env_states, env_obs_new, reward, done = step_fn(env_states, action)

                # Auto-reset done environments
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
                    obs=env_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )
                return (env_states, env_obs_next, rng), transition

            (env_states, env_obs, rng), transitions = jax.lax.scan(
                _env_step, (env_states, env_obs, rng), None, length=train_config.num_steps
            )

            # Track wins/episodes from rollout
            step_wins = (transitions.done & (transitions.reward > 5.0)).sum()
            step_episodes = transitions.done.sum()
            total_wins = total_wins + step_wins
            total_episodes = total_episodes + step_episodes

            # Compute last value for GAE
            _, last_value = jax.vmap(lambda o: network(o))(env_obs)

            # Compute advantages
            advantages, returns = _compute_gae(
                transitions.reward, transitions.value, transitions.done.astype(jnp.float32), last_value
            )

            # === PPO update phase ===
            # Flatten rollout: (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
            batch_size = train_config.num_steps * train_config.num_envs
            flat_obs = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[2:]), transitions.obs)
            flat_actions = transitions.action.reshape(batch_size)
            flat_log_probs = transitions.log_prob.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)

            def _epoch(carry, _):
                network, opt_state, rng = carry
                rng, k_perm = jax.random.split(rng)

                # Shuffle and split into minibatches
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
                      jnp.int32(0), jnp.int32(0))
        (network, opt_state, env_states, env_obs, rng, total_wins, total_episodes), _ = jax.lax.scan(
            _update_step, init_carry, None, length=train_config.num_updates
        )

        win_rate = jnp.where(total_episodes > 0, total_wins / total_episodes, 0.0)

        return network, {
            "total_wins": total_wins,
            "total_episodes": total_episodes,
            "win_rate": win_rate,
        }

    return _train_loop(network, opt_state, env_states, env_obs, k3)
