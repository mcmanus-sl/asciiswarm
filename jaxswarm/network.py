"""Equinox ActorCritic network for PureJaxRL PPO."""

import jax
import jax.numpy as jnp
import equinox as eqx


class ActorCritic(eqx.Module):
    # CNN for grid observation
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    # MLP for scalar observation
    scalar_fc: eqx.nn.Linear

    # Shared trunk
    shared_fc: eqx.nn.Linear

    # Heads
    actor: eqx.nn.Linear
    critic: eqx.nn.Linear

    def __init__(self, config, *, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        # CNN: num_tags channels -> 16 -> 32
        self.conv1 = eqx.nn.Conv2d(config.num_tags, 16, kernel_size=3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(16, 32, kernel_size=3, padding=1, key=k2)

        # CNN output: 32 * grid_h * grid_w
        cnn_out_size = 32 * config.grid_h * config.grid_w

        # Scalar MLP
        scalar_in_size = 3 + config.num_props
        self.scalar_fc = eqx.nn.Linear(scalar_in_size, 32, key=k3)

        # Shared trunk
        self.shared_fc = eqx.nn.Linear(cnn_out_size + 32, 64, key=k4)

        # Actor and critic heads
        self.actor = eqx.nn.Linear(64, config.num_actions, key=k5)
        self.critic = eqx.nn.Linear(64, 1, key=k6)

    def __call__(self, obs):
        """obs: dict with 'grid' (num_tags, H, W) and 'scalars' (N,)."""
        # CNN path
        x = obs['grid']  # (C, H, W) — already channel-first
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = x.ravel()  # flatten

        # Scalar path
        s = jax.nn.relu(self.scalar_fc(obs['scalars']))

        # Combine
        combined = jnp.concatenate([x, s])
        hidden = jax.nn.relu(self.shared_fc(combined))

        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)

        return logits, value
