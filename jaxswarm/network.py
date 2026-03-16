"""Equinox ActorCritic networks for PureJaxRL PPO."""

import jax
import jax.numpy as jnp
import equinox as eqx


# Padded grid size for ScaledActorCritic — all grids zero-padded to this
SCALED_GRID_SIZE = 32


def select_network(config):
    """Return the appropriate network class for a game config."""
    if config.grid_h > 16 or config.grid_w > 16 or config.max_entities > 128:
        return ScaledActorCritic
    return ActorCritic


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


class ScaledActorCritic(eqx.Module):
    """Larger network for games 11-14. Zero-pads grid to 32x32, uses strided convs."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    scalar_fc: eqx.nn.Linear

    shared_fc: eqx.nn.Linear

    actor: eqx.nn.Linear
    critic: eqx.nn.Linear

    def __init__(self, config, *, key):
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        # 3 conv layers: tags→32 (stride 1), 32→64 (stride 2), 64→64 (stride 2)
        # On 32x32 input: 32x32 → 32x32 → 16x16 → 8x8
        # Flatten: 64 * 8 * 8 = 4096
        # Then add one more stride-2 to get to 4x4: 64 * 4 * 4 = 1024
        self.conv1 = eqx.nn.Conv2d(config.num_tags, 32, kernel_size=3, stride=1, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, key=k2)
        self.conv3 = eqx.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, key=k3)
        # 32x32 → 32x32 → 16x16 → 8x8 → flatten = 64*8*8 = 4096
        cnn_out_size = 64 * 8 * 8

        # Scalar MLP (same structure, same width)
        scalar_in_size = 3 + config.num_props
        self.scalar_fc = eqx.nn.Linear(scalar_in_size, 32, key=k4)

        # Wider shared trunk
        self.shared_fc = eqx.nn.Linear(cnn_out_size + 32, 128, key=k5)

        # Actor and critic heads
        self.actor = eqx.nn.Linear(128, config.num_actions, key=k6)
        self.critic = eqx.nn.Linear(128, 1, key=k7)

    def __call__(self, obs):
        """obs: dict with 'grid' (num_tags, H, W) and 'scalars' (N,)."""
        x = obs['grid']  # (C, H, W)

        # Zero-pad to SCALED_GRID_SIZE x SCALED_GRID_SIZE
        _, h, w = x.shape
        pad_h = SCALED_GRID_SIZE - h
        pad_w = SCALED_GRID_SIZE - w
        x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))

        # Strided CNN
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = x.ravel()

        # Scalar path
        s = jax.nn.relu(self.scalar_fc(obs['scalars']))

        # Combine
        combined = jnp.concatenate([x, s])
        hidden = jax.nn.relu(self.shared_fc(combined))

        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)

        return logits, value
