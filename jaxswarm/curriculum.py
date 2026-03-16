"""Curriculum learning — multi-phase weight transfer across games."""

import importlib
import time

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxswarm.network import ScaledActorCritic, select_network
from jaxswarm.train import train, auto_train_config, TrainConfig


def _pad_network_for_game(network, source_config, target_config, *, key):
    """
    Transfer weights from a network trained on source_config to one for target_config.

    Handles mismatched observation/action dimensions by:
    - Keeping shared conv layers (zero-pad grid channels if needed)
    - Keeping shared scalar layers (zero-pad input if needed)
    - Re-initializing actor head if action count changes
    - Keeping critic head (always 1 output)

    Both networks must be the same class (ScaledActorCritic for curriculum).
    """
    # Create fresh target network with correct dimensions
    target_network = ScaledActorCritic(target_config, key=key)

    # Transfer conv layers — if input channels differ, zero-pad the first conv
    source_conv1_w = network.conv1.weight  # [out_c, in_c, kH, kW]
    target_conv1_w = target_network.conv1.weight

    if source_conv1_w.shape[1] != target_conv1_w.shape[1]:
        # Pad input channels
        pad_c = target_conv1_w.shape[1] - source_conv1_w.shape[1]
        if pad_c > 0:
            padded_w = jnp.pad(source_conv1_w, ((0, 0), (0, pad_c), (0, 0), (0, 0)))
        else:
            # Truncate (unlikely in curriculum — games get bigger)
            padded_w = source_conv1_w[:, :target_conv1_w.shape[1]]
        target_network = eqx.tree_at(
            lambda n: n.conv1.weight, target_network, padded_w
        )
        # Keep source bias
        target_network = eqx.tree_at(
            lambda n: n.conv1.bias, target_network, network.conv1.bias
        )
    else:
        target_network = eqx.tree_at(lambda n: n.conv1, target_network, network.conv1)

    # Conv2 and Conv3: same architecture, transfer directly
    target_network = eqx.tree_at(lambda n: n.conv2, target_network, network.conv2)
    target_network = eqx.tree_at(lambda n: n.conv3, target_network, network.conv3)

    # Scalar FC — pad if input size differs
    source_scalar_in = network.scalar_fc.weight.shape[1]  # [out, in]
    target_scalar_in = target_network.scalar_fc.weight.shape[1]

    if source_scalar_in != target_scalar_in:
        sw = network.scalar_fc.weight
        pad_in = target_scalar_in - source_scalar_in
        if pad_in > 0:
            padded_sw = jnp.pad(sw, ((0, 0), (0, pad_in)))
        else:
            padded_sw = sw[:, :target_scalar_in]
        target_network = eqx.tree_at(
            lambda n: n.scalar_fc.weight, target_network, padded_sw
        )
        target_network = eqx.tree_at(
            lambda n: n.scalar_fc.bias, target_network, network.scalar_fc.bias
        )
    else:
        target_network = eqx.tree_at(lambda n: n.scalar_fc, target_network, network.scalar_fc)

    # Shared FC: transfer (CNN output + scalar output → hidden)
    # The shared FC input size may differ if scalar dims change, but CNN output is fixed
    # (64*8*8 + 32 for ScaledActorCritic)
    source_shared_in = network.shared_fc.weight.shape[1]
    target_shared_in = target_network.shared_fc.weight.shape[1]

    if source_shared_in == target_shared_in:
        target_network = eqx.tree_at(lambda n: n.shared_fc, target_network, network.shared_fc)
    else:
        # Pad the weight matrix
        sw = network.shared_fc.weight
        pad_in = target_shared_in - source_shared_in
        if pad_in > 0:
            padded_sw = jnp.pad(sw, ((0, 0), (0, pad_in)))
        else:
            padded_sw = sw[:, :target_shared_in]
        target_network = eqx.tree_at(
            lambda n: n.shared_fc.weight, target_network, padded_sw
        )
        target_network = eqx.tree_at(
            lambda n: n.shared_fc.bias, target_network, network.shared_fc.bias
        )

    # Actor head: re-initialize if action count differs (different game)
    source_actions = network.actor.weight.shape[0]
    target_actions = target_network.actor.weight.shape[0]

    if source_actions == target_actions:
        target_network = eqx.tree_at(lambda n: n.actor, target_network, network.actor)
    else:
        # Keep fresh random init for actor — new action space
        pass

    # Critic head: always 1 output, transfer directly
    target_network = eqx.tree_at(lambda n: n.critic, target_network, network.critic)

    return target_network


def curriculum_train(
    phases: list[list[str]],
    *,
    key,
    updates_per_phase: int = None,
    available_vram_gb: float = None,
):
    """
    Multi-phase curriculum training with weight transfer.

    Args:
        phases: List of phases, each phase is a list of game module paths.
            Example: [
                ["jaxswarm.games.game_01_empty_exit", "jaxswarm.games.game_03_lock_and_key"],
                ["jaxswarm.games.game_07_hunger_clock", "jaxswarm.games.game_10_farming_growth"],
                ["jaxswarm.games.game_11_digging_deep", "jaxswarm.games.game_13_siege_architecture"],
                ["jaxswarm.games.game_14_inf_fortress"],
            ]
        key: PRNG key
        updates_per_phase: Override num_updates per game in each phase
        available_vram_gb: VRAM budget

    Returns:
        (final_network, all_metrics)
    """
    network = None
    all_metrics = {}

    for phase_idx, game_paths in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"CURRICULUM PHASE {phase_idx + 1}/{len(phases)}: {', '.join(g.split('.')[-1] for g in game_paths)}")
        print(f"{'='*60}\n")

        for game_path in game_paths:
            game_module = importlib.import_module(game_path)
            game_name = game_path.split('.')[-1]
            config = game_module.CONFIG

            key, k_train, k_pad = jax.random.split(key, 3)

            # Auto-configure training
            tc_overrides = {}
            if updates_per_phase is not None:
                tc_overrides['num_updates'] = updates_per_phase
            tc = auto_train_config(config, available_vram_gb, **tc_overrides)

            print(f"--- Training: {game_name} ---")
            print(f"    {tc.num_updates} updates × {tc.num_steps} steps × {tc.num_envs} envs")
            print(f"    Total steps: {tc.num_updates * tc.num_steps * tc.num_envs:,}")

            # Transfer weights from previous phase
            network_cls = ScaledActorCritic  # Always use ScaledActorCritic for curriculum
            if network is not None:
                print(f"    Transferring weights from previous phase...")
                prev_config = prev_game_config
                network = _pad_network_for_game(network, prev_config, config, key=k_pad)

            t0 = time.time()
            network, metrics = train(
                game_module, tc, key=k_train,
                network_cls=network_cls,
            )
            elapsed = time.time() - t0

            win_rate = float(metrics["win_rate"])
            print(f"    Done in {elapsed:.1f}s — win rate: {win_rate:.4f}")

            all_metrics[game_name] = {
                "phase": phase_idx + 1,
                "win_rate": win_rate,
                "total_episodes": int(metrics["total_episodes"]),
                "total_wins": int(metrics["total_wins"]),
                "training_time_s": round(elapsed, 1),
            }

            prev_game_config = config

    return network, all_metrics


# Pre-defined curriculum pipelines

CURRICULUM_S1 = [
    # Phase 1: Basic navigation
    ["jaxswarm.games.game_01_empty_exit", "jaxswarm.games.game_03_lock_and_key"],
    # Phase 2: Hunger + Farming
    ["jaxswarm.games.game_07_hunger_clock", "jaxswarm.games.game_10_farming_growth"],
    # Phase 3: Mining + Siege
    ["jaxswarm.games.game_11_digging_deep", "jaxswarm.games.game_13_siege_architecture"],
    # Phase 4: Capstone
    ["jaxswarm.games.game_14_inf_fortress"],
]

CURRICULUM_S2 = [
    # Phase 1: Survival basics
    ["jaxswarm.games.game_15_the_campfire"],
    # Phase 2: Expansion
    ["jaxswarm.games.game_16_the_golden_field"],
    # Phase 3: Ecology
    ["jaxswarm.games.game_17_the_shepherd"],
    # Phase 4: Processing
    ["jaxswarm.games.game_18_the_oven"],
    # Phase 5: Island effects
    ["jaxswarm.games.game_19_the_deep"],
    # Phase 6: Capstone
    ["jaxswarm.games.game_20_fishing_for_islands"],
]
