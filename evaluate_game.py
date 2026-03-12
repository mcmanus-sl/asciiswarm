#!/usr/bin/env python3
"""RL Evaluation Pipeline — judges whether a game module is mechanically sound and learnable.

Three evaluation layers:
  1. Random agent (1000 episodes) — crash-free, terminates properly
  2. PPO training (500k steps) — learning delta > 0
  3. Invariant tests — all pass

Usage:
    python evaluate_game.py 01_empty_exit
    python evaluate_game.py 02_dodge
    python evaluate_game.py 03_lock_and_key
"""

import argparse
import importlib
import json
import sys
import traceback
from collections import deque

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from asciiswarm.kernel.env import GridGameEnv
from asciiswarm.kernel.invariants import run_invariants


# ---- Custom CNN Feature Extractor ----

class SmallGridCNN(BaseFeaturesExtractor):
    """CNN for channel-first grid obs + scalar concatenation.

    SB3's default CombinedExtractor won't apply a CNN to our grid because
    it's float32 [0,1] not uint8 [0,255]. This extractor handles both
    the 'grid' and 'scalars' keys from our Dict observation space.
    """

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 64):
        # Must call super with the final combined feature dimension
        # We'll compute it after building the network
        super().__init__(observation_space, features_dim=1)  # placeholder

        grid_space = observation_space['grid']
        scalars_space = observation_space['scalars']
        n_channels = grid_space.shape[0]
        n_scalars = scalars_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
        )

        combined_dim = 64 + n_scalars
        self.final_linear = nn.Linear(combined_dim, features_dim)
        self.final_relu = nn.ReLU()

        # Update the features_dim attribute
        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grid_features = self.cnn(observations['grid'])
        scalars = observations['scalars']
        combined = torch.cat([grid_features, scalars], dim=1)
        return self.final_relu(self.final_linear(combined))


# ---- Helpers ----

def load_game_module(game_name: str):
    """Import a game module from the games/ directory."""
    module_name = f"games.{game_name}"
    return importlib.import_module(module_name)


def make_env(game_module, seed=42):
    """Create a GridGameEnv for the given game module."""
    return GridGameEnv(game_module, seed=seed)


class RandomSeedWrapper(gymnasium.Wrapper):
    """Wrapper that randomizes the env seed on each reset.

    This gives PPO diverse layouts during training, preventing overfitting
    to a single RNG stream.
    """

    def __init__(self, env, seed=42):
        super().__init__(env)
        self._seed_rng = np.random.RandomState(seed)

    def reset(self, **kwargs):
        kwargs['seed'] = int(self._seed_rng.randint(0, 2**31))
        return self.env.reset(**kwargs)


class DistanceShapingWrapper(gymnasium.Wrapper):
    """Adds potential-based reward shaping: bonus for getting closer to the current goal.

    Goal priority: nearest pickup (if any exist), then nearest exit.
    This handles multi-step games (get key → reach exit) without game-specific logic.
    """

    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self._scale = scale
        self._prev_dist = None

    def _get_goal_distance(self):
        """Manhattan distance from player to current goal.

        Targets nearest pickup if any exist, then nearest exit (only if
        exit is reachable — no solid entities blocking the BFS path).
        Returns 0 if no valid goal found, disabling shaping.
        """
        players = self.unwrapped.get_entities_by_tag('player')
        if not players:
            return 0.0
        p = players[0]

        # Priority: pickup first
        pickups = self.unwrapped.get_entities_by_tag('pickup')
        if pickups:
            return min(abs(p.x - e.x) + abs(p.y - e.y) for e in pickups)

        # Then exit — but only shape toward exit if it's on the same
        # connected component (no solid wall blocking the path).
        # This avoids pulling the agent toward unreachable exits.
        exits = self.unwrapped.get_entities_by_tag('exit')
        if not exits:
            return 0.0

        # Quick BFS check for reachability
        w, h = self.unwrapped.config['grid']
        solid_positions = set()
        for ent in self.unwrapped.get_all_entities():
            if ent.has_tag('solid'):
                solid_positions.add((ent.x, ent.y))

        exit_positions = {(e.x, e.y) for e in exits}
        visited = {(p.x, p.y)}
        queue = deque([(p.x, p.y)])
        reachable_exit = None
        while queue:
            x, y = queue.popleft()
            if (x, y) in exit_positions:
                reachable_exit = (x, y)
                break
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    if (nx, ny) not in solid_positions or (nx, ny) in exit_positions:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        if reachable_exit:
            return abs(p.x - reachable_exit[0]) + abs(p.y - reachable_exit[1])

        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = self._get_goal_distance()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_dist = self._get_goal_distance()
        # Reward shaping: positive reward for getting closer to goal
        shaping = (self._prev_dist - curr_dist) * self._scale
        self._prev_dist = curr_dist
        return obs, reward + shaping, terminated, truncated, info


# ---- Layer 1: Random Agent ----

def run_random_agent(game_module, episodes=1000, seed=42):
    """Run random agent for N episodes. Returns result dict."""
    env = make_env(game_module, seed=seed)
    rng = np.random.RandomState(seed)

    crashes = 0
    terminated_count = 0
    win_count = 0
    total_length = 0

    for ep in range(episodes):
        try:
            obs, info = env.reset(seed=seed + ep)
            done = False
            length = 0
            while not done:
                action = rng.randint(env.action_space.n)
                obs, reward, terminated, truncated, info = env.step(action)
                length += 1
                done = terminated or truncated
            total_length += length
            if terminated or truncated:
                terminated_count += 1
            if info.get('status') == 'won':
                win_count += 1
        except Exception:
            crashes += 1
            traceback.print_exc()

    termination_rate = terminated_count / episodes if episodes > 0 else 0
    win_rate = win_count / episodes if episodes > 0 else 0
    avg_length = total_length / max(episodes - crashes, 1)

    result = {
        'episodes': episodes,
        'crashes': crashes,
        'termination_rate': round(termination_rate, 4),
        'win_rate': round(win_rate, 4),
        'avg_length': round(avg_length, 1),
        'pass': crashes == 0 and termination_rate > 0.8,
    }
    return result


# ---- Layer 2: PPO Training ----

def evaluate_win_rate(model, game_module, episodes=100, seed=9999, deterministic=False):
    """Evaluate a trained model's win rate over N episodes."""
    env = make_env(game_module, seed=seed)
    wins = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        if info.get('status') == 'won':
            wins += 1
    return wins / episodes


def run_ppo_training(game_module, total_steps=500_000, seed=42):
    """Train PPO in two phases and measure learning delta.

    Uses RandomSeedWrapper for layout diversity and DistanceShapingWrapper
    (goal-aware: nearest pickup first, then exit) to overcome sparse rewards.
    """
    env = DistanceShapingWrapper(RandomSeedWrapper(make_env(game_module, seed=seed), seed=seed))

    model = PPO(
        'MultiInputPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=64),
        ),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    # Phase 1: train 10% of budget, evaluate baseline
    phase1_steps = total_steps // 10
    model.learn(total_timesteps=phase1_steps, reset_num_timesteps=True)
    win_rate_early = evaluate_win_rate(model, game_module)

    # Phase 2: train remaining steps, evaluate final
    remaining_steps = total_steps - phase1_steps
    model.learn(total_timesteps=remaining_steps, reset_num_timesteps=False)
    win_rate_final = evaluate_win_rate(model, game_module)

    learning_delta = win_rate_final - win_rate_early

    result = {
        'total_steps': total_steps,
        'win_rate_early': round(win_rate_early, 4),
        'win_rate_final': round(win_rate_final, 4),
        'learning_delta': round(learning_delta, 4),
        'pass': learning_delta > 0,
    }
    return result


# ---- Layer 3: Invariant Tests ----

def run_invariants_check(game_module, seed=42):
    """Run all invariants against a freshly-reset env."""
    env = make_env(game_module, seed=seed)
    env.reset(seed=seed)

    results = run_invariants(env, game_module)

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [name for name, ok, _ in results if not ok]
    failed_details = {name: msg for name, ok, msg in results if not ok}

    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'failed_details': failed_details if failed_details else {},
        'pass': len(failed) == 0,
    }


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description='RL Evaluation Pipeline')
    parser.add_argument('game', help='Game module name (e.g. 01_empty_exit)')
    parser.add_argument('--random-episodes', type=int, default=1000)
    parser.add_argument('--ppo-steps', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-ppo', action='store_true', help='Skip PPO training layer')
    args = parser.parse_args()

    game_module = load_game_module(args.game)
    print(f"Evaluating game: {args.game}", file=sys.stderr)

    # Layer 1: Random Agent
    print("Layer 1: Random agent...", file=sys.stderr)
    random_result = run_random_agent(game_module, episodes=args.random_episodes, seed=args.seed)
    print(f"  crashes={random_result['crashes']}, "
          f"term_rate={random_result['termination_rate']}, "
          f"win_rate={random_result['win_rate']}, "
          f"pass={random_result['pass']}", file=sys.stderr)

    # Layer 2: PPO Training
    if args.skip_ppo:
        ppo_result = {'skipped': True, 'pass': True}
    else:
        print(f"Layer 2: PPO training ({args.ppo_steps // 1000}k steps)...", file=sys.stderr)
        ppo_result = run_ppo_training(game_module, total_steps=args.ppo_steps, seed=args.seed)
        print(f"  win_rate_early={ppo_result['win_rate_early']}, "
              f"win_rate_final={ppo_result['win_rate_final']}, "
              f"delta={ppo_result['learning_delta']}, "
              f"pass={ppo_result['pass']}", file=sys.stderr)

    # Layer 3: Invariants
    print("Layer 3: Invariant tests...", file=sys.stderr)
    invariants_result = run_invariants_check(game_module, seed=args.seed)
    print(f"  {invariants_result['passed']}/{invariants_result['total']} passed, "
          f"pass={invariants_result['pass']}", file=sys.stderr)

    # Overall verdict
    overall_pass = (
        random_result['pass']
        and ppo_result['pass']
        and invariants_result['pass']
    )

    verdict = {
        'game': args.game,
        'random_agent': random_result,
        'ppo': ppo_result,
        'invariants': invariants_result,
        'overall_pass': overall_pass,
    }

    print(json.dumps(verdict, indent=2))
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
