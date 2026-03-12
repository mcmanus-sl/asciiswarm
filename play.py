#!/usr/bin/env python3
"""Interactive terminal player for any game module.

Usage:
    python play.py games/01_empty_exit.py
    python play.py games/02_dodge.py --seed 7
    python play.py games/03_lock_and_key.py
"""

import argparse
import importlib.util
import os
import sys

from asciiswarm.kernel.env import GridGameEnv


KEY_MAP = {
    'w': 'move_n', 'k': 'move_n',
    's': 'move_s', 'j': 'move_s',
    'd': 'move_e', 'l': 'move_e',
    'a': 'move_w', 'h': 'move_w',
    'e': 'interact',
    '.': 'wait',
    ' ': 'wait',
}


def load_game(path):
    spec = importlib.util.spec_from_file_location('game', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    parser = argparse.ArgumentParser(description='Play an AsciiSwarm game')
    parser.add_argument('game', help='Path to game module (e.g. games/01_empty_exit.py)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    game = load_game(args.game)
    env = GridGameEnv(game, seed=args.seed, render_mode='ansi')
    obs, info = env.reset()

    actions = env.config['actions']
    action_map = {v: k for k, v in env.ACTION_MAP.items()}

    while True:
        clear()
        print(env.render())
        print()
        print(f"Turn: {env.turn_number}/{env.config['max_turns']}  Status: {env.status}")

        # Show player properties
        players = env.get_entities_by_tag('player')
        if players:
            p = players[0]
            if p.properties:
                props = ', '.join(f"{k}={v}" for k, v in p.properties.items())
                print(f"Player: ({p.x},{p.y})  {props}")
            else:
                print(f"Player: ({p.x},{p.y})")

        print()
        print("Controls: WASD/HJKL=move  E=interact  .=wait  Q=quit  R=restart")

        if env.status != 'playing':
            result = 'YOU WIN!' if env.status == 'won' else 'YOU LOST!'
            print(f"\n*** {result} ***  (R=restart, Q=quit)")

        try:
            key = input('> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key == 'q':
            break
        if key == 'r':
            obs, info = env.reset()
            continue

        if env.status != 'playing':
            continue

        action_str = KEY_MAP.get(key)
        if action_str and action_str in action_map:
            obs, reward, terminated, truncated, info = env.step(action_map[action_str])
            if reward != 0:
                # Will show on next render
                pass
        else:
            pass  # invalid key, just re-render


if __name__ == '__main__':
    main()
