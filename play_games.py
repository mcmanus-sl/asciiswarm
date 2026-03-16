"""Interactive curses-based terminal player for trained JAX games.

Usage:
    python play_games.py          # Start at game 01, auto-advance on win
    python play_games.py 3        # Start at game 03
    python play_games.py game_04  # Start at game 04
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import curses
import importlib

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxswarm.network import ActorCritic

# ---------------------------------------------------------------------------
# Game registry (ordered by level)
# ---------------------------------------------------------------------------

GAMES = [
    ("jaxswarm.games.game_01_empty_exit",     "Game 01: Empty Exit"),
    ("jaxswarm.games.game_02_dodge",          "Game 02: Dodge"),
    ("jaxswarm.games.game_03_lock_and_key",   "Game 03: Lock & Key"),
    ("jaxswarm.games.game_04_dungeon_crawl",  "Game 04: Dungeon Crawl"),
    ("jaxswarm.games.game_05_pac_man_collect","Game 05: Pac-Man Collect"),
    ("jaxswarm.games.game_06_ice_sliding",    "Game 06: Ice Sliding"),
    ("jaxswarm.games.game_07_hunger_clock",   "Game 07: Hunger Clock"),
    ("jaxswarm.games.game_08_block_push",     "Game 08: Block Push"),
    ("jaxswarm.games.game_09_inventory_crafting","Game 09: Inventory & Crafting"),
    ("jaxswarm.games.game_10_farming_growth",  "Game 10: Farming & Growth"),
]

# entity_type -> (glyph, RGB color)
GLYPH_MAPS = {
    "game_01_empty_exit": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
    },
    "game_02_dodge": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
        3: ("w", (255, 80, 80)),
    },
    "game_03_lock_and_key": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
        3: ("#", (128, 128, 128)),
        4: ("+", (180, 100, 40)),
        5: ("k", (0, 255, 255)),
    },
    "game_04_dungeon_crawl": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
        3: ("#", (128, 128, 128)),
        4: ("w", (255, 80, 80)),
        5: ("c", (255, 40, 40)),
        6: ("s", (200, 60, 60)),
        7: ("!", (0, 255, 255)),
    },
    "game_05_pac_man_collect": {
        1: ("@", (0, 255, 0)),
        2: (".", (255, 255, 100)),
        3: ("C", (255, 40, 40)),
        4: ("P", (255, 100, 255)),
        5: ("#", (128, 128, 128)),
    },
    "game_06_ice_sliding": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
        3: ("O", (160, 160, 160)),
    },
    "game_07_hunger_clock": {
        1: ("@", (0, 255, 0)),
        3: ("f", (100, 255, 100)),
        4: ("#", (128, 128, 128)),
    },
    "game_08_block_push": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
        3: ("#", (128, 128, 128)),
        4: ("B", (180, 140, 60)),
        5: ("x", (255, 200, 0)),
    },
    "game_09_inventory_crafting": {
        1: ("@", (0, 255, 0)),
        2: (">", (255, 255, 0)),
        3: ("#", (128, 128, 128)),
        4: ("t", (140, 100, 50)),
        5: ("o", (180, 180, 220)),
        6: ("W", (200, 150, 50)),
        7: ("R", (160, 80, 80)),
    },
    "game_10_farming_growth": {
        1: ("@", (0, 255, 0)),
        3: ("~", (140, 100, 50)),
        4: ("s", (255, 200, 0)),
        5: (".", (100, 200, 100)),
        6: ("*", (50, 255, 50)),
        7: ("B", (180, 140, 60)),
    },
}


# ---------------------------------------------------------------------------
# Color setup
# ---------------------------------------------------------------------------

def _rgb_to_curses(r, g, b):
    """Map RGB to nearest curses color constant."""
    if r > 200 and g < 100 and b < 100:
        return curses.COLOR_RED
    if r > 150 and g < 130 and b > 200:
        return curses.COLOR_MAGENTA
    if r < 100 and g > 200 and b > 200:
        return curses.COLOR_CYAN
    if r < 100 and g > 150 and b < 100:
        return curses.COLOR_GREEN
    if r > 200 and g > 200 and b < 120:
        return curses.COLOR_YELLOW
    if r > 100 and g > 60 and b < 80:
        return curses.COLOR_YELLOW  # brown -> yellow
    if r < 180 and g < 180 and b < 180 and r > 80:
        return curses.COLOR_WHITE  # gray
    if r > 200 and g > 200 and b > 200:
        return curses.COLOR_WHITE
    return curses.COLOR_WHITE


def setup_colors(glyph_map):
    """Create curses color pairs for a glyph map. Returns {entity_type: pair_number}."""
    pair_map = {}
    pair_id = 1
    for etype, (glyph, rgb) in glyph_map.items():
        cc = _rgb_to_curses(*rgb)
        curses.init_pair(pair_id, cc, curses.COLOR_BLACK)
        pair_map[etype] = pair_id
        pair_id += 1
    # Empty cell
    curses.init_pair(pair_id, 8, curses.COLOR_BLACK)  # 8 = dark gray on 256-color terms
    empty_pair = pair_id
    return pair_map, empty_pair


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_game(stdscr, state, config, glyph_map, pair_map, empty_pair,
                game_title, cum_reward, autopilot, level_num, total_levels):
    """Render the game grid and HUD via curses."""
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    # HUD line 0
    status_val = int(state.status)
    status_str = {1: "WON!", -1: "DEAD", 0: "playing"}.get(status_val, "?")
    mode_str = "  [AI]" if autopilot else "  [MANUAL]"

    hud = f" {game_title}  |  Level {level_num}/{total_levels}  |  Turn {int(state.turn_number)}  |  R={cum_reward:+.2f}  |  {status_str}{mode_str}"
    stdscr.addnstr(0, 0, hud, max_x - 1, curses.A_BOLD)

    # Grid
    grid_y_offset = 2
    grid_h, grid_w = config.grid_h, config.grid_w

    # Clamp grid to available terminal space — render what fits
    render_h = min(grid_h, max_y - grid_y_offset - 2)
    render_w = min(grid_w, max_x - 2)
    if render_h < 1 or render_w < 1:
        stdscr.addnstr(2, 0, "Terminal too small!", max_x - 1)
        stdscr.refresh()
        return

    # Build glyph/color grid
    alive_np = state.alive.tolist()
    etype_np = state.entity_type.tolist()
    xs_np = state.x.tolist()
    ys_np = state.y.tolist()

    grid_chars = [['.' for _ in range(grid_w)] for _ in range(grid_h)]
    grid_pairs = [[empty_pair for _ in range(grid_w)] for _ in range(grid_h)]
    grid_attrs = [[curses.A_DIM for _ in range(grid_w)] for _ in range(grid_h)]

    # Sort entities so player draws last (on top)
    entities = []
    n_ents = len(alive_np)
    for i in range(n_ents):
        if alive_np[i]:
            entities.append((int(etype_np[i]), int(xs_np[i]), int(ys_np[i])))
    entities.sort(key=lambda e: (1 if e[0] == 1 else 0, e[0]))

    for et, ex, ey in entities:
        if 0 <= ey < grid_h and 0 <= ex < grid_w and et in glyph_map:
            glyph, _ = glyph_map[et]
            grid_chars[ey][ex] = glyph
            grid_pairs[ey][ex] = pair_map.get(et, empty_pair)
            grid_attrs[ey][ex] = curses.A_BOLD if et == 1 else curses.A_NORMAL

    # Draw grid
    for row in range(render_h):
        for col in range(render_w):
            try:
                stdscr.addch(
                    grid_y_offset + row, col + 1,
                    grid_chars[row][col],
                    curses.color_pair(grid_pairs[row][col]) | grid_attrs[row][col]
                )
            except curses.error:
                pass

    # Bottom controls
    ctrl_y = grid_y_offset + grid_h + 1
    controls = " Arrows/WASD:Move  e:Interact  .:Wait  Space:Autopilot  r:Restart  q:Quit"
    if ctrl_y < max_y:
        stdscr.addnstr(ctrl_y, 0, controls, max_x - 1, curses.A_DIM)

    stdscr.refresh()


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------

def map_key(key):
    """Map keypress to action index (0-5) or special command string."""
    if key == curses.KEY_UP or key == ord('w'):
        return 0  # N
    if key == curses.KEY_DOWN or key == ord('s'):
        return 1  # S
    if key == curses.KEY_RIGHT or key == ord('d'):
        return 2  # E
    if key == curses.KEY_LEFT or key == ord('a'):
        return 3  # W
    if key == ord('e') or key == ord('5'):
        return 4  # interact
    if key == ord('.'):
        return 5  # wait
    if key == ord(' '):
        return 'ai_toggle'
    if key == ord('q'):
        return 'quit'
    if key == ord('r'):
        return 'restart'
    return None


# ---------------------------------------------------------------------------
# Load game + weights
# ---------------------------------------------------------------------------

def load_game(game_path):
    """Import game module and load weights. Returns (module, network) or (module, None)."""
    game_module = importlib.import_module(game_path)
    config = game_module.CONFIG
    game_key = game_path.split(".")[-1]
    weight_path = f"weights/{game_key}.eqx"

    network = None
    if os.path.exists(weight_path):
        skeleton = ActorCritic(config, key=jax.random.PRNGKey(0))
        network = eqx.tree_deserialise_leaves(weight_path, skeleton)

    return game_module, network, game_key


# ---------------------------------------------------------------------------
# Play one game
# ---------------------------------------------------------------------------

def play_game(stdscr, game_module, network, game_key, glyph_map, level_num, total_levels, autopilot=False):
    """Play a single game. Returns (result, autopilot) where result is 'won', 'died', 'quit', or 'restart'."""
    config = game_module.CONFIG
    game_title = GAMES[level_num - 1][1]

    pair_map, empty_pair = setup_colors(glyph_map)

    step_jit = jax.jit(game_module.step)

    rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))
    state, obs = game_module.reset(rng)
    cum_reward = 0.0

    # Restore halfdelay if autopilot carried from previous level
    if autopilot and network is not None:
        curses.halfdelay(3)
    else:
        autopilot = False

    # Initial render
    render_game(stdscr, state, config, glyph_map, pair_map, empty_pair,
                game_title, cum_reward, autopilot, level_num, total_levels)

    while True:
        key = stdscr.getch()

        # Handle timeout (ERR) during autopilot — AI picks the action
        if key == curses.ERR:
            if not autopilot:
                continue
            # AI picks action from policy (use obs tracked from step/reset)
            logits, _ = network(obs)
            action = int(jnp.argmax(logits))
        else:
            action = map_key(key)

            if action is None:
                continue
            if action == 'quit':
                return 'quit', autopilot
            if action == 'restart':
                curses.cbreak()
                return 'restart', autopilot
            if action == 'ai_toggle':
                if network is not None:
                    autopilot = not autopilot
                    if autopilot:
                        curses.halfdelay(3)  # 300ms per tick
                    else:
                        curses.cbreak()
                render_game(stdscr, state, config, glyph_map, pair_map, empty_pair,
                            game_title, cum_reward, autopilot, level_num, total_levels)
                continue

        # Game already over — ignore movement
        if int(state.status) != 0:
            continue

        # Step the game
        state, obs, reward, done = step_jit(state, jnp.int32(action))
        cum_reward += float(reward)

        render_game(stdscr, state, config, glyph_map, pair_map, empty_pair,
                    game_title, cum_reward, autopilot, level_num, total_levels)

        # Check game over
        status = int(state.status)
        if status == 1:
            if not autopilot:
                curses.cbreak()
            _flash_message(stdscr, "YOU WON! Advancing...", curses.COLOR_GREEN)
            return 'won', autopilot
        elif status == -1:
            if not autopilot:
                curses.cbreak()
            _flash_message(stdscr, "YOU DIED! Back to level 1...", curses.COLOR_RED)
            return 'died', autopilot


def _flash_message(stdscr, msg, color):
    """Show a message briefly."""
    curses.init_pair(50, color, curses.COLOR_BLACK)
    max_y, max_x = stdscr.getmaxyx()
    row = max_y // 2
    col = max(0, (max_x - len(msg)) // 2)
    try:
        stdscr.addnstr(row, col, msg, max_x - col - 1, curses.color_pair(50) | curses.A_BOLD)
    except curses.error:
        pass
    stdscr.refresh()
    curses.napms(1500)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(stdscr, start_level):
    curses.curs_set(0)
    stdscr.keypad(True)
    curses.use_default_colors()

    # Find playable games (those with weights)
    playable = []
    for i, (game_path, title) in enumerate(GAMES):
        game_key = game_path.split(".")[-1]
        if os.path.exists(f"weights/{game_key}.eqx"):
            playable.append(i)

    if not playable:
        stdscr.addstr(0, 0, "No weights found in weights/. Train games first.")
        stdscr.refresh()
        stdscr.getch()
        return

    total_levels = len(playable)

    # Clamp start_level
    level_idx = max(0, min(start_level - 1, total_levels - 1))
    autopilot = False

    while True:
        game_idx = playable[level_idx]
        game_path, game_title = GAMES[game_idx]
        game_key = game_path.split(".")[-1]
        glyph_map = GLYPH_MAPS.get(game_key, {})

        game_module, network, game_key = load_game(game_path)

        result, autopilot = play_game(stdscr, game_module, network, game_key, glyph_map,
                                      level_idx + 1, total_levels, autopilot)

        if result == 'quit':
            return
        elif result == 'won':
            if level_idx + 1 < total_levels:
                level_idx += 1
            else:
                _flash_message(stdscr, "ALL LEVELS COMPLETE! You win!", curses.COLOR_YELLOW)
                return
        elif result == 'died':
            level_idx = 0
        elif result == 'restart':
            pass  # replay same level


def parse_start_level():
    """Parse CLI arg to determine starting level (1-indexed)."""
    if len(sys.argv) < 2:
        return 1
    arg = sys.argv[1]
    # Try as plain number
    try:
        return int(arg)
    except ValueError:
        pass
    # Try as game_XX pattern
    for i, (game_path, _) in enumerate(GAMES):
        game_key = game_path.split(".")[-1]
        if arg in game_key or arg == game_path:
            return i + 1
    # Try as just the number part
    import re
    m = re.search(r'(\d+)', arg)
    if m:
        return int(m.group(1))
    return 1


if __name__ == "__main__":
    start = parse_start_level()
    curses.wrapper(lambda stdscr: main(stdscr, start))
