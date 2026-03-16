"""Pygame-based ASCII game player for trained JAX games.

Same game loop as play_games.py (reset → step → render) but with a proper
windowed UI: entity legend, per-game status panel, and controls reference.

Usage:
    python play_pygame.py          # Start at game 01, auto-advance on win
    python play_pygame.py 3        # Start at game 03
    python play_pygame.py game_04  # Start at game 04
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import re
import time
import importlib

import pygame

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxswarm.network import ActorCritic

# ---------------------------------------------------------------------------
# Game registry (same as play_games.py)
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
# Per-game metadata for sidebar
# ---------------------------------------------------------------------------

GAME_META = {
    "game_01_empty_exit": {
        "entities": {1: "Player", 2: "Exit"},
        "status": [],
    },
    "game_02_dodge": {
        "entities": {1: "Player", 2: "Exit", 3: "Wanderer"},
        "status": [],
    },
    "game_03_lock_and_key": {
        "entities": {1: "Player", 2: "Exit", 3: "Wall", 4: "Door", 5: "Key"},
        "status": [("Has Key", "prop", 0)],
    },
    "game_04_dungeon_crawl": {
        "entities": {1: "Player", 2: "Exit", 3: "Wall", 4: "Wanderer", 5: "Chaser", 6: "Sentinel", 7: "Potion"},
        "status": [("HP", "prop", 0), ("ATK", "prop", 1), ("Kills", "gs", 0)],
    },
    "game_05_pac_man_collect": {
        "entities": {1: "Player", 2: "Dot", 3: "Chaser", 4: "Patroller", 5: "Wall"},
        "status": [("Dots Left", "gs", 0), ("Collected", "gs", 1)],
    },
    "game_06_ice_sliding": {
        "entities": {1: "Player", 2: "Exit", 3: "Rock"},
        "status": [],
    },
    "game_07_hunger_clock": {
        "entities": {1: "Player", 3: "Food", 4: "Wall"},
        "status": [("Food", "prop", 0)],
    },
    "game_08_block_push": {
        "entities": {1: "Player", 2: "Exit", 3: "Wall", 4: "Block", 5: "Target"},
        "status": [("On Target", "gs", 0), ("Pushes", "gs", 1)],
    },
    "game_09_inventory_crafting": {
        "entities": {1: "Player", 2: "Exit", 3: "Wall", 4: "Wood", 5: "Ore", 6: "Workbench", 7: "Rubble"},
        "status": [("Wood", "prop", 0), ("Ore", "prop", 1), ("Pickaxe", "prop", 2)],
    },
    "game_10_farming_growth": {
        "entities": {1: "Player", 3: "Soil", 4: "Seedbag", 5: "Sprout", 6: "Mature", 7: "Bin"},
        "status": [("Seeds", "prop", 0), ("Crops", "prop", 1), ("Delivered", "prop", 2)],
    },
}

# ---------------------------------------------------------------------------
# Colors / layout constants
# ---------------------------------------------------------------------------

BG_COLOR = (0, 0, 0)
HUD_BG = (20, 20, 30)
TITLEBAR_BG = (20, 20, 40)
STATUSBAR_BG = (20, 20, 40)
CELL_DARK = (16, 16, 16)
CELL_LIGHT = (22, 22, 22)
EMPTY_COLOR = (60, 60, 60)
TEXT_COLOR = (200, 200, 200)
DIM_TEXT = (120, 120, 120)
HIGHLIGHT_COLOR = (255, 255, 255)
BORDER_COLOR = (50, 50, 50)

FRAME_W, FRAME_H = 1280, 800
TITLEBAR_H = 36
HUD_H = 160
STATUSBAR_H = 28
GRID_PAD = 16
HUD_FONT_SIZE = 18

AUTOPILOT_INTERVAL = 0.3  # seconds between AI steps
FLASH_DURATION = 1.5  # seconds for win/death overlay


# ---------------------------------------------------------------------------
# Load game + weights (same as play_games.py)
# ---------------------------------------------------------------------------

def load_game(game_path):
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
# Key mapping
# ---------------------------------------------------------------------------

def map_key(key):
    """Map pygame key to action index (0-5) or special command string."""
    if key in (pygame.K_UP, pygame.K_w):
        return 0  # N
    if key in (pygame.K_DOWN, pygame.K_s):
        return 1  # S
    if key in (pygame.K_RIGHT, pygame.K_d):
        return 2  # E
    if key in (pygame.K_LEFT, pygame.K_a):
        return 3  # W
    if key in (pygame.K_e, pygame.K_5):
        return 4  # interact
    if key == pygame.K_PERIOD:
        return 5  # wait
    if key == pygame.K_SPACE:
        return "ai_toggle"
    if key == pygame.K_q:
        return "quit"
    if key == pygame.K_r:
        return "restart"
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def build_grid(state, config, glyph_map):
    """Build 2D grid of (glyph, color) tuples from game state."""
    grid_h, grid_w = config.grid_h, config.grid_w

    alive_np = state.alive.tolist()
    etype_np = state.entity_type.tolist()
    xs_np = state.x.tolist()
    ys_np = state.y.tolist()

    grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]

    # Sort so player draws last (on top)
    entities = []
    for i in range(len(alive_np)):
        if alive_np[i]:
            entities.append((int(etype_np[i]), int(xs_np[i]), int(ys_np[i])))
    entities.sort(key=lambda e: (1 if e[0] == 1 else 0, e[0]))

    for et, ex, ey in entities:
        if 0 <= ey < grid_h and 0 <= ex < grid_w and et in glyph_map:
            glyph, color = glyph_map[et]
            grid[ey][ex] = (glyph, color)

    return grid


def get_status_values(state, meta):
    """Read status fields from state based on GAME_META spec."""
    values = []
    # Find player index (entity_type == 1, alive)
    player_idx = None
    alive = state.alive.tolist()
    etypes = state.entity_type.tolist()
    for i in range(len(alive)):
        if alive[i] and int(etypes[i]) == 1:
            player_idx = i
            break

    for label, source, idx in meta.get("status", []):
        val = "?"
        if source == "prop" and player_idx is not None:
            try:
                val = int(state.properties[player_idx, idx])
            except (IndexError, AttributeError):
                val = "?"
        elif source == "gs":
            try:
                val = int(state.game_state[idx])
            except (IndexError, AttributeError):
                val = "?"
        values.append((label, val))
    return values


def render(screen, grid_font, hud_font, cell_size, state, config, glyph_map, game_key,
           game_title, cum_reward, autopilot, level_num, total_levels):
    """Render scaled grid + bottom HUD on 1280x800 logical surface."""
    meta = GAME_META.get(game_key, {"entities": {}, "status": []})
    grid_h, grid_w = config.grid_h, config.grid_w

    grid_pixel_w = cell_size * grid_w
    grid_pixel_h = cell_size * grid_h

    play_area_h = FRAME_H - TITLEBAR_H - HUD_H - STATUSBAR_H
    grid_x0 = (FRAME_W - grid_pixel_w) // 2
    grid_y0 = TITLEBAR_H + (play_area_h - grid_pixel_h) // 2

    # -- Background --
    screen.fill(BG_COLOR)

    # -- Title bar --
    pygame.draw.rect(screen, TITLEBAR_BG, (0, 0, FRAME_W, TITLEBAR_H))
    title_font = hud_font
    # Left: game title
    title_surf = title_font.render(f"  {game_title}", True, HIGHLIGHT_COLOR)
    screen.blit(title_surf, (4, (TITLEBAR_H - title_surf.get_height()) // 2))
    # Center: level
    lv_surf = title_font.render(f"Lv {level_num}/{total_levels}", True, TEXT_COLOR)
    screen.blit(lv_surf, ((FRAME_W - lv_surf.get_width()) // 2,
                           (TITLEBAR_H - lv_surf.get_height()) // 2))
    # Right: mode
    mode_str = "[AI]" if autopilot else "[MANUAL]"
    mode_surf = title_font.render(f"{mode_str}  ", True, HIGHLIGHT_COLOR)
    screen.blit(mode_surf, (FRAME_W - mode_surf.get_width() - 4,
                             (TITLEBAR_H - mode_surf.get_height()) // 2))

    # -- Grid --
    grid = build_grid(state, config, glyph_map)

    for row in range(grid_h):
        for col in range(grid_w):
            px = grid_x0 + col * cell_size
            py = grid_y0 + row * cell_size

            # Checkerboard cell background
            bg = CELL_LIGHT if (row + col) % 2 == 0 else CELL_DARK
            pygame.draw.rect(screen, bg, (px, py, cell_size, cell_size))

            # Glyph
            cell = grid[row][col]
            if cell is not None:
                glyph, color = cell
            else:
                glyph, color = ".", EMPTY_COLOR
            surf = grid_font.render(glyph, True, color)
            # Center glyph in cell
            gx = px + (cell_size - surf.get_width()) // 2
            gy = py + (cell_size - surf.get_height()) // 2
            screen.blit(surf, (gx, gy))

    # Grid border
    border_rect = pygame.Rect(grid_x0 - 1, grid_y0 - 1,
                              grid_pixel_w + 2, grid_pixel_h + 2)
    pygame.draw.rect(screen, BORDER_COLOR, border_rect, 1)

    # -- Bottom HUD --
    hud_y = FRAME_H - HUD_H - STATUSBAR_H
    pygame.draw.rect(screen, HUD_BG, (0, hud_y, FRAME_W, HUD_H))
    # Top border
    pygame.draw.line(screen, BORDER_COLOR, (0, hud_y), (FRAME_W, hud_y))

    col_w = FRAME_W // 3
    line_h = hud_font.get_linesize() + 2
    pad_x = 16
    pad_y = 8

    # Column dividers
    pygame.draw.line(screen, BORDER_COLOR, (col_w, hud_y + 4), (col_w, hud_y + HUD_H - 4))
    pygame.draw.line(screen, BORDER_COLOR, (col_w * 2, hud_y + 4), (col_w * 2, hud_y + HUD_H - 4))

    # -- Column 1: LEGEND --
    cx = pad_x
    cy = hud_y + pad_y
    header = hud_font.render("LEGEND", True, HIGHLIGHT_COLOR)
    screen.blit(header, (cx, cy))
    cy += line_h + 2

    entity_items = sorted(meta["entities"].items())
    # Two-column sub-layout for legend
    sub_col_w = (col_w - pad_x * 2) // 2
    for i, (etype, name) in enumerate(entity_items):
        sub_col = i % 2
        sub_row = i // 2
        ex = cx + sub_col * sub_col_w
        ey = cy + sub_row * line_h
        if etype in glyph_map:
            glyph, color = glyph_map[etype]
            g_surf = hud_font.render(glyph, True, color)
            n_surf = hud_font.render(f" {name}", True, TEXT_COLOR)
            screen.blit(g_surf, (ex, ey))
            screen.blit(n_surf, (ex + g_surf.get_width(), ey))
        else:
            n_surf = hud_font.render(f"  {name}", True, TEXT_COLOR)
            screen.blit(n_surf, (ex, ey))

    # -- Column 2: STATUS --
    cx = col_w + pad_x
    cy = hud_y + pad_y
    header = hud_font.render("STATUS", True, HIGHLIGHT_COLOR)
    screen.blit(header, (cx, cy))
    cy += line_h + 2

    status_vals = get_status_values(state, meta)
    for label, val in status_vals:
        text = hud_font.render(f"{label}: {val}", True, TEXT_COLOR)
        screen.blit(text, (cx, cy))
        cy += line_h

    text = hud_font.render(f"Turn: {int(state.turn_number)}", True, TEXT_COLOR)
    screen.blit(text, (cx, cy))
    cy += line_h
    text = hud_font.render(f"Score: {cum_reward:+.2f}", True, TEXT_COLOR)
    screen.blit(text, (cx, cy))

    # -- Column 3: CONTROLS --
    cx = col_w * 2 + pad_x
    cy = hud_y + pad_y
    header = hud_font.render("CONTROLS", True, HIGHLIGHT_COLOR)
    screen.blit(header, (cx, cy))
    cy += line_h + 2

    controls = [
        ("Arrows", "Move"),
        ("E", "Interact"),
        (".", "Wait"),
        ("Space", "Autopilot"),
        ("R", "Restart"),
        ("Q", "Quit"),
    ]
    for key_label, desc in controls:
        key_surf = hud_font.render(f"{key_label:<7}", True, DIM_TEXT)
        desc_surf = hud_font.render(desc, True, TEXT_COLOR)
        screen.blit(key_surf, (cx, cy))
        screen.blit(desc_surf, (cx + key_surf.get_width(), cy))
        cy += line_h

    # -- Status bar --
    pygame.draw.rect(screen, STATUSBAR_BG, (0, FRAME_H - STATUSBAR_H, FRAME_W, STATUSBAR_H))
    status_val = int(state.status)
    status_str = {1: "WON!", -1: "DEAD", 0: "playing"}.get(status_val, "?")
    bar_surf = hud_font.render(f"  {status_str}", True, TEXT_COLOR)
    screen.blit(bar_surf, (4, FRAME_H - STATUSBAR_H + (STATUSBAR_H - bar_surf.get_height()) // 2))
    right_surf = hud_font.render(f"Turn {int(state.turn_number)}  ", True, DIM_TEXT)
    screen.blit(right_surf, (FRAME_W - right_surf.get_width() - 4,
                             FRAME_H - STATUSBAR_H + (STATUSBAR_H - right_surf.get_height()) // 2))

    pygame.display.flip()


def render_flash(screen, font, message, color):
    """Semi-transparent overlay with centered text."""
    sw, sh = screen.get_size()
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    text_surf = font.render(message, True, color)
    tx = (sw - text_surf.get_width()) // 2
    ty = (sh - text_surf.get_height()) // 2
    screen.blit(text_surf, (tx, ty))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Play one game
# ---------------------------------------------------------------------------

def play_game(screen, game_module, network, game_key, glyph_map, level_num, total_levels, autopilot=False):
    """Play a single game. Returns (result, autopilot)."""
    config = game_module.CONFIG
    game_title = GAMES[level_num - 1][1]

    pygame.display.set_caption(f"{game_title} - ASCII Swarm")
    clock = pygame.time.Clock()

    # Compute cell size for this game's grid
    play_area_h = FRAME_H - TITLEBAR_H - HUD_H - STATUSBAR_H
    play_area_w = FRAME_W - GRID_PAD * 2
    cell_size = min(play_area_w // config.grid_w, play_area_h // config.grid_h)

    # Fonts: grid glyph font scaled to cell, HUD font fixed
    grid_font = pygame.font.SysFont("DejaVu Sans Mono,Courier New,monospace", max(12, int(cell_size * 0.7)))
    hud_font = pygame.font.SysFont("DejaVu Sans Mono,Courier New,monospace", HUD_FONT_SIZE)
    flash_font = pygame.font.SysFont("DejaVu Sans Mono,Courier New,monospace", 36)

    step_jit = jax.jit(game_module.step)

    rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))
    state, obs = game_module.reset(rng)
    cum_reward = 0.0

    if network is None:
        autopilot = False

    last_ai_step = time.time()

    running = True
    result = "quit"

    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", autopilot
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit", autopilot
                mapped = map_key(event.key)
                if mapped is None:
                    continue
                if mapped == "quit":
                    return "quit", autopilot
                if mapped == "restart":
                    return "restart", autopilot
                if mapped == "ai_toggle":
                    if network is not None:
                        autopilot = not autopilot
                        last_ai_step = time.time()
                    continue
                # Movement/action key
                if int(state.status) == 0:
                    action = mapped

        # Autopilot tick
        if autopilot and int(state.status) == 0:
            now = time.time()
            if now - last_ai_step >= AUTOPILOT_INTERVAL:
                logits, _ = network(obs)
                action = int(jnp.argmax(logits))
                last_ai_step = now

        # Step if we have an action
        if action is not None and int(state.status) == 0:
            state, obs, reward, done = step_jit(state, jnp.int32(action))
            cum_reward += float(reward)

        # Render
        render(screen, grid_font, hud_font, cell_size, state, config, glyph_map, game_key,
               game_title, cum_reward, autopilot, level_num, total_levels)

        # Check game over
        status = int(state.status)
        if status == 1:
            render_flash(screen, flash_font, "YOU WON! Advancing...", (0, 255, 0))
            pygame.time.wait(int(FLASH_DURATION * 1000))
            return "won", autopilot
        elif status == -1:
            render_flash(screen, flash_font, "YOU DIED! Back to level 1...", (255, 60, 60))
            pygame.time.wait(int(FLASH_DURATION * 1000))
            return "died", autopilot

        clock.tick(30)

    return result, autopilot


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(start_level):
    # Find playable games (those with weights)
    playable = []
    for i, (game_path, title) in enumerate(GAMES):
        game_key = game_path.split(".")[-1]
        if os.path.exists(f"weights/{game_key}.eqx"):
            playable.append(i)

    if not playable:
        print("No weights found in weights/. Train games first.")
        return

    # Init pygame once — persistent window across levels
    pygame.init()
    screen = pygame.display.set_mode((FRAME_W, FRAME_H), pygame.SCALED)

    total_levels = len(playable)
    level_idx = max(0, min(start_level - 1, total_levels - 1))
    autopilot = False

    while True:
        game_idx = playable[level_idx]
        game_path, game_title = GAMES[game_idx]
        game_key = game_path.split(".")[-1]
        glyph_map = GLYPH_MAPS.get(game_key, {})

        game_module, network, game_key = load_game(game_path)

        result, autopilot = play_game(screen, game_module, network, game_key, glyph_map,
                                      level_idx + 1, total_levels, autopilot)

        if result == "quit":
            break
        elif result == "won":
            if level_idx + 1 < total_levels:
                level_idx += 1
            else:
                # All levels complete — show briefly then exit
                font = pygame.font.SysFont("DejaVu Sans Mono,Courier New,monospace", 36)
                render_flash(screen, font, "ALL LEVELS COMPLETE!", (255, 255, 0))
                pygame.time.wait(2000)
                break
        elif result == "died":
            level_idx = 0
        elif result == "restart":
            pass  # replay same level

    pygame.quit()


def parse_start_level():
    if len(sys.argv) < 2:
        return 1
    arg = sys.argv[1]
    try:
        return int(arg)
    except ValueError:
        pass
    for i, (game_path, _) in enumerate(GAMES):
        game_key = game_path.split(".")[-1]
        if arg in game_key or arg == game_path:
            return i + 1
    m = re.search(r'(\d+)', arg)
    if m:
        return int(m.group(1))
    return 1


if __name__ == "__main__":
    start = parse_start_level()
    main(start)
