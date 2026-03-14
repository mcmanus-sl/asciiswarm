"""Replay trained PPO agents on Games 01-06, render ASCII grids → MP4 video."""

import os
import subprocess
import importlib
import tempfile
import shutil

import jax
import jax.numpy as jnp
import equinox as eqx
from PIL import Image, ImageDraw, ImageFont

import sys

from jaxswarm.network import ActorCritic


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------

GAMES = [
    ("jaxswarm.games.game_01_empty_exit",    "Game 01: Empty Exit"),
    ("jaxswarm.games.game_02_dodge",         "Game 02: Dodge"),
    ("jaxswarm.games.game_03_lock_and_key",  "Game 03: Lock & Key"),
    ("jaxswarm.games.game_04_dungeon_crawl", "Game 04: Dungeon Crawl"),
    ("jaxswarm.games.game_05_pac_man_collect","Game 05: Pac-Man Collect"),
    ("jaxswarm.games.game_06_ice_sliding",   "Game 06: Ice Sliding"),
]

# entity_type → (glyph, RGB color)
GLYPH_MAPS = {
    "game_01_empty_exit": {
        1: ("@", (0, 255, 0)),    # player - bright green
        2: (">", (255, 255, 0)),  # exit - yellow
    },
    "game_02_dodge": {
        1: ("@", (0, 255, 0)),    # player
        2: (">", (255, 255, 0)),  # exit
        3: ("w", (255, 80, 80)),  # wanderer - red
    },
    "game_03_lock_and_key": {
        1: ("@", (0, 255, 0)),    # player
        2: (">", (255, 255, 0)),  # exit
        3: ("#", (128, 128, 128)),# wall - gray
        4: ("+", (180, 100, 40)), # door - brown
        5: ("k", (0, 255, 255)),  # key - cyan
    },
    "game_04_dungeon_crawl": {
        1: ("@", (0, 255, 0)),    # player
        2: (">", (255, 255, 0)),  # exit
        3: ("#", (128, 128, 128)),# wall
        4: ("w", (255, 80, 80)),  # wanderer
        5: ("c", (255, 40, 40)),  # chaser
        6: ("s", (200, 60, 60)),  # sentinel
        7: ("!", (0, 255, 255)),  # potion
    },
    "game_05_pac_man_collect": {
        1: ("@", (0, 255, 0)),    # player
        2: (".", (255, 255, 100)),# dot - light yellow
        3: ("C", (255, 40, 40)),  # chaser ghost
        4: ("P", (255, 100, 255)),# patroller ghost
        5: ("#", (128, 128, 128)),# wall
    },
    "game_06_ice_sliding": {
        1: ("@", (0, 255, 0)),    # player
        2: (">", (255, 255, 0)),  # exit
        3: ("O", (160, 160, 160)),# rock
    },
}

# ---------------------------------------------------------------------------
# Episode recording
# ---------------------------------------------------------------------------

def run_episode(game_module, network, key):
    """Run one episode with trained policy, return list of (state, reward, done) per step."""
    config = game_module.CONFIG
    k_reset, k_act = jax.random.split(key)
    state, obs = game_module.reset(k_reset)

    states = [state]
    rewards = []
    dones = []
    cumulative_reward = 0.0

    step_jit = jax.jit(game_module.step)

    for t in range(config.max_turns):
        logits, _ = network(obs)
        rng, k_act = jax.random.split(k_act)
        action = jax.random.categorical(rng, logits)

        state, obs, reward, done = step_jit(state, action)
        r = float(reward)
        d = bool(done)
        cumulative_reward += r

        states.append(state)
        rewards.append(r)
        dones.append(d)

        if d:
            break

    return states, rewards, dones, cumulative_reward


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

CELL_SIZE = 28
FONT_SIZE = 20
HEADER_HEIGHT = 60
BG_COLOR = (10, 10, 10)
EMPTY_COLOR = (40, 40, 40)
EMPTY_GLYPH = "."

def _get_font(size):
    """Try to load a monospace font."""
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def render_frame(state, config, glyph_map, title, turn, reward, status, font, header_font):
    """Render a single game state as a PIL Image."""
    grid_h, grid_w = config.grid_h, config.grid_w
    img_w = grid_w * CELL_SIZE + 20  # 10px padding each side
    img_h = grid_h * CELL_SIZE + HEADER_HEIGHT + 30  # header + bottom padding

    img = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((10, 5), title, fill=(200, 200, 200), font=header_font)

    # HUD line
    status_str = {1: "WON", -1: "LOST", 0: "playing"}.get(int(status), "?")
    hud = f"Turn {turn:3d}  R={reward:+.2f}  [{status_str}]"
    draw.text((10, 30), hud, fill=(150, 150, 150), font=font)

    # Build grid from state
    alive = state.alive          # [max_entities]
    etype = state.entity_type    # [max_entities]
    xs = state.x                 # [max_entities]
    ys = state.y                 # [max_entities]

    # Build a 2D lookup: for each cell, find the highest-priority entity
    # Priority: player (type 1) > enemies > items > walls > exit
    grid_glyphs = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    grid_colors = [[(40, 40, 40) for _ in range(grid_w)] for _ in range(grid_h)]

    n_ents = int(alive.shape[0])
    alive_np = alive.tolist()
    etype_np = etype.tolist()
    xs_np = xs.tolist()
    ys_np = ys.tolist()

    # Sort entities by priority (player last so it overwrites)
    entities = []
    for i in range(n_ents):
        if alive_np[i]:
            entities.append((etype_np[i], xs_np[i], ys_np[i]))

    # Sort: type 1 (player) goes last so it draws on top
    entities.sort(key=lambda e: (1 if e[0] == 1 else 0, e[0]))

    for et, ex, ey in entities:
        et = int(et)
        ex = int(ex)
        ey = int(ey)
        if 0 <= ey < grid_h and 0 <= ex < grid_w and et in glyph_map:
            glyph, color = glyph_map[et]
            grid_glyphs[ey][ex] = glyph
            grid_colors[ey][ex] = color

    # Draw grid
    ox, oy = 10, HEADER_HEIGHT
    for row in range(grid_h):
        for col in range(grid_w):
            cx = ox + col * CELL_SIZE + CELL_SIZE // 2
            cy = oy + row * CELL_SIZE + CELL_SIZE // 2
            glyph = grid_glyphs[row][col]
            color = grid_colors[row][col]
            if glyph is None:
                glyph = EMPTY_GLYPH
                color = EMPTY_COLOR
            # Center the glyph in the cell
            bbox = font.getbbox(glyph)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2), glyph, fill=color, font=font)

    return img


def render_separator(width, height, text, header_font):
    """Render a separator frame between episodes."""
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    bbox = header_font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(200, 200, 200), font=header_font)
    return img


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    font = _get_font(FONT_SIZE)
    header_font = _get_font(FONT_SIZE + 2)

    tmpdir = tempfile.mkdtemp(prefix="replay_")
    frame_idx = 0
    fps = 5

    print(f"Temp frames dir: {tmpdir}")

    # Pre-compute uniform frame size from the largest game grid (game 04: 16x16)
    max_grid_w = max(importlib.import_module(g[0]).CONFIG.grid_w for g in GAMES)
    max_grid_h = max(importlib.import_module(g[0]).CONFIG.grid_h for g in GAMES)
    canvas_w = max_grid_w * CELL_SIZE + 20
    canvas_h = max_grid_h * CELL_SIZE + HEADER_HEIGHT + 30
    # Ensure even dimensions for h264
    canvas_w += canvas_w % 2
    canvas_h += canvas_h % 2
    print(f"Uniform frame size: {canvas_w}x{canvas_h}")

    def save_frame(img, idx):
        """Pad image to canvas size and save."""
        if img.width != canvas_w or img.height != canvas_h:
            padded = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
            px = (canvas_w - img.width) // 2
            py = (canvas_h - img.height) // 2
            padded.paste(img, (px, py))
            img = padded
        img.save(os.path.join(tmpdir, f"frame_{idx:06d}.png"))

    for game_idx, (game_path, game_title) in enumerate(GAMES):
        print(f"\n{'='*60}")
        print(f"  {game_title}")
        print(f"{'='*60}")

        game_module = importlib.import_module(game_path)
        config = game_module.CONFIG
        game_key = game_path.split(".")[-1]
        glyph_map = GLYPH_MAPS[game_key]

        # Load weights
        weight_path = f"weights/{game_key}.eqx"
        if not os.path.exists(weight_path):
            print(f"  ERROR: No weights found at {weight_path}")
            print(f"  Run: python evaluate_game.py {game_path}")
            sys.exit(1)
        print(f"  Loading weights: {weight_path}")
        skeleton = ActorCritic(config, key=jax.random.PRNGKey(0))
        network = eqx.tree_deserialise_leaves(weight_path, skeleton)

        # Run 3 episodes
        for ep in range(3):
            ep_key = jax.random.PRNGKey(ep * 100 + 7)
            states, rewards, dones, cum_r = run_episode(game_module, network, ep_key)

            print(f"  Episode {ep+1}: {len(states)-1} steps, reward={cum_r:.2f}, "
                  f"status={int(states[-1].status)}")

            # Render frames
            cum = 0.0
            for t in range(len(states)):
                s = states[t]
                r = rewards[t] if t < len(rewards) else 0.0
                cum += r

                img = render_frame(s, config, glyph_map, game_title,
                                   int(s.turn_number), cum, int(s.status),
                                   font, header_font)
                save_frame(img, frame_idx)
                frame_idx += 1

            # Separator between episodes (~1 sec)
            sep_text = f"Episode {ep+1} done  |  reward: {cum_r:+.2f}"
            sep_img = render_separator(canvas_w, canvas_h, sep_text, header_font)
            for _ in range(fps):
                sep_img.save(os.path.join(tmpdir, f"frame_{frame_idx:06d}.png"))
                frame_idx += 1

        # Game separator (~1.5 sec)
        if game_idx < len(GAMES) - 1:
            next_title = GAMES[game_idx + 1][1]
            sep_img = render_separator(canvas_w, canvas_h, f"Next: {next_title}", header_font)
            for _ in range(int(fps * 1.5)):
                sep_img.save(os.path.join(tmpdir, f"frame_{frame_idx:06d}.png"))
                frame_idx += 1

    print(f"\nTotal frames: {frame_idx}")
    print("Encoding video with ffmpeg...")

    output_path = "replay.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmpdir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
    else:
        print(f"\nVideo saved: {output_path}")

    # Cleanup
    shutil.rmtree(tmpdir)
    print("Done!")


if __name__ == "__main__":
    main()
