# Game Spec 17: Siege & Defense

## Overview
A tower-defense-meets-DF-siege game. The player defends a fortress against 5 waves of increasingly powerful enemies by building walls, placing traps, and commanding archers. Between waves, the player gathers resources and fortifies. Survive all 5 waves to win. This is Dwarf Fortress's siege mechanic: goblin armies vs. your defenses.

## Grid
- Dimensions: 24×20

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'build_wall', 'build_trap', 'order_archer'],
    'grid': (24, 20),
    'max_turns': 800,
    'step_penalty': -0.002,
    'player_properties': [
        {'key': 'health', 'max': 10},
        {'key': 'stone', 'max': 20},
        {'key': 'wave', 'max': 5},
        {'key': 'archers_alive', 'max': 3},
        {'key': 'fort_hp', 'max': 10},
    ],
}
```

## Layout

- **Fortress** (x=0–8): Player's base. Contains stockpile, archer posts, and the fort core.
- **No-man's-land** (x=9–15): Open field with scattered stone deposits.
- **Spawn zone** (x=16–23): Enemies spawn here at wave start.

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Inside fortress |
| `wall` | `solid` | `#` | 1 | Fortress walls (pre-built), outer boundary |
| `fort_core` | `npc` | `C` | 5 | Center of fortress (x=4, y=10). If destroyed, game lost. |
| `archer` | `npc` | `A` | 8 | 3 archers inside fortress. Range attack during waves. |
| `stone_deposit` | `pickup` | `o` | 3 | 6–10 in no-man's-land. Player collects for building. |
| `built_wall` | `solid` | `=` | 5 | Created by player via `build_wall`. HP=3. |
| `trap` | `npc` | `^` | 3 | Created by player via `build_trap`. Destroys first enemy that walks onto it, then self-destructs. |
| `grunt` | `hazard` | `g` | 5 | Wave enemies. HP=2, ATK=1. Move toward fort_core. |
| `brute` | `hazard` | `B` | 5 | Heavy enemies (waves 3+). HP=5, ATK=2. Move toward fort_core. Can destroy built_walls (1 damage per hit). |
| `sapper` | `hazard` | `s` | 5 | Sapper enemies (waves 4+). HP=1, ATK=1. Ignores built_walls (teleport through). Move toward fort_core. |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `health` | 10 | 10 | Player HP |
| `stone` | 5 | 20 | Building material |
| `wave` | 0 | 5 | Current wave (0 = prep phase) |
| `archers_alive` | 3 | 3 | Living archers |
| `fort_hp` | 10 | 10 | Fort core HP. Enemies attacking it deal damage. |

## Wave System

Waves trigger every 120 turns (turns 120, 240, 360, 480, 600). Between waves: prep phase (no enemies).

| Wave | Enemies spawned |
|------|----------------|
| 1 | 4 grunts |
| 2 | 6 grunts |
| 3 | 5 grunts + 2 brutes |
| 4 | 4 grunts + 2 brutes + 2 sappers |
| 5 | 6 grunts + 3 brutes + 3 sappers |

A wave ends when all enemies from that wave are destroyed. If all 5 waves are survived (checked after wave 5 enemies cleared): `env.end_game('won')`.

## Behaviors

### `grunt`
Move toward `fort_core` (Manhattan pathfinding, prefer axis with greatest distance). On collision with `built_wall`: cancel move, deal 1 damage to wall (tracked in wall's `hp` property). If wall `hp <= 0`, destroy it. On reaching `fort_core`: deal 1 damage to `fort_hp`.

### `brute`
Same as grunt but 2 damage to walls and fort_core.

### `sapper`
Move toward fort_core. Built_walls do not block sappers (sapper's `before_move` handler allows passage through `built_wall` entities). Normal walls still block.

### `archer`
During wave phase: each turn, if any enemy within Manhattan distance 6 of archer, deal 1 damage to the nearest enemy. Archers don't move.

## Event Handlers

### `input` (Movement)
Standard movement.

### `input` (Build Wall)
If action is `build_wall` and `stone >= 3`:
- Check the cell the player is facing (last move direction, default east). If empty and not inside fortress (x > 8):
  - Create `built_wall` at that cell with `hp=3`. Decrement `stone` by 3.
  - Emit `reward` `{ 'amount': 0.1 }`.
- Simplification: build on the cell 1 tile east of the player if it's empty.

### `input` (Build Trap)
If action is `build_trap` and `stone >= 2`:
- Place trap 1 tile east of player if empty and x > 8.
- Decrement `stone` by 2. Emit `reward` `{ 'amount': 0.1 }`.

### `input` (Order Archer)
`order_archer`: nearest archer relocates to player's position (swap). Allows repositioning archers.

### `collision` (pickup)
Stone deposit: increment `stone` by 3 (cap at 20). Destroy deposit. Emit reward.

### `collision` (trap triggers)
Enemy walks onto trap: destroy enemy, destroy trap. Emit `reward` `{ 'amount': 0.2 }`.

### `collision` (combat)
Player vs enemy: mutual damage, cancel move (same as spec 04).

### `turn_start` (Wave spawning)
Check turn number. If wave trigger turn and `wave < 5`: increment `wave`, spawn enemies at random positions in spawn zone.

### `turn_end` (Fort HP check)
If `fort_hp <= 0`: `env.end_game('lost')`.
If `wave == 5` and no enemies remain: `env.end_game('won')`.

### `before_move`
Standard solid blocking. Exception: sappers pass through `built_wall` (but not regular `wall`).

## Win Condition
Survive all 5 waves (all wave-5 enemies destroyed with fort_core intact).

## Lose Condition
- Fort core HP reaches 0.
- Player HP reaches 0.

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0% |
| PPO win rate at 1M steps | >2% |
| PPO learning delta (1M - 200k) | >1.5% |

## Invariant Tests

1. Fort core exists at start with hp=10.
2. Exactly 3 archers at start.
3. 6–10 stone deposits exist.
4. No enemies exist at start (wave 0 is prep).
5. Player starts inside fortress (x < 9).
6. Wave 0 at start.
7. Build zone is x > 8 (no building inside fortress).

## Notes
- The prep-vs-wave rhythm forces the agent to learn resource gathering between attacks. Pure turtling fails because stone is in no-man's-land.
- Brutes punish weak walls; sappers punish wall-only strategies. The agent must learn to combine walls, traps, and archer positioning.
- The increasing enemy variety across waves creates a curriculum within a single episode.
- Fort core HP is a shared resource — a single breach can cascade into a loss, much like DF sieges.
