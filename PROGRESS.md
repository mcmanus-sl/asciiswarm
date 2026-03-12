# Progress

Tracking file for agent swarm progress. Agents update this file with status of their work.

## Game 04: Dungeon Crawl — DONE
- Implemented in `games/04_dungeon_crawl.py`
- Tests in `tests/games/test_04_dungeon_crawl.py` (11 tests, all passing)
- Features: procedural room generation (3-5 rooms), 3 enemy types (wanderer, chaser, sentinel), combat system, health potions, exit goal
- Invariants: rooms connected, potions exist, enemy count in range, player healthy, no enemy on player spawn, exit reachable
- Exit invariants (EXIT_INVARIANTS) included

## Game 05: Pac-Man Collect
- **Status**: COMPLETE
- **File**: `games/05_pac_collect.py`
- **Tests**: `tests/games/test_05_pac_collect.py` — 11/11 pass
- **Full suite**: 203/203 pass
- **Notes**: 12x12 grid with cross-pattern interior walls. Player at (6,6), chaser at (1,1), patroller at (10,1). Dots fill all empty cells. Chaser uses Manhattan distance pursuit (prefers horizontal on ties). Patroller follows rectangular patrol path (9 steps per direction). Collision with ghost = loss. Collect all dots = win. Per-dot reward +0.05.

## Game 06: Ice Sliding
- **Status:** DONE
- All 207 tests pass (full suite including kernel + all games)
- Random agent: 0 crashes, 100% termination rate, ~80% win rate (1000 episodes)
- Solvability verified via ice-sliding BFS on every seed
- Hardcoded fallback layout if procedural generation fails after 100 attempts
- Exit invariants + 5 game-specific invariants

## Game 07: Hunger Clock
- **Status:** COMPLETE
- **File:** `games/07_hunger_clock.py`
- **Tests:** `tests/games/test_07_hunger_clock.py` (15 tests, all passing)
- **Full suite:** 207/207 passed
- **Notes:** Player at (0,13) must reach exit at (13,0) before food runs out. 10-15 food pickups restore 5 food each (cap 20). 3-5 wall clusters with BFS reachability check. Food decreases by 1 per turn. Manhattan distance is 26 but starting food is 20, so player must eat to survive.

## Game 09: Inventory & Crafting
- **Status:** COMPLETE
- **File:** `games/09_inventory_crafting.py`
- **Tests:** `tests/games/test_09_inventory_crafting.py` (14 tests, all passing)
- **Full suite:** 258/258 passed
- **Notes:** 16x16 grid. Player in left third collects wood (4-6) and ore (3-5) scattered in the left section. Workbench in center area (6-9, 6-9). Vertical wall at x=12 with one rubble gap. Exit behind wall (x>=13). Chain: gather 2 wood + 2 ore → craft pickaxe at workbench → mine rubble → reach exit. Pickup rewards +0.05, craft/mine rewards +0.3 each. Items tagged 'pickup' for distance shaping.
