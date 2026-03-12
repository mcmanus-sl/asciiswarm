# Progress

Tracking file for agent swarm progress. Agents update this file with status of their work.

## Game 05: Pac-Man Collect
- **Status**: COMPLETE
- **File**: `games/05_pac_collect.py`
- **Tests**: `tests/games/test_05_pac_collect.py` — 11/11 pass
- **Full suite**: 203/203 pass
- **Notes**: 12x12 grid with cross-pattern interior walls. Player at (6,6), chaser at (1,1), patroller at (10,1). Dots fill all empty cells. Chaser uses Manhattan distance pursuit (prefers horizontal on ties). Patroller follows rectangular patrol path (9 steps per direction). Collision with ghost = loss. Collect all dots = win. Per-dot reward +0.05.

## Game 07: Hunger Clock
- **Status:** COMPLETE
- **File:** `games/07_hunger_clock.py`
- **Tests:** `tests/games/test_07_hunger_clock.py` (15 tests, all passing)
- **Full suite:** 207/207 passed
- **Notes:** Player at (0,13) must reach exit at (13,0) before food runs out. 10-15 food pickups restore 5 food each (cap 20). 3-5 wall clusters with BFS reachability check. Food decreases by 1 per turn. Manhattan distance is 26 but starting food is 20, so player must eat to survive.
