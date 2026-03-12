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

## Game 11: Multi-Floor Dungeon
- **Status:** COMPLETE
- **File:** `games/11_multi_floor_dungeon.py`
- **Tests:** `tests/games/test_11_multi_floor_dungeon.py` — 16/16 pass
- **Full suite:** 286/286 passed
- **Notes:** 36x12 grid (three 12x12 zones side by side). Player starts in floor 1. Stairs_down/stairs_up pairs connect floors via interact action (direct teleport, no move_entity). Floor 1: 1-2 wanderers, 1 potion. Floor 2: 2-3 chasers, 1-2 potions, 1 key. Floor 3: 1 sentinel + 2 chasers, 1 potion, locked exit + exit. Combat same as game 04 (player atk=2). Chaser chases on same floor within Manhattan 6. Sentinel stationary, alerts within Manhattan 3. Intermediate rewards: +0.2 stairs, +0.1 pickup/kill, +0.3 unlock door. Tags: pickup for key/potions, exit for goal. Exit invariant (exists only, not reachable — blocked by locked_exit).

## Game 09: Inventory & Crafting
- **Status:** COMPLETE
- **File:** `games/09_inventory_crafting.py`
- **Tests:** `tests/games/test_09_inventory_crafting.py` (14 tests, all passing)
- **Full suite:** 258/258 passed
- **Notes:** 16x16 grid. Player in left third collects wood (4-6) and ore (3-5) scattered in the left section. Workbench in center area (6-9, 6-9). Vertical wall at x=12 with one rubble gap. Exit behind wall (x>=13). Chain: gather 2 wood + 2 ore → craft pickaxe at workbench → mine rubble → reach exit. Pickup rewards +0.05, craft/mine rewards +0.3 each. Items tagged 'pickup' for distance shaping.

## Game 12: NPC Allies & Orders
- **Status:** COMPLETE
- **File:** `games/12_npc_allies_and_orders.py`
- **Tests:** `tests/games/test_12_npc_allies_and_orders.py` — 14/14 pass
- **Full suite:** 317/317 passed
- **RL Eval:** Random agent 0 crashes, 100% termination, 4.7% win rate. PPO early=44%, final=98%, delta=0.54. All 10 invariants pass.
- **Notes:** 20x20 grid. Player at (2,10) in base room (left side). 3 allies start in follow mode inside base. 6-8 trees in middle zone (5<=x<=13). 2-3 raiders on right side (x>=15). Exit at (18,10). Stockpile in base, 3 barricade slots at x=14. Order actions: order_follow, order_guard, order_harvest. Allies harvest trees to stockpile (+1 wood each), guard barricade zone (attack nearby raiders), or follow player. Player can build barricades (2 wood each) at slots via interact. Combat: player HP=5, raider HP=2, ally HP=3. Intermediate rewards: +0.2 barricade build, +0.15 raider kill, +0.1 wood deposit. Tags: pickup for trees, exit for goal.

## Game 10: Farming & Growth
- **Status:** COMPLETE
- **File:** `games/10_farming_and_growth.py`
- **Tests:** `tests/games/test_10_farming_and_growth.py` (19 tests, all passing)
- **Full suite:** 263/263 passed
- **Notes:** 14x14 grid. Player starts in farmhouse (top-left). Seedbag at (2,5) gives 6 seeds. 3x3 soil patch at center (5-7, 5-7). Plant seeds via interact on soil, sprouts grow to mature after 15 turns. Harvest by walking onto mature crops. Deliver crops to bin at (12,1) via interact. Win by delivering 5 crops. No exit entity — game uses pickup tags on seedbag/mature for distance shaping. 5 game-specific invariants.

## Game 13: Ecology & Spawning
- **Status:** COMPLETE
- **File:** `games/13_ecology_and_spawning.py`
- **Tests:** `tests/games/test_13_ecology_and_spawning.py` (15 tests, all passing)
- **Full suite:** 318/318 passed
- **RL Eval:** Random agent: 0 crashes, 100% termination, 0% win. PPO: 14% early -> 41% final, delta=0.27. Invariants: 9/9 passed.
- **Notes:** 20x20 grid. Player at bottom-left, exit at top-right. 8-10 rabbits (prey, reproduce near bushes every 20 turns, cap 15), 2-3 wolves (predators, hunt rabbits within 5 tiles, reproduce every 40 turns, cap 5). Hunger clock: food -1 every 3 turns, hunting rabbits +3 food. Wolf combat: player -2hp, wolf -1hp per hit. Rabbits flee wolves within 3 tiles. 5-7 bushes enable rabbit reproduction. Exit tagged for distance shaping. 6 game-specific invariants + exit_exists.

## Game 14: Fluid & Pressure
- **Status:** COMPLETE
- **File:** `games/14_fluid_and_pressure.py`
- **Tests:** `tests/games/test_14_fluid_and_pressure.py` (17 tests, all passing)
- **Full suite:** 349/349 passed
- **RL Eval:** Random agent: 0 crashes, 100% termination, 4% win. PPO: 98% early -> 100% final, delta=0.02. Invariants: 10/10 passed.
- **Notes:** 20x16 grid. Player at left (1, 8), exit at right (18, 8). 1-2 water sources in upper area spawn water each turn. Water spreads to adjacent empty cells (30% chance per turn, 3-turn maturation delay, capped at 60 tiles). 2-3 pumps (toggle via interact, destroy water within Manhattan 2 when active). 2-3 drains (passively destroy water at end of turn). 1 valve (interact to permanently destroy a water source). Air=10, decreases by 1 per turn on water, resets on dry land. Drowning at air=0 loses. Intermediate rewards: +0.1 pump toggle, +0.3 valve use. Exit tagged for distance shaping. 6 game-specific invariants + exit_exists.

## Game 15: Trade & Economy
- **Status:** COMPLETE
- **File:** `games/15_trade_and_economy.py`
- **Tests:** `tests/games/test_15_trade_and_economy.py` (19 tests, all passing)
- **Full suite:** 368/368 passed
- **RL Eval:** Random agent: 0 crashes, 100% termination, 3.8% win. PPO: 22% early -> 73% final, delta=0.51. Invariants: 10/10 passed.
- **Notes:** 20x10 grid. Player at (2,5) left side, harbor (exit) at (18,5) right side. Two merchants (merchant_a at left, merchant_b at right) with buy/sell actions — goods_a bought cheaply at merchant_a, sold at premium at merchant_b, and vice versa. Prices escalate with sold_count/bought_count. 8 gold pile pickups (5 gold each) scattered across map provide a direct collection path to 50 gold. 3-4 bandits on road with chase behavior (Manhattan distance 4), combat drops 5 gold. 2 rest stops heal +2 HP. Win by reaching harbor with 50+ gold. Gold milestone rewards (+0.3 per 10 gold). Pickups tagged for distance shaping. 6 game-specific invariants + exit_exists.
