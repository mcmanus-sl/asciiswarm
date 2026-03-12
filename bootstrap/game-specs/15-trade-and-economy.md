# Game Spec 15: Trade & Economy

## Overview
A merchant game. The player buys goods cheaply in one town, transports them across a dangerous road, and sells them at a profit in another town. The player must accumulate enough gold to purchase a ship ticket (the "exit"). Prices fluctuate based on supply. This introduces economic reasoning — buy low, sell high, manage inventory — the core loop of DF's trade depot.

## Grid
- Dimensions: 30×12

## GAME_CONFIG

```python
GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'buy', 'sell'],
    'grid': (30, 12),
    'max_turns': 600,
    'step_penalty': -0.003,
    'player_properties': [
        {'key': 'gold', 'max': 100},
        {'key': 'goods_a', 'max': 5},
        {'key': 'goods_b', 'max': 5},
    ],
}
```

## Layout

- **Town A** (x=0–7): Western town. Sells Goods A cheaply, buys Goods B at premium.
- **Road** (x=8–21): Dangerous path with bandits. Scattered rest stops.
- **Town B** (x=22–29): Eastern town. Sells Goods B cheaply, buys Goods A at premium.

## Entities

| Type | Tags | Glyph | Z-Order | Spawning |
|------|------|-------|---------|----------|
| `player` | `player` | `@` | 10 | Town A center |
| `wall` | `solid` | `#` | 1 | Outer boundary, town walls, road barriers |
| `merchant_a` | `npc` | `M` | 5 | Inside Town A. Sells Goods A for 5 gold, buys Goods B for 12 gold. |
| `merchant_b` | `npc` | `M` | 5 | Inside Town B. Sells Goods B for 5 gold, buys Goods A for 12 gold. |
| `bandit` | `hazard` | `b` | 5 | 3–4 on the road. Chase behavior within Manhattan distance 4. |
| `rest_stop` | `npc` | `R` | 3 | 2 along the road. Interact to heal. |
| `harbor` | `exit` | `H` | 5 | Town B harbor. Requires 50 gold to board (interact). |

## Player Properties

| Key | Initial Value | Max | Description |
|-----|--------------|-----|-------------|
| `gold` | 15 | 100 | Currency. Start with enough for 3 goods. |
| `goods_a` | 0 | 5 | Goods from Town A |
| `goods_b` | 0 | 5 | Goods from Town B |

## Behaviors

### `bandit`
If player within Manhattan distance 4, move toward player. Otherwise, patrol: random cardinal movement. HP=2, Attack=1. On death, drop 3 gold (increase player's `gold`).

## Price System

Prices are deterministic and based on remaining stock:
- `merchant_a` sells Goods A: base price 5. Each purchase increases price by 1 (tracked as `sold_count` property on merchant). Buys Goods B: base price 12, decreases by 1 per sale.
- `merchant_b` mirrors: sells Goods B at 5+sold_count, buys Goods A at 12-bought_count.
- Prices floor at 3 (minimum) and cap at 20 (maximum).

## Event Handlers

### `input` (Movement)
Standard 4-direction movement.

### `input` (Buy)
If action is `buy` and player is adjacent to a merchant:
- If `merchant_a`: calculate price. If `gold >= price` and `goods_a < 5`: decrement gold, increment goods_a, increment merchant's sold_count.
- If `merchant_b`: same for goods_b.
- Emit `reward` `{ 'amount': 0.05 }` on successful purchase.

### `input` (Sell)
If action is `sell` and player is adjacent to a merchant:
- If `merchant_a` and `goods_b >= 1`: calculate buy price. Increment gold, decrement goods_b, increment merchant's bought_count.
- If `merchant_b` and `goods_a >= 1`: same logic.
- Emit `reward` `{ 'amount': 0.1 }` on sale.

### `input` (Interact)
- Adjacent to `rest_stop`: heal 2 HP (player has health property implicitly via combat — add `health` max 5 to player_properties if needed, start at 5).
- Standing on `harbor` and `gold >= 50`: `env.end_game('won')`. Emit `reward` `{ 'amount': 1.0 }`.

### `collision` (combat)
Bandit collision: cancel move, mutual damage (same as spec 04). Player HP=5.

### `before_move`
Standard solid blocking.

## Win Condition
Accumulate 50 gold and interact with the harbor.

## Lose Condition
Player health reaches 0 (bandit combat).

## RL Evaluation Criteria

| Metric | Expected Range |
|--------|---------------|
| Random agent win rate | 0% |
| PPO win rate at 500k steps | >3% |
| PPO learning delta (500k - 100k) | >2% |

## Invariant Tests

1. Exactly one merchant in each town.
2. Player starts with 15 gold.
3. Harbor exists in Town B.
4. 3–4 bandits exist on the road.
5. Merchant starting prices are correct (sold_count=0, bought_count=0).

## Notes
- The optimal strategy requires 3–4 round trips (buy 5 goods for ~25-35 gold, sell for ~50-60, minus bandit encounters). The agent must learn that the road is a means, not an end.
- Diminishing returns on prices force the agent to make round trips rather than buying everything at once.
- Bandits dropping gold creates an alternative income source for combat-oriented strategies.
- This is the first game where the `buy` and `sell` actions matter — purely economic reasoning.
- The 50-gold target is calibrated so that 2-3 successful trade runs (buying at ~5-7, selling at ~10-12 each) are needed.
