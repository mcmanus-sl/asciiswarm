"""Game 15: Trade & Economy — collect gold, trade goods, and board the ship."""

from asciiswarm.kernel.invariants import (
    invariant, InvariantError, check_exit_exists, Invariant,
)

# Exit exists but requires gold to use.
INVARIANTS = [Invariant('exit_exists', check_exit_exists)]

GAME_CONFIG = {
    'tags': ['player', 'solid', 'hazard', 'pickup', 'exit', 'npc'],
    'actions': ['move_n', 'move_s', 'move_e', 'move_w', 'interact', 'wait',
                'buy', 'sell'],
    'grid': (20, 10),
    'max_turns': 400,
    'step_penalty': -0.005,
    'player_properties': [
        {'key': 'gold', 'max': 100},
        {'key': 'goods_a', 'max': 5},
        {'key': 'goods_b', 'max': 5},
        {'key': 'health', 'max': 5},
    ],
}


def setup(env):
    w, h = env.config['grid']  # 20, 10

    # ---- Outer boundary walls only ----
    for x in range(w):
        env.create_entity('wall', x, 0, '#', ['solid'], z_order=1)
        env.create_entity('wall', x, h - 1, '#', ['solid'], z_order=1)
    for y in range(1, h - 1):
        env.create_entity('wall', 0, y, '#', ['solid'], z_order=1)
        env.create_entity('wall', w - 1, y, '#', ['solid'], z_order=1)

    occupied = set()
    # Add boundary to occupied
    for x in range(w):
        occupied.add((x, 0))
        occupied.add((x, h - 1))
    for y in range(h):
        occupied.add((0, y))
        occupied.add((w - 1, y))

    # ---- Player at left side ----
    px, py = 2, 5
    player = env.create_entity('player', px, py, '@', ['player'], z_order=10,
                               properties={'gold': 15, 'goods_a': 0, 'goods_b': 0, 'health': 5})
    occupied.add((px, py))

    # ---- Merchant A near left (Town A area) ----
    mx_a, my_a = 3, 3
    env.create_entity('merchant_a', mx_a, my_a, 'M', ['npc'], z_order=5,
                      properties={'sold_count': 0, 'bought_count': 0})
    occupied.add((mx_a, my_a))

    # ---- Merchant B near right (Town B area) ----
    mx_b, my_b = 16, 3
    env.create_entity('merchant_b', mx_b, my_b, 'M', ['npc'], z_order=5,
                      properties={'sold_count': 0, 'bought_count': 0})
    occupied.add((mx_b, my_b))

    # ---- Harbor (exit) at right side ----
    hx, hy = 18, 5
    env.create_entity('harbor', hx, hy, 'H', ['exit'], z_order=5)
    occupied.add((hx, hy))

    # ---- Rest stops ----
    for _ in range(2):
        for _attempt in range(50):
            rx = 7 + int(env.random() * 6)  # x in [7, 12]
            ry = 1 + int(env.random() * (h - 2))
            if (rx, ry) not in occupied:
                occupied.add((rx, ry))
                env.create_entity('rest_stop', rx, ry, 'R', ['npc'], z_order=3)
                break

    # ---- Gold pickups scattered across the map (main path to winning) ----
    # 8 pickups × 5 gold = 40. With starting 15 gold, need ~7 pickups to reach 50.
    num_gold_piles = 8
    for i in range(num_gold_piles):
        for _attempt in range(100):
            # Spread gold across the map, biased toward middle and right
            gx = 3 + int(env.random() * (w - 5))
            gy = 1 + int(env.random() * (h - 2))
            if (gx, gy) not in occupied:
                occupied.add((gx, gy))
                env.create_entity('gold_pile', gx, gy, '$', ['pickup'], z_order=3,
                                  properties={'gold_amount': 5})
                break

    # ---- Bandits (fewer, on the road) ----
    num_bandits = 3 + int(env.random() * 2)  # 3-4
    for _ in range(num_bandits):
        for _attempt in range(50):
            bx = 6 + int(env.random() * 10)  # x in [6, 15]
            by = 1 + int(env.random() * (h - 2))
            if (bx, by) not in occupied:
                occupied.add((bx, by))
                env.create_entity('bandit', bx, by, 'b', ['hazard'], z_order=5,
                                  properties={'health': 2, 'attack': 1})
                break

    # Track gold for milestone rewards
    last_gold_milestone = [15]  # starting gold, mutable via closure

    # ---- Input handler ----
    def on_input(event):
        p = env.get_entities_by_tag('player')
        if not p:
            return
        p = p[0]
        action = event.payload['action']

        moves = {
            'move_n': (0, -1),
            'move_s': (0, 1),
            'move_e': (1, 0),
            'move_w': (-1, 0),
        }

        if action in moves:
            dx, dy = moves[action]
            env.move_entity(p.id, p.x + dx, p.y + dy)
            # Check gold milestones after move (pickup may have triggered)
            _check_gold_milestone(env, p, last_gold_milestone)
            return

        if action == 'interact':
            _handle_interact(env, p)
            _check_gold_milestone(env, p, last_gold_milestone)
            return

        if action == 'buy':
            _handle_buy(env, p)
            _check_gold_milestone(env, p, last_gold_milestone)
            return

        if action == 'sell':
            _handle_sell(env, p)
            _check_gold_milestone(env, p, last_gold_milestone)
            return

    env.on('input', on_input)

    # ---- Before move: solids block ----
    def on_before_move(event):
        tx, ty = event.payload['to_x'], event.payload['to_y']
        for ent in env.get_entities_at(tx, ty):
            if ent.has_tag('solid'):
                event.cancel()
                return

    env.on('before_move', on_before_move)

    # ---- Collision handler ----
    def on_collision(event):
        mover = event.payload['mover']
        occupants = event.payload['occupants']

        if mover.has_tag('player'):
            for occ in occupants:
                if occ.has_tag('hazard'):
                    # Combat with bandit
                    event.cancel()
                    occ_atk = occ.properties.get('attack', 1)
                    p_atk = 2  # player attack power

                    mover.properties['health'] = mover.properties.get('health', 5) - occ_atk
                    occ.properties['health'] = occ.properties.get('health', 2) - p_atk

                    if occ.properties['health'] <= 0:
                        # Bandit dies, drop 5 gold
                        mover.properties['gold'] = min(
                            mover.properties.get('gold', 0) + 5, 100)
                        env.destroy_entity(occ.id)
                        env.emit('reward', {'amount': 0.2})

                    if mover.properties['health'] <= 0:
                        env.end_game('lost')
                    return

                elif occ.has_tag('pickup'):
                    # Gold pile pickup
                    gold_amt = occ.properties.get('gold_amount', 5)
                    mover.properties['gold'] = min(
                        mover.properties.get('gold', 0) + gold_amt, 100)
                    env.destroy_entity(occ.id)
                    env.emit('reward', {'amount': 0.2})

                elif occ.has_tag('exit'):
                    # Harbor: need 50 gold to board
                    if mover.properties.get('gold', 0) >= 50:
                        env.emit('reward', {'amount': 1.0})
                        env.end_game('won')

        elif mover.has_tag('hazard'):
            for occ in occupants:
                if occ.has_tag('player'):
                    event.cancel()
                    m_atk = mover.properties.get('attack', 1)
                    p_atk = 2

                    occ.properties['health'] = occ.properties.get('health', 5) - m_atk
                    mover.properties['health'] = mover.properties.get('health', 2) - p_atk

                    if mover.properties['health'] <= 0:
                        occ.properties['gold'] = min(
                            occ.properties.get('gold', 0) + 5, 100)
                        env.destroy_entity(mover.id)

                    if occ.properties['health'] <= 0:
                        env.end_game('lost')
                    return

    env.on('collision', on_collision)

    # ---- Bandit behavior ----
    def bandit_behavior(entity, env):
        players = env.get_entities_by_tag('player')
        if not players:
            return
        p = players[0]
        dist = abs(entity.x - p.x) + abs(entity.y - p.y)

        if dist <= 4:
            # Chase player
            dx_abs = abs(p.x - entity.x)
            dy_abs = abs(p.y - entity.y)
            if dx_abs >= dy_abs:
                step_x = 1 if p.x > entity.x else -1
                env.move_entity(entity.id, entity.x + step_x, entity.y)
            else:
                step_y = 1 if p.y > entity.y else -1
                env.move_entity(entity.id, entity.x, entity.y + step_y)
        else:
            # Random patrol
            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            idx = int(env.random() * 4)
            dx, dy = directions[idx]
            env.move_entity(entity.id, entity.x + dx, entity.y + dy)

    env.register_behavior('bandit', bandit_behavior)


def _check_gold_milestone(env, player, last_milestone):
    """Emit reward for every 10 gold gained above last milestone."""
    gold = player.properties.get('gold', 0)
    while gold >= last_milestone[0] + 10:
        last_milestone[0] += 10
        env.emit('reward', {'amount': 0.3})


def _get_adjacent_merchant(env, player, merchant_type):
    """Find a merchant of given type adjacent to player."""
    merchants = env.get_entities_by_type(merchant_type)
    for m in merchants:
        if abs(player.x - m.x) + abs(player.y - m.y) <= 1:
            return m
    return None


def _handle_interact(env, player):
    """Handle interact action: rest stops and harbor."""
    # Check adjacent rest stops
    for ent in _get_adjacent_entities(env, player):
        if ent.type == 'rest_stop':
            player.properties['health'] = min(
                player.properties.get('health', 5) + 2, 5)
            env.emit('reward', {'amount': 0.05})
            return

    # Check standing on harbor
    for ent in env.get_entities_at(player.x, player.y):
        if ent.has_tag('exit'):
            if player.properties.get('gold', 0) >= 50:
                env.emit('reward', {'amount': 1.0})
                env.end_game('won')
            return


def _handle_buy(env, player):
    """Handle buy action: purchase goods from adjacent merchant."""
    # Check merchant_a
    m = _get_adjacent_merchant(env, player, 'merchant_a')
    if m:
        sold_count = m.properties.get('sold_count', 0)
        price = max(3, min(20, 5 + sold_count))
        gold = player.properties.get('gold', 0)
        goods_a = player.properties.get('goods_a', 0)
        if gold >= price and goods_a < 5:
            player.properties['gold'] = gold - price
            player.properties['goods_a'] = goods_a + 1
            m.properties['sold_count'] = sold_count + 1
            env.emit('reward', {'amount': 0.05})
        return

    # Check merchant_b
    m = _get_adjacent_merchant(env, player, 'merchant_b')
    if m:
        sold_count = m.properties.get('sold_count', 0)
        price = max(3, min(20, 5 + sold_count))
        gold = player.properties.get('gold', 0)
        goods_b = player.properties.get('goods_b', 0)
        if gold >= price and goods_b < 5:
            player.properties['gold'] = gold - price
            player.properties['goods_b'] = goods_b + 1
            m.properties['sold_count'] = sold_count + 1
            env.emit('reward', {'amount': 0.05})
        return


def _handle_sell(env, player):
    """Handle sell action: sell goods to adjacent merchant."""
    # Merchant A buys Goods B
    m = _get_adjacent_merchant(env, player, 'merchant_a')
    if m:
        bought_count = m.properties.get('bought_count', 0)
        buy_price = max(3, min(20, 12 - bought_count))
        goods_b = player.properties.get('goods_b', 0)
        if goods_b >= 1:
            player.properties['gold'] = min(
                player.properties.get('gold', 0) + buy_price, 100)
            player.properties['goods_b'] = goods_b - 1
            m.properties['bought_count'] = bought_count + 1
            env.emit('reward', {'amount': 0.1})
        return

    # Merchant B buys Goods A
    m = _get_adjacent_merchant(env, player, 'merchant_b')
    if m:
        bought_count = m.properties.get('bought_count', 0)
        buy_price = max(3, min(20, 12 - bought_count))
        goods_a = player.properties.get('goods_a', 0)
        if goods_a >= 1:
            player.properties['gold'] = min(
                player.properties.get('gold', 0) + buy_price, 100)
            player.properties['goods_a'] = goods_a - 1
            m.properties['bought_count'] = bought_count + 1
            env.emit('reward', {'amount': 0.1})
        return


def _get_adjacent_entities(env, entity):
    """Get all entities adjacent to (including same cell as) the given entity."""
    results = []
    for dx, dy in [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]:
        nx, ny = entity.x + dx, entity.y + dy
        for ent in env.get_entities_at(nx, ny):
            if ent.id != entity.id:
                results.append(ent)
    return results


# ---- Game-specific invariants ----

@invariant('one_merchant_per_town')
def check_merchants(env):
    ma = env.get_entities_by_type('merchant_a')
    mb = env.get_entities_by_type('merchant_b')
    if len(ma) != 1:
        raise InvariantError(f"Expected 1 merchant_a, found {len(ma)}")
    if len(mb) != 1:
        raise InvariantError(f"Expected 1 merchant_b, found {len(mb)}")


@invariant('player_starts_with_15_gold')
def check_player_gold(env):
    p = env.get_entities_by_tag('player')[0]
    if p.properties.get('gold') != 15:
        raise InvariantError(f"Player gold is {p.properties.get('gold')}, expected 15")


@invariant('harbor_exists')
def check_harbor(env):
    harbors = env.get_entities_by_type('harbor')
    if len(harbors) != 1:
        raise InvariantError(f"Expected 1 harbor, found {len(harbors)}")


@invariant('bandits_exist')
def check_bandits(env):
    bandits = env.get_entities_by_type('bandit')
    if not (3 <= len(bandits) <= 4):
        raise InvariantError(f"Expected 3-4 bandits, found {len(bandits)}")


@invariant('merchant_prices_correct')
def check_merchant_prices(env):
    for mtype in ['merchant_a', 'merchant_b']:
        m = env.get_entities_by_type(mtype)[0]
        if m.properties.get('sold_count', 0) != 0:
            raise InvariantError(f"{mtype} sold_count is {m.properties.get('sold_count')}, expected 0")
        if m.properties.get('bought_count', 0) != 0:
            raise InvariantError(f"{mtype} bought_count is {m.properties.get('bought_count')}, expected 0")


@invariant('gold_piles_exist')
def check_gold_piles(env):
    piles = env.get_entities_by_type('gold_pile')
    if len(piles) < 6:
        raise InvariantError(f"Expected at least 6 gold piles, found {len(piles)}")
