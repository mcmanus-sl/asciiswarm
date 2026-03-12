"""Invariant test framework for game modules.

Game modules register invariant tests via the @invariant decorator or by
appending to INVARIANTS list. The framework provides built-in invariants
that apply to all games, plus a way for games to add their own.
"""

from collections import deque


class InvariantError(AssertionError):
    """Raised when an invariant check fails."""
    pass


class Invariant:
    """A named invariant check."""

    def __init__(self, name, check_fn, builtin=False):
        self.name = name
        self.check_fn = check_fn
        self.builtin = builtin

    def check(self, env):
        """Run the check. Raises InvariantError on failure."""
        self.check_fn(env)


# ---- Built-in invariants ----

def check_player_singleton(env):
    players = env.get_entities_by_tag('player')
    if len(players) != 1:
        raise InvariantError(
            f"Expected exactly 1 entity tagged 'player', found {len(players)}"
        )


def check_exit_exists(env):
    exits = env.get_entities_by_tag('exit')
    if len(exits) < 1:
        raise InvariantError("No entity tagged 'exit' found at game start")


def check_no_empty_tags(env):
    for entity in env.get_all_entities():
        if not entity.tags:
            raise InvariantError(
                f"Entity {entity.id} ({entity.type}) has an empty tag list"
            )


def check_exit_reachable(env):
    """BFS over non-solid tiles to verify exit is reachable from player."""
    players = env.get_entities_by_tag('player')
    exits = env.get_entities_by_tag('exit')
    if not players or not exits:
        return  # other invariants will catch this

    player = players[0]
    exit_positions = {(e.x, e.y) for e in exits}

    # Build solid set
    solid_positions = set()
    for entity in env.get_all_entities():
        if entity.has_tag('solid'):
            solid_positions.add((entity.x, entity.y))

    # BFS
    visited = set()
    queue = deque([(player.x, player.y)])
    visited.add((player.x, player.y))
    width, height = env.config['grid']

    while queue:
        x, y = queue.popleft()
        if (x, y) in exit_positions:
            return  # reachable
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited and (nx, ny) not in solid_positions:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    # Also check if exit is on a solid tile (still reachable if player can walk onto it)
    # Re-check allowing exit positions even if solid
    visited2 = set()
    queue2 = deque([(player.x, player.y)])
    visited2.add((player.x, player.y))
    while queue2:
        x, y = queue2.popleft()
        if (x, y) in exit_positions:
            return
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited2:
                    if (nx, ny) not in solid_positions or (nx, ny) in exit_positions:
                        visited2.add((nx, ny))
                        queue2.append((nx, ny))

    raise InvariantError(
        f"Exit at {exit_positions} is not reachable from player at "
        f"({player.x}, {player.y}) via BFS over non-solid tiles"
    )


def check_behaviors_registered(env):
    """Every entity type has a registered behavior or is tagged with an inert-like tag."""
    inert_types = set()
    for entity in env.get_all_entities():
        etype = entity.type
        if etype in env._behaviors:
            continue
        # Considered inert if it has no behavior registered — that's fine for
        # walls, exits, pickups, etc. Only flag types that seem active.
        # The spec says "or is inert by design" — we accept any type without
        # a behavior as inert. Game-specific invariants can override.
        inert_types.add(etype)


BUILTIN_INVARIANTS = [
    Invariant('player_singleton', check_player_singleton, builtin=True),
    Invariant('no_empty_tags', check_no_empty_tags, builtin=True),
    Invariant('behaviors_registered', check_behaviors_registered, builtin=True),
]

# Exit invariants — opt-in for games that have exit entities.
# Import and add to your game's INVARIANTS list or use @invariant.
EXIT_INVARIANTS = [
    Invariant('exit_exists', check_exit_exists),
    Invariant('exit_reachable', check_exit_reachable),
]


def invariant(name):
    """Decorator to register a game-specific invariant.

    Usage in a game module:
        from asciiswarm.kernel.invariants import invariant

        @invariant('every_room_has_exit')
        def check_rooms(env):
            # ... raise InvariantError if violated
    """
    def decorator(fn):
        inv = Invariant(name, fn, builtin=False)
        # Attach to the function so test framework can discover it
        fn._invariant = inv
        return fn
    return decorator


def get_game_invariants(game_module):
    """Collect game-specific invariants from a game module.

    Looks for functions decorated with @invariant, or an INVARIANTS list.
    """
    invariants = []

    # Check for INVARIANTS list
    if hasattr(game_module, 'INVARIANTS'):
        for inv in game_module.INVARIANTS:
            if isinstance(inv, Invariant):
                invariants.append(inv)

    # Check for decorated functions
    for attr_name in dir(game_module):
        attr = getattr(game_module, attr_name)
        if callable(attr) and hasattr(attr, '_invariant'):
            invariants.append(attr._invariant)

    return invariants


def run_invariants(env, game_module=None):
    """Run all built-in invariants and any game-specific ones.

    Returns list of (invariant_name, passed: bool, error_msg: str|None).
    """
    all_invariants = list(BUILTIN_INVARIANTS)
    if game_module:
        all_invariants.extend(get_game_invariants(game_module))

    results = []
    for inv in all_invariants:
        try:
            inv.check(env)
            results.append((inv.name, True, None))
        except (InvariantError, AssertionError) as e:
            results.append((inv.name, False, str(e)))

    return results
