import json
import pytest
from types import SimpleNamespace

from asciiswarm.kernel.env import GridGameEnv


def _make_module(config=None, setup=None):
    mod = SimpleNamespace()
    mod.GAME_CONFIG = config or {'grid': (8, 8)}
    mod.setup = setup or (lambda env: env.create_entity('player', 0, 0, '@', ['player']))
    return mod


def test_serialize_round_trip():
    """Serialize → load → serialize produces identical output."""
    mod = _make_module()
    env = GridGameEnv(mod)
    env.reset(seed=42)

    state1 = env.serialize_state()
    json1 = json.dumps(state1, sort_keys=True)

    env.load_state(state1)
    state2 = env.serialize_state()
    json2 = json.dumps(state2, sort_keys=True)

    assert json1 == json2


def test_load_state_entities_have_methods():
    """Entities after load_state have working set/get methods."""
    mod = _make_module()
    env = GridGameEnv(mod)
    env.reset(seed=42)

    player = env.get_entities_by_tag('player')[0]
    player.set('health', 5)
    state = env.serialize_state()

    env.load_state(state)
    loaded_player = env.get_entities_by_tag('player')[0]
    assert loaded_player.get('health') == 5
    loaded_player.set('armor', 3)
    assert loaded_player.get('armor') == 3


def test_prng_survives_round_trip():
    """PRNG state is preserved: random() sequence continues correctly."""
    mod = _make_module()
    env = GridGameEnv(mod)
    env.reset(seed=42)

    # Advance PRNG a few times
    for _ in range(10):
        env.random()

    state = env.serialize_state()

    # Get next values from original
    expected = [env.random() for _ in range(5)]

    # Load state and get same values
    env.load_state(state)
    actual = [env.random() for _ in range(5)]

    assert expected == actual


def test_entity_id_counter_survives():
    """Entity ID counter is preserved across serialize/load."""
    mod = _make_module()
    env = GridGameEnv(mod)
    env.reset(seed=42)

    # Create some more entities
    env.create_entity('wall', 1, 1, '#', ['solid'])
    env.create_entity('wall', 2, 2, '#', ['solid'])
    state = env.serialize_state()

    env.load_state(state)
    new_ent = env.create_entity('wall', 3, 3, '#', ['solid'])
    # Should continue from where we left off (e1=player, e2=wall, e3=wall, so next is e4)
    assert new_ent.id == 'e4'


def test_turn_number_survives():
    mod = _make_module()
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.handle_input('wait')
    env.handle_input('wait')
    assert env.turn_number == 2

    state = env.serialize_state()
    env.load_state(state)
    assert env.turn_number == 2


def test_game_status_survives():
    mod = _make_module()
    env = GridGameEnv(mod)
    env.reset(seed=42)
    env.end_game('won')

    state = env.serialize_state()
    env.load_state(state)
    assert env.status == 'won'


def test_load_state_does_not_call_setup():
    """load_state must NOT re-run setup — no duplicate entities."""
    call_count = [0]

    def counting_setup(env):
        call_count[0] += 1
        env.create_entity('player', 0, 0, '@', ['player'])

    mod = _make_module(setup=counting_setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)
    assert call_count[0] == 1

    state = env.serialize_state()
    env.load_state(state)
    assert call_count[0] == 1  # setup was NOT called again

    # Should have exactly 1 player, not 2
    players = env.get_entities_by_tag('player')
    assert len(players) == 1


def test_load_state_determinism():
    """Load state, run N turns, serialize. Repeat. Output identical."""
    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('exit', 7, 7, '>', ['exit'])

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)

    state = env.serialize_state()

    # Run 10 turns from this state
    env.load_state(state)
    for _ in range(10):
        env.handle_input('wait')
    result1 = json.dumps(env.serialize_state(), sort_keys=True)

    # Repeat from same state
    env.load_state(state)
    for _ in range(10):
        env.handle_input('wait')
    result2 = json.dumps(env.serialize_state(), sort_keys=True)

    assert result1 == result2


def test_serialized_entities_sorted_by_id():
    """Entities in serialized output are sorted by numeric ID."""
    def setup(env):
        env.create_entity('player', 0, 0, '@', ['player'])
        env.create_entity('wall', 1, 0, '#', ['solid'])
        env.create_entity('exit', 2, 0, '>', ['exit'])

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)

    state = env.serialize_state()
    ids = [e['id'] for e in state['entities']]
    assert ids == ['e1', 'e2', 'e3']


def test_serialized_properties_sorted():
    """Property keys within entities are sorted alphabetically."""
    def setup(env):
        p = env.create_entity('player', 0, 0, '@', ['player'])
        p.set('zebra', 1)
        p.set('alpha', 2)
        p.set('middle', 3)

    mod = _make_module(setup=setup)
    env = GridGameEnv(mod)
    env.reset(seed=42)

    state = env.serialize_state()
    props = state['entities'][0]['properties']
    keys = list(props.keys())
    assert keys == ['alpha', 'middle', 'zebra']
