import pytest
from asciiswarm.kernel.entity import Entity


def test_create_entity():
    e = Entity('e1', 'player', 3, 4, '@', ['player'], z_order=10)
    assert e.id == 'e1'
    assert e.type == 'player'
    assert e.x == 3
    assert e.y == 4
    assert e.glyph == '@'
    assert e.tags == ['player']
    assert e.z_order == 10
    assert e.properties == {}


def test_entity_set_get():
    e = Entity('e1', 'player', 0, 0, '@', ['player'])
    e.set('health', 10)
    assert e.get('health') == 10


def test_entity_get_default():
    e = Entity('e1', 'player', 0, 0, '@', ['player'])
    assert e.get('missing') is None
    assert e.get('missing', 42) == 42


def test_entity_has_tag():
    e = Entity('e1', 'wall', 0, 0, '#', ['solid', 'hazard'])
    assert e.has_tag('solid')
    assert e.has_tag('hazard')
    assert not e.has_tag('player')


def test_entity_properties_from_constructor():
    e = Entity('e1', 'player', 0, 0, '@', ['player'], properties={'health': 10, 'attack': 2})
    assert e.get('health') == 10
    assert e.get('attack') == 2


def test_entity_validate_serializable():
    e = Entity('e1', 'player', 0, 0, '@', ['player'])
    e.set('health', 10)
    e.set('name', 'hero')
    e.validate_properties_serializable()  # should not raise


def test_entity_validate_non_serializable():
    e = Entity('e1', 'player', 0, 0, '@', ['player'])
    e.set('callback', lambda: None)
    with pytest.raises(ValueError, match="non-JSON-serializable"):
        e.validate_properties_serializable()


def test_entity_repr():
    e = Entity('e1', 'player', 3, 4, '@', ['player'])
    r = repr(e)
    assert 'e1' in r
    assert 'player' in r
