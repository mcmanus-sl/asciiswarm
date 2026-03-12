import pytest
from asciiswarm.kernel.events import EventSystem, Event


def test_handler_fires():
    es = EventSystem()
    results = []
    es.on('test', lambda e: results.append(e.payload))
    es.emit('test', {'val': 1})
    assert results == [{'val': 1}]


def test_handlers_fire_in_registration_order():
    es = EventSystem()
    order = []
    es.on('test', lambda e: order.append('a'))
    es.on('test', lambda e: order.append('b'))
    es.on('test', lambda e: order.append('c'))
    es.emit('test')
    assert order == ['a', 'b', 'c']


def test_unsubscribe():
    es = EventSystem()
    results = []
    unsub = es.on('test', lambda e: results.append(1))
    es.emit('test')
    assert results == [1]
    unsub()
    es.emit('test')
    assert results == [1]  # handler was removed


def test_unsubscribe_mid_iteration():
    """Unsubscribing during handler execution doesn't crash."""
    es = EventSystem()
    results = []
    unsub = [None]

    def handler_a(e):
        results.append('a')
        unsub[0]()  # unsubscribe self during execution

    unsub[0] = es.on('test', handler_a)
    es.on('test', lambda e: results.append('b'))
    es.emit('test')
    assert results == ['a', 'b']
    # Second emit: handler_a should be gone
    results.clear()
    es.emit('test')
    assert results == ['b']


def test_cancellable_before_move():
    es = EventSystem()
    es.on('before_move', lambda e: e.cancel())
    event = es.emit('before_move', {'entity': None})
    assert event.cancelled


def test_cancellable_collision():
    es = EventSystem()
    es.on('collision', lambda e: e.cancel())
    event = es.emit('collision', {'mover': None, 'occupants': []})
    assert event.cancelled


def test_non_cancellable_event():
    es = EventSystem()
    es.on('test', lambda e: e.cancel())
    event = es.emit('test')
    assert not event.cancelled  # cancel() is a no-op for non-cancellable


def test_custom_event():
    es = EventSystem()
    results = []
    es.on('my_custom_event', lambda e: results.append(e.payload['data']))
    es.emit('my_custom_event', {'data': 'hello'})
    assert results == ['hello']


def test_cascade_depth_guard():
    es = EventSystem(max_cascade_depth=5)

    def recursive_handler(e):
        es.emit('test')

    es.on('test', recursive_handler)
    with pytest.raises(RuntimeError, match="cascade depth"):
        es.emit('test')


def test_clear():
    es = EventSystem()
    results = []
    es.on('test', lambda e: results.append(1))
    es.emit('test')
    assert results == [1]
    es.clear()
    es.emit('test')
    assert results == [1]  # handlers cleared


def test_empty_payload():
    es = EventSystem()
    results = []
    es.on('test', lambda e: results.append(e.payload))
    es.emit('test')
    assert results == [{}]
