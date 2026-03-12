class Event:
    """An event object passed to handlers. Some events support cancellation."""

    def __init__(self, name, payload, cancellable=False):
        self.name = name
        self.payload = payload
        self.cancellable = cancellable
        self.cancelled = False

    def cancel(self):
        if self.cancellable:
            self.cancelled = True


CANCELLABLE_EVENTS = frozenset({'before_move', 'collision'})


class EventSystem:
    """Simple pub/sub event system with ordered handlers and cancellation support."""

    def __init__(self, max_cascade_depth=1000):
        self._handlers = {}  # event_name -> list of handlers
        self._cascade_depth = 0
        self.max_cascade_depth = max_cascade_depth

    def on(self, event_name, handler):
        """Register a handler. Returns an unsubscribe callable."""
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)

        def unsubscribe():
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def emit(self, event_name, payload=None):
        """Emit an event. Returns the Event object (check .cancelled for cancellable events)."""
        self._cascade_depth += 1
        if self._cascade_depth > self.max_cascade_depth:
            self._cascade_depth -= 1
            raise RuntimeError(
                f"Event cascade depth exceeded {self.max_cascade_depth}. "
                f"Possible infinite event loop."
            )

        try:
            cancellable = event_name in CANCELLABLE_EVENTS
            event = Event(event_name, payload or {}, cancellable=cancellable)

            handlers = list(self._handlers.get(event_name, []))
            for handler in handlers:
                handler(event)
            return event
        finally:
            self._cascade_depth -= 1

    def clear(self):
        """Remove all handlers. Called on reset to prevent handler accumulation."""
        self._handlers.clear()
        self._cascade_depth = 0
