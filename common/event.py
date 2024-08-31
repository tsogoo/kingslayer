class EventManager:
    def __init__(self):
        self._listeners = {}

    def register(self, event_type, listener):
        """Register a listener for a specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(listener)

    def unregister(self, event_type, listener):
        """Unregister a listener from a specific event type."""
        if event_type in self._listeners:
            self._listeners[event_type].remove(listener)

    def trigger(self, event_type, *args, **kwargs):
        """Trigger an event and notify all listeners."""
        if event_type in self._listeners:
            for listener in self._listeners[event_type]:
                listener(*args, **kwargs)