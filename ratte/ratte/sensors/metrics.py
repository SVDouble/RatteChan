import asyncio

__all__ = ["Metrics"]


class Metrics:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._counters = {}

    async def increment(self, name: str, value: int = 1):
        """Increment the specified counter by a given value."""
        async with self._lock:
            if name not in self._counters:
                self._counters[name] = 0
            self._counters[name] += value

    async def reset(self):
        """Reset all counters and return their values."""
        async with self._lock:
            snapshot = self._counters.copy()
            self._counters = {name: 0 for name in self._counters}
        return snapshot

    async def get(self, name: str):
        """Get the current value of a specific counter."""
        async with self._lock:
            return self._counters.get(name, 0)
