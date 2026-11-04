"""Wall-clock timer."""
import time


class Timer:
    """Context manager; read .seconds after exit."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
