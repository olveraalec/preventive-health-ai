"""Thread-safe in-process runtime metrics.

No patient identifiers or raw telemetry are stored here.
"""

from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class MetricsRegistry:
    def __init__(self):
        self._lock = Lock()
        self._counters = defaultdict(int)
        self._timings = defaultdict(list)

    def increment(self, name, amount=1):
        if amount < 0:
            raise ValueError("Counter increment must be nonnegative")

        with self._lock:
            self._counters[name] += amount

    def observe(self, name, seconds):
        if seconds < 0:
            raise ValueError("Duration cannot be negative")

        with self._lock:
            self._timings[name].append(float(seconds))

    @contextmanager
    def track(self, name):
        start = perf_counter()
        try:
            yield
        except Exception:
            self.increment(f"{name}_failures_total")
            raise
        finally:
            self.observe(
                f"{name}_latency_seconds",
                perf_counter() - start,
            )

    def snapshot(self):
        with self._lock:
            return {
                "counters": dict(self._counters),
                "timings": {
                    name: {
                        "count": len(values),
                        "mean_seconds": sum(values) / len(values),
                        "max_seconds": max(values),
                    }
                    for name, values in self._timings.items()
                    if values
                },
            }


registry = MetricsRegistry()
