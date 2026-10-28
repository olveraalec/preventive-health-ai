import pytest

from healthcare_ai_lab.observability.registry import (
    MetricsRegistry,
)


def test_counter_increment():
    metrics = MetricsRegistry()

    metrics.increment("requests_total")
    metrics.increment("requests_total")

    assert (
        metrics.snapshot()["counters"]["requests_total"] == 2
    )


def test_latency_recorded():
    metrics = MetricsRegistry()

    with metrics.track("inference"):
        sum(range(100))

    snapshot = metrics.snapshot()

    assert (
        snapshot["timings"]["inference_latency_seconds"]["count"]
        == 1
    )


def test_failure_recorded():
    metrics = MetricsRegistry()

    with pytest.raises(RuntimeError):
        with metrics.track("inference"):
            raise RuntimeError("Simulated failure")

    assert (
        metrics.snapshot()["counters"][
            "inference_failures_total"
        ] == 1
    )
