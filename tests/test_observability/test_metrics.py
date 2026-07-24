import time
from typing import cast

from prometheus_client.core import Metric

from src.observability.metrics import ACTIVE_RUNS, REQUEST_COUNT, REQUEST_LATENCY, LatencyTracker


def _get_samples(metric):
    return cast(list[Metric], list(metric.collect()))[0].samples


def test_latency_tracker_records_histogram():
    with LatencyTracker("rag", "semantic_search"):
        time.sleep(0.001)
    samples = [
        s
        for s in _get_samples(REQUEST_LATENCY)
        if s.name == "lewis_request_duration_seconds_bucket"
    ]
    assert any(
        s.labels["component"] == "rag" and s.labels["method"] == "semantic_search" for s in samples
    )


def test_latency_tracker_increments_counter():
    before = sum(s.value for s in _get_samples(REQUEST_COUNT) if s.labels["component"] == "sql")
    with LatencyTracker("sql", "execute"):
        pass
    after = sum(s.value for s in _get_samples(REQUEST_COUNT) if s.labels["component"] == "sql")
    assert after == before + 1


def test_latency_tracker_error_status():
    try:
        with LatencyTracker("timeline", "get"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    error_samples = [s for s in _get_samples(REQUEST_COUNT) if s.labels.get("status") == "error"]
    assert error_samples


def test_gauge_set():
    ACTIVE_RUNS.set(3)
    value = next(s.value for s in _get_samples(ACTIVE_RUNS))
    assert value == 3.0
