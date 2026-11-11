"""Counters, structured logging with correlation ids, and optional CloudWatch export."""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)


@contextmanager
def correlation_context(correlation_id: str | None) -> Iterator[None]:
    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


def current_correlation_id() -> str | None:
    return _correlation_id.get()


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit one JSON log line carrying the current correlation id."""
    payload = {"event": event, "correlation_id": current_correlation_id(), **fields}
    logger.log(level, json.dumps(payload, default=str, sort_keys=True))


class Metrics:
    """Thread-safe counters. ``flush`` publishes deltas to CloudWatch when a client is configured."""

    def __init__(self, namespace: str = "SqsMlPipeline", cloudwatch_client: Any | None = None) -> None:
        self.namespace = namespace
        self._client = cloudwatch_client
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counts[name] += value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()

    def flush(self) -> int:
        """Publish and clear counters. Returns the number of metrics sent."""
        if self._client is None:
            return 0
        with self._lock:
            pending = {k: v for k, v in self._counts.items() if v}
            self._counts.clear()
        data = [{"MetricName": k, "Value": float(v), "Unit": "Count"} for k, v in pending.items()]
        for i in range(0, len(data), 20):  # CloudWatch accepts at most 20 metrics per call
            self._client.put_metric_data(Namespace=self.namespace, MetricData=data[i : i + 20])
        return len(data)


metrics = Metrics()
