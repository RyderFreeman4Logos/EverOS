"""Metrics lifespan provider.

Confirms the metrics registry is ready and logs that the ``/metrics`` HTTP
endpoint is mounted on the main API. ``EVEROS_TRACEMALLOC=1`` observes an
already-enabled tracer for at most 20 minutes; it never controls process-global
tracemalloc. Start Python with ``PYTHONTRACEMALLOC=1`` to enable tracing, which
adds overhead for the process lifetime. The sampling window limits only these
gauges, not tracing; operators must end a live trial by restarting the process.
"""

from __future__ import annotations

import asyncio
import os
from types import ModuleType
from typing import Any

from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Gauge, get_metrics_registry

from .base import LifespanProvider

logger = get_logger(__name__)

_TRACEMALLOC_WINDOW_SECONDS = 20 * 60
_TRACEMALLOC_OPT_IN_ENV = "EVEROS_TRACEMALLOC"


class _TracemallocMetrics:
    """Expose numeric allocation values until the bounded sample expires."""

    def __init__(
        self,
        tracemalloc: ModuleType,
        window_seconds: float,
    ) -> None:
        self._tracemalloc = tracemalloc
        self._loop = asyncio.get_running_loop()
        self._deadline = self._loop.time() + window_seconds
        self._closed = False
        self._current_bytes = 0
        self._peak_bytes = 0
        self._timer = self._loop.call_later(window_seconds, self.close)

    def current_bytes(self) -> float:
        self._sample()
        return float(self._current_bytes)

    def peak_bytes(self) -> float:
        self._sample()
        return float(self._peak_bytes)

    def _snapshot(self) -> None:
        if self._tracemalloc.is_tracing():
            self._current_bytes, self._peak_bytes = (
                self._tracemalloc.get_traced_memory()
            )

    def _sample(self) -> None:
        if self._closed:
            return
        if self._loop.time() >= self._deadline:
            self.close()
        else:
            self._snapshot()

    def close(self) -> None:
        if self._closed:
            return
        self._snapshot()
        self._closed = True
        self._timer.cancel()


class MetricsLifespanProvider(LifespanProvider):
    """Warm the registry and optionally sample bounded Python allocations."""

    def __init__(self, order: int = 0) -> None:
        super().__init__(name="metrics", order=order)
        self._tracemalloc_metrics: _TracemallocMetrics | None = None
        self._tracemalloc_registry: CollectorRegistry | None = None
        self._tracemalloc_gauges: list[Gauge] = []

    async def startup(self, app: FastAPI) -> Any:
        if self._tracemalloc_registry is not None:
            return self._tracemalloc_registry

        registry = get_metrics_registry()
        if os.environ.get(_TRACEMALLOC_OPT_IN_ENV) == "1":
            import tracemalloc

            if tracemalloc.is_tracing():
                self._tracemalloc_registry = registry
                try:
                    current_gauge = Gauge(
                        "everos_python_traced_memory_bytes",
                        "Current Python traced allocation size in bytes; "
                        "frozen at expiry.",
                    )
                    self._tracemalloc_gauges.append(current_gauge)
                    peak_gauge = Gauge(
                        "everos_python_traced_memory_peak_bytes",
                        "Peak Python traced allocation size in bytes; "
                        "frozen at expiry.",
                    )
                    self._tracemalloc_gauges.append(peak_gauge)
                    self._tracemalloc_metrics = _TracemallocMetrics(
                        tracemalloc,
                        _TRACEMALLOC_WINDOW_SECONDS,
                    )
                    current_gauge.set_function(self._tracemalloc_metrics.current_bytes)
                    peak_gauge.set_function(self._tracemalloc_metrics.peak_bytes)
                except BaseException:
                    self._release_tracemalloc_metrics()
                    raise
        logger.info("metrics_registry_ready", endpoint="/metrics")
        return registry

    async def shutdown(self, app: FastAPI) -> None:
        self._release_tracemalloc_metrics()
        logger.info("metrics_lifespan_shutdown")

    def _release_tracemalloc_metrics(self) -> None:
        if self._tracemalloc_metrics is not None:
            self._tracemalloc_metrics.close()
            self._tracemalloc_metrics = None
        if self._tracemalloc_registry is not None:
            for gauge in self._tracemalloc_gauges:
                self._tracemalloc_registry.unregister(gauge._gauge)
        self._tracemalloc_gauges.clear()
        self._tracemalloc_registry = None
