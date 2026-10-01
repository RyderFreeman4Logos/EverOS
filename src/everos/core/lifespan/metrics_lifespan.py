"""Metrics lifespan provider.

Confirms the metrics registry is ready and logs that the ``/metrics`` HTTP
endpoint is mounted on the main API. Setting ``EVEROS_TRACEMALLOC=1`` opts into
a 20-minute sample; EverOS starts a one-frame tracer only when needed and
exposes two numeric gauges.
"""

from __future__ import annotations

import asyncio
import os
from types import ModuleType
from typing import Any

from fastapi import FastAPI

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
        owns_tracing: bool,
        window_seconds: float,
    ) -> None:
        self._tracemalloc = tracemalloc
        self._owns_tracing = owns_tracing
        self._loop = asyncio.get_running_loop()
        self._deadline = self._loop.time() + window_seconds
        self._closed = False
        self._current_bytes = 0
        self._peak_bytes = 0
        # Stop tracing even if metrics scraping pauses for the whole window.
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
        if self._owns_tracing and self._tracemalloc.is_tracing():
            self._tracemalloc.stop()


class MetricsLifespanProvider(LifespanProvider):
    """Warm the registry and optionally sample bounded Python allocations."""

    def __init__(self, order: int = 0) -> None:
        super().__init__(name="metrics", order=order)
        self._tracemalloc_metrics: _TracemallocMetrics | None = None

    async def startup(self, app: FastAPI) -> Any:
        registry = get_metrics_registry()
        if os.environ.get(_TRACEMALLOC_OPT_IN_ENV) == "1":
            import tracemalloc

            current_gauge = Gauge(
                "everos_python_traced_memory_bytes",
                "Current Python traced allocation size in bytes; frozen at expiry.",
            )
            peak_gauge = Gauge(
                "everos_python_traced_memory_peak_bytes",
                "Peak Python traced allocation size in bytes; frozen at expiry.",
            )
            owns_tracing = not tracemalloc.is_tracing()
            if owns_tracing:
                tracemalloc.start(1)
            self._tracemalloc_metrics = _TracemallocMetrics(
                tracemalloc,
                owns_tracing,
                _TRACEMALLOC_WINDOW_SECONDS,
            )
            current_gauge.set_function(self._tracemalloc_metrics.current_bytes)
            peak_gauge.set_function(self._tracemalloc_metrics.peak_bytes)
        logger.info("metrics_registry_ready", endpoint="/metrics")
        return registry

    async def shutdown(self, app: FastAPI) -> None:
        if self._tracemalloc_metrics is not None:
            self._tracemalloc_metrics.close()
        logger.info("metrics_lifespan_shutdown")
