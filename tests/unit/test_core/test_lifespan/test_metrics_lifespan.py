"""``MetricsLifespanProvider`` — startup returns registry, shutdown logs."""

from __future__ import annotations

import asyncio
import tracemalloc

import pytest
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from everos.core.lifespan import metrics_lifespan as metrics_lifespan_module
from everos.core.lifespan.metrics_lifespan import MetricsLifespanProvider
from everos.core.lifespan.tracing_lifespan import TracingLifespanProvider
from everos.core.observability.metrics import (
    generate_metrics_response,
    reset_metrics_registry,
    set_metrics_registry,
)


async def test_startup_returns_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVEROS_TRACEMALLOC", raising=False)
    fresh = CollectorRegistry()
    set_metrics_registry(fresh)
    try:
        p = MetricsLifespanProvider()
        result = await p.startup(FastAPI())
        assert result is fresh
    finally:
        reset_metrics_registry()


async def test_shutdown_is_noop() -> None:
    # Smoke test — must not raise.
    p = MetricsLifespanProvider()
    await p.shutdown(FastAPI())


def test_provider_metadata() -> None:
    p = MetricsLifespanProvider(order=42)
    assert p.name == "metrics"
    assert p.order == 42


def test_metrics_start_before_tracing_lifespan() -> None:
    assert MetricsLifespanProvider().order < TracingLifespanProvider().order


async def test_opt_in_exposes_numeric_tracemalloc_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in starts one-frame tracing and exposes byte gauges on scrape."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    registry = CollectorRegistry()
    set_metrics_registry(registry)
    tracing = False
    started: list[int] = []
    stopped: list[bool] = []

    def start(nframe: int) -> None:
        nonlocal tracing
        started.append(nframe)
        tracing = True

    def stop() -> None:
        nonlocal tracing
        stopped.append(tracing)
        tracing = False

    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: tracing)
    monkeypatch.setattr(tracemalloc, "start", start)
    monkeypatch.setattr(tracemalloc, "stop", stop)
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (1234, 5678))

    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        exposition = generate_metrics_response().decode()
        assert "everos_python_traced_memory_bytes 1234.0" in exposition
        assert "everos_python_traced_memory_peak_bytes 5678.0" in exposition
        assert started == [1]
        await provider.shutdown(FastAPI())
        assert stopped == [True]
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()


async def test_opt_in_tracemalloc_stops_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifetime timer stops tracing even without another metrics scrape."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    monkeypatch.setattr(
        metrics_lifespan_module, "_TRACEMALLOC_WINDOW_SECONDS", 0.01, raising=False
    )
    registry = CollectorRegistry()
    set_metrics_registry(registry)
    tracing = False
    stopped = asyncio.Event()

    def start(nframe: int) -> None:
        nonlocal tracing
        assert nframe == 1
        tracing = True

    def stop() -> None:
        nonlocal tracing
        tracing = False
        stopped.set()

    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: tracing)
    monkeypatch.setattr(tracemalloc, "start", start)
    monkeypatch.setattr(tracemalloc, "stop", stop)
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (1, 2))

    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        await asyncio.wait_for(stopped.wait(), timeout=1)
        assert not tracing
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()


async def test_preexisting_tracemalloc_is_not_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use existing tracing without stopping it."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    registry = CollectorRegistry()
    set_metrics_registry(registry)
    stopped: list[bool] = []

    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(
        tracemalloc, "start", lambda _nframe: pytest.fail("already tracing")
    )
    monkeypatch.setattr(tracemalloc, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (77, 88))

    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        exposition = generate_metrics_response().decode()
        assert "everos_python_traced_memory_bytes 77.0" in exposition
        assert "everos_python_traced_memory_peak_bytes 88.0" in exposition
        await provider.shutdown(FastAPI())
        assert stopped == []
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()


async def test_tracemalloc_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default startup neither starts tracing nor registers diagnostic gauges."""
    monkeypatch.delenv("EVEROS_TRACEMALLOC", raising=False)
    registry = CollectorRegistry()
    set_metrics_registry(registry)
    calls: list[str] = []

    def unexpected_call() -> None:
        calls.append("tracemalloc")
        pytest.fail("tracemalloc used while diagnostics are disabled")

    monkeypatch.setattr(tracemalloc, "is_tracing", unexpected_call)
    monkeypatch.setattr(tracemalloc, "start", lambda _nframe: unexpected_call())
    monkeypatch.setattr(tracemalloc, "stop", unexpected_call)
    monkeypatch.setattr(tracemalloc, "get_traced_memory", unexpected_call)
    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        exposition = generate_metrics_response().decode()
        assert "everos_python_traced_memory" not in exposition
        assert calls == []
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()
