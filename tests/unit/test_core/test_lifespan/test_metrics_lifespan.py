"""``MetricsLifespanProvider`` — startup returns registry, shutdown logs."""

from __future__ import annotations

import asyncio
import tracemalloc
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY, CollectorRegistry

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


async def test_opt_in_observes_existing_tracemalloc_without_owning_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in observes existing tracing and exposes byte gauges on scrape."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    registry = CollectorRegistry()
    set_metrics_registry(registry)
    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(
        tracemalloc, "start", lambda *_args: pytest.fail("must not start tracing")
    )
    monkeypatch.setattr(
        tracemalloc, "stop", lambda: pytest.fail("must not stop tracing")
    )
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (1234, 5678))

    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        exposition = generate_metrics_response().decode()
        assert "everos_python_traced_memory_bytes 1234.0" in exposition
        assert "everos_python_traced_memory_peak_bytes 5678.0" in exposition
        await provider.shutdown(FastAPI())
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()


async def test_opt_in_default_registry_supports_sequential_lifespans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed providers release collectors for the next app lifespan."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    set_metrics_registry(REGISTRY)
    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(
        tracemalloc, "start", lambda *_args: pytest.fail("must not start tracing")
    )
    monkeypatch.setattr(
        tracemalloc, "stop", lambda: pytest.fail("must not stop tracing")
    )
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (123, 456))

    try:
        for _ in range(2):
            provider = MetricsLifespanProvider()
            await provider.startup(FastAPI())
            exposition = generate_metrics_response().decode()
            assert "everos_python_traced_memory_bytes 123.0" in exposition
            await provider.shutdown(FastAPI())
        assert "everos_python_traced_memory_bytes" not in (
            generate_metrics_response().decode()
        )
    finally:
        reset_metrics_registry()


async def test_opt_in_partial_startup_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed setup releases only its collectors for a retry."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    set_metrics_registry(REGISTRY)
    set_function = metrics_lifespan_module.Gauge.set_function
    calls = 0

    def fail_peak_gauge(
        self: metrics_lifespan_module.Gauge, function: Callable[[], float]
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected gauge setup failure")
        set_function(self, function)

    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(
        tracemalloc, "start", lambda *_args: pytest.fail("must not start tracing")
    )
    monkeypatch.setattr(
        tracemalloc, "stop", lambda: pytest.fail("must not stop tracing")
    )
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (123, 456))
    monkeypatch.setattr(metrics_lifespan_module.Gauge, "set_function", fail_peak_gauge)

    try:
        provider = MetricsLifespanProvider()
        with pytest.raises(RuntimeError, match="injected gauge setup failure"):
            await provider.startup(FastAPI())
        assert "everos_python_traced_memory" not in (
            generate_metrics_response().decode()
        )
        await provider.startup(FastAPI())
        await provider.shutdown(FastAPI())
    finally:
        reset_metrics_registry()


async def test_same_instance_startup_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated startup on one provider leaves its collectors registered once."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    set_metrics_registry(REGISTRY)
    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(
        tracemalloc, "start", lambda *_args: pytest.fail("must not start tracing")
    )
    monkeypatch.setattr(
        tracemalloc, "stop", lambda: pytest.fail("must not stop tracing")
    )
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (123, 456))

    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        assert await provider.startup(FastAPI()) is REGISTRY
        exposition = generate_metrics_response().decode()
        assert "everos_python_traced_memory_bytes 123.0" in exposition
        assert "everos_python_traced_memory_peak_bytes 456.0" in exposition
        await provider.shutdown(FastAPI())
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()


async def test_opt_in_sampling_expires_without_stopping_tracer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sample freezes at expiry while external tracing remains untouched."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    monkeypatch.setattr(
        metrics_lifespan_module, "_TRACEMALLOC_WINDOW_SECONDS", 0.01, raising=False
    )
    registry = CollectorRegistry()
    set_metrics_registry(registry)
    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(
        tracemalloc, "start", lambda *_args: pytest.fail("must not start tracing")
    )
    monkeypatch.setattr(
        tracemalloc, "stop", lambda: pytest.fail("must not stop tracing")
    )
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (1, 2))

    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        await asyncio.sleep(0.02)
        exposition = generate_metrics_response().decode()
        assert "everos_python_traced_memory_bytes 1.0" in exposition
    finally:
        await provider.shutdown(FastAPI())
        reset_metrics_registry()


async def test_external_tracemalloc_restart_is_never_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cannot stop a tracer that another owner restarted."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    set_metrics_registry(CollectorRegistry())
    tracing = True
    calls: list[str] = []

    def is_tracing() -> bool:
        return tracing

    def record_start(*_args: int) -> None:
        calls.append("start")

    def record_stop() -> None:
        calls.append("stop")

    monkeypatch.setattr(tracemalloc, "is_tracing", is_tracing)
    monkeypatch.setattr(tracemalloc, "start", record_start)
    monkeypatch.setattr(tracemalloc, "stop", record_stop)
    monkeypatch.setattr(tracemalloc, "get_traced_memory", lambda: (77, 88))
    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        tracing = False
        tracing = True
        await provider.shutdown(FastAPI())
        assert tracing
        assert calls == []
    finally:
        reset_metrics_registry()


async def test_opt_in_without_tracer_registers_no_diagnostic_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in alone does not start tracing or publish empty measurements."""
    monkeypatch.setenv("EVEROS_TRACEMALLOC", "1")
    set_metrics_registry(CollectorRegistry())
    monkeypatch.setattr(tracemalloc, "is_tracing", lambda: False)
    monkeypatch.setattr(
        tracemalloc, "start", lambda *_args: pytest.fail("must not start tracing")
    )
    monkeypatch.setattr(
        tracemalloc, "stop", lambda: pytest.fail("must not stop tracing")
    )
    monkeypatch.setattr(
        tracemalloc,
        "get_traced_memory",
        lambda: pytest.fail("inactive tracer must not be sampled"),
    )
    provider = MetricsLifespanProvider()
    try:
        await provider.startup(FastAPI())
        assert "everos_python_traced_memory" not in (
            generate_metrics_response().decode()
        )
        await provider.shutdown(FastAPI())
    finally:
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
