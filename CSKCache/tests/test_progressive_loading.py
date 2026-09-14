from __future__ import annotations

import pytest

from cskcache.runtime.progressive_loading import (
    ProgressiveLoadCoordinator,
    ProgressiveLoadingConfig,
)


def test_coordinator_barrier_observes_ordered_layer_progress() -> None:
    coordinator = ProgressiveLoadCoordinator(
        ProgressiveLoadingConfig(
            ssd_bandwidth_bytes_per_ms=1_000.0,
            h2d_bandwidth_bytes_per_ms=2_000.0,
        )
    )
    try:
        coordinator.register("call-1", (100, 200, 300, 400))
        coordinator.mark_host_ready("call-1", 1)
        first = coordinator.snapshot("call-1")
        assert first.host_ready_layers == (1,)
        assert first.host_ready_prefix == 0
        assert first.completed_ssd_bytes == 200

        coordinator.mark_execution_selected("call-1")
        coordinator.mark_host_ready("call-1", 0)
        coordinator.mark_h2d_complete(
            "call-1", transferred_bytes=50, duration_ms=0.5
        )
        second = coordinator.snapshot("call-1")
        assert second.host_ready_prefix == 2
        assert second.completed_ssd_bytes == 300
        assert second.completed_h2d_bytes == 50
        assert second.execution_selected_at_ns is not None
    finally:
        coordinator.close()


def test_coordinator_failure_and_release_are_explicit() -> None:
    coordinator = ProgressiveLoadCoordinator()
    coordinator.register("call-1", (100,))
    coordinator.mark_failed("call-1", "ssd_failed")
    assert coordinator.snapshot("call-1").failure_reason == "ssd_failed"
    coordinator.release("call-1")
    with pytest.raises(KeyError, match="unknown progressive ticket"):
        coordinator.snapshot("call-1")
    coordinator.close()


def test_coordinator_carries_h2d_ewma_across_requests() -> None:
    coordinator = ProgressiveLoadCoordinator(
        ProgressiveLoadingConfig(
            h2d_bandwidth_bytes_per_ms=100.0,
            bandwidth_ewma_alpha=0.5,
        )
    )
    try:
        coordinator.register("first", (100, 100))
        coordinator.mark_h2d_complete(
            "first", transferred_bytes=1_000, duration_ms=2.0
        )
        coordinator.release("first")
        coordinator.register("second", (100, 100))

        snapshot = coordinator.snapshot("second")

        assert snapshot.h2d_bandwidth_bytes_per_ms == 300.0
    finally:
        coordinator.close()
