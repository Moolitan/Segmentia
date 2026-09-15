from __future__ import annotations

import json
from pathlib import Path

import pytest

from cskcache.runtime.system_profile import (
    SystemPerformanceTracker,
    deployment_fingerprint,
    resolve_system_profile_path,
)


PROFITABILITY_ENABLED = True


def environment() -> dict[str, object]:
    return {
        "model_fingerprint": "model-a",
        "gpu_name": "test-gpu",
        "world_size": 1,
        "storage_backend": "raw_block",
    }


def test_default_profile_path_is_above_skill_pool(tmp_path: Path) -> None:
    fingerprint = deployment_fingerprint(environment())
    path = resolve_system_profile_path(
        tmp_path / "pool" / "model" / "catalog.json", fingerprint
    )
    assert path == (
        tmp_path
        / "pool"
        / "cskcache_metadata"
        / "system_profiles"
        / f"{fingerprint}.json"
    )


def test_first_request_skips_gate_then_persists_ready_profile(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.json"
    tracker = SystemPerformanceTracker(
        enabled=PROFITABILITY_ENABLED,
        environment=environment(),
        metadata_path=tmp_path / "pool" / "catalog.json",
        explicit_path=profile_path,
    )
    assert tracker.estimate(
        skill_tokens=1000,
        storage_bytes=1_000_000,
        prefetch_elapsed_ms=0.0,
        strategy="ratio_prefix",
        ratio=0.05,
    ) is None
    message = tracker.first_use_message("ratio_prefix", 0.05)
    assert message is not None
    assert "gating is skipped for the current request" in message
    assert tracker.first_use_message("ratio_prefix", 0.05) is None

    tracker.record_ssd(1_000_000, 10.0)
    tracker.record_h2d(1_000_000, 2.0)
    tracker.record_prefill(50, 5.0)
    tracker.record_composition(
        strategy="ratio_prefix",
        ratio=0.05,
        tokens=1000,
        duration_ms=40.0,
    )
    estimate = tracker.estimate(
        skill_tokens=1000,
        storage_bytes=1_000_000,
        prefetch_elapsed_ms=4.0,
        strategy="ratio_prefix",
        ratio=0.05,
    )
    assert estimate is not None
    assert estimate.t_pf_ms == pytest.approx(100.0)
    assert estimate.t_ready_ms == pytest.approx(6.0)
    assert estimate.t_comp_ms == pytest.approx(40.0)
    assert estimate.gain_ms == pytest.approx(54.0)
    assert estimate.profitable
    assert profile_path.is_file()

    reloaded = SystemPerformanceTracker(
        enabled=PROFITABILITY_ENABLED,
        environment=environment(),
        metadata_path=tmp_path / "pool" / "catalog.json",
        explicit_path=profile_path,
    )
    assert reloaded.profile.is_ready("ratio_prefix", 0.05)


def test_corrupt_or_mismatched_profile_reenters_learning(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("not-json", encoding="utf-8")
    tracker = SystemPerformanceTracker(
        enabled=PROFITABILITY_ENABLED,
        environment=environment(),
        metadata_path=tmp_path / "catalog.json",
        explicit_path=profile_path,
    )
    assert not tracker.profile.is_ready("ratio_prefix", 0.05)

    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_fingerprint": "wrong",
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    mismatched = SystemPerformanceTracker(
        enabled=PROFITABILITY_ENABLED,
        environment=environment(),
        metadata_path=tmp_path / "catalog.json",
        explicit_path=profile_path,
    )
    assert mismatched.first_use_message("ratio_prefix", 0.05) is not None


def test_disabled_tracker_never_persists_or_estimates(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    tracker = SystemPerformanceTracker(
        enabled=False,
        environment=environment(),
        metadata_path=tmp_path / "catalog.json",
        explicit_path=profile_path,
    )
    tracker.record_ssd(100, 1.0)
    tracker.record_prefill(10, 1.0)
    tracker.record_composition(
        strategy="ratio_prefix", ratio=0.05, tokens=10, duration_ms=1.0
    )
    assert not profile_path.exists()
    assert tracker.first_use_message("ratio_prefix", 0.05) is None


def test_non_writer_does_not_print_first_use_message(tmp_path: Path) -> None:
    tracker = SystemPerformanceTracker(
        enabled=PROFITABILITY_ENABLED,
        environment=environment(),
        metadata_path=tmp_path / "catalog.json",
        writer=False,
    )
    assert tracker.first_use_message("ratio_prefix", 0.05) is None
