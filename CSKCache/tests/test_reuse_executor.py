from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cskcache import (
    CSKCacheReuseExecutor,
    ContextAwareKVCorrector,
    ReusePlan,
)


def plan() -> ReusePlan:
    return ReusePlan(
        ticket="call-1",
        cache_object_id="skill-v1",
        request_id="request-1",
        segment_start=10,
        segment_end=22,
        reuse_start=16,
        reuse_end=20,
        source_reuse_start=6,
        source_reuse_end=10,
        calibration_start=12,
        calibration_end=14,
        correction_alpha=0.6,
        block_alignment=2,
    )


def test_context_aware_corrector_changes_only_suffix_key() -> None:
    staged_key = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    original_key = staged_key.clone()
    untouched_value = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    original_value = untouched_value.clone()
    recomputed = staged_key[:2] + torch.tensor([1.0, 2.0, 3.0, 4.0])

    offset = ContextAwareKVCorrector().correct_key_(
        staged_key,
        recomputed,
        calibration_tokens=2,
        suffix_offset=4,
        alpha=0.6,
    )

    assert torch.allclose(offset, torch.tensor([0.6, 1.2, 1.8, 2.4]))
    assert torch.equal(staged_key[:4], original_key[:4])
    assert torch.allclose(staged_key[4:], original_key[4:] + offset)
    assert torch.equal(untouched_value, original_value)


@pytest.mark.parametrize(
    ("suffix_offset", "alpha"),
    ((1, 0.6), (4, float("nan")), (8, 0.6)),
)
def test_context_aware_corrector_rejects_invalid_policy(
    suffix_offset: int, alpha: float
) -> None:
    staged = torch.zeros((8, 4), dtype=torch.float32)
    recomputed = torch.zeros((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError):
        ContextAwareKVCorrector().correct_key_(
            staged,
            recomputed,
            calibration_tokens=2,
            suffix_offset=suffix_offset,
            alpha=alpha,
        )


class FakeLayerStream:
    def __init__(self, layers: int, *, fail_layer: int | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.keys = [torch.zeros((8, 4), dtype=torch.float32) for _ in range(layers)]
        self.fail_layer = fail_layer

    def stage_layer(self, layer_id: int) -> None:
        self.calls.append(("stage", layer_id))

    def staged_key(self, layer_id: int) -> torch.Tensor:
        self.calls.append(("staged_key", layer_id))
        return self.keys[layer_id]

    def recomputed_key(
        self, layer_id: int, start: int, end: int
    ) -> torch.Tensor:
        self.calls.append(("recomputed_key", layer_id, start, end))
        if layer_id == self.fail_layer:
            return torch.zeros((1, 4), dtype=torch.float32)
        return torch.full((end - start, 4), float(layer_id + 1))

    def commit_layer(self, layer_id: int) -> None:
        self.calls.append(("commit", layer_id))

    def finish(self) -> None:
        self.calls.append(("finish",))


class FakeDataPlane:
    def __init__(self, layers: int, *, fail_layer: int | None = None) -> None:
        self.buffers = tuple(SimpleNamespace(layer_id=i) for i in range(layers))
        self.stream = FakeLayerStream(layers, fail_layer=fail_layer)
        self.calls: list[tuple[object, ...]] = []

    def get_active_layer_buffers(self, ticket: str, request_id: str):
        self.calls.append(("get_buffers", ticket, request_id))
        return self.buffers

    def open_layer_stream(
        self,
        reuse_plan,
        buffers,
        *,
        kvcaches,
        slot_mapping,
        profile_t0_event=None,
    ):
        assert profile_t0_event is None
        self.calls.append(
            (
                "open",
                reuse_plan.ticket,
                tuple(item.layer_id for item in buffers),
                len(kvcaches),
                len(slot_mapping),
            )
        )
        return self.stream

    def mark_layer_loaded(self, ticket: str, request_id: str, layer_id: int):
        self.calls.append(("loaded", ticket, request_id, layer_id))

    def mark_layer_corrected(self, ticket: str, request_id: str, layer_id: int):
        self.calls.append(("corrected", ticket, request_id, layer_id))


def test_executor_runs_stage_correct_commit_in_layer_order() -> None:
    data_plane = FakeDataPlane(2)
    executor = CSKCacheReuseExecutor(data_plane, expected_layers=2)

    result = executor.execute(
        plan(),
        kvcaches=(torch.empty(0), torch.empty(0)),
        slot_mapping=torch.arange(20),
    )

    assert result.processed_layers == 2
    assert result.correction_alpha == 0.6
    assert data_plane.calls == [
        ("get_buffers", "call-1", "request-1"),
        ("open", "call-1", (0, 1), 2, 20),
        ("loaded", "call-1", "request-1", 0),
        ("corrected", "call-1", "request-1", 0),
        ("loaded", "call-1", "request-1", 1),
        ("corrected", "call-1", "request-1", 1),
    ]
    assert data_plane.stream.calls == [
        ("stage", 0),
        ("staged_key", 0),
        ("recomputed_key", 0, 12, 14),
        ("commit", 0),
        ("stage", 1),
        ("staged_key", 1),
        ("recomputed_key", 1, 12, 14),
        ("commit", 1),
        ("finish",),
    ]
    assert torch.allclose(
        data_plane.stream.keys[0][4:], torch.full((4, 4), 0.6)
    )
    assert torch.allclose(
        data_plane.stream.keys[1][4:], torch.full((4, 4), 1.2)
    )


def test_executor_stops_at_first_failed_layer_without_false_completion() -> None:
    data_plane = FakeDataPlane(2, fail_layer=1)
    executor = CSKCacheReuseExecutor(data_plane, expected_layers=2)

    with pytest.raises(ValueError, match="shapes differ"):
        executor.execute(
            plan(),
            kvcaches=(torch.empty(0), torch.empty(0)),
            slot_mapping=torch.arange(20),
        )

    assert ("corrected", "call-1", "request-1", 0) in data_plane.calls
    assert ("loaded", "call-1", "request-1", 1) in data_plane.calls
    assert ("corrected", "call-1", "request-1", 1) not in data_plane.calls
    assert ("commit", 1) not in data_plane.stream.calls
    assert ("finish",) not in data_plane.stream.calls


def test_reuse_plan_round_trip_validates_transport_mapping() -> None:
    original = plan()
    assert ReusePlan.from_dict(original.to_dict()) == original

    malformed = original.to_dict()
    malformed["reuse_end"] = 19
    malformed["source_reuse_end"] = 9
    with pytest.raises(ValueError, match="block aligned"):
        ReusePlan.from_dict(malformed)
