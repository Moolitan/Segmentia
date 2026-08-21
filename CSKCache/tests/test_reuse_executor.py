from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cskcache import (
    CSKCacheReuseExecutor,
    ContextAwareKVCorrector,
    ChunkLayerBuffer,
    SingleLayerChunkBuffers,
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
        calibration_start=14,
        calibration_end=16,
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
        self.keys = [torch.zeros((6, 4), dtype=torch.float32) for _ in range(layers)]
        self.fail_layer = fail_layer

    def submit_layer(self, layer_id: int) -> None:
        self.calls.append(("submit", layer_id))

    def wait_layer(self, layer_id: int) -> None:
        self.calls.append(("wait", layer_id))

    def staged_key(self, layer_id: int) -> torch.Tensor:
        self.calls.append(("staged_key", layer_id))
        return self.keys[layer_id]

    def commit_calibration(self, layer_id: int, key, value) -> None:
        self.calls.append(
            ("commit_calibration", layer_id, tuple(key.shape), tuple(value.shape))
        )

    def commit_layer(self, layer_id: int) -> None:
        self.calls.append(("commit", layer_id))

    def finish(self) -> None:
        self.calls.append(("finish",))

    def abort(self) -> None:
        self.calls.append(("abort",))


class FakeDataPlane:
    def __init__(
        self,
        layers: int,
        *,
        chunk_single_layer: bool = False,
        fail_layer: int | None = None,
    ) -> None:
        if chunk_single_layer:
            self.buffers = tuple(
                SingleLayerChunkBuffers(
                    (
                        ChunkLayerBuffer(
                            chunk_id=0,
                            token_start=0,
                            token_end=1,
                            memory_obj=SimpleNamespace(layer_id=layer_id),
                        ),
                    )
                )
                for layer_id in range(layers)
            )
        else:
            self.buffers = tuple(
                SimpleNamespace(layer_id=i) for i in range(layers)
            )
        self.stream = FakeLayerStream(layers, fail_layer=fail_layer)
        self.calls: list[tuple[object, ...]] = []

    def open_calibration_model(self, reuse_plan, token_ids):
        self.calls.append(
            (
                "open_model",
                tuple(token_ids[reuse_plan.calibration_start : reuse_plan.calibration_end]),
            )
        )
        stream = self.stream

        class ModelExecutor:
            def __init__(self):
                self.layer_id = 0

            def __next__(self):
                layer_id = self.layer_id
                if layer_id >= len(self_outer.buffers):
                    raise StopIteration
                self.layer_id += 1
                stream.calls.append(("forward", layer_id))
                rows = 1 if layer_id == stream.fail_layer else 2
                key = torch.full((rows, 4), float(layer_id + 1))
                value = torch.full((rows, 4), float(10 + layer_id))
                return key, value

            def close(self):
                stream.calls.append(("model_close",))

        self_outer = self
        return ModelExecutor()

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
                len(buffers),
                len(kvcaches),
                len(slot_mapping),
            )
        )
        return self.stream

    def mark_layer_loaded(self, ticket: str, request_id: str, layer_id: int):
        self.calls.append(("loaded", ticket, request_id, layer_id))

    def mark_layer_corrected(self, ticket: str, request_id: str, layer_id: int):
        self.calls.append(("corrected", ticket, request_id, layer_id))


def test_executor_runs_h2d_first_for_packed_layer_buffers() -> None:
    data_plane = FakeDataPlane(2)
    executor = CSKCacheReuseExecutor(data_plane, expected_layers=2)

    result = executor.execute(
        plan(),
        token_ids=tuple(range(20)),
        kvcaches=(torch.empty(0), torch.empty(0)),
        slot_mapping=torch.arange(20),
    )

    assert result.processed_layers == 2
    assert result.correction_alpha == 0.6
    assert data_plane.calls == [
        ("get_buffers", "call-1", "request-1"),
        ("open", "call-1", 2, 2, 20),
        ("open_model", (14, 15)),
        ("loaded", "call-1", "request-1", 0),
        ("corrected", "call-1", "request-1", 0),
        ("loaded", "call-1", "request-1", 1),
        ("corrected", "call-1", "request-1", 1),
    ]
    assert data_plane.stream.calls == [
        ("submit", 0),
        ("wait", 0),
        ("submit", 1),
        ("forward", 0),
        ("staged_key", 0),
        ("commit_calibration", 0, (2, 4), (2, 4)),
        ("commit", 0),
        ("wait", 1),
        ("forward", 1),
        ("staged_key", 1),
        ("commit_calibration", 1, (2, 4), (2, 4)),
        ("commit", 1),
        ("finish",),
        ("model_close",),
    ]
    assert torch.allclose(
        data_plane.stream.keys[0][2:], torch.full((4, 4), 0.6)
    )
    assert torch.allclose(
        data_plane.stream.keys[1][2:], torch.full((4, 4), 1.2)
    )


def test_executor_runs_compute_first_for_chunk_single_layer_buffers() -> None:
    data_plane = FakeDataPlane(2, chunk_single_layer=True)
    executor = CSKCacheReuseExecutor(
        data_plane,
        expected_layers=2,
        execution_order="compute_first",
    )

    result = executor.execute(
        plan(),
        token_ids=tuple(range(20)),
        kvcaches=(torch.empty(0), torch.empty(0)),
        slot_mapping=torch.arange(20),
    )

    assert result.processed_layers == 2
    assert data_plane.stream.calls == [
        ("submit", 0),
        ("wait", 0),
        ("forward", 0),
        ("staged_key", 0),
        ("commit_calibration", 0, (2, 4), (2, 4)),
        ("commit", 0),
        ("submit", 1),
        ("wait", 1),
        ("forward", 1),
        ("staged_key", 1),
        ("commit_calibration", 1, (2, 4), (2, 4)),
        ("commit", 1),
        ("finish",),
        ("model_close",),
    ]


def test_executor_rejects_unknown_execution_order() -> None:
    with pytest.raises(ValueError, match="unsupported CSKCache execution order"):
        CSKCacheReuseExecutor(
            FakeDataPlane(2),
            expected_layers=2,
            execution_order="automatic",
        )


@pytest.mark.parametrize(
    ("chunk_single_layer", "execution_order", "earlier", "later"),
    (
        (False, "compute_first", ("forward", 0), ("submit", 1)),
        (True, "h2d_first", ("submit", 1), ("forward", 0)),
    ),
)
def test_execution_order_is_independent_of_host_layout(
    chunk_single_layer: bool,
    execution_order: str,
    earlier: tuple[object, ...],
    later: tuple[object, ...],
) -> None:
    data_plane = FakeDataPlane(2, chunk_single_layer=chunk_single_layer)
    executor = CSKCacheReuseExecutor(
        data_plane,
        expected_layers=2,
        execution_order=execution_order,
    )

    executor.execute(
        plan(),
        token_ids=tuple(range(20)),
        kvcaches=(torch.empty(0), torch.empty(0)),
        slot_mapping=torch.arange(20),
    )

    assert data_plane.stream.calls.index(earlier) < data_plane.stream.calls.index(
        later
    )


def test_executor_stops_at_first_failed_layer_without_false_completion() -> None:
    data_plane = FakeDataPlane(2, fail_layer=1)
    executor = CSKCacheReuseExecutor(data_plane, expected_layers=2)

    with pytest.raises(ValueError, match="shapes differ"):
        executor.execute(
            plan(),
            token_ids=tuple(range(20)),
            kvcaches=(torch.empty(0), torch.empty(0)),
            slot_mapping=torch.arange(20),
        )

    assert ("corrected", "call-1", "request-1", 0) in data_plane.calls
    assert ("loaded", "call-1", "request-1", 1) in data_plane.calls
    assert ("corrected", "call-1", "request-1", 1) not in data_plane.calls
    assert ("commit", 1) not in data_plane.stream.calls
    assert ("finish",) not in data_plane.stream.calls
    assert ("abort",) in data_plane.stream.calls


def test_reuse_plan_round_trip_validates_transport_mapping() -> None:
    original = plan()
    assert ReusePlan.from_dict(original.to_dict()) == original

    malformed = original.to_dict()
    malformed["reuse_end"] = 19
    malformed["source_reuse_end"] = 9
    with pytest.raises(ValueError, match="block aligned"):
        ReusePlan.from_dict(malformed)
