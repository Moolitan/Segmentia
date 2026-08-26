"""CSKCache worker tests over LMCache's physical layer interface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cskcache import (
    ChunkLayerBuffer,
    CorrectionStrategy,
    ReusePlan,
    SingleLayerChunkBuffers,
)
from cskcache.integrations.lmcache.worker import (
    _CSKCalibrationBlender,
    _LMCacheCSKLayerStream,
)


class _Buffer:
    def __init__(self, tokens: int) -> None:
        self.tensor = torch.zeros((2, tokens, 4), dtype=torch.bfloat16)
        self.metadata = SimpleNamespace(cached_positions=None)


class _GPUConnector:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.calls = []
        self.received = []

    def batched_to_gpu(self, starts, ends, **kwargs):
        self.calls.append((starts, ends, kwargs))
        for _ in range(self.num_layers):
            self.received.append((yield))
            if kwargs.get("split_stage", False):
                yield
        yield
        yield


def _plan() -> dict[str, object]:
    return {
        "ticket": "call-1",
        "cache_object_id": "skill-v1",
        "request_id": "request-1",
        "segment_start": 100,
        "segment_end": 110,
        "reuse_start": 104,
        "reuse_end": 108,
        "source_reuse_start": 4,
        "source_reuse_end": 8,
        "calibration_start": 101,
        "calibration_end": 104,
        "correction_alpha": 0.6,
        "block_alignment": 4,
        "source_token_count": 10,
    }


def test_lmcache_data_plane_maps_one_complete_layer_stream() -> None:
    buffers = [_Buffer(10), _Buffer(10)]
    gpu_connector = _GPUConnector(num_layers=2)
    gpu_connector.keys = [torch.zeros((7, 4)), torch.zeros((7, 4))]
    gpu_connector.get_kv = lambda layer_id: (
        gpu_connector.keys[layer_id],
        torch.zeros_like(gpu_connector.keys[layer_id]),
    )
    gpu_connector.get_paged_kv = lambda layer_id, start, end: (
        torch.zeros((end - start, 4)),
        torch.zeros((end - start, 4)),
    )
    gpu_connector.commits = []
    gpu_connector.commit_staged_kv = lambda layer_id: gpu_connector.commits.append(
        layer_id
    )
    gpu_connector.commit_recomputed_kv = lambda *_args: None

    stream = _LMCacheCSKLayerStream(
        gpu_connector,
        ReusePlan.from_dict(_plan()),
        buffers,
        kvcaches=[torch.empty(0), torch.empty(0)],
        slot_mapping=torch.arange(108),
    )
    for layer_id in range(2):
        stream.submit_layer(layer_id)
        assert gpu_connector.received[layer_id] == [buffers[layer_id]]
        with pytest.raises(ValueError, match="has not been staged"):
            stream.staged_key(layer_id)
        stream.wait_layer(layer_id)
        stream.commit_calibration(
            layer_id, torch.zeros((3, 4)), torch.zeros((3, 4))
        )
        stream.commit_layer(layer_id)
    stream.finish()

    assert gpu_connector.commits == [0, 1]
    physical = gpu_connector.calls[0][2]
    assert physical["split_stage"] is True
    assert physical["staged_range_start"] == 101
    assert physical["staged_range_end"] == 108
    assert physical["deferred_write_start"] == 104


def test_lmcache_layer_stream_sends_chunk_segments_for_each_layer() -> None:
    layer_objects = [
        [_Buffer(4), _Buffer(4), _Buffer(2)],
        [_Buffer(4), _Buffer(4), _Buffer(2)],
    ]
    groups = [
        SingleLayerChunkBuffers(
            tuple(
                ChunkLayerBuffer(chunk_id, start, end, memory_obj)
                for chunk_id, ((start, end), memory_obj) in enumerate(zip(
                    ((0, 4), (4, 8), (8, 10)), objects, strict=True
                ))
            )
        )
        for objects in layer_objects
    ]
    gpu_connector = _GPUConnector(num_layers=2)
    stream = _LMCacheCSKLayerStream(
        gpu_connector,
        ReusePlan.from_dict(_plan()),
        groups,
        kvcaches=[torch.empty(0), torch.empty(0)],
        slot_mapping=torch.arange(108),
    )

    for layer_id in range(2):
        stream.submit_layer(layer_id)
        stream.wait_layer(layer_id)
    stream.finish()

    assert gpu_connector.calls[0][0] == [100, 104]
    assert gpu_connector.calls[0][1] == [104, 108]
    assert gpu_connector.received[0] == layer_objects[0][:2]
    assert gpu_connector.received[1] == layer_objects[1][:2]


def test_partial_prefix_binds_the_complete_source_object() -> None:
    buffers = [_Buffer(10), _Buffer(10)]
    gpu_connector = _GPUConnector(num_layers=2)
    partial = dict(_plan())
    partial["segment_end"] = 108

    stream = _LMCacheCSKLayerStream(
        gpu_connector,
        ReusePlan.from_dict(partial),
        buffers,
        kvcaches=[torch.empty(0), torch.empty(0)],
        slot_mapping=torch.arange(108),
    )
    for layer_id in range(2):
        stream.submit_layer(layer_id)
        stream.wait_layer(layer_id)
    stream.finish()

    assert gpu_connector.received == [[buffers[0]], [buffers[1]]]
    assert gpu_connector.calls[0][2]["staged_range_end"] == 108


def test_calibration_blender_assembles_dynamic_prefix_and_fresh_kv() -> None:
    prefix_key = torch.arange(404, dtype=torch.float32).reshape(101, 4)
    prefix_value = prefix_key + 1000
    gpu_connector = SimpleNamespace(
        get_paged_kv=lambda layer_id, start, end: (
            prefix_key[start:end], prefix_value[start:end]
        )
    )
    rotary = lambda positions, q, k: (q + positions[:, None], k + 5)
    model = SimpleNamespace(
        vllm_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))]
            )
        )
    )
    blender = _CSKCalibrationBlender(gpu_connector)
    blender.bind_model(model)
    blender.begin(ReusePlan.from_dict(_plan()))
    q = torch.zeros((3, 4))
    k = torch.ones((3, 4))
    v = torch.full((3, 4), 2.0)

    _, key_bank, value_bank, _, _, _ = blender.process_qkv(
        q, k, v, torch.zeros_like(q), 0, None, SimpleNamespace()
    )
    fresh_key, fresh_value = blender.take_result(0)

    assert key_bank.shape == (104, 4)
    assert value_bank.shape == (104, 4)
    assert torch.equal(key_bank[:101], prefix_key)
    assert torch.equal(value_bank[:101], prefix_value)
    assert torch.equal(fresh_key, torch.full((3, 4), 6.0))
    assert torch.equal(fresh_value, torch.full((3, 4), 2.0))
    blender.finish()


def test_deviation_topk_selects_once_and_updates_only_selected_rows() -> None:
    staged_keys = [torch.zeros((4, 4)) for _ in range(3)]
    staged_values = [torch.zeros((4, 4)) for _ in range(3)]
    gpu_connector = SimpleNamespace(
        get_kv=lambda layer_id: (
            staged_keys[layer_id], staged_values[layer_id]
        ),
        get_paged_kv=lambda layer_id, start, end: (
            torch.zeros((end - start, 4)),
            torch.zeros((end - start, 4)),
        ),
    )
    observed_positions: list[tuple[int, ...]] = []

    def rotary(positions, q, k):
        observed_positions.append(tuple(int(value) for value in positions))
        return q, k

    model = SimpleNamespace(
        vllm_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(rotary_emb=rotary)
                    )
                    for _ in range(3)
                ]
            )
        )
    )
    plan = ReusePlan(
        ticket="topk-ticket",
        cache_object_id="skill-v1",
        request_id="topk-request",
        segment_start=4,
        segment_end=10,
        reuse_start=6,
        reuse_end=10,
        source_reuse_start=2,
        source_reuse_end=6,
        calibration_start=6,
        calibration_end=6,
        correction_alpha=0.6,
        block_alignment=2,
        source_token_count=6,
        correction_strategy=CorrectionStrategy.DEVIATION_TOPK,
        deviation_recompute_ratio=0.5,
        deviation_check_layer=1,
    )
    metadata = SimpleNamespace(
        selected=[],
        update_from_top_indices=lambda indices: metadata.selected.append(
            tuple(int(value) for value in indices)
        ),
    )
    blender = _CSKCalibrationBlender(gpu_connector)
    blender.bind_model(model)
    blender.begin(plan)

    full_q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    full_key = torch.tensor(
        [[1.0, 0, 0, 0], [10.0, 0, 0, 0],
         [2.0, 0, 0, 0], [8.0, 0, 0, 0]]
    )
    full_value = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100
    residual = full_q + 1000

    blender.process_qkv(
        full_q, full_key, full_value, residual, 0, None, metadata
    )
    first = blender.take_deviation_result(0)
    assert first.recomputed_tokens == 4
    assert first.selection_applied is False
    assert torch.equal(staged_keys[0], full_key)

    selected_q, key_bank, value_bank, selected_residual, _, _ = (
        blender.process_qkv(
            full_q, full_key, full_value, residual, 1, None, metadata
        )
    )
    checked = blender.take_deviation_result(1)
    assert checked.recomputed_tokens == 2
    assert checked.selection_applied is True
    assert metadata.selected == [(1, 3)]
    assert torch.equal(selected_q, full_q[[1, 3]])
    assert torch.equal(selected_residual, residual[[1, 3]])
    assert key_bank.shape == (10, 4)
    assert value_bank.shape == (10, 4)
    assert torch.equal(staged_keys[1][1], full_key[1])
    assert torch.equal(staged_keys[1][3], full_key[3])
    assert torch.count_nonzero(staged_keys[1][0]).item() == 0
    assert torch.count_nonzero(staged_keys[1][2]).item() == 0

    later_key = torch.full((2, 4), 7.0)
    later_value = torch.full((2, 4), 9.0)
    blender.process_qkv(
        selected_q,
        later_key,
        later_value,
        selected_residual,
        2,
        None,
        metadata,
    )
    later = blender.take_deviation_result(2)
    assert later.recomputed_tokens == 2
    assert later.selection_applied is False
    assert torch.equal(staged_keys[2][1], later_key[0])
    assert torch.equal(staged_keys[2][3], later_key[1])
    assert observed_positions == [(6, 7, 8, 9), (6, 7, 8, 9), (7, 9)]
    blender.finish()
