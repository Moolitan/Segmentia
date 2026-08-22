"""CPU-only correctness checks for the shared chunk/layout coordinate system."""

from __future__ import annotations

import pytest

from cskcache import (
    CacheObjectMetadata,
    ChunkingSpec,
    ContainerMetadata,
    KVLayout,
    LayerExtent,
    ReadStrategy,
    build_chunk_plan,
    build_layout_plan,
)
from cskcache.host_memory.base import SingleLayerKVBuffer
from cskcache.host_memory.transfers import (
    bind_layer_buffers,
    build_h2d_transfer_plan,
)
from cskcache.storage.loads import build_storage_load_plan


def test_chunk_size_equal_to_skill_is_one_chunk() -> None:
    plan = build_chunk_plan(8000, ChunkingSpec(8000))

    assert plan.chunk_count == 1
    assert plan.effective_chunk_size_tokens == 8000
    assert (plan.chunks[0].token_start, plan.chunks[0].token_end) == (0, 8000)


def test_fixed_size_keeps_the_exact_tail() -> None:
    plan = build_chunk_plan(
        8000,
        ChunkingSpec(256),
    )

    assert plan.chunk_count == 32
    assert plan.chunks[-1].token_count == 64
    assert plan.chunks[-1].token_end == 8000


@pytest.mark.parametrize(
    ("layout", "expected_regions", "expected_load_groups", "h2d_slices"),
    (
        (KVLayout.CHUNK_ALL_LAYERS, 32, 32, 32),
        (KVLayout.CHUNK_SINGLE_LAYER, 1280, 32, 32),
        (KVLayout.PACKED_CHUNKS_SINGLE_LAYER, 40, 40, 1),
        (KVLayout.PACKED_CHUNKS_ALL_LAYERS, 1, 1, 1),
    ),
)
def test_four_layouts_share_one_chunk_plan(
    layout: KVLayout,
    expected_regions: int,
    expected_load_groups: int,
    h2d_slices: int,
) -> None:
    chunk_plan = build_chunk_plan(
        8000,
        ChunkingSpec(256),
    )
    layout_plan = build_layout_plan(layout, chunk_plan, 40)
    load_plan = build_storage_load_plan(layout_plan)
    transfer_plan = build_h2d_transfer_plan(layout_plan)

    assert layout_plan.chunk_plan is chunk_plan
    assert len(layout_plan.regions) == expected_regions
    assert len(load_plan.groups) == expected_load_groups
    assert len(transfer_plan.steps) == 40
    assert all(len(step.slices) == h2d_slices for step in transfer_plan.steps)
    assert all(
        step.slices[0].token_start == 0
        and step.slices[-1].token_end == 8000
        for step in transfer_plan.steps
    )


def test_chunk_count_is_derived_not_configured() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ChunkingSpec(0)
    with pytest.raises(ValueError, match="canonical schema"):
        ChunkingSpec.from_dict(
            {"mode": "whole_skill", "chunk_size_tokens": None}
        )


def _packed_layer_metadata(layout: KVLayout) -> CacheObjectMetadata:
    chunking = ChunkingSpec(256)
    layers = tuple(
        LayerExtent(
            layer_id=layer_id,
            backend_key=f"layer-{layer_id}",
            offset_bytes=4096 + layer_id * 4096,
            length_bytes=4096,
            dtype="bfloat16",
            shape=(2, 8000, 1),
            memory_layout="KV_2TD",
            payload_sha256=f"{layer_id + 1:064x}",
        )
        for layer_id in range(2)
    )
    return CacheObjectMetadata(
        object_id="skill-v1",
        skill_name="skill",
        skill_version="v1",
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        token_count=8000,
        source_position_start=0,
        token_ids_sha256="a" * 64,
        start_marker_token_ids=(1,),
        container_id="container-v1",
        read_strategy=ReadStrategy.CONTIGUOUS,
        layers=layers,
        chunking=chunking,
        storage_layout=layout,
    )


def test_catalog_round_trip_records_fixed_chunks_and_packed_layers() -> None:
    container = ContainerMetadata(
        container_id="container-v1",
        raw_file_path="/tmp/skill-kv.bin",
        container_format_version=1,
        storage_generation="generation-v1",
        capacity_bytes=16384,
        alignment_bytes=4096,
        header_bytes=4096,
    )
    metadata = _packed_layer_metadata(KVLayout.PACKED_CHUNKS_SINGLE_LAYER)

    metadata.validate(2, container)
    restored = CacheObjectMetadata.from_dict(metadata.to_dict())

    assert restored == metadata
    assert restored.chunking.chunk_size_tokens == 256
    assert restored.storage_layout is KVLayout.PACKED_CHUNKS_SINGLE_LAYER


def test_catalog_rejects_unencoded_fixed_chunk_regions() -> None:
    container = ContainerMetadata(
        container_id="container-v1",
        raw_file_path="/tmp/skill-kv.bin",
        container_format_version=1,
        storage_generation="generation-v1",
        capacity_bytes=16384,
        alignment_bytes=4096,
        header_bytes=4096,
    )
    metadata = _packed_layer_metadata(KVLayout.CHUNK_SINGLE_LAYER)

    with pytest.raises(ValueError, match="one complete Skill region per model layer"):
        metadata.validate(2, container)


def test_packed_host_layers_preserve_fixed_chunk_plan_for_h2d() -> None:
    chunk_plan = build_chunk_plan(
        8000,
        ChunkingSpec(256),
    )
    buffers = tuple(
        SingleLayerKVBuffer(
            layout=KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
            chunk_plan=chunk_plan,
            memory_obj=object(),
        )
        for _ in range(2)
    )

    bound = bind_layer_buffers(buffers, token_count=8000)

    assert bound.plan.layout_plan.chunk_plan is chunk_plan
    assert bound.plan.layout is KVLayout.PACKED_CHUNKS_SINGLE_LAYER
    assert all(len(step.slices) == 1 for step in bound.plan.steps)
