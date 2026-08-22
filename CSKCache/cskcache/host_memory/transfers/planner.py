"""Build one layer-ordered H2D plan from any shared KV layout."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...chunking import ChunkingSpec, build_chunk_plan
from ...layouts import KVLayout, KVLayoutPlan, build_layout_plan
from ..base import SingleLayerChunkBuffers, SingleLayerKVBuffer
from .base import (
    BoundH2DTransferPlan,
    H2DRegionSlice,
    H2DTransferPlan,
    H2DTransferStep,
)


def build_h2d_transfer_plan(layout_plan: KVLayoutPlan) -> H2DTransferPlan:
    """Derive one source-slice step per model layer for any KV layout."""

    steps: list[H2DTransferStep] = []
    for layer_id in range(layout_plan.num_layers):
        slices = tuple(
            H2DRegionSlice(
                region_id=region.region_id,
                layer_id=layer_id,
                chunk_start=region.chunk_start,
                chunk_end=region.chunk_end,
                token_start=region.token_start,
                token_end=region.token_end,
            )
            for region in sorted(
                layout_plan.regions_for_layer(layer_id),
                key=lambda item: item.chunk_start,
            )
        )
        steps.append(H2DTransferStep(layer_id, slices))
    return H2DTransferPlan(layout_plan.layout, layout_plan, tuple(steps))


def bind_layer_buffers(
    buffers: Sequence[Any],
    *,
    token_count: int,
) -> BoundH2DTransferPlan:
    """Bind current LMCache layer buffers to their source-slice plan."""

    if not buffers:
        raise ValueError("host buffers must contain at least one layer")
    chunked = tuple(isinstance(item, SingleLayerChunkBuffers) for item in buffers)
    packed = tuple(isinstance(item, SingleLayerKVBuffer) for item in buffers)
    if (any(chunked) and not all(chunked)) or (any(packed) and not all(packed)):
        raise ValueError("host buffers use mixed layouts")
    if any(chunked) and any(packed):
        raise ValueError("host buffers use mixed layouts")

    if all(chunked):
        first = buffers[0]
        assert isinstance(first, SingleLayerChunkBuffers)
        if not first.chunks:
            raise ValueError("a chunk-single-layer buffer must contain chunks")
        chunk_size = first.chunks[0].token_end - first.chunks[0].token_start
        chunk_plan = build_chunk_plan(token_count, ChunkingSpec(chunk_size))
        expected = tuple(
            (chunk.chunk_id, chunk.token_start, chunk.token_end)
            for chunk in chunk_plan.chunks
        )
        layer_objects = []
        for layer in buffers:
            assert isinstance(layer, SingleLayerChunkBuffers)
            actual = tuple(
                (chunk.chunk_id, chunk.token_start, chunk.token_end)
                for chunk in layer.chunks
            )
            if actual != expected:
                raise ValueError("layer buffers disagree with the Skill ChunkPlan")
            layer_objects.append(tuple(chunk.memory_obj for chunk in layer.chunks))
        layout_plan = build_layout_plan(
            KVLayout.CHUNK_SINGLE_LAYER,
            chunk_plan,
            len(buffers),
        )
    elif all(packed):
        first = buffers[0]
        assert isinstance(first, SingleLayerKVBuffer)
        chunk_plan = first.chunk_plan
        layout = first.layout
        if chunk_plan.skill_token_count != token_count:
            raise ValueError("host ChunkPlan differs from the authenticated Skill")
        layer_objects = []
        for layer in buffers:
            assert isinstance(layer, SingleLayerKVBuffer)
            if layer.layout is not layout or layer.chunk_plan != chunk_plan:
                raise ValueError("pinned layers disagree on layout or ChunkPlan")
            layer_objects.append((layer.memory_obj,))
        layout_plan = build_layout_plan(layout, chunk_plan, len(buffers))
    else:
        # Compatibility for caller-owned complete-Skill layer MemoryObjs.
        chunk_plan = build_chunk_plan(token_count, ChunkingSpec(token_count))
        layout_plan = build_layout_plan(
            KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
            chunk_plan,
            len(buffers),
        )
        layer_objects = [(item,) for item in buffers]

    transfer_plan = build_h2d_transfer_plan(layout_plan)
    return BoundH2DTransferPlan(transfer_plan, tuple(layer_objects))
