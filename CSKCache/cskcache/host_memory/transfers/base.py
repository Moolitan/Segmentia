"""Pinned-to-GPU transfer plans expressed in chunk/layer coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from ...chunking import ChunkingMode, ChunkingSpec, build_chunk_plan
from ...layouts import KVLayout, KVLayoutPlan
from ...layouts import build_layout_plan
from ..base import SingleLayerChunkBuffers, SingleLayerKVBuffer


@dataclass(frozen=True)
class H2DRegionSlice:
    """The part of one host region needed by one model layer."""

    region_id: int
    layer_id: int
    chunk_start: int
    chunk_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class H2DTransferStep:
    """All source slices submitted while staging one model layer."""

    layer_id: int
    slices: tuple[H2DRegionSlice, ...]

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if not self.slices:
            raise ValueError("an H2D transfer step must contain source data")


@dataclass(frozen=True)
class H2DTransferPlan:
    """Complete layerwise H2D plan for one host layout."""

    layout: KVLayout
    layout_plan: KVLayoutPlan
    steps: tuple[H2DTransferStep, ...]

    def __post_init__(self) -> None:
        if self.layout is not self.layout_plan.layout:
            raise ValueError("H2D layout differs from its layout plan")
        if tuple(step.layer_id for step in self.steps) != tuple(
            range(self.layout_plan.num_layers)
        ):
            raise ValueError("H2D steps must cover every layer in order")
        expected_chunks = tuple(range(self.layout_plan.chunk_plan.chunk_count))
        for step in self.steps:
            actual: list[int] = []
            cursor = 0
            for source in step.slices:
                if source.layer_id != step.layer_id:
                    raise ValueError("H2D slice belongs to another layer")
                if source.token_start != cursor:
                    raise ValueError("H2D slices must cover tokens without gaps")
                actual.extend(range(source.chunk_start, source.chunk_end))
                cursor = source.token_end
            if tuple(actual) != expected_chunks:
                raise ValueError("H2D slices must cover every chunk once")
            if cursor != self.layout_plan.chunk_plan.skill_token_count:
                raise ValueError("H2D slices must cover the complete Skill")


@dataclass(frozen=True)
class BoundH2DTransferPlan:
    """A transfer plan paired with the concrete pinned source buffers."""

    plan: H2DTransferPlan
    layer_objects: tuple[tuple[Any, ...], ...]

    def __post_init__(self) -> None:
        if len(self.layer_objects) != len(self.plan.steps):
            raise ValueError("bound H2D layers differ from the transfer plan")
        if any(
            len(objects) != len(step.slices)
            for step, objects in zip(
                self.plan.steps,
                self.layer_objects,
                strict=True,
            )
        ):
            raise ValueError("bound H2D sources differ from their layer slices")


def build_layerwise_transfer_plan(layout_plan: KVLayoutPlan) -> H2DTransferPlan:
    """Derive layer steps from the exact regions of any 2x2 layout."""

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
    """Bind current LMCache layer buffers to the shared transfer plan."""

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
        chunk_plan = build_chunk_plan(
            token_count,
            ChunkingSpec(ChunkingMode.FIXED_SIZE, chunk_size),
        )
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
        # Compatibility for caller-owned whole-Skill MemoryObjs. Production
        # StorageManager buffers always carry an explicit plan.
        chunk_plan = build_chunk_plan(
            token_count,
            ChunkingSpec(ChunkingMode.WHOLE_SKILL),
        )
        layout_plan = build_layout_plan(
            KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
            chunk_plan,
            len(buffers),
        )
        layer_objects = [(item,) for item in buffers]

    transfer_plan = build_layerwise_transfer_plan(layout_plan)
    return BoundH2DTransferPlan(transfer_plan, tuple(layer_objects))
