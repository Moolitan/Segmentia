"""Data contracts for Pinned-to-GPU transfer planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ...layouts import KVLayout, KVLayoutPlan


class H2DCopyStrategy(str, Enum):
    """How one layer's pinned source objects are submitted to the GPU."""

    # Implemented: LMCache enqueues one Key copy and one Value copy for each
    # source object supplied by the transfer plan.
    PER_OBJECT_COPY = "per_object_copy"

    # Reserved: a future primitive may submit multiple source pointers through
    # one fused gather operation.  No runtime implementation exists yet.
    FUSED_GATHER = "fused_gather"


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


class H2DCopySession(Protocol):
    """Lifecycle shared by concrete host-to-GPU copy primitives."""

    strategy: H2DCopyStrategy

    def submit(self, memory_objects: Sequence[Any]) -> None: ...

    def wait(self) -> None: ...

    def finish(self) -> None: ...

    def close(self) -> None: ...
