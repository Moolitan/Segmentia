"""Device-independent chunk/layer layout contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..chunking import SkillChunkPlan


class KVLayout(str, Enum):
    """Physical region layout over the chunk and model-layer dimensions.

    A layout only states which chunk/layer coordinates share one contiguous
    region in storage or host memory.  It does not select an SSD read policy,
    an H2D transfer implementation, the order in which H2D is submitted, or
    how H2D overlaps with model computation.  Those policies belong to the
    storage-load, host-memory-transfer, and execution layers respectively.

    The diagrams below describe object composition, not exact tensor strides;
    the selected ``MemoryFormat`` determines the concrete byte ordering.
    """

    # One object per logical chunk.  Each object contains that chunk's KV for
    # every model layer:
    #
    #   obj_chunk0 = [L0.KV(chunk0) | L1.KV(chunk0) | ... | Ln.KV(chunk0)]
    #   obj_chunk1 = [L0.KV(chunk1) | L1.KV(chunk1) | ... | Ln.KV(chunk1)]
    CHUNK_ALL_LAYERS = "chunk_all_layers"

    # One object per (logical chunk, model layer) pair:
    #
    #   obj_l0_c0 = [L0.K(chunk0) | L0.V(chunk0)]
    #   obj_l0_c1 = [L0.K(chunk1) | L0.V(chunk1)]
    #   ...
    #   obj_ln_cm = [Ln.K(chunkM) | Ln.V(chunkM)]
    CHUNK_SINGLE_LAYER = "chunk_single_layer"

    # One object per model layer.  All logical chunks of that layer are packed
    # into the same contiguous object; chunk boundaries remain metadata:
    #
    #   obj_l0 = [
    #       L0.K(chunk0) | L0.K(chunk1) | ... | L0.K(chunkM)
    #       L0.V(chunk0) | L0.V(chunk1) | ... | L0.V(chunkM)
    #   ]
    #   ...
    #   obj_ln = [Ln.K(all chunks), Ln.V(all chunks)]
    PACKED_CHUNKS_SINGLE_LAYER = "packed_chunks_single_layer"

    # One object for the complete Skill KV.  It packs every logical chunk of
    # every model layer into one contiguous region:
    #
    #   one_obj = [
    #       L0.K(all chunks), L0.V(all chunks)
    #       L1.K(all chunks), L1.V(all chunks)
    #       ...
    #       Ln.K(all chunks), Ln.V(all chunks)
    #   ]
    PACKED_CHUNKS_ALL_LAYERS = "packed_chunks_all_layers"


@dataclass(frozen=True)
class KVRegion:
    """One contiguous logical region in storage or host memory."""

    region_id: int
    chunk_start: int
    chunk_end: int
    token_start: int
    token_end: int
    layer_start: int
    layer_end: int

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must be non-negative")
        if not 0 <= self.chunk_start < self.chunk_end:
            raise ValueError("region chunk interval must be non-empty")
        if not 0 <= self.token_start < self.token_end:
            raise ValueError("region token interval must be non-empty")
        if not 0 <= self.layer_start < self.layer_end:
            raise ValueError("region layer interval must be non-empty")

    @property
    def chunk_count(self) -> int:
        return self.chunk_end - self.chunk_start

    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start


@dataclass(frozen=True)
class KVLayoutPlan:
    """Physical region coverage derived from one authoritative ChunkPlan."""

    layout: KVLayout
    chunk_plan: SkillChunkPlan
    num_layers: int
    regions: tuple[KVRegion, ...]

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not self.regions:
            raise ValueError("a KV layout plan must contain at least one region")
        if tuple(region.region_id for region in self.regions) != tuple(
            range(len(self.regions))
        ):
            raise ValueError("region IDs must be dense and ordered")

        coverage = [
            [0 for _layer in range(self.num_layers)]
            for _chunk in range(self.chunk_plan.chunk_count)
        ]
        for region in self.regions:
            if region.chunk_end > self.chunk_plan.chunk_count:
                raise ValueError("region exceeds the chunk plan")
            if region.layer_end > self.num_layers:
                raise ValueError("region exceeds the model layer count")
            first = self.chunk_plan.chunks[region.chunk_start]
            last = self.chunk_plan.chunks[region.chunk_end - 1]
            if (
                region.token_start != first.token_start
                or region.token_end != last.token_end
            ):
                raise ValueError("region token interval disagrees with its chunks")
            for chunk_id in range(region.chunk_start, region.chunk_end):
                for layer_id in range(region.layer_start, region.layer_end):
                    coverage[chunk_id][layer_id] += 1
        if any(count != 1 for row in coverage for count in row):
            raise ValueError("layout regions must cover every chunk/layer pair once")

    def regions_for_layer(self, layer_id: int) -> tuple[KVRegion, ...]:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError("layer_id is outside the layout")
        return tuple(
            region
            for region in self.regions
            if region.layer_start <= layer_id < region.layer_end
        )
