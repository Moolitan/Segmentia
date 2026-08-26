"""Layout mapping and summary helpers for true vectored SSD reads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence


CSKCACHE_ROOT = Path(__file__).resolve().parents[2]
if str(CSKCACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(CSKCACHE_ROOT))

from cskcache.chunking import ChunkingSpec, build_chunk_plan
from cskcache.layouts import KVLayout, build_layout_plan


LAYOUTS = tuple(layout.value for layout in KVLayout)
IO_ENGINES = ("posix", "io_uring")
DIRECT_MODES = (False, True)
SUBMISSION_MODES = ("multi_read", "readv")


@dataclass(frozen=True)
class ScatterSegment:
    source_relative_offset: int
    destination_offset: int
    length_bytes: int
    region_id: int


@dataclass(frozen=True)
class VectorGroup:
    source_relative_offset: int
    segments: tuple[ScatterSegment, ...]

    @property
    def length_bytes(self) -> int:
        return sum(segment.length_bytes for segment in self.segments)


@dataclass(frozen=True)
class ScatterPlan:
    layout: str
    payload_bytes: int
    region_lengths: tuple[int, ...]
    region_offsets: tuple[int, ...]
    segments: tuple[ScatterSegment, ...]
    vector_groups: tuple[VectorGroup, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    layout: str
    io_engine: str
    use_odirect: bool
    submission_mode: str


def expected_dataset(cfg: Any) -> dict[str, int]:
    return {
        "token_count": int(cfg.TOKEN_COUNT),
        "chunk_size_tokens": int(cfg.CHUNK_SIZE_TOKENS),
        "num_layers": int(cfg.NUM_LAYERS),
        "hidden_dim": int(cfg.KV_HEAD_DIM),
        "dtype_bytes": int(cfg.DTYPE_BYTES),
        "seed": int(cfg.DATASET_SEED),
    }


def load_layout_manifest(cfg: Any, layout: str) -> dict[str, Any]:
    path = Path(cfg.DATA_ROOT) / f"{layout}.manifest.json"
    if not path.is_file():
        raise RuntimeError(f"missing prepared layout manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"layout artifact is incomplete: {path}")
    if payload.get("dataset") != expected_dataset(cfg):
        raise RuntimeError(f"layout artifact dataset differs from config: {path}")
    artifact = payload.get("layout_artifact", {})
    if artifact.get("layout") != layout:
        raise RuntimeError(f"layout artifact identity differs from filename: {path}")
    raw_path = Path(str(artifact.get("raw_file", "")))
    if not raw_path.is_file() or raw_path.stat().st_size != int(
        artifact.get("capacity_bytes", -1)
    ):
        raise RuntimeError(f"prepared raw file is missing or has the wrong size: {raw_path}")
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise RuntimeError(f"layout manifest has no verified regions: {path}")
    return payload


def build_scatter_plan(cfg: Any, layout: str) -> ScatterPlan:
    """Map the packed-all source byte stream directly into one host layout."""
    token_bytes = int(cfg.KV_HEAD_DIM) * int(cfg.DTYPE_BYTES)
    chunk_plan = build_chunk_plan(
        int(cfg.TOKEN_COUNT),
        ChunkingSpec(chunk_size_tokens=int(cfg.CHUNK_SIZE_TOKENS)),
    )
    layout_plan = build_layout_plan(layout, chunk_plan, int(cfg.NUM_LAYERS))
    region_lengths = tuple(
        2
        * region.layer_count
        * (region.token_end - region.token_start)
        * token_bytes
        for region in layout_plan.regions
    )
    region_offsets_list = []
    cursor = 0
    for length in region_lengths:
        region_offsets_list.append(cursor)
        cursor += length
    region_offsets = tuple(region_offsets_list)
    payload_bytes = 2 * int(cfg.NUM_LAYERS) * int(cfg.TOKEN_COUNT) * token_bytes
    if cursor != payload_bytes:
        raise AssertionError("destination regions do not cover the complete KV")

    region_by_coordinate = {}
    for region in layout_plan.regions:
        for chunk_id in range(region.chunk_start, region.chunk_end):
            for layer_id in range(region.layer_start, region.layer_end):
                region_by_coordinate[(chunk_id, layer_id)] = region

    atoms = []
    for layer_id in range(int(cfg.NUM_LAYERS)):
        for kv_index in range(2):
            for chunk in chunk_plan.chunks:
                region = region_by_coordinate[(chunk.chunk_id, layer_id)]
                region_tokens = region.token_end - region.token_start
                local_layer = layer_id - region.layer_start
                local_token = chunk.token_start - region.token_start
                destination_offset = region_offsets[region.region_id] + (
                    ((local_layer * 2 + kv_index) * region_tokens + local_token)
                    * token_bytes
                )
                source_offset = (
                    ((layer_id * 2 + kv_index) * int(cfg.TOKEN_COUNT))
                    + chunk.token_start
                ) * token_bytes
                atoms.append(
                    ScatterSegment(
                        source_relative_offset=source_offset,
                        destination_offset=destination_offset,
                        length_bytes=chunk.token_count * token_bytes,
                        region_id=region.region_id,
                    )
                )

    segments: list[ScatterSegment] = []
    for atom in atoms:
        if segments:
            previous = segments[-1]
            if (
                previous.region_id == atom.region_id
                and previous.source_relative_offset + previous.length_bytes
                == atom.source_relative_offset
                and previous.destination_offset + previous.length_bytes
                == atom.destination_offset
            ):
                segments[-1] = ScatterSegment(
                    source_relative_offset=previous.source_relative_offset,
                    destination_offset=previous.destination_offset,
                    length_bytes=previous.length_bytes + atom.length_bytes,
                    region_id=previous.region_id,
                )
                continue
        segments.append(atom)

    _validate_segment_coverage(segments, payload_bytes, int(cfg.ALIGNMENT_BYTES))
    groups = tuple(
        VectorGroup(
            source_relative_offset=segments[start].source_relative_offset,
            segments=tuple(segments[start : start + int(cfg.MAX_IOVECS_PER_CALL)]),
        )
        for start in range(0, len(segments), int(cfg.MAX_IOVECS_PER_CALL))
    )
    for group in groups:
        source_cursor = group.source_relative_offset
        for segment in group.segments:
            if segment.source_relative_offset != source_cursor:
                raise AssertionError("one readv group must cover contiguous source bytes")
            source_cursor += segment.length_bytes
    return ScatterPlan(
        layout=layout,
        payload_bytes=payload_bytes,
        region_lengths=region_lengths,
        region_offsets=region_offsets,
        segments=tuple(segments),
        vector_groups=groups,
    )


def _validate_segment_coverage(
    segments: Sequence[ScatterSegment], payload_bytes: int, alignment: int
) -> None:
    source_cursor = 0
    for segment in segments:
        if segment.source_relative_offset != source_cursor:
            raise AssertionError("scatter source coverage has a gap or overlap")
        source_cursor += segment.length_bytes
        if (
            segment.source_relative_offset % alignment
            or segment.destination_offset % alignment
            or segment.length_bytes % alignment
        ):
            raise ValueError("all scatter segments must be O_DIRECT aligned")
    if source_cursor != payload_bytes:
        raise AssertionError("scatter source coverage is incomplete")
    destination_cursor = 0
    for start, end in sorted(
        (segment.destination_offset, segment.destination_offset + segment.length_bytes)
        for segment in segments
    ):
        if start != destination_cursor:
            raise AssertionError("scatter destination coverage has a gap or overlap")
        destination_cursor = end
    if destination_cursor != payload_bytes:
        raise AssertionError("scatter destination coverage is incomplete")


def build_case_matrix() -> tuple[BenchmarkCase, ...]:
    cases = []
    for layout in LAYOUTS:
        for io_engine in IO_ENGINES:
            for use_odirect in DIRECT_MODES:
                for submission_mode in SUBMISSION_MODES:
                    access = "odirect" if use_odirect else "buffered"
                    cases.append(
                        BenchmarkCase(
                            case_id=(
                                f"{layout}__{io_engine}__{access}__"
                                f"{submission_mode}"
                            ),
                            layout=layout,
                            io_engine=io_engine,
                            use_odirect=use_odirect,
                            submission_mode=submission_mode,
                        )
                    )
    return tuple(cases)


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("invalid percentile inputs")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["case_id"]), []).append(sample)
    rows = []
    for case in build_case_matrix():
        case_samples = grouped.get(case.case_id, [])
        if not case_samples:
            raise ValueError(f"missing samples for {case.case_id}")
        durations = [float(sample["duration_ms"]) for sample in case_samples]
        rates = [float(sample["gib_per_second"]) for sample in case_samples]
        rows.append(
            {
                **asdict(case),
                "sample_count": len(case_samples),
                "segment_count": int(case_samples[0]["segment_count"]),
                "request_count": int(case_samples[0]["request_count"]),
                "payload_bytes": int(case_samples[0]["payload_bytes"]),
                "p50_ms": percentile(durations, 0.50),
                "p95_ms": percentile(durations, 0.95),
                "p50_gib_s": percentile(rates, 0.50),
            }
        )
    return rows
