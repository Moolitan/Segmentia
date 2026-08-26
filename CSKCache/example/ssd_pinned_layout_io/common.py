"""Pure planning and artifact helpers shared by preparation and measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


CSKCACHE_ROOT = Path(__file__).resolve().parents[2]
if str(CSKCACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(CSKCACHE_ROOT))

from cskcache.chunking import ChunkingSpec, build_chunk_plan
from cskcache.layouts import KVLayout, build_layout_plan


LAYOUTS = tuple(layout.value for layout in KVLayout)
IO_ENGINES = ("posix", "io_uring")
DIRECT_MODES = (False, True)
ARTIFACT_TYPE = "cskcache_ssd_pinned_layout_io_v1"


def round_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("value must be non-negative and alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class RegionExtent:
    region_id: int
    chunk_start: int
    chunk_end: int
    token_start: int
    token_end: int
    layer_start: int
    layer_end: int
    offset_bytes: int
    length_bytes: int


@dataclass(frozen=True)
class LayoutArtifact:
    layout: str
    raw_file: str
    alignment_bytes: int
    header_bytes: int
    metadata_bytes: int
    slot_bytes: int
    capacity_bytes: int
    payload_bytes: int
    extents: tuple[RegionExtent, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extents"] = [asdict(extent) for extent in self.extents]
        return payload


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    layout: str
    io_engine: str
    use_odirect: bool


def build_artifact(
    *,
    layout: str,
    raw_file: Path,
    token_count: int,
    chunk_size_tokens: int,
    num_layers: int,
    hidden_dim: int,
    dtype_bytes: int,
    alignment_bytes: int,
    header_bytes: int,
    metadata_bytes: int,
) -> LayoutArtifact:
    """Map the authoritative CSK layout plan onto aligned raw-file slots."""
    if metadata_bytes % alignment_bytes or header_bytes % alignment_bytes:
        raise ValueError("metadata and header sizes must be alignment multiples")
    chunk_plan = build_chunk_plan(
        token_count, ChunkingSpec(chunk_size_tokens=chunk_size_tokens)
    )
    plan = build_layout_plan(layout, chunk_plan, num_layers)
    lengths = [
        2
        * region.layer_count
        * (region.token_end - region.token_start)
        * hidden_dim
        * dtype_bytes
        for region in plan.regions
    ]
    slot_bytes = round_up(header_bytes + max(lengths), alignment_bytes)
    extents = tuple(
        RegionExtent(
            region_id=region.region_id,
            chunk_start=region.chunk_start,
            chunk_end=region.chunk_end,
            token_start=region.token_start,
            token_end=region.token_end,
            layer_start=region.layer_start,
            layer_end=region.layer_end,
            offset_bytes=(
                metadata_bytes + region.region_id * slot_bytes + header_bytes
            ),
            length_bytes=length,
        )
        for region, length in zip(plan.regions, lengths, strict=True)
    )
    payload_bytes = 2 * num_layers * token_count * hidden_dim * dtype_bytes
    if sum(extent.length_bytes for extent in extents) != payload_bytes:
        raise AssertionError("layout plan does not cover the complete logical KV")
    if any(
        extent.offset_bytes % alignment_bytes
        or extent.length_bytes % alignment_bytes
        for extent in extents
    ):
        raise ValueError(
            "this dataset produces an unaligned extent and cannot be compared "
            "with O_DIRECT without padding the logical transfer"
        )
    return LayoutArtifact(
        layout=layout,
        raw_file=str(raw_file.resolve()),
        alignment_bytes=alignment_bytes,
        header_bytes=header_bytes,
        metadata_bytes=metadata_bytes,
        slot_bytes=slot_bytes,
        capacity_bytes=metadata_bytes + len(extents) * slot_bytes,
        payload_bytes=payload_bytes,
        extents=extents,
    )


def build_case_matrix() -> tuple[BenchmarkCase, ...]:
    cases = []
    for layout in LAYOUTS:
        for io_engine in IO_ENGINES:
            for use_odirect in DIRECT_MODES:
                access = "odirect" if use_odirect else "buffered"
                cases.append(
                    BenchmarkCase(
                        case_id=f"{layout}__{io_engine}__{access}",
                        layout=layout,
                        io_engine=io_engine,
                        use_odirect=use_odirect,
                    )
                )
    return tuple(cases)


def dataset_spec(cfg: Any) -> dict[str, int]:
    return {
        "token_count": int(cfg.TOKEN_COUNT),
        "chunk_size_tokens": int(cfg.CHUNK_SIZE_TOKENS),
        "num_layers": int(cfg.NUM_LAYERS),
        "hidden_dim": int(cfg.KV_HEAD_DIM),
        "dtype_bytes": int(cfg.DTYPE_BYTES),
        "seed": int(cfg.DATASET_SEED),
    }


def artifact_for_layout(cfg: Any, layout: str) -> LayoutArtifact:
    return build_artifact(
        layout=layout,
        raw_file=Path(cfg.DATA_ROOT) / f"{layout}.raw",
        token_count=int(cfg.TOKEN_COUNT),
        chunk_size_tokens=int(cfg.CHUNK_SIZE_TOKENS),
        num_layers=int(cfg.NUM_LAYERS),
        hidden_dim=int(cfg.KV_HEAD_DIM),
        dtype_bytes=int(cfg.DTYPE_BYTES),
        alignment_bytes=int(cfg.ALIGNMENT_BYTES),
        header_bytes=int(cfg.HEADER_BYTES),
        metadata_bytes=int(cfg.METADATA_BYTES),
    )


def manifest_path(cfg: Any, layout: str) -> Path:
    return Path(cfg.DATA_ROOT) / f"{layout}.manifest.json"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_manifest(cfg: Any, layout: str) -> dict[str, Any]:
    path = manifest_path(cfg, layout)
    if not path.is_file():
        raise RuntimeError(f"missing prepared manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = artifact_for_layout(cfg, layout).to_dict()
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(f"unexpected artifact type in {path}")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"layout preparation is incomplete: {path}")
    if manifest.get("dataset") != dataset_spec(cfg):
        raise RuntimeError(f"dataset config changed after preparing {path}")
    if manifest.get("layout_artifact") != expected:
        raise RuntimeError(f"layout geometry changed after preparing {path}")
    regions = manifest.get("regions")
    if not isinstance(regions, list) or len(regions) != len(expected["extents"]):
        raise RuntimeError(f"prepared region records are incomplete: {path}")
    expected_geometry = [
        {
            "region_id": extent["region_id"],
            "offset_bytes": extent["offset_bytes"],
            "length_bytes": extent["length_bytes"],
        }
        for extent in expected["extents"]
    ]
    actual_geometry = [
        {
            "region_id": record.get("region_id"),
            "offset_bytes": record.get("offset_bytes"),
            "length_bytes": record.get("length_bytes"),
        }
        for record in regions
    ]
    if actual_geometry != expected_geometry or any(
        not isinstance(record.get("payload_sha256"), str)
        or len(record["payload_sha256"]) != 64
        for record in regions
    ):
        raise RuntimeError(f"prepared region records are invalid: {path}")
    return manifest


def resolve_mount(path: Path) -> tuple[str, Path, set[str]]:
    """Return source, longest matching mount point, and mount options."""
    resolved = path.resolve()
    matches: list[tuple[str, Path, set[str]]] = []
    for line in Path("/proc/self/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        source, mount_raw, _fstype, options = fields[:4]
        mount_point = Path(mount_raw.replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append((source, mount_point, set(options.split(","))))
    if not matches:
        raise RuntimeError(f"no mount contains {resolved}")
    return max(matches, key=lambda item: len(item[1].parts))


def require_writable_data_mount(cfg: Any) -> None:
    source, mount_point, options = resolve_mount(Path(cfg.DATA_ROOT))
    if mount_point != Path(cfg.DATA_MOUNT).resolve():
        raise RuntimeError(
            f"DATA_ROOT resolves through {mount_point}, expected {cfg.DATA_MOUNT}"
        )
    expected = str(cfg.EXPECTED_DEVICE)
    if Path(source).resolve() != Path(expected).resolve():
        raise RuntimeError(f"DATA_ROOT is backed by {source}, expected {expected}")
    if "rw" not in options:
        raise RuntimeError(f"{mount_point} must be mounted read-write")


def fill_byte(seed: int, layer_id: int, kv_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{layer_id}:{kv_index}".encode()).digest()
    return digest[0]


def iter_region_blocks(
    extent: RegionExtent,
    *,
    hidden_dim: int,
    dtype_bytes: int,
    seed: int,
    maximum_block_bytes: int = 8 * 1024**2,
) -> Iterable[bytes]:
    """Yield layer-major, then K/V-major deterministic region bytes."""
    segment_bytes = (
        (extent.token_end - extent.token_start) * hidden_dim * dtype_bytes
    )
    for layer_id in range(extent.layer_start, extent.layer_end):
        for kv_index in range(2):
            value = fill_byte(seed, layer_id, kv_index)
            remaining = segment_bytes
            block = bytes([value]) * min(maximum_block_bytes, segment_bytes)
            while remaining:
                take = min(len(block), remaining)
                yield block[:take]
                remaining -= take


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
    rows = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_case.setdefault(str(sample["case_id"]), []).append(sample)
    for case in build_case_matrix():
        case_samples = by_case.get(case.case_id, [])
        if not case_samples:
            raise ValueError(f"missing samples for {case.case_id}")
        durations = [float(sample["duration_ms"]) for sample in case_samples]
        rates = [float(sample["gib_per_second"]) for sample in case_samples]
        rows.append(
            {
                **asdict(case),
                "sample_count": len(case_samples),
                "region_count": int(case_samples[0]["region_count"]),
                "payload_bytes": int(case_samples[0]["payload_bytes"]),
                "p50_ms": percentile(durations, 0.50),
                "p95_ms": percentile(durations, 0.95),
                "p5_gib_s": percentile(rates, 0.05),
                "p50_gib_s": percentile(rates, 0.50),
                "p95_gib_s": percentile(rates, 0.95),
            }
        )
    return rows
