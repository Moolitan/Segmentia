from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


from CSKCache.example.ssd_pinned_scatter_gather.common import (
    LAYOUTS,
    build_case_matrix,
    build_scatter_plan,
    summarize_samples,
)
from CSKCache.example.ssd_pinned_scatter_gather.analyze import plot_latency


CFG = SimpleNamespace(
    TOKEN_COUNT=12518,
    CHUNK_SIZE_TOKENS=256,
    NUM_LAYERS=40,
    KV_HEAD_DIM=1024,
    DTYPE_BYTES=2,
    DATASET_SEED=20260824,
    ALIGNMENT_BYTES=4096,
    MAX_IOVECS_PER_CALL=1024,
)


def test_scatter_plans_cover_same_payload_without_overlap() -> None:
    expected = {
        "chunk_all_layers": (3920, 4),
        "chunk_single_layer": (3920, 4),
        "packed_chunks_single_layer": (40, 1),
        "packed_chunks_all_layers": (1, 1),
    }
    for layout in LAYOUTS:
        plan = build_scatter_plan(CFG, layout)
        assert plan.payload_bytes == 2_050_949_120
        assert (len(plan.segments), len(plan.vector_groups)) == expected[layout]
        assert sum(plan.region_lengths) == plan.payload_bytes
        assert sum(segment.length_bytes for segment in plan.segments) == plan.payload_bytes


def test_scatter_gather_case_matrix_has_32_unique_cases() -> None:
    cases = build_case_matrix()
    assert len(cases) == 32
    assert len({case.case_id for case in cases}) == 32
    assert {case.submission_mode for case in cases} == {"multi_read", "readv"}


def test_scatter_gather_summary_and_plot(tmp_path: Path) -> None:
    samples = []
    for case in build_case_matrix():
        for repetition, duration in enumerate((100.0, 80.0)):
            samples.append(
                {
                    "case_id": case.case_id,
                    "duration_ms": duration,
                    "gib_per_second": 5.0 + repetition,
                    "segment_count": 1,
                    "request_count": 1,
                    "payload_bytes": 1024,
                }
            )
    rows = summarize_samples(samples)
    assert len(rows) == 32
    assert all(row["p50_ms"] == 90.0 for row in rows)
    output = tmp_path / "latency.png"
    plot_latency(rows, output, 12518)
    assert output.is_file() and output.stat().st_size > 0
