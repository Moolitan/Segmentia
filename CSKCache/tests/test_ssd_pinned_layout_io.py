from __future__ import annotations

from pathlib import Path
import sys


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "ssd_pinned_layout_io"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from common import LAYOUTS, build_artifact, build_case_matrix, summarize_samples
from analyze import plot_latency_bars
from run import execute_read


def _artifact(layout: str):
    return build_artifact(
        layout=layout,
        raw_file=Path(f"/{layout}.raw"),
        token_count=12518,
        chunk_size_tokens=256,
        num_layers=40,
        hidden_dim=1024,
        dtype_bytes=2,
        alignment_bytes=4096,
        header_bytes=4096,
        metadata_bytes=64 * 1024**2,
    )


def test_four_layout_geometries_cover_equal_aligned_payloads() -> None:
    expected_counts = {
        "chunk_all_layers": 49,
        "chunk_single_layer": 1960,
        "packed_chunks_single_layer": 40,
        "packed_chunks_all_layers": 1,
    }
    artifacts = {layout: _artifact(layout) for layout in LAYOUTS}
    assert {
        layout: len(value.extents) for layout, value in artifacts.items()
    } == expected_counts
    assert {value.payload_bytes for value in artifacts.values()} == {2_050_949_120}
    for artifact in artifacts.values():
        assert (
            sum(extent.length_bytes for extent in artifact.extents)
            == artifact.payload_bytes
        )
        assert all(extent.offset_bytes % 4096 == 0 for extent in artifact.extents)
        assert all(extent.length_bytes % 4096 == 0 for extent in artifact.extents)


def test_case_matrix_is_four_by_two_by_two() -> None:
    cases = build_case_matrix()
    assert len(cases) == 16
    assert len({case.case_id for case in cases}) == 16
    assert {case.layout for case in cases} == set(LAYOUTS)
    assert {case.io_engine for case in cases} == {"posix", "io_uring"}
    assert {case.use_odirect for case in cases} == {False, True}


def test_summary_keeps_all_sixteen_cases() -> None:
    samples = []
    for case in build_case_matrix():
        for repetition, duration in enumerate((100.0, 80.0)):
            samples.append(
                {
                    "case_id": case.case_id,
                    "duration_ms": duration,
                    "gib_per_second": 5.0 if repetition == 0 else 6.0,
                    "region_count": 1,
                    "payload_bytes": 1024,
                }
            )
    rows = summarize_samples(samples)
    assert len(rows) == 16
    assert all(row["sample_count"] == 2 for row in rows)
    assert all(row["p50_ms"] == 90.0 for row in rows)
    assert all(row["p50_gib_s"] == 5.5 for row in rows)


def test_execute_read_submits_one_complete_extent_batch() -> None:
    artifact = _artifact("packed_chunks_single_layer")

    class FakeCore:
        calls = []

        def read_extents_into(self, offsets, lengths, objects):
            self.calls.append((offsets, lengths, objects))
            return [True] * len(objects)

    core = FakeCore()
    objects = [object() for _ in artifact.extents]
    assert execute_read(core, artifact, objects) >= 0
    assert len(core.calls) == 1
    assert len(core.calls[0][0]) == len(artifact.extents)
    assert len(core.calls[0][1]) == len(artifact.extents)
    assert core.calls[0][2] == objects


def test_latency_bar_chart_smoke(tmp_path: Path) -> None:
    rows = []
    for case in build_case_matrix():
        rows.append(
            {
                "layout": case.layout,
                "io_engine": case.io_engine,
                "use_odirect": case.use_odirect,
                "p50_ms": 100.0,
            }
        )
    output = tmp_path / "latency-bar-chart.png"
    plot_latency_bars(rows, output, token_count=12518)
    assert output.is_file()
    assert output.stat().st_size > 0
