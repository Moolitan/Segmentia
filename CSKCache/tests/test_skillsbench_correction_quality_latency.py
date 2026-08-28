from __future__ import annotations

import json

import pytest

from paper_evaluation.skillsbench_correction_sweep.quality_latency.analyze import (
    _draw_ratio_metrics,
    _pair_samples,
    _summaries,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.metrics import (
    extract_thinking,
    rouge_l_recall,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.profile import (
    parse_csk_profile,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.preflight import (
    parse_gpu_rows,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.recover import (
    reparse_sample,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.run import (
    SECTION,
    _open_run,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.schema import (
    write_sample_tables,
)
from paper_evaluation.skillsbench_correction_sweep.quality_latency.workload import (
    write_catalog_view,
)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_extract_thinking_prefers_structured_reasoning():
    response = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "structured reasoning",
                    "content": "<think>embedded reasoning</think>answer",
                }
            }
        ]
    }
    assert extract_thinking(response) == (
        "structured reasoning",
        "reasoning_content",
    )


def test_extract_thinking_accepts_truncated_think_tag():
    response = {"choices": [{"message": {"content": "x<think>unfinished"}}]}
    assert extract_thinking(response) == ("unfinished", "content_think_tag")


def test_rouge_l_recall_uses_full_prefill_as_denominator():
    assert rouge_l_recall("A B C D", "a c d") == 0.75
    assert rouge_l_recall("A B", "x y") == 0.0


def test_parse_fixed_layer_calibration_compute(tmp_path):
    path = tmp_path / "profile.jsonl"
    request_id = "chatcmpl-measured"
    _write_jsonl(
        path,
        [
            {
                "event": "csk_request_bind",
                "request_id": request_id,
                "matched_tokens": 1539,
            },
            {
                "event": "csk_reuse_registered",
                "request_id": request_id,
                "reuse_start": 128,
                "reuse_end": 1504,
            },
            {
                "event": "csk_correction_complete",
                "request_id": request_id,
                "calibration_tokens": 77,
            },
            {
                "event": "cskcache_layer_compute",
                "request_id": request_id,
                "calibration_correct_install": [
                    {
                        "layer": 8,
                        "calibration_forward_ms": 1.25,
                        "residual_correction_ms": 0.20,
                        "gpu_ms": 1.55,
                    }
                ],
            },
            {
                "event": "cskcache_layer_compute",
                "request_id": "chatcmpl-unrelated",
                "calibration_correct_install": [],
            },
        ],
    )
    parsed = parse_csk_profile(path, request_id=request_id, profile_layer=8)
    assert parsed["matched_tokens"] == 1539
    assert parsed["reused_tokens"] == 1376
    assert parsed["actual_calibration_tokens"] == 77
    assert parsed["calibration_compute_ms"] == 1.45


def test_parse_profile_accepts_vllm_engine_child_request_id(tmp_path):
    path = tmp_path / "profile.jsonl"
    api_request_id = "chatcmpl-measured"
    engine_request_id = f"{api_request_id}-92b02043"
    _write_jsonl(
        path,
        [
            {
                "event": "csk_request_bind",
                "request_id": engine_request_id,
                "matched_tokens": 1000,
            },
            {
                "event": "csk_reuse_registered",
                "request_id": engine_request_id,
                "reuse_start": 100,
                "reuse_end": 900,
            },
            {
                "event": "csk_correction_complete",
                "request_id": engine_request_id,
                "calibration_tokens": 50,
            },
            {
                "event": "cskcache_layer_compute",
                "request_id": engine_request_id,
                "calibration_correct_install": [
                    {
                        "layer": 8,
                        "calibration_forward_ms": 1.0,
                        "residual_correction_ms": 0.1,
                        "gpu_ms": 1.2,
                    }
                ],
            },
        ],
    )
    parsed = parse_csk_profile(
        path, request_id=api_request_id, profile_layer=8
    )
    assert parsed["actual_calibration_tokens"] == 50
    assert parsed["calibration_compute_ms"] == 1.1


def test_catalog_view_keeps_only_the_exact_object(tmp_path):
    master = {
        "catalog_version": 3,
        "expected_layers": 40,
        "containers": [{"container_id": "pool"}],
        "objects": [
            {"object_id": "xlsx:v1"},
            {"object_id": "xlsx:v2"},
        ],
    }
    output = tmp_path / "catalog.json"
    write_catalog_view(master, "xlsx:v2", output)
    view = json.loads(output.read_text(encoding="utf-8"))
    assert view["containers"] == master["containers"]
    assert view["objects"] == [{"object_id": "xlsx:v2"}]


def test_parse_gpu_rows_uses_nounit_csv():
    assert parse_gpu_rows("0, 12, 49140\n1, 0, 81920\n") == {
        0: (12, 49140),
        1: (0, 81920),
    }


def test_pairing_uses_full_thinking_as_reference(tmp_path):
    full_path = tmp_path / "full.txt"
    ratio_path = tmp_path / "ratio.txt"
    full_path.write_text("inspect data then verify result", encoding="utf-8")
    ratio_path.write_text("inspect data and verify", encoding="utf-8")
    common = {
        "platform_id": "a6000",
        "task_id": "task",
        "status": "valid",
        "thinking_words": 5,
    }
    samples = [
        {
            **common,
            "case_id": "full",
            "system": "Full",
            "system_family": "full",
            "thinking_path": full_path.name,
        },
        {
            **common,
            "case_id": "ratio",
            "system": "Ratio-5%",
            "system_family": "cskcache",
            "thinking_path": ratio_path.name,
            "requested_calibration_ratio": 0.05,
            "calibration_compute_ms": 2.0,
            "calibration_forward_ms": 1.8,
            "residual_correction_ms": 0.2,
            "actual_calibration_tokens": 50,
            "reused_tokens": 900,
        },
    ]
    paired = _pair_samples(tmp_path, samples)
    assert len(paired) == 1
    assert paired[0]["reference_case_id"] == "full"
    assert paired[0]["rouge_l_recall"] == 0.6
    summaries = _summaries(paired)
    assert summaries[0]["median_rouge_l_recall"] == 0.6
    assert summaries[0]["median_calibration_compute_ms"] == 2.0


def test_recovery_reparses_engine_child_profile_without_model_rerun(tmp_path):
    case_id = "a6000__task__ratio-5"
    attempt = tmp_path / "cases" / case_id / "attempt-001"
    server = attempt / "server"
    server.mkdir(parents=True)
    thinking = attempt / "thinking.txt"
    thinking.write_text("inspect and verify", encoding="utf-8")
    from paper_evaluation.common.driver import _benchmark_request_id

    api_request_id = f"chatcmpl-{_benchmark_request_id(case_id)}"
    engine_request_id = f"{api_request_id}-92b02043"
    _write_jsonl(
        server / "cskcache_profile.jsonl",
        [
            {
                "event": "csk_request_bind",
                "request_id": engine_request_id,
                "matched_tokens": 1000,
            },
            {
                "event": "csk_reuse_registered",
                "request_id": engine_request_id,
                "reuse_start": 100,
                "reuse_end": 900,
            },
            {
                "event": "csk_correction_complete",
                "request_id": engine_request_id,
                "calibration_tokens": 50,
            },
            {
                "event": "cskcache_layer_compute",
                "request_id": engine_request_id,
                "calibration_correct_install": [
                    {
                        "layer": 8,
                        "calibration_forward_ms": 1.0,
                        "residual_correction_ms": 0.1,
                        "gpu_ms": 1.2,
                    }
                ],
            },
        ],
    )
    recovered = reparse_sample(
        tmp_path,
        {
            "case_id": case_id,
            "system_family": "cskcache",
            "attempt_dir": attempt.relative_to(tmp_path).as_posix(),
            "thinking_path": thinking.relative_to(tmp_path).as_posix(),
            "skill_tokens": 1000,
            "expected_calibration_tokens": 50,
            "status": "invalid",
        },
    )
    assert recovered["status"] == "valid"
    assert recovered["reused_tokens"] == 800
    assert recovered["calibration_compute_ms"] == 1.1


def test_custom_sample_csv_includes_common_schema_version(tmp_path):
    write_sample_tables(tmp_path, [{"case_id": "case"}])
    header = (tmp_path / "samples.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("schema_version,")


def test_explicit_resume_accepts_code_change_but_not_config_change(
    tmp_path, monkeypatch
):
    from paper_evaluation.skillsbench_correction_sweep.quality_latency import run

    output_root = tmp_path / "output"
    run_dir = output_root / SECTION / "run-001"
    run_dir.mkdir(parents=True)
    values = {"ratio": [0.01, 0.03]}
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "section": SECTION,
                "run_id": "run-001",
                "input_fingerprint": "original",
                "config": values,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "section": SECTION,
                "run_id": "run-001",
                "input_fingerprint": "original",
                "cases": {"done": {"status": "completed"}},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "code.py"
    config_path.write_text("new parser code\n", encoding="utf-8")
    monkeypatch.setattr(run, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("SKILLSBENCH_SWEEP_RESUME_DIR", str(run_dir))

    resumed = _open_run(config_paths=[config_path], config_values=values)
    assert resumed.run_dir == run_dir
    assert resumed.completed("done")
    history = json.loads(
        (run_dir / "resume_history.jsonl").read_text(encoding="utf-8")
    )
    assert history["original_input_fingerprint"] == "original"
    assert history["current_code_fingerprint"] != "original"

    with pytest.raises(RuntimeError, match="scientific config changed"):
        _open_run(
            config_paths=[config_path],
            config_values={"ratio": [0.05]},
        )


def test_dual_axis_ratio_plot_writes_png_and_pdf(tmp_path):
    paired = []
    for task_index, task_id in enumerate(("3d-scan-calc", "citation-check")):
        for ratio in (0.01, 0.05, 0.10):
            paired.append(
                {
                    "platform_id": "a6000",
                    "task_id": task_id,
                    "requested_calibration_ratio": ratio,
                    "calibration_compute_ms": 1.0 + task_index + ratio,
                    "rouge_l_recall": 0.3 + ratio,
                }
            )
    _draw_ratio_metrics(tmp_path, paired, summaries=[{"unused": True}])
    assert (tmp_path / "ratio_quality_latency.png").stat().st_size > 0
    assert (tmp_path / "ratio_quality_latency.pdf").stat().st_size > 0
