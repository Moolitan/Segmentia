from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_evaluation.common.driver import make_server_config
from paper_evaluation.common.schema import read_csv, write_csv
from paper_evaluation.common.workloads import BLEND_SEPARATOR
from paper_evaluation.config import PLATFORMS
from paper_evaluation.section_6_3_latency_scaling import config as local
from paper_evaluation.section_6_3_latency_scaling import run as latency_run
from paper_evaluation.section_6_3_latency_scaling.analyze import (
    analyze,
    bootstrap_mean_ci,
)
from paper_evaluation.section_6_3_latency_scaling.profile import (
    parse_deviation_topk_profile,
)
from paper_evaluation.section_6_3_latency_scaling.schema import SAMPLE_COLUMNS
from paper_evaluation.section_6_3_latency_scaling.workload import (
    Workload,
    bucket_for_tokens,
    eligible_for_all_ratios,
    load_curated_task,
    load_fixed_workloads,
    sha256_file,
    write_catalog_view,
)


def _workload(tmp_path: Path, *, object_id: str = "skill:object") -> Workload:
    task_path = tmp_path / f"{object_id.replace(':', '-')}-task.md"
    skill_path = tmp_path / f"{object_id.replace(':', '-')}-SKILL.md"
    task_path.write_text("Do the task.", encoding="utf-8")
    skill_path.write_text("Use this Skill.", encoding="utf-8")
    return Workload(
        task_id="task",
        source_type="skillsbench",
        skill_name="skill",
        skill_version="v1",
        object_id=object_id,
        skill_tokens=498,
        length_bucket="<1K",
        task_path=task_path,
        skill_path=skill_path,
        relative_skill_path="task/environment/skills/skill/SKILL.md",
    )


def test_launcher_does_not_shadow_python_statistics() -> None:
    launcher = (
        Path(__file__).parents[1]
        / "example/paper_evaluation/section_6_3_latency_scaling/run.sh"
    ).read_text(encoding="utf-8")
    assert "$SUITE_DIR/common" not in launcher
    assert 'CSKCACHE_DIR="$(cd -- "$EXAMPLE_DIR/.." && pwd)"' in launcher
    assert 'PYTHONPATH="$CSKCACHE_DIR:' in launcher


def test_system_matrix_uses_full_prefill_and_two_external_kv_arms(tmp_path) -> None:
    assert [variant.name for variant in local.SYSTEMS] == [
        "Full", "CacheBlend-15%", "CSKCache-5%"
    ]
    full, baseline, method = local.SYSTEMS
    full_config = make_server_config(
        platform=PLATFORMS["a6000_qwen3_14b"],
        variant=full,
        port=8300,
        case_root=tmp_path,
        chunk_tokens=256,
    )
    assert full.family == "full"
    assert full_config.connector is None
    assert baseline.family == "cskcache"
    assert baseline.correction_strategy == "deviation_topk"
    assert baseline.cacheblend_ratio == 0.15
    assert method.calibration_ratio == 0.05
    assert local.CHUNK_TOKENS == 256
    assert local.HOST_PAGE_TOKENS == 512


def test_section_server_config_passes_independent_host_page_tokens(
    tmp_path, monkeypatch
) -> None:
    captured = {}
    expected = object()

    def fake_make_server_config(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        latency_run, "make_server_config", fake_make_server_config
    )
    actual = latency_run._server_config(
        platform_id="a6000_qwen3_14b",
        variant=local.SYSTEMS[2],
        server_dir=tmp_path / "server",
        catalog_view=tmp_path / "catalog.json",
    )
    assert actual is expected
    assert captured["chunk_tokens"] == 256
    assert captured["host_page_tokens"] == 512


def test_request_clears_before_a_and_preserves_prefix_for_b(
    tmp_path, monkeypatch
) -> None:
    workload = _workload(tmp_path)
    events = []

    class FakeServer:
        def reset_prefix_cache(self):
            events.append("reset-before-a")

    def fake_pair(server, **kwargs):
        events.append("request-a-then-b")
        assert kwargs["selection_prompt"] == f"Do the task.\n{BLEND_SEPARATOR}"
        assert kwargs["task_prompt"] == "Do the task."
        assert kwargs["reset_prefix_cache"] is False
        return "result"

    monkeypatch.setattr(latency_run, "run_request_pair", fake_pair)
    result = latency_run._request(
        server=FakeServer(),
        variant=local.SYSTEMS[0],
        workload=workload,
        case_id="case",
    )
    assert result == "result"
    assert events == ["reset-before-a", "request-a-then-b"]


def test_case_id_contains_measurement_repetition(tmp_path) -> None:
    workload = _workload(tmp_path)
    first = latency_run._case_id("a6000_qwen3_14b", workload, "Full", 0)
    fifth = latency_run._case_id("a6000_qwen3_14b", workload, "Full", 4)
    assert first.endswith("rep-01")
    assert fifth.endswith("rep-05")
    assert first != fifth


def test_deviation_profile_validates_all_layers(tmp_path) -> None:
    request_id = "chatcmpl-measure"
    engine_request_id = f"{request_id}-1234abcd"
    records = [
        {"event": "csk_request_bind", "request_id": engine_request_id, "matched_tokens": 327},
        {"event": "csk_reuse_registered", "request_id": engine_request_id, "reuse_start": 100, "reuse_end": 356},
        {"event": "csk_correction_complete", "request_id": engine_request_id, "correction_strategy": "deviation_topk", "execution_method": "deviation_topk", "calibration_tokens": 0},
    ]
    for layer in range(4):
        records.append(
            {
                "event": "cskcache_deviation_topk_layer",
                "request_id": engine_request_id,
                "layer": layer,
                "candidate_tokens": 256,
                "recomputed_tokens": 256 if layer < 1 else 38,
                "selection_applied": layer == 1,
                "recompute_ratio": 0.15,
                "check_layer": 1,
            }
        )
    path = tmp_path / "profile.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    profile = parse_deviation_topk_profile(
        path,
        request_id=request_id,
        expected_layers=4,
        expected_ratio=0.15,
        expected_check_layer=1,
    )
    assert profile["matched_tokens"] == 327
    assert profile["reused_tokens"] == 256
    assert profile["selected_tokens"] == 38


def test_fixed_bucket_boundaries_are_left_closed_right_open() -> None:
    cases = {
        999: "<1K",
        1000: "1K-3K",
        2999: "1K-3K",
        3000: "3K-5K",
        4999: "3K-5K",
        5000: "5K-8K",
        7999: "5K-8K",
        8000: "8K-10K",
        9999: "8K-10K",
        10_000: ">10K",
    }
    for token_count, expected in cases.items():
        assert bucket_for_tokens(token_count, local.LENGTH_BUCKETS) == expected


def test_reuse_eligibility_includes_full_recompute_and_worst_case_alignment():
    assert not eligible_for_all_ratios(
        300,
        max_ratio=0.05,
        minimum_full_recompute_tokens=32,
        block_alignment=16,
        minimum_reuse_tokens=256,
    )
    assert eligible_for_all_ratios(
        498,
        max_ratio=0.05,
        minimum_full_recompute_tokens=32,
        block_alignment=16,
        minimum_reuse_tokens=256,
    )


def test_curated_proof_task_is_frozen_and_over_10k() -> None:
    metadata_path = (
        Path(__file__).parents[1]
        / "example/paper_evaluation/section_6_3_latency_scaling/curated_tasks"
        / "proof-gradient-descent-audit/metadata.json"
    )
    task = load_curated_task(metadata_path)
    assert task.task_id == "proof-gradient-descent-audit"
    assert task.source_type == "curated_repository_task"
    assert task.skill_name == "proof-checker"
    assert task.skill_tokens == 13_314
    assert task.length_bucket == ">10K"
    assert "synthetic" not in task.task_id
    assert "synthetic" not in task.source_type


def test_bootstrap_mean_is_deterministic_and_brackets_mean():
    values = [10.0, 20.0, 30.0, 40.0]
    first = bootstrap_mean_ci(values, resamples=2000, seed=7)
    second = bootstrap_mean_ci(values, resamples=2000, seed=7)
    assert first == second
    assert first[0] <= 25.0 <= first[1]


def test_catalog_view_rejects_ambiguous_selected_skill_names(tmp_path):
    catalog = {
        "catalog_version": 1,
        "expected_layers": 1,
        "containers": [{"container_id": "pool"}],
        "objects": [
            {"object_id": "one", "skill_name": "same"},
            {"object_id": "two", "skill_name": "same"},
        ],
    }
    first = _workload(tmp_path, object_id="one")
    second = _workload(tmp_path, object_id="two")
    first = Workload(**{**first.__dict__, "skill_name": "same"})
    second = Workload(**{**second.__dict__, "skill_name": "same", "task_id": "two"})
    with pytest.raises(RuntimeError, match="ambiguous"):
        write_catalog_view(catalog, [first, second], tmp_path / "view.json")


def test_catalog_view_contains_only_selected_objects(tmp_path):
    catalog = {
        "catalog_version": 1,
        "expected_layers": 1,
        "containers": [{"container_id": "pool"}],
        "objects": [
            {"object_id": "one", "skill_name": "first"},
            {"object_id": "two", "skill_name": "skill"},
        ],
    }
    output = tmp_path / "view.json"
    write_catalog_view(catalog, [_workload(tmp_path, object_id="two")], output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [item["object_id"] for item in payload["objects"]] == ["two"]


def test_verified_pool_loader_requires_exactly_one_workload_per_bucket(tmp_path):
    pool_root = tmp_path / "pool"
    catalog_path = pool_root / "raw/catalog.json"
    catalog_path.parent.mkdir(parents=True)
    counts = (498, 2004, 4056, 6390, 8285, 13_314)
    records = []
    objects = []
    for index, (bucket, token_count) in enumerate(zip(local.BUCKET_ORDER, counts)):
        task = tmp_path / f"task-{index}.md"
        skill = tmp_path / f"skill-{index}.md"
        task.write_text(f"task {index}", encoding="utf-8")
        skill.write_text(f"skill {index}", encoding="utf-8")
        digest = hashlib.sha256(f"tokens-{index}".encode()).hexdigest()
        object_id = f"skill-{index}:{digest[:16]}:model"
        objects.append(
            {
                "object_id": object_id,
                "skill_name": f"skill-{index}",
                "skill_version": f"version-{index}",
                "token_count": token_count,
                "token_ids_sha256": digest,
                "layers": [{"layer_id": 0}],
            }
        )
        records.append(
            {
                "task_id": f"task-{index}",
                "source_type": "skillsbench" if index < 5 else "curated_repository_task",
                "task_path": str(task),
                "task_text_sha256": sha256_file(task),
                "skill_name": f"skill-{index}",
                "skill_path": str(skill),
                "skill_text_sha256": sha256_file(skill),
                "relative_skill_path": f"skills/skill-{index}/SKILL.md",
                "skill_version": f"version-{index}",
                "skill_tokens": token_count,
                "skill_token_ids_sha256": digest,
                "length_bucket": bucket,
                "object_id": object_id,
            }
        )
    catalog = {
        "catalog_version": 3,
        "expected_layers": 1,
        "containers": [{"container_id": "pool"}],
        "objects": objects,
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    manifest = {
        "status": "verified",
        "model_id": "Qwen3-14B",
        "catalog_sha256": sha256_file(catalog_path),
        "workloads": records,
    }
    (pool_root / "fixed_length_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    workloads, _, metadata = load_fixed_workloads(
        pool_root=pool_root,
        expected_model_id="Qwen3-14B",
        buckets=local.LENGTH_BUCKETS,
        max_ratio=0.05,
        minimum_full_recompute_tokens=32,
        block_alignment=16,
        minimum_reuse_tokens=256,
    )
    assert [item.length_bucket for item in workloads] == list(local.BUCKET_ORDER)
    assert metadata["workload_count"] == 6


def test_complete_single_model_matrix_generates_grouped_bucket_figure(tmp_path):
    rows = []
    systems = [variant.name for variant in local.SYSTEMS]
    for bucket_index, bucket in enumerate(local.BUCKET_ORDER):
        for system_index, system in enumerate(systems):
            for repetition in range(1, local.REPETITIONS + 1):
                rows.append(
                    {
                        "status": "valid",
                        "invalid_reason": "",
                        "platform_id": "a6000_qwen3_14b",
                        "model_id": "Qwen3-14B",
                        "length_bucket": bucket,
                        "repetition": repetition,
                        "system": system,
                        "task_id": f"task-{bucket_index}",
                        "ttft_ms": 10 + bucket_index * 20 + system_index + repetition,
                        "prompt_tokens": 1000 + bucket_index * 1000,
                        "skill_tokens": (498, 2004, 4056, 6390, 8285, 13_314)[bucket_index],
                        "reuse_ratio": 0.0 if system == "Full" else 0.8,
                        "fallback": False,
                    }
                )
    write_csv(tmp_path / "samples.csv", rows, SAMPLE_COLUMNS)
    analyze(tmp_path)
    summary = read_csv(tmp_path / "summary.csv")
    assert len(summary) == len(local.BUCKET_ORDER) * len(local.SYSTEMS)
    assert {int(row["sample_count"]) for row in summary} == {5}
    assert (tmp_path / "ttft_a6000_qwen3_14b.pdf").stat().st_size > 0
    assert (tmp_path / "ttft_a6000_qwen3_14b.png").stat().st_size > 0
