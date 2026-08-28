"""Run isolated two-request SkillsBench quality--latency cases."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from paper_evaluation import config as suite
from paper_evaluation.common.driver import (
    SystemVariant,
    _benchmark_request_id,
    make_server_config,
    run_request_pair,
)
from paper_evaluation.common.run_state import RunContext, input_fingerprint, utc_now
from paper_evaluation.common.schema import append_jsonl
from paper_evaluation.common.server import VLLMServer
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS

from . import config as local
from .metrics import extract_thinking, finish_reason, rouge_tokens
from .profile import parse_csk_profile
from .schema import atomic_json, collect_samples
from .workload import Workload, load_workloads, write_catalog_view


SECTION = "skillsbench_correction_quality_latency"


def _execution_limit() -> str:
    value = os.environ.get("SKILLSBENCH_SWEEP_LIMIT", "core").strip().lower()
    if value not in {"smoke", "core", "all"}:
        raise ValueError("SKILLSBENCH_SWEEP_LIMIT must be smoke, core, or all")
    return value


def _finalize_partial() -> bool:
    value = os.environ.get("SKILLSBENCH_SWEEP_FINALIZE", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("SKILLSBENCH_SWEEP_FINALIZE must be 0 or 1")
    return value == "1"


def _selected_workloads(
    workloads: list[Workload], execution_limit: str
) -> list[Workload]:
    if execution_limit == "smoke":
        return [workloads[0]]
    if execution_limit == "core":
        return [workload for workload in workloads if workload.tier == "core"]
    return workloads


def _open_run(
    *,
    config_paths: Iterable[Path],
    config_values: Mapping[str, Any],
) -> RunContext:
    resume_value = os.environ.get("SKILLSBENCH_SWEEP_RESUME_DIR", "").strip()
    if not resume_value:
        return RunContext.open(
            output_root=OUTPUT_ROOT,
            section=SECTION,
            config_paths=config_paths,
            config_values=config_values,
        )

    run_dir = Path(resume_value).expanduser().resolve()
    expected_parent = (OUTPUT_ROOT / SECTION).resolve()
    if run_dir.parent != expected_parent:
        raise ValueError(
            "SKILLSBENCH_SWEEP_RESUME_DIR must name a direct run directory under "
            f"{expected_parent}"
        )
    manifest_path = run_dir / "manifest.json"
    state_path = run_dir / "run_state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"resume directory lacks manifest.json or run_state.json: {run_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if manifest.get("section") != SECTION or state.get("section") != SECTION:
        raise ValueError(f"resume directory belongs to a different section: {run_dir}")
    if state.get("status") == "completed":
        raise RuntimeError(f"cannot resume a finalized run: {run_dir}")
    original_fingerprint = str(manifest.get("input_fingerprint", ""))
    if not original_fingerprint or state.get("input_fingerprint") != original_fingerprint:
        raise RuntimeError("resume manifest and run state fingerprints disagree")
    normalized_values = json.loads(
        json.dumps(dict(config_values), ensure_ascii=False, sort_keys=True, default=str)
    )
    if manifest.get("config") != normalized_values:
        raise RuntimeError(
            "explicit resume rejected because the scientific config changed"
        )
    current_fingerprint = input_fingerprint(config_paths, config_values)
    append_jsonl(
        run_dir / "resume_history.jsonl",
        {
            "resumed_utc": utc_now(),
            "run_id": str(manifest["run_id"]),
            "original_input_fingerprint": original_fingerprint,
            "current_code_fingerprint": current_fingerprint,
            "scientific_config_equal": True,
        },
    )
    print(
        f"[resume] run={run_dir} completed_cases="
        f"{sum(case.get('status') == 'completed' for case in state['cases'].values())}",
        flush=True,
    )
    return RunContext(
        section=SECTION,
        run_id=str(manifest["run_id"]),
        run_dir=run_dir,
        fingerprint=original_fingerprint,
        state=state,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def _case_id(platform_id: str, workload: Workload, system: str) -> str:
    return "__".join(
        (_slug(platform_id), _slug(workload.task_id), _slug(system))
    )


def _next_attempt(case_root: Path) -> Path:
    case_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in case_root.glob("attempt-*"):
        try:
            indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = case_root / f"attempt-{max(indices, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def _relative(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _content_text(response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _run_case(
    *,
    run: RunContext,
    platform_id: str,
    workload: Workload,
    variant: SystemVariant,
    master_catalog: dict[str, Any],
) -> dict[str, Any]:
    platform = PLATFORMS[platform_id]
    case_id = _case_id(platform_id, workload, variant.name)
    case_root = run.run_dir / "cases" / case_id
    attempt = _next_attempt(case_root)
    catalog_view = attempt / "catalog.json"
    write_catalog_view(master_catalog, workload.object_id, catalog_view)
    run.mark(
        case_id,
        "running",
        attempt_dir=_relative(attempt, run.run_dir),
    )
    started = utc_now()
    server_cfg = make_server_config(
        platform=platform,
        variant=variant,
        port=BASE_PORT,
        case_root=attempt / "server",
        chunk_tokens=local.CHUNK_TOKENS,
        correction_alpha=local.CORRECTION_ALPHA,
        minimum_full_recompute_tokens=local.MINIMUM_FULL_RECOMPUTE_TOKENS,
        minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
        catalog_override=catalog_view,
        raw_slot_bytes=local.RAW_SLOT_BYTES,
        raw_metadata_bytes=local.RAW_METADATA_BYTES,
    )
    task_prompt = workload.task_path.read_text(encoding="utf-8")
    skill_text = workload.skill_path.read_text(encoding="utf-8")
    with VLLMServer(server_cfg) as server:
        result = run_request_pair(
            server,
            variant=variant,
            skill_name=workload.skill_name,
            skill_text=skill_text,
            task_prompt=task_prompt,
            case_id=case_id,
            max_tokens=local.MAX_TOKENS,
            stream=False,
            selection_prompt=task_prompt,
            enable_thinking=True,
            seed=local.SEED,
        )

    response = (
        dict(result.completion.response)
        if result.completion.response is not None
        else None
    )
    thinking, extraction_source = extract_thinking(response)
    content = _content_text(response)
    response_path = attempt / "response.json"
    thinking_path = attempt / "thinking.txt"
    content_path = attempt / "content.txt"
    atomic_json(response_path, response)
    thinking_path.write_text(thinking, encoding="utf-8")
    content_path.write_text(content, encoding="utf-8")

    profile_values: dict[str, int | float] = {}
    profile_error = ""
    expected_calibration_tokens: int | str = ""
    if variant.family == "cskcache":
        assert variant.calibration_ratio is not None
        expected_calibration_tokens = max(
            1, math.ceil(workload.skill_tokens * variant.calibration_ratio)
        )
        request_id = f"chatcmpl-{_benchmark_request_id(case_id)}"
        try:
            profile_values = parse_csk_profile(
                attempt / "server/cskcache_profile.jsonl",
                request_id=request_id,
                profile_layer=local.PROFILE_LAYER,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            profile_error = f"{type(exc).__name__}: {exc}"

    invalid_reasons = []
    if not thinking.strip():
        invalid_reasons.append("thinking_empty")
    if variant.family == "cskcache":
        if result.fallback:
            invalid_reasons.append(f"fallback:{result.fallback_reason}")
        if profile_error:
            invalid_reasons.append(f"profile:{profile_error}")
        else:
            if int(profile_values["matched_tokens"]) != workload.skill_tokens:
                invalid_reasons.append("matched_tokens_mismatch")
            if (
                int(profile_values["actual_calibration_tokens"])
                != expected_calibration_tokens
            ):
                invalid_reasons.append("calibration_tokens_mismatch")
            if int(profile_values["reused_tokens"]) < local.MINIMUM_REUSE_TOKENS:
                invalid_reasons.append("reuse_interval_too_short")

    actual_calibration_tokens = profile_values.get(
        "actual_calibration_tokens", ""
    )
    sample = {
        "run_id": run.run_id,
        "section": SECTION,
        "case_id": case_id,
        "status": "valid" if not invalid_reasons else "invalid",
        "invalid_reason": ";".join(invalid_reasons),
        "platform_id": platform_id,
        "gpu_name": platform.gpu_name,
        "model_id": platform.model_id,
        "model_path": str(platform.model_path),
        "task_id": workload.task_id,
        "tier": workload.tier,
        "smoke": workload.smoke,
        "skill_name": workload.skill_name,
        "skill_version": workload.skill_version,
        "object_id": workload.object_id,
        "skill_tokens": workload.skill_tokens,
        "system": variant.name,
        "system_family": variant.family,
        "correction_strategy": variant.correction_strategy,
        "requested_calibration_ratio": (
            variant.calibration_ratio
            if variant.calibration_ratio is not None
            else ""
        ),
        "expected_calibration_tokens": expected_calibration_tokens,
        "actual_calibration_tokens": actual_calibration_tokens,
        "actual_calibration_ratio": (
            float(actual_calibration_tokens) / workload.skill_tokens
            if actual_calibration_tokens != ""
            else ""
        ),
        "matched_tokens": profile_values.get("matched_tokens", ""),
        "reused_tokens": profile_values.get("reused_tokens", ""),
        "reuse_start": profile_values.get("reuse_start", ""),
        "reuse_end": profile_values.get("reuse_end", ""),
        "profile_layer": profile_values.get("profile_layer", ""),
        "calibration_forward_ms": profile_values.get(
            "calibration_forward_ms", ""
        ),
        "residual_correction_ms": profile_values.get(
            "residual_correction_ms", ""
        ),
        "calibration_compute_ms": profile_values.get(
            "calibration_compute_ms", ""
        ),
        "layer_gpu_ms": profile_values.get("layer_gpu_ms", ""),
        "prompt_tokens": result.prompt_tokens,
        "vllm_cached_tokens": result.cached_tokens,
        "ttft_ms": result.server_ttft_ms,
        "request_latency_ms": result.completion.client_latency_ms,
        "output_tokens": result.completion.output_tokens,
        "finish_reason": finish_reason(response),
        "thinking_extraction_source": extraction_source,
        "thinking_chars": len(thinking),
        "thinking_words": len(rouge_tokens(thinking)),
        "fallback": result.fallback,
        "fallback_reason": result.fallback_reason,
        "attempt_dir": _relative(attempt, run.run_dir),
        "response_path": _relative(response_path, run.run_dir),
        "thinking_path": _relative(thinking_path, run.run_dir),
        "content_path": _relative(content_path, run.run_dir),
        "catalog_view_path": _relative(catalog_view, run.run_dir),
        "started_utc": started,
        "completed_utc": utc_now(),
    }
    atomic_json(attempt / "sample.json", sample)
    run.mark(
        case_id,
        "completed",
        attempt_dir=_relative(attempt, run.run_dir),
        sample_status=sample["status"],
    )
    return sample


def _smoke_gate(samples: list[dict[str, Any]], platform_id: str) -> None:
    selected = [
        sample
        for sample in samples
        if sample.get("platform_id") == platform_id
        and sample.get("task_id") == local.SMOKE_TASK_ID
    ]
    expected_systems = {variant.name for variant in local.SYSTEMS}
    actual_systems = {str(sample.get("system")) for sample in selected}
    if actual_systems != expected_systems:
        raise RuntimeError(
            f"smoke gate is incomplete: expected={sorted(expected_systems)} "
            f"actual={sorted(actual_systems)}"
        )
    invalid = [sample for sample in selected if sample.get("status") != "valid"]
    if invalid:
        details = ", ".join(
            f"{sample['system']}={sample['invalid_reason']}" for sample in invalid
        )
        raise RuntimeError(f"smoke gate has invalid cases: {details}")
    calibration_tokens = {
        int(sample["actual_calibration_tokens"])
        for sample in selected
        if sample.get("system_family") == "cskcache"
    }
    if len(calibration_tokens) != 5:
        raise RuntimeError("smoke calibration ratios did not produce five budgets")


def main() -> None:
    execution_limit = _execution_limit()
    finalize_partial = _finalize_partial()
    workloads, master_catalog = load_workloads(
        workload_path=local.WORKLOAD_FILE,
        skillsbench_root=local.SKILLSBENCH_ROOT,
        manifest_path=local.MANIFEST_PATH,
        catalog_path=local.MASTER_CATALOG_PATH,
        expected_commit=local.SKILLSBENCH_COMMIT,
        expected_catalog_sha256=local.EXPECTED_CATALOG_SHA256,
    )
    selected_workloads = _selected_workloads(workloads, execution_limit)
    values = {
        "platform_ids": list(local.PLATFORM_IDS),
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "workloads": [
            {
                "task_id": workload.task_id,
                "skill_name": workload.skill_name,
                "tier": workload.tier,
                "smoke": workload.smoke,
                "skill_version": workload.skill_version,
                "object_id": workload.object_id,
                "skill_tokens": workload.skill_tokens,
            }
            for workload in workloads
        ],
        "skillsbench_commit": local.SKILLSBENCH_COMMIT,
        "catalog_sha256": local.EXPECTED_CATALOG_SHA256,
        "max_tokens": local.MAX_TOKENS,
        "seed": local.SEED,
        "chunk_tokens": local.CHUNK_TOKENS,
        "correction_alpha": local.CORRECTION_ALPHA,
        "minimum_full_recompute_tokens": local.MINIMUM_FULL_RECOMPUTE_TOKENS,
        "minimum_reuse_tokens": local.MINIMUM_REUSE_TOKENS,
        "profile_layer": local.PROFILE_LAYER,
        "raw_slot_bytes": local.RAW_SLOT_BYTES,
        "raw_metadata_bytes": local.RAW_METADATA_BYTES,
    }
    config_paths = [
        Path(__file__),
        Path(local.__file__),
        Path(suite.__file__),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("profile.py"),
        Path(__file__).with_name("schema.py"),
        Path(__file__).with_name("workload.py"),
        Path(__file__).resolve().parents[2] / "common/csk_config.py",
        Path(__file__).resolve().parents[2] / "common/driver.py",
        Path(__file__).resolve().parents[2] / "common/server.py",
        local.WORKLOAD_FILE,
        local.MANIFEST_PATH,
        *(workload.task_path for workload in workloads),
        *(workload.skill_path for workload in workloads),
    ]
    run = _open_run(
        config_paths=config_paths,
        config_values=values,
    )
    failures: list[str] = []
    for platform_id in local.PLATFORM_IDS:
        platform = PLATFORMS[platform_id]
        if not platform.model_path.is_dir():
            raise FileNotFoundError(f"model does not exist: {platform.model_path}")
        for workload in selected_workloads:
            for variant in local.SYSTEMS:
                case_id = _case_id(platform_id, workload, variant.name)
                if run.completed(case_id):
                    print(f"[skip] {case_id}", flush=True)
                    continue
                print(f"[case] {case_id}", flush=True)
                try:
                    sample = _run_case(
                        run=run,
                        platform_id=platform_id,
                        workload=workload,
                        variant=variant,
                        master_catalog=master_catalog,
                    )
                    print(
                        f"[{sample['status']}] {case_id} "
                        f"thinking_words={sample['thinking_words']} "
                        f"calibration_ms={sample['calibration_compute_ms']}",
                        flush=True,
                    )
                except Exception as exc:
                    state = run.state["cases"].get(case_id, {})
                    attempt_dir = str(state.get("attempt_dir", ""))
                    if attempt_dir:
                        atomic_json(
                            run.run_dir / attempt_dir / "error.json",
                            {
                                "case_id": case_id,
                                "error": f"{type(exc).__name__}: {exc}",
                                "failed_utc": utc_now(),
                            },
                        )
                    run.mark(
                        case_id,
                        "failed",
                        attempt_dir=attempt_dir,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    failures.append(f"{case_id}: {type(exc).__name__}: {exc}")
                    print(f"[failed] {failures[-1]}", flush=True)
            if workload.smoke:
                from .analyze import analyze

                analyze(run.run_dir)
                _smoke_gate(collect_samples(run.run_dir), platform_id)

    from .analyze import analyze

    analyze(run.run_dir)
    if failures:
        raise RuntimeError(
            f"{len(failures)} cases failed; rerun the same command to create "
            "new attempts: " + " | ".join(failures)
        )
    if execution_limit == "all" or finalize_partial:
        run.finish()
    print(f"results={run.run_dir}")
    if execution_limit != "all" and not finalize_partial:
        print(
            "run remains active for resume; set SKILLSBENCH_SWEEP_LIMIT=all "
            "to continue or SKILLSBENCH_SWEEP_FINALIZE=1 to close this subset"
        )


if __name__ == "__main__":
    main()
