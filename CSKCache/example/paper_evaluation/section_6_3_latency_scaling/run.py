"""Measure TTFT for six fixed Skill-length workloads and three systems."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from paper_evaluation import config as suite
from paper_evaluation.common.driver import (
    SystemVariant,
    _benchmark_request_id,
    make_server_config,
    run_request_pair,
)
from paper_evaluation.common.run_state import RunContext, utc_now
from paper_evaluation.common.schema import write_csv
from paper_evaluation.common.server import VLLMServer
from paper_evaluation.common.workloads import BLEND_SEPARATOR
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS
from paper_evaluation.skillsbench_correction_sweep.quality_latency.profile import (
    parse_csk_profile,
)

from . import config as local
from .profile import parse_deviation_topk_profile
from .schema import (
    WORKLOAD_COLUMNS,
    atomic_json,
    collect_samples,
    write_sample_tables,
)
from .workload import Workload, load_fixed_workloads, write_catalog_view


SECTION = "section_6_3_latency_scaling"


def _execution_limit() -> str:
    value = os.environ.get("LATENCY_SCALING_LIMIT", "all").strip().lower()
    if value not in {"smoke", "all"}:
        raise ValueError("LATENCY_SCALING_LIMIT must be smoke or all")
    return value


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def _case_id(
    platform_id: str, workload: Workload, system: str, repetition: int
) -> str:
    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    return "__".join(
        (
            _slug(platform_id),
            _slug(workload.task_id),
            _slug(system),
            f"rep-{repetition + 1:02d}",
        )
    )


def _next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("attempt-*"):
        try:
            indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    result = root / f"attempt-{max(indices, default=0) + 1:03d}"
    result.mkdir()
    return result


def _relative(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _with_deviation_topk_config(server_cfg):
    environment = dict(server_cfg.extra_env)
    extra = json.loads(environment["LMCACHE_EXTRA_CONFIG"])
    extra["csk_deviation_recompute_ratio"] = local.DEVIATION_RECOMPUTE_RATIO
    extra["csk_deviation_check_layer"] = local.DEVIATION_CHECK_LAYER
    environment["LMCACHE_EXTRA_CONFIG"] = json.dumps(
        extra, separators=(",", ":")
    )
    return replace(server_cfg, extra_env=environment)


def _server_config(
    *,
    platform_id: str,
    variant: SystemVariant,
    server_dir: Path,
    catalog_view: Path,
):
    cfg = make_server_config(
        platform=PLATFORMS[platform_id],
        variant=variant,
        port=BASE_PORT,
        case_root=server_dir,
        chunk_tokens=local.CHUNK_TOKENS,
        correction_alpha=local.CORRECTION_ALPHA,
        minimum_full_recompute_tokens=local.MINIMUM_FULL_RECOMPUTE_TOKENS,
        minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
        host_page_tokens=local.HOST_PAGE_TOKENS,
        catalog_override=catalog_view,
        raw_slot_bytes=local.RAW_SLOT_BYTES,
        raw_metadata_bytes=local.RAW_METADATA_BYTES,
    )
    if variant.correction_strategy == "deviation_topk":
        return _with_deviation_topk_config(cfg)
    return cfg


def _request(
    *,
    server: VLLMServer,
    variant: SystemVariant,
    workload: Workload,
    case_id: str,
):
    task_prompt = workload.task_path.read_text(encoding="utf-8")
    skill_text = workload.skill_path.read_text(encoding="utf-8")
    selection_prompt = f"{task_prompt}\n{BLEND_SEPARATOR}"

    # Isolate this workload from the previous one while retaining request A's
    # ordinary vLLM prefix KV for the immediately following measured request B.
    # External offline Skill KV is intentionally retained by reset_external=false.
    server.reset_prefix_cache()
    return run_request_pair(
        server,
        variant=variant,
        skill_name=workload.skill_name,
        skill_text=skill_text,
        task_prompt=task_prompt,
        case_id=case_id,
        max_tokens=local.MAX_TOKENS,
        stream=True,
        reset_prefix_cache=False,
        selection_prompt=selection_prompt,
        enable_thinking=False,
    )


def _warm_server(
    server: VLLMServer,
    *,
    variant: SystemVariant,
    workload: Workload,
    server_attempt: Path,
) -> None:
    warmup_id = "__".join(
        ("warmup", _slug(variant.name), _slug(server_attempt.name))
    )
    _request(
        server=server,
        variant=variant,
        workload=workload,
        case_id=warmup_id,
    )


def _run_case(
    *,
    run: RunContext,
    server: VLLMServer,
    server_dir: Path,
    platform_id: str,
    variant: SystemVariant,
    workload: Workload,
    repetition: int,
    expected_layers: int,
) -> dict[str, Any]:
    platform = PLATFORMS[platform_id]
    case_id = _case_id(platform_id, workload, variant.name, repetition)
    attempt = _next_attempt(run.run_dir / "cases" / case_id)
    run.mark(
        case_id,
        "running",
        attempt_dir=_relative(attempt, run.run_dir),
        server_dir=_relative(server_dir, run.run_dir),
    )
    started = utc_now()
    result = _request(
        server=server,
        variant=variant,
        workload=workload,
        case_id=case_id,
    )

    expected_calibration_tokens: int | str = ""
    actual_calibration_tokens: int | str = ""
    requested_recompute_ratio: float | str = ""
    expected_recomputed_tokens: int | str = ""
    actual_recomputed_tokens: int | str = ""
    deviation_check_layer: int | str = ""
    profile_reused_tokens = 0
    invalid_reasons = []
    if result.cached_tokens <= 0:
        invalid_reasons.append("request_b_prefix_cache_miss")
    if variant.family == "full":
        host_cache_mode = "none"
        host_cache_prepared = False
    elif variant.correction_strategy == "deviation_topk":
        host_cache_mode = "cskcache_t0_pinned"
        host_cache_prepared = True
        requested_recompute_ratio = local.DEVIATION_RECOMPUTE_RATIO
        deviation_check_layer = local.DEVIATION_CHECK_LAYER
        request_id = f"chatcmpl-{_benchmark_request_id(case_id)}"
        try:
            profile = parse_deviation_topk_profile(
                server_dir / "cskcache_profile.jsonl",
                request_id=request_id,
                expected_layers=expected_layers,
                expected_ratio=local.DEVIATION_RECOMPUTE_RATIO,
                expected_check_layer=local.DEVIATION_CHECK_LAYER,
            )
            expected_recomputed_tokens = int(profile["selected_tokens"])
            actual_recomputed_tokens = int(profile["selected_tokens"])
            profile_reused_tokens = int(profile["reused_tokens"])
            if int(profile["matched_tokens"]) != workload.skill_tokens:
                invalid_reasons.append("matched_tokens_mismatch")
            if profile_reused_tokens < local.MINIMUM_REUSE_TOKENS:
                invalid_reasons.append("reuse_interval_too_short")
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            invalid_reasons.append(f"profile:{type(exc).__name__}:{exc}")
    else:
        host_cache_mode = "cskcache_t0_pinned"
        host_cache_prepared = True
        assert variant.calibration_ratio is not None
        expected_calibration_tokens = max(
            1, math.ceil(workload.skill_tokens * variant.calibration_ratio)
        )
        request_id = f"chatcmpl-{_benchmark_request_id(case_id)}"
        try:
            profile = parse_csk_profile(
                server_dir / "cskcache_profile.jsonl",
                request_id=request_id,
                profile_layer=local.PROFILE_LAYER,
            )
            actual_calibration_tokens = int(profile["actual_calibration_tokens"])
            profile_reused_tokens = int(profile["reused_tokens"])
            if int(profile["matched_tokens"]) != workload.skill_tokens:
                invalid_reasons.append("matched_tokens_mismatch")
            if actual_calibration_tokens != expected_calibration_tokens:
                invalid_reasons.append("calibration_tokens_mismatch")
            if profile_reused_tokens < local.MINIMUM_REUSE_TOKENS:
                invalid_reasons.append("reuse_interval_too_short")
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            invalid_reasons.append(f"profile:{type(exc).__name__}:{exc}")

    if variant.family == "cskcache":
        if result.fallback:
            invalid_reasons.append(f"fallback:{result.fallback_reason}")

    result_path = attempt / "result.json"
    atomic_json(
        result_path,
        {
            "case_id": case_id,
            "text": result.completion.text,
            "client_ttft_ms": result.completion.client_ttft_ms,
            "client_latency_ms": result.completion.client_latency_ms,
            "server_ttft_ms": result.server_ttft_ms,
            "prompt_tokens": result.prompt_tokens,
            "cached_tokens": result.cached_tokens,
        },
    )
    sample = {
        "schema_version": 1,
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
        "source_type": workload.source_type,
        "skill_name": workload.skill_name,
        "skill_version": workload.skill_version,
        "object_id": workload.object_id,
        "skill_tokens": workload.skill_tokens,
        "length_bucket": workload.length_bucket,
        "repetition": repetition + 1,
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
        "requested_recompute_ratio": requested_recompute_ratio,
        "expected_recomputed_tokens": expected_recomputed_tokens,
        "actual_recomputed_tokens": actual_recomputed_tokens,
        "deviation_check_layer": deviation_check_layer,
        "prompt_tokens": result.prompt_tokens,
        "vllm_cached_tokens": result.cached_tokens,
        "profile_reused_tokens": profile_reused_tokens,
        "reuse_ratio": (
            profile_reused_tokens / result.prompt_tokens
            if result.prompt_tokens
            else 0.0
        ),
        "host_cache_mode": host_cache_mode,
        "host_cache_prepared": host_cache_prepared,
        "ttft_ms": result.server_ttft_ms,
        "client_ttft_ms": result.completion.client_ttft_ms,
        "request_latency_ms": result.completion.client_latency_ms,
        "output_tokens": result.completion.output_tokens,
        "fallback": result.fallback,
        "fallback_reason": result.fallback_reason,
        "attempt_dir": _relative(attempt, run.run_dir),
        "server_dir": _relative(server_dir, run.run_dir),
        "started_utc": started,
        "completed_utc": utc_now(),
    }
    atomic_json(attempt / "sample.json", sample)
    run.mark(
        case_id,
        "completed" if sample["status"] == "valid" else "failed",
        attempt_dir=_relative(attempt, run.run_dir),
        server_dir=_relative(server_dir, run.run_dir),
        sample_status=sample["status"],
    )
    write_sample_tables(run.run_dir, collect_samples(run.run_dir))
    return sample


def _write_workloads(run_dir: Path, workloads: list[Workload]) -> None:
    rows = []
    for index, workload in enumerate(workloads):
        rows.append(
            {
                "selection_order": index,
                "source_type": workload.source_type,
                "length_bucket": workload.length_bucket,
                "task_id": workload.task_id,
                "skill_name": workload.skill_name,
                "skill_version": workload.skill_version,
                "object_id": workload.object_id,
                "skill_tokens": workload.skill_tokens,
                "relative_skill_path": workload.relative_skill_path,
            }
        )
    write_csv(run_dir / "selected_workloads.csv", rows, WORKLOAD_COLUMNS)


def _smoke_gate(run_dir: Path, platform_id: str, workload: Workload) -> None:
    expected = set(local.SMOKE_SYSTEM_NAMES)
    rows = [
        row
        for row in collect_samples(run_dir)
        if row["platform_id"] == platform_id and row["task_id"] == workload.task_id
        and row["system"] in expected and int(row["repetition"]) == 1
    ]
    if {row["system"] for row in rows} != expected:
        raise RuntimeError("smoke gate is incomplete")
    invalid = [row for row in rows if row["status"] != "valid"]
    if invalid:
        detail = " | ".join(
            f"{row['system']}={row['invalid_reason']}" for row in invalid
        )
        raise RuntimeError(f"smoke gate failed: {detail}")


def _full_gate(run_dir: Path, platform_id: str, workloads: list[Workload]) -> None:
    rows = [
        row for row in collect_samples(run_dir) if row["platform_id"] == platform_id
    ]
    expected = len(workloads) * len(local.SYSTEMS) * local.REPETITIONS
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} samples, found {len(rows)}")
    invalid = [row for row in rows if row["status"] != "valid"]
    if invalid:
        raise RuntimeError(f"full gate has {len(invalid)} invalid samples")
    identities = {
        (row["task_id"], row["system"], int(row["repetition"])) for row in rows
    }
    if len(identities) != expected:
        raise RuntimeError("full gate has duplicate or missing repetition identities")


def main() -> None:
    limit = _execution_limit()
    platform_id = local.ACTIVE_PLATFORM_IDS[0]
    if tuple(local.ACTIVE_PLATFORM_IDS) != (platform_id,):
        raise RuntimeError("this phase must run exactly one active platform")
    pool_root = local.PLATFORM_CACHE_POOLS[platform_id]
    if pool_root is None:
        raise RuntimeError(f"no offline cache pool configured for {platform_id}")
    workloads, master_catalog, selection_metadata = load_fixed_workloads(
        pool_root=pool_root,
        expected_model_id=PLATFORMS[platform_id].model_id,
        buckets=local.LENGTH_BUCKETS,
        max_ratio=local.MAX_CALIBRATION_RATIO,
        minimum_full_recompute_tokens=local.MINIMUM_FULL_RECOMPUTE_TOKENS,
        block_alignment=local.VLLM_BLOCK_ALIGNMENT_TOKENS,
        minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
    )
    values = {
        "active_platform_ids": list(local.ACTIVE_PLATFORM_IDS),
        "future_platform_ids": list(local.FUTURE_PLATFORM_IDS),
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "skillsbench_commit": local.SKILLSBENCH_COMMIT,
        "catalog_sha256": selection_metadata["catalog_sha256"],
        "selection": selection_metadata,
        "workloads": [
            {
                "task_id": workload.task_id,
                "source_type": workload.source_type,
                "skill_name": workload.skill_name,
                "skill_version": workload.skill_version,
                "object_id": workload.object_id,
                "skill_tokens": workload.skill_tokens,
                "length_bucket": workload.length_bucket,
            }
            for workload in workloads
        ],
        "max_tokens": local.MAX_TOKENS,
        "chunk_tokens": local.CHUNK_TOKENS,
        "host_page_tokens": local.HOST_PAGE_TOKENS,
        "vllm_block_alignment_tokens": local.VLLM_BLOCK_ALIGNMENT_TOKENS,
        "minimum_full_recompute_tokens": local.MINIMUM_FULL_RECOMPUTE_TOKENS,
        "minimum_reuse_tokens": local.MINIMUM_REUSE_TOKENS,
        "deviation_recompute_ratio": local.DEVIATION_RECOMPUTE_RATIO,
        "deviation_check_layer": local.DEVIATION_CHECK_LAYER,
        "repetitions": local.REPETITIONS,
        "smoke_bucket": local.SMOKE_BUCKET,
        "request_pair_prefix_policy": "clear_before_a_preserve_a_to_b",
    }
    source_dir = Path(__file__).resolve().parent
    config_paths = [
        Path(__file__),
        Path(local.__file__),
        source_dir / "analyze.py",
        source_dir / "schema.py",
        source_dir / "profile.py",
        source_dir / "workload.py",
        Path(suite.__file__),
        source_dir / "workloads.json",
        pool_root / "fixed_length_manifest.json",
        pool_root / "raw" / "catalog.json",
        *(workload.task_path for workload in workloads),
        *(workload.skill_path for workload in workloads),
    ]
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=config_paths,
        config_values=values,
    )
    _write_workloads(run.run_dir, workloads)
    catalog_view = run.run_dir / "selected_catalog.json"
    write_catalog_view(master_catalog, workloads, catalog_view)
    expected_layers = int(master_catalog["expected_layers"])

    variants = list(local.SYSTEMS)
    if limit == "smoke":
        variants = [
            variant for variant in variants if variant.name in local.SMOKE_SYSTEM_NAMES
        ]
        selected_workloads = [
            workload
            for workload in workloads
            if workload.length_bucket == local.SMOKE_BUCKET
        ]
        repetitions = (0,)
    else:
        selected_workloads = workloads
        repetitions = tuple(range(local.REPETITIONS))

    platform = PLATFORMS[platform_id]
    if not platform.model_path.is_dir():
        raise FileNotFoundError(f"model does not exist: {platform.model_path}")
    for variant in variants:
        pending = [
            (workload, repetition)
            for workload in selected_workloads
            for repetition in repetitions
            if not run.completed(
                _case_id(platform_id, workload, variant.name, repetition)
            )
        ]
        if not pending:
            continue
        server_attempt = _next_attempt(
            run.run_dir / "servers" / platform_id / _slug(variant.name)
        )
        cfg = _server_config(
            platform_id=platform_id,
            variant=variant,
            server_dir=server_attempt,
            catalog_view=catalog_view,
        )
        print(
            f"[server] platform={platform_id} system={variant.name} "
            f"pending={len(pending)}",
            flush=True,
        )
        try:
            with VLLMServer(cfg) as server:
                _warm_server(
                    server,
                    variant=variant,
                    workload=workloads[0],
                    server_attempt=server_attempt,
                )
                for workload, repetition in pending:
                    case_id = _case_id(
                        platform_id, workload, variant.name, repetition
                    )
                    print(f"[case] {case_id}", flush=True)
                    try:
                        sample = _run_case(
                            run=run,
                            server=server,
                            server_dir=server_attempt,
                            platform_id=platform_id,
                            variant=variant,
                            workload=workload,
                            repetition=repetition,
                            expected_layers=expected_layers,
                        )
                    except Exception as exc:
                        state = run.state["cases"].get(case_id, {})
                        attempt_value = str(state.get("attempt_dir", ""))
                        if attempt_value:
                            atomic_json(
                                run.run_dir / attempt_value / "error.json",
                                {
                                    "case_id": case_id,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "failed_utc": utc_now(),
                                },
                            )
                        run.mark(
                            case_id,
                            "failed",
                            attempt_dir=attempt_value,
                            server_dir=_relative(server_attempt, run.run_dir),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise
                    print(
                        f"[{sample['status']}] {case_id} "
                        f"ttft_ms={sample['ttft_ms']:.3f} "
                        f"cached_tokens={sample['vllm_cached_tokens']}",
                        flush=True,
                    )
                    if sample["status"] != "valid":
                        raise RuntimeError(
                            f"invalid sample {case_id}: {sample['invalid_reason']}"
                        )
        except Exception:
            print(f"[abort-arm] system={variant.name}", flush=True)
            raise

    from .analyze import analyze

    analyze(run.run_dir)
    if limit == "smoke":
        _smoke_gate(run.run_dir, platform_id, selected_workloads[0])
        print("[smoke-pass] Full, CacheBlend-15%, and CSKCache-5%", flush=True)
    else:
        _full_gate(run.run_dir, platform_id, workloads)
        run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
