"""Run frozen rule-adherence tasks for correction strategies."""

from __future__ import annotations

import json
from pathlib import Path

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS, ROOT
from paper_evaluation.config import RAW_POOL_ROOT
from common.csk_config import catalog_path, require_catalog_skills
from common.driver import evaluate_rules, make_server_config, run_request_pair
from common.run_state import RunContext, utc_now
from common.server import VLLMServer


SECTION = "section_6_2_correction_quality"


def load_workloads() -> list[dict]:
    workloads = json.loads(local.WORKLOAD_FILE.read_text(encoding="utf-8"))
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads.json must contain a non-empty list")
    for workload in workloads:
        if not workload.get("rules"):
            raise ValueError(f"workload has no rules: {workload}")
    return workloads


def main() -> None:
    workloads = load_workloads()
    values = {
        "platform_ids": list(local.PLATFORM_IDS),
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "workload_file": str(local.WORKLOAD_FILE),
        "max_tokens": local.MAX_TOKENS,
        "chunk_tokens": local.CHUNK_TOKENS,
        "repetitions": local.REPETITIONS,
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(
            Path(__file__), Path(local.__file__), Path(suite.__file__),
            local.WORKLOAD_FILE,
            *(ROOT / workload["skill_path"] for workload in workloads),
        ),
        config_values=values,
    )
    from transformers import AutoTokenizer

    for platform_index, platform_id in enumerate(local.PLATFORM_IDS):
        platform = PLATFORMS[platform_id]
        if not platform.model_path.is_dir():
            raise FileNotFoundError(f"model does not exist: {platform.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            platform.model_path, local_files_only=True
        )
        require_catalog_skills(
            catalog_path(RAW_POOL_ROOT, platform.model_id, "raw_block"),
            {str(workload["skill_name"]) for workload in workloads},
        )
        for system_index, variant in enumerate(local.SYSTEMS):
            server_root = run.run_dir / "servers" / platform_id / variant.name
            server_cfg = make_server_config(
                platform=platform,
                variant=variant,
                port=BASE_PORT + platform_index * 20 + system_index,
                case_root=server_root,
                chunk_tokens=local.CHUNK_TOKENS,
                correction_alpha=local.CORRECTION_ALPHA,
                minimum_full_recompute_tokens=(
                    local.MINIMUM_FULL_RECOMPUTE_TOKENS
                ),
                minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
            )
            with VLLMServer(server_cfg) as server:
                for workload in workloads:
                    skill_path = ROOT / workload["skill_path"]
                    skill_text = skill_path.read_text(encoding="utf-8")
                    skill_tokens = len(
                        tokenizer.encode(skill_text, add_special_tokens=False)
                    )
                    for repetition in range(local.REPETITIONS):
                        case_id = (
                            f"{platform_id}__{variant.name}__"
                            f"{workload['skill_name']}__{workload['task_id']}__"
                            f"r{repetition}"
                        )
                        if run.completed(case_id):
                            continue
                        case_dir = run.run_dir / "cases" / case_id
                        case_dir.mkdir(parents=True, exist_ok=True)
                        run.mark(case_id, "running", case_dir=str(case_dir))
                        started = utc_now()
                        try:
                            result = run_request_pair(
                                server,
                                variant=variant,
                                skill_name=workload["skill_name"],
                                skill_text=skill_text,
                                task_prompt=workload["task_prompt"],
                                case_id=case_id,
                                max_tokens=local.MAX_TOKENS,
                                stream=False,
                            )
                            passed, total, rule_results = evaluate_rules(
                                result.completion.response,
                                result.completion.text,
                                workload["rules"],
                            )
                            raw = {
                                "case_id": case_id,
                                "workload": workload,
                                "response": result.completion.response,
                                "text": result.completion.text,
                                "rules": rule_results,
                            }
                            (case_dir / "result.json").write_text(
                                json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                            budget = (
                                variant.calibration_tokens
                                if variant.calibration_tokens
                                else ""
                            )
                            row = {
                                "case_id": case_id,
                                "status": "completed",
                                "platform_id": platform_id,
                                "gpu_name": platform.gpu_name,
                                "model_id": platform.model_id,
                                "model_path": str(platform.model_path),
                                "tensor_parallel_size": platform.tensor_parallel_size,
                                "system": variant.name,
                                "skill_name": workload["skill_name"],
                                "skill_tokens": skill_tokens,
                                "task_id": workload["task_id"],
                                "correction_strategy": variant.correction_strategy,
                                "correction_budget_tokens": budget,
                                "correction_ratio": (
                                    variant.calibration_ratio
                                    if variant.calibration_ratio is not None
                                    else variant.cacheblend_ratio
                                ),
                                "repetition": repetition,
                                "warmup": False,
                                "prompt_tokens": result.prompt_tokens,
                                "reused_tokens": result.cached_tokens,
                                "reuse_ratio": (
                                    result.cached_tokens / result.prompt_tokens
                                    if result.prompt_tokens else 0.0
                                ),
                                "ttft_ms": result.server_ttft_ms,
                                "latency_ms": result.completion.client_latency_ms,
                                "rule_adherence": passed / total,
                                "rule_passed": passed,
                                "rule_total": total,
                                "output_tokens": result.completion.output_tokens,
                                "fallback": result.fallback,
                                "fallback_reason": result.fallback_reason,
                                "started_utc": started,
                                "completed_utc": utc_now(),
                            }
                            run.record(row)
                            run.mark(case_id, "completed", case_dir=str(case_dir))
                        except Exception as exc:
                            run.mark(
                                case_id,
                                "failed",
                                case_dir=str(case_dir),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            raise
    from analyze import analyze

    analyze(run.run_dir)
    run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
