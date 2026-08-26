"""Measure server-side TTFT across models, platforms, and Skill lengths."""

from __future__ import annotations

import json
from pathlib import Path

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS, RAW_POOL_ROOT
from common.csk_config import catalog_path, require_catalog_skills
from common.driver import make_server_config, run_request_pair
from common.run_state import RunContext, utc_now
from common.server import VLLMServer


SECTION = "section_6_3_latency_scaling"


def _skill_tokens(tokenizer, skill_path: Path) -> int:
    return len(
        tokenizer.encode(
            skill_path.read_text(encoding="utf-8"), add_special_tokens=False
        )
    )


def main() -> None:
    values = {
        "platform_ids": list(local.PLATFORM_IDS),
        "workloads": [name for name, _skill, _prompt in local.WORKLOADS],
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "chunk_tokens": local.CHUNK_TOKENS,
        "replicas": local.REPLICAS,
        "warmups": local.WARMUPS,
        "repetitions": local.REPETITIONS,
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(
            Path(__file__), Path(local.__file__), Path(suite.__file__),
            *(path for _name, skill, prompt in local.WORKLOADS for path in (skill, prompt)),
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
            {name for name, _skill, _prompt in local.WORKLOADS},
        )
        for skill_index, (skill_name, skill_path, prompt_path) in enumerate(
            local.WORKLOADS
        ):
            skill_text = skill_path.read_text(encoding="utf-8")
            task_prompt = prompt_path.read_text(encoding="utf-8").strip()
            token_count = _skill_tokens(tokenizer, skill_path)
            for system_index, variant in enumerate(local.SYSTEMS):
                for replica in range(local.REPLICAS):
                    all_ordinals = [
                        (True, index) for index in range(local.WARMUPS)
                    ] + [
                        (False, index) for index in range(local.REPETITIONS)
                    ]
                    pending = []
                    for warmup, repetition in all_ordinals:
                        case_id = (
                            f"{platform_id}__{variant.name}__{skill_name}__"
                            f"replica{replica}__"
                            f"{'warmup' if warmup else 'measure'}{repetition}"
                        )
                        if not run.completed(case_id):
                            pending.append((case_id, warmup, repetition))
                    if not pending:
                        continue
                    server_root = (
                        run.run_dir / "servers" / platform_id / variant.name
                        / skill_name / f"replica_{replica}"
                    )
                    port = (
                        BASE_PORT + platform_index * 100 + skill_index * 20
                        + system_index * 4 + replica
                    )
                    server_cfg = make_server_config(
                        platform=platform,
                        variant=variant,
                        port=port,
                        case_root=server_root,
                        chunk_tokens=local.CHUNK_TOKENS,
                        correction_alpha=local.CORRECTION_ALPHA,
                        minimum_full_recompute_tokens=(
                            local.MINIMUM_FULL_RECOMPUTE_TOKENS
                        ),
                        minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
                    )
                    with VLLMServer(server_cfg) as server:
                        for case_id, warmup, repetition in pending:
                            case_dir = run.run_dir / "cases" / case_id
                            case_dir.mkdir(parents=True, exist_ok=True)
                            run.mark(case_id, "running", case_dir=str(case_dir))
                            started = utc_now()
                            try:
                                result = run_request_pair(
                                    server,
                                    variant=variant,
                                    skill_name=skill_name,
                                    skill_text=skill_text,
                                    task_prompt=task_prompt,
                                    case_id=case_id,
                                    max_tokens=local.MAX_TOKENS,
                                    stream=True,
                                )
                                (case_dir / "result.json").write_text(
                                    json.dumps(
                                        {
                                            "case_id": case_id,
                                            "text": result.completion.text,
                                            "client_ttft_ms": (
                                                result.completion.client_ttft_ms
                                            ),
                                            "server_ttft_ms": result.server_ttft_ms,
                                        },
                                        indent=2,
                                        ensure_ascii=False,
                                    )
                                    + "\n",
                                    encoding="utf-8",
                                )
                                run.record(
                                    {
                                        "case_id": case_id,
                                        "status": "completed",
                                        "platform_id": platform_id,
                                        "gpu_name": platform.gpu_name,
                                        "model_id": platform.model_id,
                                        "model_path": str(platform.model_path),
                                        "tensor_parallel_size": (
                                            platform.tensor_parallel_size
                                        ),
                                        "system": variant.name,
                                        "skill_name": skill_name,
                                        "skill_tokens": token_count,
                                        "task_id": prompt_path.stem,
                                        "chunk_tokens": local.CHUNK_TOKENS,
                                        "correction_strategy": (
                                            variant.correction_strategy
                                        ),
                                        "correction_budget_tokens": (
                                            variant.calibration_tokens or ""
                                        ),
                                        "correction_ratio": (
                                            variant.calibration_ratio
                                            if variant.calibration_ratio is not None
                                            else variant.cacheblend_ratio
                                        ),
                                        "replica": replica,
                                        "repetition": repetition,
                                        "warmup": warmup,
                                        "prompt_tokens": result.prompt_tokens,
                                        "reused_tokens": result.cached_tokens,
                                        "reuse_ratio": (
                                            result.cached_tokens / result.prompt_tokens
                                            if result.prompt_tokens else 0.0
                                        ),
                                        "ttft_ms": result.server_ttft_ms,
                                        "latency_ms": (
                                            result.completion.client_latency_ms
                                        ),
                                        "output_tokens": (
                                            result.completion.output_tokens
                                        ),
                                        "fallback": result.fallback,
                                        "fallback_reason": result.fallback_reason,
                                        "started_utc": started,
                                        "completed_utc": utc_now(),
                                    }
                                )
                                run.mark(
                                    case_id, "completed", case_dir=str(case_dir)
                                )
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
