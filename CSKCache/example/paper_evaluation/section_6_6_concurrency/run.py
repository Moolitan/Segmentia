"""Measure synchronized concurrent request-B arrivals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS, RAW_POOL_ROOT
from common.csk_config import catalog_path, require_catalog_skills
from common.driver import (
    execute_prepared_request,
    make_server_config,
    prepare_request_pair,
)
from common.run_state import RunContext, utc_now
from common.server import VLLMServer


SECTION = "section_6_6_concurrency"


def _execute_at_barrier(server, variant, prepared, case_id, barrier):
    barrier.wait()
    return execute_prepared_request(
        server,
        variant=variant,
        prepared=prepared,
        skill_name=local.SKILL_NAME,
        case_id=case_id,
        max_tokens=local.MAX_TOKENS,
        stream=True,
        reset_prefix_cache=False,
    )


def main() -> None:
    values = {
        "platform_ids": list(local.PLATFORM_IDS),
        "skill_path": str(local.SKILL_PATH),
        "task_prompt_path": str(local.TASK_PROMPT_PATH),
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "concurrencies": list(local.CONCURRENCIES),
        "chunk_tokens": local.CHUNK_TOKENS,
        "replicas": local.REPLICAS,
        "warmup_batches": local.WARMUP_BATCHES,
        "measured_batches": local.MEASURED_BATCHES,
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(
            Path(__file__), Path(local.__file__), Path(suite.__file__),
            local.SKILL_PATH, local.TASK_PROMPT_PATH,
        ),
        config_values=values,
    )
    from transformers import AutoTokenizer

    skill_text = local.SKILL_PATH.read_text(encoding="utf-8")
    base_task = local.TASK_PROMPT_PATH.read_text(encoding="utf-8").strip()
    for platform_index, platform_id in enumerate(local.PLATFORM_IDS):
        platform = PLATFORMS[platform_id]
        if not platform.model_path.is_dir():
            raise FileNotFoundError(f"model does not exist: {platform.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            platform.model_path, local_files_only=True
        )
        require_catalog_skills(
            catalog_path(RAW_POOL_ROOT, platform.model_id, "raw_block"),
            {local.SKILL_NAME},
        )
        skill_tokens = len(tokenizer.encode(skill_text, add_special_tokens=False))
        for system_index, variant in enumerate(local.SYSTEMS):
            for concurrency_index, concurrency in enumerate(local.CONCURRENCIES):
                for replica in range(local.REPLICAS):
                    ordinals = [
                        (True, value) for value in range(local.WARMUP_BATCHES)
                    ] + [
                        (False, value) for value in range(local.MEASURED_BATCHES)
                    ]
                    pending = []
                    for warmup, batch in ordinals:
                        case_ids = tuple(
                            f"{platform_id}__{variant.name}__c{concurrency}__"
                            f"replica{replica}__{'warmup' if warmup else 'measure'}"
                            f"{batch}__slot{slot}"
                            for slot in range(concurrency)
                        )
                        if not all(run.completed(case_id) for case_id in case_ids):
                            pending.append((warmup, batch, case_ids))
                    if not pending:
                        continue
                    server_root = (
                        run.run_dir / "servers" / platform_id / variant.name
                        / f"concurrency_{concurrency}" / f"replica_{replica}"
                    )
                    port = (
                        BASE_PORT + platform_index * 100
                        + system_index * 20 + concurrency_index * 4 + replica
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
                        for warmup, batch, case_ids in pending:
                            prepared = []
                            for slot, case_id in enumerate(case_ids):
                                if run.completed(case_id):
                                    raise RuntimeError(
                                        "partial concurrent batch cannot be resumed; "
                                        "delete this active run and restart the subsection"
                                    )
                                case_dir = run.run_dir / "cases" / case_id
                                case_dir.mkdir(parents=True, exist_ok=True)
                                run.mark(case_id, "running", case_dir=str(case_dir))
                                prepared.append(
                                    prepare_request_pair(
                                        server,
                                        variant=variant,
                                        skill_name=local.SKILL_NAME,
                                        skill_text=skill_text,
                                        task_prompt=(
                                            f"{base_task}\n\nConcurrent slot: {slot}. "
                                            f"Replica: {replica}. Batch: {batch}."
                                        ),
                                        case_id=case_id,
                                    )
                                )
                            server.reset_prefix_cache()
                            barrier = threading.Barrier(concurrency)
                            batch_started = time.perf_counter_ns()
                            with ThreadPoolExecutor(
                                max_workers=concurrency
                            ) as executor:
                                futures = [
                                    executor.submit(
                                        _execute_at_barrier,
                                        server,
                                        variant,
                                        item,
                                        case_id,
                                        barrier,
                                    )
                                    for item, case_id in zip(
                                        prepared, case_ids, strict=True
                                    )
                                ]
                                results = [future.result() for future in futures]
                            batch_elapsed_ms = (
                                time.perf_counter_ns() - batch_started
                            ) / 1e6
                            throughput = concurrency / (batch_elapsed_ms / 1000.0)
                            for case_id, result in zip(
                                case_ids, results, strict=True
                            ):
                                case_dir = run.run_dir / "cases" / case_id
                                (case_dir / "result.json").write_text(
                                    json.dumps(
                                        {
                                            "case_id": case_id,
                                            "server_ttft_ms": result.server_ttft_ms,
                                            "client_ttft_ms": (
                                                result.completion.client_ttft_ms
                                            ),
                                            "batch_elapsed_ms": batch_elapsed_ms,
                                            "throughput_requests_per_s": throughput,
                                            "text": result.completion.text,
                                        },
                                        indent=2,
                                        ensure_ascii=False,
                                    ) + "\n",
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
                                        "skill_name": local.SKILL_NAME,
                                        "skill_tokens": skill_tokens,
                                        "task_id": local.TASK_PROMPT_PATH.stem,
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
                                        "concurrency": concurrency,
                                        "replica": replica,
                                        "repetition": batch,
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
                                        "batch_elapsed_ms": batch_elapsed_ms,
                                        "throughput_requests_per_s": throughput,
                                        "output_tokens": (
                                            result.completion.output_tokens
                                        ),
                                        "fallback": result.fallback,
                                        "fallback_reason": result.fallback_reason,
                                        "started_utc": utc_now(),
                                        "completed_utc": utc_now(),
                                    }
                                )
                                run.mark(
                                    case_id, "completed", case_dir=str(case_dir)
                                )
    from analyze import analyze

    analyze(run.run_dir)
    run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
