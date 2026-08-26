"""Ingest four SSD layouts and measure blocking versus prefetched SSD."""

from __future__ import annotations

import json
from pathlib import Path

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import (
    BASE_PORT, OUTPUT_ROOT, PLATFORMS, RAW_POOL_ROOT,
)
from common.csk_config import catalog_path, require_catalog_skills
from common.driver import make_server_config, run_request_pair
from common.run_state import RunContext, utc_now
from common.server import VLLMServer


SECTION = "section_6_5_layout_storage"


def _ingest_ssd(run: RunContext) -> None:
    source_path = local.SSD_LAYOUT_RUN / "samples.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"SSD layout samples do not exist: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = payload.get("samples") or []
    for source in rows:
        if source["layout"] not in local.SSD_LAYOUTS:
            continue
        case_id = f"ssd__{source['case_id']}__r{source['repetition']}"
        if run.completed(case_id):
            continue
        run.mark(case_id, "running", source=str(source_path))
        run.record(
            {
                "case_id": case_id,
                "status": "completed",
                "platform_id": "nvme_990_pro",
                "model_id": "Qwen3-14B-shaped-KV",
                "system": "SSD-to-Pinned",
                "skill_name": "synthetic-12518",
                "skill_tokens": local.SKILL_TOKENS,
                "storage_layout": source["layout"],
                "host_layout": source["layout"],
                "io_engine": source["io_engine"],
                "use_odirect": bool(source["use_odirect"]),
                "repetition": int(source["repetition"]),
                "warmup": False,
                "latency_ms": float(source["duration_ms"]),
                "started_utc": utc_now(),
                "completed_utc": utc_now(),
            }
        )
        run.mark(case_id, "completed", source=str(source_path))


def _measure_hierarchy(run: RunContext) -> None:
    platform = PLATFORMS[local.PLATFORM_ID]
    if not platform.model_path.is_dir():
        raise FileNotFoundError(f"model does not exist: {platform.model_path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        platform.model_path, local_files_only=True
    )
    require_catalog_skills(
        catalog_path(RAW_POOL_ROOT, platform.model_id, "raw_block"),
        {local.SKILL_NAME},
    )
    skill_text = local.SKILL_PATH.read_text(encoding="utf-8")
    skill_tokens = len(tokenizer.encode(skill_text, add_special_tokens=False))
    task_prompt = local.TASK_PROMPT_PATH.read_text(encoding="utf-8").strip()
    for mode_index, (mode_name, wait_for_prefetch) in enumerate(
        local.HIERARCHY_MODES
    ):
        for replica in range(local.REPLICAS):
            ordinals = [(True, index) for index in range(local.WARMUPS)] + [
                (False, index) for index in range(local.REPETITIONS)
            ]
            pending = []
            for warmup, repetition in ordinals:
                case_id = (
                    f"{local.PLATFORM_ID}__{mode_name}__replica{replica}__"
                    f"{'warmup' if warmup else 'measure'}{repetition}"
                )
                if not run.completed(case_id):
                    pending.append((case_id, warmup, repetition))
            if not pending:
                continue
            server_root = (
                run.run_dir / "servers" / local.PLATFORM_ID / mode_name
                / f"replica_{replica}"
            )
            server_cfg = make_server_config(
                platform=platform,
                variant=local.SYSTEM,
                port=BASE_PORT + mode_index * 4 + replica,
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
                        # Reset before request A. Resetting between A and B would
                        # accidentally give the "Blocking SSD" arm prefetch lead.
                        server.reset_prefix_cache()
                        result = run_request_pair(
                            server,
                            variant=local.SYSTEM,
                            skill_name=local.SKILL_NAME,
                            skill_text=skill_text,
                            task_prompt=task_prompt,
                            case_id=case_id,
                            max_tokens=local.MAX_TOKENS,
                            stream=True,
                            wait_for_prefetch=wait_for_prefetch,
                            reset_prefix_cache=False,
                        )
                        run.record(
                            {
                                "case_id": case_id,
                                "status": "completed",
                                "platform_id": local.PLATFORM_ID,
                                "gpu_name": platform.gpu_name,
                                "model_id": platform.model_id,
                                "model_path": str(platform.model_path),
                                "tensor_parallel_size": platform.tensor_parallel_size,
                                "system": mode_name,
                                "skill_name": local.SKILL_NAME,
                                "skill_tokens": skill_tokens,
                                "task_id": local.TASK_PROMPT_PATH.stem,
                                "chunk_tokens": local.CHUNK_TOKENS,
                                "storage_layout": "packed_chunks_single_layer",
                                "host_layout": "packed_chunks_single_layer",
                                "io_engine": "io_uring",
                                "use_odirect": True,
                                "correction_strategy": local.SYSTEM.correction_strategy,
                                "correction_budget_tokens": local.SYSTEM.calibration_tokens,
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
                                "latency_ms": result.completion.client_latency_ms,
                                "output_tokens": result.completion.output_tokens,
                                "fallback": result.fallback,
                                "fallback_reason": result.fallback_reason,
                                "started_utc": started,
                                "completed_utc": utc_now(),
                            }
                        )
                        run.mark(case_id, "completed", case_dir=str(case_dir))
                    except Exception as exc:
                        run.mark(
                            case_id,
                            "failed",
                            case_dir=str(case_dir),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise


def main() -> None:
    values = {
        "ssd_layout_run": str(local.SSD_LAYOUT_RUN),
        "ssd_layouts": list(local.SSD_LAYOUTS),
        "platform_id": local.PLATFORM_ID,
        "skill_path": str(local.SKILL_PATH),
        "task_prompt_path": str(local.TASK_PROMPT_PATH),
        "hierarchy_modes": list(local.HIERARCHY_MODES),
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
            local.SSD_LAYOUT_RUN / "samples.json",
            local.SKILL_PATH,
            local.TASK_PROMPT_PATH,
        ),
        config_values=values,
    )
    _ingest_ssd(run)
    _measure_hierarchy(run)
    from analyze import analyze

    analyze(run.run_dir)
    run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
