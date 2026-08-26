"""Measure logical invalidation granularity independently of packed layout."""

from __future__ import annotations

import json
from pathlib import Path

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS, RAW_POOL_ROOT
from common.csk_config import catalog_path, require_catalog_skills
from common.driver import SystemVariant, make_server_config, run_request_pair
from common.logical_catalog import derive_catalog
from common.run_state import RunContext, utc_now
from common.server import VLLMServer
from common.workloads import longest_full_chunk_prefix, mutate_tokens


SECTION = "section_6_4_chunk_granularity"


def _mutation_label(mutation: str, position: float) -> str:
    return mutation if mutation in {"exact", "append"} else f"replace-{int(position * 100)}"


def main() -> None:
    platform = PLATFORMS[local.PLATFORM_ID]
    source_catalog = catalog_path(RAW_POOL_ROOT, platform.model_id, "raw_block")
    values = {
        "platform_id": local.PLATFORM_ID,
        "workloads": [name for name, _skill, _prompt in local.WORKLOADS],
        "chunk_tokens": list(local.CHUNK_TOKENS),
        "mutations": list(local.MUTATIONS),
        "repetitions": local.REPETITIONS,
        "include_native_cacheblend": local.INCLUDE_NATIVE_CACHEBLEND,
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(
            Path(__file__), Path(local.__file__), Path(suite.__file__),
            *(path for _name, skill, prompt in local.WORKLOADS for path in (skill, prompt)),
            *((source_catalog,) if source_catalog.is_file() else ()),
        ),
        config_values=values,
    )
    from transformers import AutoTokenizer
    from cskcache import build_skill_token_identity

    tokenizer = AutoTokenizer.from_pretrained(
        platform.model_path, local_files_only=True
    )
    skill_paths = {name: path for name, path, _prompt in local.WORKLOADS}
    require_catalog_skills(source_catalog, set(skill_paths))
    catalogs = {
        chunk: derive_catalog(
            source_catalog=source_catalog,
            output_catalog=run.run_dir / "catalogs" / f"chunk_{chunk}.json",
            tokenizer_path=platform.model_path,
            skill_paths=skill_paths,
            chunk_tokens=chunk,
        )
        for chunk in local.CHUNK_TOKENS
    }

    for skill_index, (skill_name, skill_path, prompt_path) in enumerate(local.WORKLOADS):
        original_text = skill_path.read_text(encoding="utf-8")
        task_prompt = prompt_path.read_text(encoding="utf-8").strip()
        body_tokens = tokenizer.encode(original_text, add_special_tokens=False)
        source_identity = build_skill_token_identity(
            tokenizer, skill_name, original_text
        )
        variants = []
        for mutation, position in local.MUTATIONS:
            mutated_body = mutate_tokens(body_tokens, mutation, position)
            mutated_text = (
                original_text
                if mutation == "exact"
                else tokenizer.decode(mutated_body, skip_special_tokens=False)
            )
            identity = build_skill_token_identity(tokenizer, skill_name, mutated_text)
            variants.append((mutation, position, mutated_text, identity.token_ids))

        # Pure logical baseline rows are deterministic and require no GPU.
        for mutation, position, _text, current_ids in variants:
            label = _mutation_label(mutation, position)
            whole_reused = (
                len(source_identity.token_ids)
                if source_identity.token_ids == current_ids else 0
            )
            logical_arms = [("Whole-Skill", len(source_identity.token_ids), whole_reused)]
            logical_arms.extend(
                (
                    f"Chunk-{chunk}",
                    chunk,
                    longest_full_chunk_prefix(
                        source_identity.token_ids, current_ids, chunk
                    ),
                )
                for chunk in local.CHUNK_TOKENS
            )
            for system, chunk, reused in logical_arms:
                case_id = f"logical__{skill_name}__{label}__{system}"
                if run.completed(case_id):
                    continue
                run.mark(case_id, "running")
                run.record(
                    {
                        "case_id": case_id,
                        "status": "completed",
                        "platform_id": local.PLATFORM_ID,
                        "gpu_name": platform.gpu_name,
                        "model_id": platform.model_id,
                        "model_path": str(platform.model_path),
                        "tensor_parallel_size": platform.tensor_parallel_size,
                        "system": system,
                        "skill_name": skill_name,
                        "skill_tokens": len(source_identity.token_ids),
                        "task_id": prompt_path.stem,
                        "chunk_tokens": chunk,
                        "mutation": mutation,
                        "mutation_position": position,
                        "repetition": -1,
                        "warmup": False,
                        "reused_tokens": reused,
                        "reuse_ratio": reused / len(source_identity.token_ids),
                        "started_utc": utc_now(),
                        "completed_utc": utc_now(),
                    }
                )
                run.mark(case_id, "completed")

        arms = [
            (
                SystemVariant(
                    f"Chunk-{chunk}", "cskcache",
                    correction_strategy="fixed_prefix",
                    calibration_tokens=local.CALIBRATION_TOKENS,
                ),
                chunk,
                catalogs[chunk],
            )
            for chunk in local.CHUNK_TOKENS
        ]
        if local.INCLUDE_NATIVE_CACHEBLEND:
            arms.append(
                (
                    SystemVariant(
                        "CacheBlend-15%", "cacheblend", cacheblend_ratio=0.15
                    ),
                    256,
                    None,
                )
            )
        for arm_index, (variant, chunk, catalog_override) in enumerate(arms):
            server_root = (
                run.run_dir / "servers" / variant.name / skill_name
            )
            server_cfg = make_server_config(
                platform=platform,
                variant=variant,
                port=BASE_PORT + skill_index * 20 + arm_index,
                case_root=server_root,
                chunk_tokens=chunk,
                catalog_override=catalog_override,
                correction_alpha=local.CORRECTION_ALPHA,
                minimum_full_recompute_tokens=(
                    local.MINIMUM_FULL_RECOMPUTE_TOKENS
                ),
                minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
            )
            with VLLMServer(server_cfg) as server:
                for mutation, position, mutated_text, current_ids in variants:
                    label = _mutation_label(mutation, position)
                    expected_reused = longest_full_chunk_prefix(
                        source_identity.token_ids, current_ids, chunk
                    )
                    for repetition in range(local.REPETITIONS):
                        case_id = (
                            f"ttft__{skill_name}__{label}__{variant.name}__"
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
                                skill_name=skill_name,
                                skill_text=mutated_text,
                                source_skill_text=original_text,
                                task_prompt=task_prompt,
                                case_id=case_id,
                                max_tokens=local.MAX_TOKENS,
                                stream=True,
                            )
                            (case_dir / "result.json").write_text(
                                json.dumps(
                                    {
                                        "case_id": case_id,
                                        "expected_reused_tokens": expected_reused,
                                        "server_cached_tokens": result.cached_tokens,
                                        "server_ttft_ms": result.server_ttft_ms,
                                    },
                                    indent=2,
                                ) + "\n",
                                encoding="utf-8",
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
                                    "system": variant.name,
                                    "skill_name": skill_name,
                                    "skill_tokens": len(source_identity.token_ids),
                                    "task_id": prompt_path.stem,
                                    "chunk_tokens": chunk,
                                    "mutation": mutation,
                                    "mutation_position": position,
                                    "correction_strategy": variant.correction_strategy,
                                    "correction_budget_tokens": (
                                        variant.calibration_tokens or ""
                                    ),
                                    "correction_ratio": variant.cacheblend_ratio,
                                    "repetition": repetition,
                                    "warmup": False,
                                    "prompt_tokens": result.prompt_tokens,
                                    "reused_tokens": (
                                        result.cached_tokens
                                        if variant.family == "cacheblend"
                                        else expected_reused
                                    ),
                                    "reuse_ratio": (
                                        (result.cached_tokens if variant.family == "cacheblend" else expected_reused)
                                        / len(source_identity.token_ids)
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
                                case_id, "failed", case_dir=str(case_dir),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            raise
    from analyze import analyze

    analyze(run.run_dir)
    run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
