"""Run one real CSKCache request with a synthetic pinned Skill KV object."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from config import (
    CALIBRATION_RATIO,
    CALIBRATION_TOKENS,
    CHUNK_SIZE_TOKENS,
    CORRECTION_ALPHA,
    EXECUTION_ORDER,
    GPU_MEMORY_UTILIZATION,
    HOST_LAYOUT,
    KV_FILL_VALUE,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    MINIMUM_FULL_RECOMPUTE_TOKENS,
    MINIMUM_REUSE_TOKENS,
    MODEL_PATH,
    OUTPUT_ROOT,
    PREFIX_TOKENS,
    PROMPT_FILL_TOKEN_ID,
    RUN_DIR_OVERRIDE,
    SKILL_NAME,
    SKILL_TOKENS,
    STORAGE_LAYOUT,
    TAIL_TOKENS,
    WARMUP_REQUESTS,
)


RUN_DIR = RUN_DIR_OVERRIDE or (
    OUTPUT_ROOT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
)
CATALOG_PATH = RUN_DIR / "catalog.json"
CONTAINER_PATH = RUN_DIR / "synthetic_kv.bin"
PROFILE_PATH = RUN_DIR / "cskcache_profile.jsonl"
WARMUP_PROFILE_PATH = RUN_DIR / "warmup_profile.jsonl"
WARMUP_RESULT_PATH = RUN_DIR / "warmup_request_result.json"


def _prepare_catalog() -> tuple[list[int], str]:
    from transformers import AutoConfig, AutoTokenizer

    from cskcache import (
        CacheObjectMetadata,
        ChunkingSpec,
        ContainerMetadata,
        LayerExtent,
        MetadataManager,
        KVLayout,
        ReadStrategy,
        fingerprint_model,
        fingerprint_token_ids,
        fingerprint_tokenizer,
        publish_generation_sidecar,
    )

    model_config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    marker = tokenizer.encode(
        f'<context_segment skill_name="{SKILL_NAME}">\n',
        add_special_tokens=False,
    )
    closing = tokenizer.encode("\n</context_segment>\n", add_special_tokens=False)
    object_token_ids = (
        marker
        + [PROMPT_FILL_TOKEN_ID] * (SKILL_TOKENS - len(marker) - len(closing))
        + closing
    )

    num_layers = int(model_config.num_hidden_layers)
    head_dim = int(
        getattr(model_config, "head_dim", 0)
        or model_config.hidden_size // model_config.num_attention_heads
    )
    kv_hidden_size = int(model_config.num_key_value_heads) * head_dim
    layer_shape = (2, SKILL_TOKENS, kv_hidden_size)
    layer_bytes = 2 * SKILL_TOKENS * kv_hidden_size * 2
    capacity_bytes = num_layers * layer_bytes
    CONTAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONTAINER_PATH.open("wb") as handle:
        handle.truncate(capacity_bytes)

    container = ContainerMetadata(
        container_id="synthetic-pinned-container",
        raw_file_path=str(CONTAINER_PATH.resolve()),
        container_format_version=1,
        storage_generation="synthetic-pinned-v1",
        capacity_bytes=capacity_bytes,
        alignment_bytes=4096,
        header_bytes=0,
    )
    publish_generation_sidecar(container)
    layers = tuple(
        LayerExtent(
            layer_id=layer_id,
            backend_key=f"synthetic-layer-{layer_id}",
            offset_bytes=layer_id * layer_bytes,
            length_bytes=layer_bytes,
            dtype="bfloat16",
            shape=layer_shape,
            memory_layout="KV_2TD",
            payload_sha256="0" * 64,
        )
        for layer_id in range(num_layers)
    )
    cache_object = CacheObjectMetadata(
        object_id="synthetic-pinned-object",
        skill_name=SKILL_NAME,
        skill_version="v1",
        model_fingerprint=fingerprint_model(MODEL_PATH),
        tokenizer_fingerprint=fingerprint_tokenizer(MODEL_PATH),
        token_count=SKILL_TOKENS,
        source_position_start=0,
        token_ids_sha256=fingerprint_token_ids(object_token_ids),
        start_marker_token_ids=tuple(marker),
        container_id=container.container_id,
        read_strategy=ReadStrategy.CONTIGUOUS,
        layers=layers,
        chunking=ChunkingSpec(CHUNK_SIZE_TOKENS),
        storage_layout=KVLayout(STORAGE_LAYOUT),
    )
    metadata = MetadataManager(CATALOG_PATH, expected_layers=num_layers)
    metadata.publish_container(container)
    metadata.publish_object(cache_object)
    return object_token_ids, str(container.raw_file_path)


def _configure_environment(container_path: str) -> None:
    extra_config = {
        "csk_t0_prefetch": True,
        "external_control_enabled": True,
        "exact_save_kv_2td": True,
        "cskcache_metadata_path": str(CATALOG_PATH),
        "cskcache_tokenizer_path": str(MODEL_PATH),
        "csk_storage_backend": "raw_block",
        "csk_chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "csk_storage_layout": STORAGE_LAYOUT,
        "csk_minimum_full_recompute_tokens": MINIMUM_FULL_RECOMPUTE_TOKENS,
        "csk_calibration_tokens": CALIBRATION_TOKENS,
        "csk_minimum_reuse_tokens": MINIMUM_REUSE_TOKENS,
        "csk_correction_alpha": CORRECTION_ALPHA,
        "csk_execution_order": EXECUTION_ORDER,
        "csk_host_layout": HOST_LAYOUT,
        "storage_plugin.raw_block.module_path": "memory_backend",
        "storage_plugin.raw_block.class_name": (
            "PinnedMemoryExtentBackend"
        ),
        "pinned_memory_extent.device_path": container_path,
        "pinned_memory_extent.capacity_bytes": CONTAINER_PATH.stat().st_size,
        "pinned_memory_extent.block_align": 4096,
        "pinned_memory_extent.header_bytes": 0,
        "pinned_memory_extent.fill_value": KV_FILL_VALUE,
    }
    os.environ.update(
        {
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_FORCE_SKIP_SAVE": "1",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
            "LMCACHE_STORAGE_PLUGINS": "raw_block",
            "LMCACHE_EXTRA_CONFIG": json.dumps(extra_config),
            "CSKCACHE_PROFILE": "1",
            "CSKCACHE_PROFILE_TRACE_PATH": str(PROFILE_PATH),
        }
    )


async def _execute_request(
    engine: object,
    object_token_ids: list[int],
    *,
    phase: str,
) -> dict[str, object]:
    from cskcache import render_skill_payload
    from cskcache.integrations.vllm.base import (
        INSPECT_TOOL_OBSERVATION,
        SUBMIT_PREFETCH,
    )
    from vllm import SamplingParams
    ticket = f"synthetic-pinned-{phase}-ticket"
    await engine.execute_connector_control(
        SUBMIT_PREFETCH,
        {"ticket": ticket, "skill_name": SKILL_NAME},
    )
    await engine.execute_connector_control(
        INSPECT_TOOL_OBSERVATION,
        {
            "ticket": ticket,
            "tool_name": "skill",
            "content": render_skill_payload(
                SKILL_NAME, "synthetic pipeline input"
            ),
        },
    )
    prompt_token_ids = (
        [PROMPT_FILL_TOKEN_ID] * PREFIX_TOKENS
        + object_token_ids
        + [PROMPT_FILL_TOKEN_ID] * TAIL_TOKENS
    )
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        extra_args={
            "kv_transfer_params": {"cskcache_candidate": {"ticket": ticket}}
        },
    )
    final_output = None
    request_start = time.perf_counter_ns()
    async for output in engine.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params,
        request_id=f"pinned-kv-pipeline-{phase}",
    ):
        final_output = output
    request_end = time.perf_counter_ns()
    generated_token_ids = []
    if final_output is not None and final_output.outputs:
        generated_token_ids = list(final_output.outputs[0].token_ids)
    return {
        "request_id": getattr(final_output, "request_id", None),
        "phase": phase,
        "prompt_tokens": len(prompt_token_ids),
        "skill_tokens": SKILL_TOKENS,
        "calibration_ratio": CALIBRATION_RATIO,
        "calibration_tokens": CALIBRATION_TOKENS,
        "execution_order": EXECUTION_ORDER,
        "host_layout": HOST_LAYOUT,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "storage_layout": STORAGE_LAYOUT,
        "warmup_requests": WARMUP_REQUESTS,
        "request_elapsed_ms": (request_end - request_start) / 1_000_000,
        "generated_token_ids": generated_token_ids,
    }


async def _run_request(object_token_ids: list[int]) -> None:
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(MODEL_PATH),
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="CSKCacheConnectorV1",
            kv_connector_module_path="cskcache.integrations.vllm.connector",
            kv_role="kv_both",
        ),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        warmup_results = []
        for index in range(WARMUP_REQUESTS):
            warmup_results.append(
                await _execute_request(
                    engine,
                    object_token_ids,
                    phase=f"warmup-{index}",
                )
            )
            if not await engine.reset_prefix_cache(
                reset_running_requests=False,
                reset_connector=False,
            ):
                raise RuntimeError("prefix cache reset failed after warm-up")
        if warmup_results:
            WARMUP_RESULT_PATH.write_text(
                json.dumps(warmup_results, indent=2) + "\n",
                encoding="utf-8",
            )
            PROFILE_PATH.replace(WARMUP_PROFILE_PATH)
            PROFILE_PATH.touch()

        measured = await _execute_request(
            engine,
            object_token_ids,
            phase="measured",
        )
        (RUN_DIR / "request_result.json").write_text(
            json.dumps(measured, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        engine.shutdown()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    PROFILE_PATH.touch()
    os.environ["CSKCACHE_PROFILE"] = "1"
    os.environ["CSKCACHE_PROFILE_TRACE_PATH"] = str(PROFILE_PATH)
    object_token_ids, container_path = _prepare_catalog()
    _configure_environment(container_path)
    asyncio.run(_run_request(object_token_ids))

    from analyze import analyze

    analyze(RUN_DIR)
    print(f"results: {RUN_DIR}")


if __name__ == "__main__":
    main()
