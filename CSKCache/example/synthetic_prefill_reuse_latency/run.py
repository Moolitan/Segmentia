#!/usr/bin/env python3
"""Measure synthetic normal-prefill and ready-Pinned-KV reuse latency."""

from __future__ import annotations

import asyncio
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import config as cfg


_EXPLICIT_RUN_DIR = os.getenv("CSK_SYNTHETIC_LATENCY_RUN_DIR")
RUN_ID = (
    Path(_EXPLICIT_RUN_DIR).name
    if _EXPLICIT_RUN_DIR
    else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
)
RUN_DIR = (
    Path(_EXPLICIT_RUN_DIR)
    if _EXPLICIT_RUN_DIR
    else cfg.OUTPUT_ROOT / RUN_ID
)
CATALOG_PATH = RUN_DIR / "catalog.json"
CONTAINER_PATH = RUN_DIR / "synthetic_kv.bin"
PROFILE_PATH = RUN_DIR / "cskcache_profile.jsonl"
SAMPLES_PATH = RUN_DIR / "samples.csv"
CONFIG_PATH = RUN_DIR / "run_config.json"
WORKLOADS_PATH = RUN_DIR / "workloads.json"

SAMPLE_FIELDS = (
    "request_id",
    "phase",
    "warmup",
    "repetition",
    "order_in_round",
    "method",
    "correction_strategy",
    "segment_tokens",
    "prompt_tokens",
    "ttft_ms",
    "generated_token_id",
    "host_ready_before_timer",
    "reuse_validated",
)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _prepare_catalog() -> dict[int, tuple[str, list[int]]]:
    from transformers import AutoConfig, AutoTokenizer

    from cskcache import (
        CacheObjectMetadata,
        ChunkingSpec,
        ContainerMetadata,
        KVLayout,
        LayerExtent,
        MetadataManager,
        ReadStrategy,
        fingerprint_model,
        fingerprint_token_ids,
        fingerprint_tokenizer,
        publish_generation_sidecar,
    )

    model_config = AutoConfig.from_pretrained(
        cfg.MODEL_PATH, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_PATH, local_files_only=True
    )
    num_layers = int(model_config.num_hidden_layers)
    head_dim = int(
        getattr(model_config, "head_dim", 0)
        or model_config.hidden_size // model_config.num_attention_heads
    )
    kv_hidden_size = int(model_config.num_key_value_heads) * head_dim
    model_fingerprint = fingerprint_model(cfg.MODEL_PATH)
    tokenizer_fingerprint = fingerprint_tokenizer(cfg.MODEL_PATH)

    cursor = 0
    objects: list[CacheObjectMetadata] = []
    workloads: dict[int, tuple[str, list[int]]] = {}
    for segment_tokens in cfg.TOKEN_LENGTHS:
        skill_name = f"synthetic-{segment_tokens}"
        marker = tokenizer.encode(
            f'<context_segment skill_name="{skill_name}">\n',
            add_special_tokens=False,
        )
        closing = tokenizer.encode(
            "\n</context_segment>\n", add_special_tokens=False
        )
        fill_tokens = segment_tokens - len(marker) - len(closing)
        if fill_tokens <= 0:
            raise ValueError(
                f"segment length {segment_tokens} cannot contain its markers"
            )
        token_ids = (
            marker
            + [cfg.PROMPT_FILL_TOKEN_ID] * fill_tokens
            + closing
        )
        layer_shape = (2, segment_tokens, kv_hidden_size)
        layer_bytes = 2 * segment_tokens * kv_hidden_size * 2
        layers = []
        for layer_id in range(num_layers):
            cursor = _align_up(cursor, 4096)
            layers.append(
                LayerExtent(
                    layer_id=layer_id,
                    backend_key=(
                        f"synthetic-{segment_tokens}-layer-{layer_id}"
                    ),
                    offset_bytes=cursor,
                    length_bytes=layer_bytes,
                    dtype="bfloat16",
                    shape=layer_shape,
                    memory_layout="KV_2TD",
                    payload_sha256="0" * 64,
                )
            )
            cursor += layer_bytes
        objects.append(
            CacheObjectMetadata(
                object_id=f"synthetic-pinned-{segment_tokens}",
                skill_name=skill_name,
                skill_version="v1",
                model_fingerprint=model_fingerprint,
                tokenizer_fingerprint=tokenizer_fingerprint,
                token_count=segment_tokens,
                source_position_start=0,
                token_ids_sha256=fingerprint_token_ids(token_ids),
                start_marker_token_ids=tuple(marker),
                container_id="synthetic-latency-container",
                read_strategy=ReadStrategy.CONTIGUOUS,
                layers=tuple(layers),
                chunking=ChunkingSpec(cfg.CHUNK_SIZE_TOKENS),
                storage_layout=KVLayout(cfg.STORAGE_LAYOUT),
            )
        )
        workloads[segment_tokens] = (skill_name, token_ids)

    capacity_bytes = _align_up(cursor, 4096)
    with CONTAINER_PATH.open("wb") as handle:
        handle.truncate(capacity_bytes)
    container = ContainerMetadata(
        container_id="synthetic-latency-container",
        raw_file_path=str(CONTAINER_PATH.resolve()),
        container_format_version=1,
        storage_generation=RUN_ID,
        capacity_bytes=capacity_bytes,
        alignment_bytes=4096,
        header_bytes=0,
    )
    publish_generation_sidecar(container)
    metadata = MetadataManager(CATALOG_PATH, expected_layers=num_layers)
    metadata.publish_container(container)
    for cache_object in objects:
        metadata.publish_object(cache_object)
    return workloads


def _configure_environment(method: str) -> None:
    from cskcache import (
        CorrectionStrategy,
        NormalPrefillMethod,
        execution_method_for,
    )

    contracts = {
        contract.name: contract
        for contract in (
            NormalPrefillMethod(),
            execution_method_for(CorrectionStrategy.DIRECT),
            execution_method_for(CorrectionStrategy.DEVIATION_TOPK),
        )
    }
    if method not in contracts or tuple(contracts) != cfg.METHODS:
        raise ValueError(f"unsupported method: {method}")
    contract = contracts[method]
    correction_strategy = (
        contract.correction_strategy.value
        if contract.correction_strategy is not None
        else CorrectionStrategy.DIRECT.value
    )
    extra_config = {
        "csk_t0_prefetch": True,
        "external_control_enabled": True,
        "exact_save_kv_2td": True,
        "cskcache_metadata_path": str(CATALOG_PATH),
        "cskcache_tokenizer_path": str(cfg.MODEL_PATH),
        "csk_storage_backend": "raw_block",
        "csk_chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
        "csk_storage_layout": cfg.STORAGE_LAYOUT,
        "csk_host_layout": cfg.HOST_LAYOUT,
        "csk_execution_order": cfg.EXECUTION_ORDER,
        "csk_correction_strategy": correction_strategy,
        "csk_minimum_full_recompute_tokens": (
            cfg.MINIMUM_FULL_RECOMPUTE_TOKENS
        ),
        "csk_calibration_tokens": cfg.CALIBRATION_TOKENS,
        "csk_deviation_recompute_ratio": cfg.DEVIATION_RECOMPUTE_RATIO,
        "csk_deviation_check_layer": cfg.DEVIATION_CHECK_LAYER,
        "csk_minimum_reuse_tokens": cfg.MINIMUM_REUSE_TOKENS,
        "csk_correction_alpha": cfg.CORRECTION_ALPHA,
        "storage_plugin.raw_block.module_path": "memory_backend",
        "storage_plugin.raw_block.class_name": "PinnedMemoryExtentBackend",
        "pinned_memory_extent.device_path": str(CONTAINER_PATH),
        "pinned_memory_extent.capacity_bytes": CONTAINER_PATH.stat().st_size,
        "pinned_memory_extent.block_align": 4096,
        "pinned_memory_extent.header_bytes": 0,
        "pinned_memory_extent.fill_value": cfg.KV_FILL_VALUE,
    }
    os.environ.update(
        {
            "LMCACHE_CHUNK_SIZE": str(cfg.CHUNK_SIZE_TOKENS),
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_FORCE_SKIP_SAVE": "1",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(
                cfg.LMCACHE_MAX_LOCAL_CPU_SIZE_GB
            ),
            "LMCACHE_STORAGE_PLUGINS": "raw_block",
            "LMCACHE_EXTRA_CONFIG": json.dumps(extra_config),
            "CSKCACHE_PROFILE": "1",
            "CSKCACHE_PROFILE_TRACE_PATH": str(PROFILE_PATH),
        }
    )


def _profile_records() -> list[dict[str, Any]]:
    records = []
    for line in PROFILE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _profile_ticket(record: dict[str, Any]) -> str:
    """Read ticket-keyed events emitted before a request ID exists."""

    return str(record.get("ticket") or record.get("request_id") or "")


def _profile_request_matches(record: dict[str, Any], request_id: str) -> bool:
    """Match vLLM's internal ``external_id-random_suffix`` request ID."""

    observed = str(record.get("request_id") or "")
    return observed == request_id or observed.startswith(f"{request_id}-")


async def _wait_for_profile(
    predicate: Callable[[dict[str, Any]], bool], description: str
) -> dict[str, Any]:
    deadline = time.monotonic() + cfg.PROFILE_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for record in _profile_records():
            if predicate(record):
                return record
        await asyncio.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {description}")


async def _reset_prefix_cache(engine: object) -> None:
    reset = await engine.reset_prefix_cache(
        reset_running_requests=False,
        reset_connector=False,
    )
    if not reset:
        raise RuntimeError("vLLM prefix-cache reset failed")


async def _prepare_reuse(
    engine: object,
    *,
    ticket: str,
    skill_name: str,
) -> None:
    from cskcache import render_skill_payload
    from cskcache.integrations.vllm.base import (
        INSPECT_TOOL_OBSERVATION,
        SUBMIT_PREFETCH,
    )

    await engine.execute_connector_control(
        SUBMIT_PREFETCH,
        {"ticket": ticket, "skill_name": skill_name},
    )
    await engine.execute_connector_control(
        INSPECT_TOOL_OBSERVATION,
        {
            "ticket": ticket,
            "tool_name": "skill",
            "content": render_skill_payload(skill_name, "synthetic"),
        },
    )
    await _wait_for_profile(
        lambda record: (
            record.get("event") == "csk_host_ready"
            and _profile_ticket(record) == ticket
        ),
        f"host-ready ticket {ticket}",
    )


async def _timed_generate(
    engine: object,
    *,
    request_id: str,
    prompt_token_ids: list[int],
    ticket: str | None,
) -> tuple[float, int | None]:
    from vllm import SamplingParams

    extra_args = (
        {"kv_transfer_params": {"cskcache_candidate": {"ticket": ticket}}}
        if ticket is not None
        else None
    )
    sampling = SamplingParams(
        max_tokens=cfg.MAX_TOKENS,
        temperature=0.0,
        extra_args=extra_args,
    )
    first_token_ns = None
    generated_token_id = None
    started_ns = time.perf_counter_ns()
    async for output in engine.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling,
        request_id=request_id,
    ):
        if output.outputs and output.outputs[0].token_ids:
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns()
            generated_token_id = int(output.outputs[0].token_ids[0])
    if first_token_ns is None:
        raise RuntimeError(f"request {request_id} produced no token")
    return (first_token_ns - started_ns) / 1_000_000, generated_token_id


async def _validate_reuse(
    request_id: str,
    ticket: str,
    *,
    method: str,
) -> None:
    await _wait_for_profile(
        lambda record: (
            record.get("event") == "csk_worker_load_complete"
            and _profile_request_matches(record, request_id)
            and record.get("ticket") == ticket
            and int(record.get("layers", 0)) == 40
        ),
        f"40-layer worker completion for {request_id}",
    )
    activation = await _wait_for_profile(
        lambda record: (
            record.get("event") == "csk_reuse_scheduler_activate"
            and _profile_request_matches(record, request_id)
            and record.get("ticket") == ticket
        ),
        f"scheduler activation for {request_id}",
    )
    if int(activation.get("external_tokens", 0)) <= 0:
        raise RuntimeError(f"request {request_id} activated zero reused tokens")
    correction = await _wait_for_profile(
        lambda record: (
            record.get("event") == "csk_correction_complete"
            and _profile_request_matches(record, request_id)
            and int(record.get("layers", 0)) == 40
        ),
        f"40-layer correction for {request_id}",
    )
    expected_strategy = (
        "deviation_topk" if method == "deviation_topk" else "direct"
    )
    if correction.get("correction_strategy") != expected_strategy:
        raise RuntimeError(
            f"request {request_id} ran unexpected correction strategy"
        )
    if correction.get("execution_method") != method:
        raise RuntimeError(
            f"request {request_id} ran unexpected execution method"
        )
    if method == "deviation_topk":
        layer_records = [
            record
            for record in _profile_records()
            if record.get("event") == "cskcache_deviation_topk_layer"
            and _profile_request_matches(record, request_id)
            and record.get("ticket") == ticket
        ]
        if len(layer_records) != 40:
            raise RuntimeError(
                f"request {request_id} produced {len(layer_records)} top-k layers"
            )
        layer_records.sort(key=lambda record: int(record["layer"]))
        candidate_tokens = int(layer_records[0]["candidate_tokens"])
        selected_tokens = max(
            1, int(candidate_tokens * cfg.DEVIATION_RECOMPUTE_RATIO)
        )
        for layer_id, record in enumerate(layer_records):
            expected_tokens = (
                candidate_tokens
                if layer_id < cfg.DEVIATION_CHECK_LAYER
                else selected_tokens
            )
            if (
                int(record["layer"]) != layer_id
                or int(record["candidate_tokens"]) != candidate_tokens
                or int(record["recomputed_tokens"]) != expected_tokens
                or bool(record["selection_applied"])
                != (layer_id == cfg.DEVIATION_CHECK_LAYER)
            ):
                raise RuntimeError(
                    f"request {request_id} failed top-k layer evidence"
                )
    release = await _wait_for_profile(
        lambda record: (
            record.get("event") == "csk_reuse_release"
            and record.get("ticket") == ticket
        ),
        f"ticket release for {request_id}",
    )
    if release.get("released") is not True:
        raise RuntimeError(f"ticket {ticket} was not released cleanly")


async def _run_arm(
    engine: object,
    *,
    phase: str,
    warmup: bool,
    repetition: int,
    order_in_round: int,
    method: str,
    segment_tokens: int,
    skill_name: str,
    object_token_ids: list[int],
) -> dict[str, object]:
    await _reset_prefix_cache(engine)
    request_id = (
        f"synthetic-latency-{phase}-{segment_tokens}-{method}-{repetition}"
    )
    ticket = (
        None
        if method == "normal_prefill"
        else f"{request_id}-ticket"
    )
    if ticket is not None:
        await _prepare_reuse(
            engine, ticket=ticket, skill_name=skill_name
        )
    prompt_token_ids = (
        [cfg.PROMPT_FILL_TOKEN_ID] * cfg.PREFIX_TOKENS
        + object_token_ids
        + [cfg.PROMPT_FILL_TOKEN_ID] * cfg.TAIL_TOKENS
    )
    ttft_ms, generated_token_id = await _timed_generate(
        engine,
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        ticket=ticket,
    )
    validated = False
    if ticket is not None:
        await _validate_reuse(request_id, ticket, method=method)
        validated = True
    return {
        "request_id": request_id,
        "phase": phase,
        "warmup": warmup,
        "repetition": repetition,
        "order_in_round": order_in_round,
        "method": method,
        "correction_strategy": (
            "none" if method == "normal_prefill" else (
                "deviation_topk" if method == "deviation_topk" else "direct"
            )
        ),
        "segment_tokens": segment_tokens,
        "prompt_tokens": len(prompt_token_ids),
        "ttft_ms": f"{ttft_ms:.6f}",
        "generated_token_id": generated_token_id,
        "host_ready_before_timer": ticket is not None,
        "reuse_validated": validated,
    }


def _write_sample(sample: dict[str, object]) -> None:
    exists = SAMPLES_PATH.is_file()
    with SAMPLES_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(sample)


async def _run_method(
    workloads: dict[int, tuple[str, list[int]]],
    *,
    method: str,
) -> None:
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(cfg.MODEL_PATH),
        dtype="auto",
        max_model_len=cfg.MAX_MODEL_LEN,
        gpu_memory_utilization=cfg.GPU_MEMORY_UTILIZATION,
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
        for segment_tokens in cfg.TOKEN_LENGTHS:
            skill_name, token_ids = workloads[segment_tokens]
            for warmup_index in range(cfg.WARMUPS_PER_MODE):
                sample = await _run_arm(
                    engine,
                    phase="warmup",
                    warmup=True,
                    repetition=warmup_index,
                    order_in_round=0,
                    method=method,
                    segment_tokens=segment_tokens,
                    skill_name=skill_name,
                    object_token_ids=token_ids,
                )
                _write_sample(sample)
                print(
                    f"warmup tokens={segment_tokens} method={method} "
                    f"ttft_ms={sample['ttft_ms']}",
                    flush=True,
                )

        for repetition in range(cfg.REPETITIONS):
            lengths = (
                cfg.TOKEN_LENGTHS
                if repetition % 2 == 0
                else tuple(reversed(cfg.TOKEN_LENGTHS))
            )
            for order_in_round, segment_tokens in enumerate(lengths):
                skill_name, token_ids = workloads[segment_tokens]
                sample = await _run_arm(
                    engine,
                    phase="measure",
                    warmup=False,
                    repetition=repetition,
                    order_in_round=order_in_round,
                    method=method,
                    segment_tokens=segment_tokens,
                    skill_name=skill_name,
                    object_token_ids=token_ids,
                )
                _write_sample(sample)
                print(
                    f"measure repeat={repetition} "
                    f"tokens={segment_tokens} method={method} "
                    f"ttft_ms={sample['ttft_ms']}",
                    flush=True,
                )
    finally:
        engine.shutdown()


def _write_run_inputs(
    workloads: dict[int, tuple[str, list[int]]]
) -> None:
    WORKLOADS_PATH.write_text(
        json.dumps(
            {
                str(tokens): {
                    "skill_name": skill_name,
                    "token_ids": token_ids,
                }
                for tokens, (skill_name, token_ids) in workloads.items()
            }
        ),
        encoding="utf-8",
    )
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "model_path": str(cfg.MODEL_PATH),
                "token_lengths": list(cfg.TOKEN_LENGTHS),
                "methods": list(cfg.METHODS),
                "warmups_per_mode": cfg.WARMUPS_PER_MODE,
                "repetitions": cfg.REPETITIONS,
                "prefix_tokens": cfg.PREFIX_TOKENS,
                "tail_tokens": cfg.TAIL_TOKENS,
                "calibration_tokens": cfg.CALIBRATION_TOKENS,
                "deviation_recompute_ratio": cfg.DEVIATION_RECOMPUTE_RATIO,
                "deviation_check_layer": cfg.DEVIATION_CHECK_LAYER,
                "chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
                "storage_layout": cfg.STORAGE_LAYOUT,
                "host_layout": cfg.HOST_LAYOUT,
                "execution_order": cfg.EXECUTION_ORDER,
                "engine_restart_boundary": "method",
                "timing_boundary": (
                    "AsyncLLM.generate_start_to_first_output_token"
                ),
                "reuse_host_boundary": (
                    "csk_host_ready is required before timer start"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_workloads() -> dict[int, tuple[str, list[int]]]:
    payload = json.loads(WORKLOADS_PATH.read_text(encoding="utf-8"))
    return {
        int(tokens): (
            str(item["skill_name"]),
            [int(token_id) for token_id in item["token_ids"]],
        )
        for tokens, item in payload.items()
    }


def _run_worker(method: str) -> None:
    if not _EXPLICIT_RUN_DIR:
        raise RuntimeError("worker requires CSK_SYNTHETIC_LATENCY_RUN_DIR")
    _configure_environment(method)
    asyncio.run(_run_method(_load_workloads(), method=method))


def _run_coordinator() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"run directory already exists: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    PROFILE_PATH.touch()
    workloads = _prepare_catalog()
    _write_run_inputs(workloads)
    child_environment = os.environ.copy()
    child_environment["CSK_SYNTHETIC_LATENCY_RUN_DIR"] = str(RUN_DIR)
    for method in cfg.METHODS:
        print(f"starting_method={method}", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", method],
            check=True,
            env=child_environment,
        )

    from analyze import analyze

    analyze(RUN_DIR, publish_root=cfg.PUBLISH_ROOT)
    print(f"results={RUN_DIR}")
    print(f"published={cfg.PUBLISH_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=cfg.METHODS)
    arguments = parser.parse_args()
    if arguments.worker is None:
        _run_coordinator()
    else:
        _run_worker(arguments.worker)


if __name__ == "__main__":
    main()
