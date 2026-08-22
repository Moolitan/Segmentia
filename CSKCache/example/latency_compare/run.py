#!/usr/bin/env python3
"""Compare one-token request latency with and without Skill KV reuse."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import statistics
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import config as cfg


def backend_root() -> Path:
    leaf = "raw" if cfg.STORAGE_BACKEND == "raw_block" else "layer_files"
    return cfg.POOL_ROOT / cfg.POOL_MODEL_DIR / leaf


def validate_config() -> None:
    if cfg.STORAGE_BACKEND not in {"raw_block", "local_disk"}:
        raise ValueError("STORAGE_BACKEND must be raw_block or local_disk")
    if cfg.CHUNK_SIZE_TOKENS <= 0:
        raise ValueError("CHUNK_SIZE_TOKENS must be positive")
    supported_layouts = {"chunk_single_layer", "packed_chunks_single_layer"}
    if (
        cfg.STORAGE_LAYOUT not in supported_layouts
        or cfg.HOST_LAYOUT not in supported_layouts
    ):
        raise ValueError("current online path requires single-layer layouts")
    if cfg.EXECUTION_ORDER not in {"h2d_first", "compute_first"}:
        raise ValueError("EXECUTION_ORDER must be h2d_first or compute_first")
    if cfg.WARMUP_PAIRS < 0 or cfg.MEASURE_PAIRS <= 0:
        raise ValueError("WARMUP_PAIRS must be nonnegative and MEASURE_PAIRS positive")
    skill_file = cfg.SKILLS_DIR / cfg.SKILL_NAME / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"Skill file does not exist: {skill_file}")
    catalog = backend_root() / "catalog.json"
    if not catalog.is_file():
        raise FileNotFoundError(f"offline Catalog does not exist: {catalog}")


def lmcache_extra_config() -> dict[str, object]:
    catalog_path = backend_root() / "catalog.json"
    result: dict[str, object] = {
        "csk_t0_prefetch": True,
        "external_control_enabled": True,
        "exact_save_kv_2td": True,
        "cskcache_metadata_path": str(catalog_path),
        "csk_storage_backend": cfg.STORAGE_BACKEND,
        "csk_chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
        "csk_storage_layout": cfg.STORAGE_LAYOUT,
        "csk_host_layout": cfg.HOST_LAYOUT,
        "csk_execution_order": cfg.EXECUTION_ORDER,
        "csk_prefetch_handle_ttl_seconds": cfg.PREFETCH_HANDLE_TTL_SECONDS,
        "csk_minimum_full_recompute_tokens": cfg.MINIMUM_FULL_RECOMPUTE_TOKENS,
        "csk_calibration_tokens": cfg.CALIBRATION_TOKENS,
        "csk_minimum_reuse_tokens": cfg.MINIMUM_REUSE_TOKENS,
        "csk_correction_alpha": cfg.CORRECTION_ALPHA,
    }
    if cfg.STORAGE_BACKEND == "raw_block":
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        containers = catalog.get("containers") or []
        if len(containers) != 1:
            raise ValueError("raw_block Catalog must describe one container")
        container = containers[0]
        result.update(
            {
                "storage_plugin.raw_block.module_path": (
                    "lmcache.v1.storage_backend.plugins.rust_raw_block_backend"
                ),
                "storage_plugin.raw_block.class_name": "RustRawBlockBackend",
                "rust_raw_block.device_path": container["raw_file_path"],
                "rust_raw_block.capacity_bytes": container["capacity_bytes"],
                "rust_raw_block.block_align": container["alignment_bytes"],
                "rust_raw_block.header_bytes": container["header_bytes"],
                "rust_raw_block.slot_bytes": cfg.RAW_SLOT_BYTES,
                "rust_raw_block.use_odirect": True,
                "rust_raw_block.enable_zero_copy": True,
                "rust_raw_block.meta_total_bytes": cfg.RAW_METADATA_BYTES,
                "rust_raw_block.meta_magic": cfg.RAW_METADATA_MAGIC,
                "rust_raw_block.meta_version": container["container_format_version"],
                "rust_raw_block.meta_enable_periodic": False,
                "rust_raw_block.load_checkpoint_on_init": True,
                "rust_raw_block.io_engine": cfg.RAW_IO_ENGINE,
                "rust_raw_block.iouring_queue_depth": cfg.RAW_QUEUE_DEPTH,
            }
        )
    return result


def export_shell_config() -> None:
    validate_config()
    connector = {
        "kv_connector": "CSKCacheConnectorV1",
        "kv_connector_module_path": "cskcache.integrations.vllm.connector",
        "kv_role": "kv_both",
    }
    values = {
        "LATENCY_MODEL_PATH": str(cfg.MODEL_PATH),
        "LATENCY_SERVED_MODEL": cfg.SERVED_MODEL,
        "LATENCY_PORT": str(cfg.PORT),
        "LATENCY_API_KEY": cfg.API_KEY,
        "LATENCY_GPU_UTIL": str(cfg.GPU_MEMORY_UTILIZATION),
        "LATENCY_MAX_MODEL_LEN": str(cfg.MAX_MODEL_LEN),
        "LATENCY_BACKEND_ROOT": str(backend_root()),
        "LATENCY_STORAGE_BACKEND": cfg.STORAGE_BACKEND,
        "LATENCY_LMCACHE_EXTRA_CONFIG": json.dumps(
            lmcache_extra_config(), separators=(",", ":")
        ),
        "LATENCY_KV_TRANSFER_CONFIG": json.dumps(
            connector, separators=(",", ":")
        ),
        "LATENCY_LOCAL_CPU_GB": str(cfg.LMCACHE_MAX_LOCAL_CPU_SIZE_GB),
        "LATENCY_LOCAL_DISK_GB": str(cfg.LMCACHE_MAX_LOCAL_DISK_SIZE_GB),
    }
    for name, value in values.items():
        print(f"export {name}={shlex.quote(value)}")


def post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"http://127.0.0.1:{cfg.PORT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg.API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=cfg.REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {error.code}: {body}") from error


def reset_prefix_cache() -> None:
    endpoint = (
        f"http://127.0.0.1:{cfg.PORT}/reset_prefix_cache?reset_external=false"
    )
    for _ in range(20):
        request = Request(
            endpoint,
            data=b"",
            headers={"Authorization": f"Bearer {cfg.API_KEY}"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        if result.get("success") is True:
            return
        time.sleep(0.05)
    raise RuntimeError("vLLM prefix cache reset did not succeed")


def skill_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "skill",
                "description": "Load one named Skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": [cfg.SKILL_NAME],
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def trigger_prefetch() -> dict[str, Any]:
    response = post_chat(
        {
            "model": cfg.SERVED_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Load the {cfg.SKILL_NAME} Skill.",
                }
            ],
            "tools": skill_tool(),
            "tool_choice": {"type": "function", "function": {"name": "skill"}},
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    tool_calls = response["choices"][0]["message"].get("tool_calls") or []
    if len(tool_calls) != 1 or tool_calls[0]["function"]["name"] != "skill":
        raise RuntimeError("prefetch trigger did not return exactly one skill call")
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    if arguments.get("name") != cfg.SKILL_NAME:
        raise RuntimeError("prefetch trigger selected the wrong Skill")
    return {
        "id": str(tool_calls[0]["id"]),
        "type": "function",
        "function": {
            "name": "skill",
            "arguments": json.dumps(
                {"name": cfg.SKILL_NAME}, separators=(",", ":")
            ),
        },
    }


def profile_events() -> list[dict[str, Any]]:
    path = Path(os.environ["CSKCACHE_PROFILE_TRACE_PATH"])
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def wait_for_event(event_name: str, ticket: str) -> None:
    deadline = time.monotonic() + cfg.PREFETCH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for event in profile_events():
            if event.get("event") == event_name and ticket in {
                event.get("ticket"),
                event.get("request_id"),
            }:
                return
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {event_name} for {ticket}")


def request_payload(tool_call: dict[str, Any], skill_content: str) -> dict[str, Any]:
    return {
        "model": cfg.SERVED_MODEL,
        "messages": [
            {"role": "user", "content": cfg.TASK_PROMPT},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {
                "role": "tool",
                "name": "skill",
                "tool_call_id": tool_call["id"],
                "content": skill_content,
            },
        ],
        "tools": skill_tool(),
        "tool_choice": "none",
        "temperature": 0,
        "max_tokens": cfg.MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def timed_request(payload: dict[str, Any]) -> float:
    started = time.perf_counter_ns()
    post_chat(payload)
    return (time.perf_counter_ns() - started) / 1_000_000


def run_pair(skill_content: str) -> tuple[float, float]:
    reset_prefix_cache()
    tool_call = trigger_prefetch()
    ticket = tool_call["id"]
    wait_for_event("csk_host_ready", ticket)
    payload = request_payload(tool_call, skill_content)

    reused_ms = timed_request(payload)
    wait_for_event("csk_worker_load_complete", ticket)
    wait_for_event("csk_reuse_release", ticket)

    reset_prefix_cache()
    no_reuse_ms = timed_request(payload)
    return no_reuse_ms, reused_ms


def main() -> None:
    validate_config()
    from cskcache import render_skill_payload

    skill_text = (cfg.SKILLS_DIR / cfg.SKILL_NAME / "SKILL.md").read_text(
        encoding="utf-8"
    )
    skill_content = render_skill_payload(cfg.SKILL_NAME, skill_text)
    for _ in range(cfg.WARMUP_PAIRS):
        run_pair(skill_content)

    no_reuse_values: list[float] = []
    reused_values: list[float] = []
    for repeat in range(cfg.MEASURE_PAIRS):
        no_reuse_ms, reused_ms = run_pair(skill_content)
        no_reuse_values.append(no_reuse_ms)
        reused_values.append(reused_ms)
        print(f"no_reuse repeat={repeat} latency_ms={no_reuse_ms:.3f}")
        print(f"cskcache repeat={repeat} latency_ms={reused_ms:.3f}")

    print(f"median no_reuse latency_ms={statistics.median(no_reuse_values):.3f}")
    print(f"median cskcache latency_ms={statistics.median(reused_values):.3f}")


if __name__ == "__main__":
    if os.environ.get("CSKCACHE_LATENCY_EXPORT_CONFIG") == "1":
        export_shell_config()
    else:
        main()
