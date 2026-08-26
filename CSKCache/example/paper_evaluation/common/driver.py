"""Controlled request-A/request-B driver shared by paper subsections."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper_evaluation.config import (
    API_KEY,
    PREFETCH_TIMEOUT_SECONDS,
    RAW_POOL_ROOT,
    REQUEST_TIMEOUT_SECONDS,
    Platform,
)
from .csk_config import build_extra_config
from .http_client import CompletionResult, nonstream_chat, stream_chat
from .server import (
    CACHEBLEND_CONNECTOR,
    CSK_CONNECTOR,
    ServerConfig,
    cacheblend_environment,
    cskcache_environment,
)
from .workloads import BLEND_SEPARATOR, skill_tool, target_messages


TIMELINE_REQUEST_MARKER = "cskcache-latency-"


def _benchmark_request_id(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", case_id).strip("-")
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"{TIMELINE_REQUEST_MARKER}{slug[:160]}-{digest}"


@dataclass(frozen=True)
class SystemVariant:
    name: str
    family: str
    correction_strategy: str = ""
    calibration_tokens: int = 0
    calibration_ratio: float | None = None
    cacheblend_ratio: float | None = None


@dataclass(frozen=True)
class RequestResult:
    completion: CompletionResult
    server_ttft_ms: float
    prompt_tokens: int
    cached_tokens: int
    tool_call_id: str
    fallback: bool
    fallback_reason: str


@dataclass(frozen=True)
class PreparedRequest:
    """State produced by excluded request A and consumed by measured request B."""

    messages: tuple[Mapping[str, Any], ...]
    tool_call_id: str
    profile_path: Path


def make_server_config(
    *,
    platform: Platform,
    variant: SystemVariant,
    port: int,
    case_root: Path,
    chunk_tokens: int,
    storage_layout: str = "packed_chunks_single_layer",
    host_layout: str = "packed_chunks_single_layer",
    execution_order: str = "h2d_first",
    correction_alpha: float = 0.6,
    minimum_full_recompute_tokens: int = 32,
    minimum_reuse_tokens: int = 256,
    backend: str = "raw_block",
    io_engine: str = "io_uring",
    use_odirect: bool = True,
    catalog_override: Path | None = None,
) -> ServerConfig:
    trace = case_root / "vllm_timeline.jsonl"
    log = case_root / "vllm.log"
    if variant.family == "full":
        return ServerConfig(
            platform=platform,
            system=variant.name,
            port=port,
            log_path=log,
            trace_path=trace,
            extra_env={"LMCACHE_LOCAL_CPU": "False"},
            connector=None,
            enable_prefix_caching=True,
        )
    if variant.family == "cacheblend":
        if variant.cacheblend_ratio is None:
            raise ValueError("CacheBlend variant requires cacheblend_ratio")
        storage_root = case_root / "cacheblend_store"
        storage_root.mkdir(parents=True, exist_ok=True)
        return ServerConfig(
            platform=platform,
            system=variant.name,
            port=port,
            log_path=log,
            trace_path=trace,
            extra_env=cacheblend_environment(
                ratio=variant.cacheblend_ratio,
                chunk_tokens=chunk_tokens,
                storage_root=storage_root,
            ),
            connector=CACHEBLEND_CONNECTOR,
            enable_prefix_caching=False,
        )
    if variant.family != "cskcache":
        raise ValueError(f"unsupported system family: {variant.family}")
    extra = build_extra_config(
        pool_root=RAW_POOL_ROOT,
        model_id=platform.model_id,
        backend=backend,
        chunk_tokens=chunk_tokens,
        storage_layout=storage_layout,
        host_layout=host_layout,
        execution_order=execution_order,
        correction_strategy=variant.correction_strategy,
        calibration_tokens=variant.calibration_tokens,
        calibration_ratio=variant.calibration_ratio,
        correction_alpha=correction_alpha,
        minimum_full_recompute_tokens=minimum_full_recompute_tokens,
        minimum_reuse_tokens=minimum_reuse_tokens,
        io_engine=io_engine,
        use_odirect=use_odirect,
        catalog_override=catalog_override,
    )
    environment = cskcache_environment(extra)
    environment["CSKCACHE_PROFILE_TRACE_PATH"] = str(
        case_root / "cskcache_profile.jsonl"
    )
    if backend == "raw_block":
        environment["LMCACHE_STORAGE_PLUGINS"] = "raw_block"
        environment.pop("LMCACHE_LOCAL_DISK", None)
    else:
        environment["LMCACHE_STORAGE_PLUGINS"] = ""
        environment["LMCACHE_LOCAL_DISK"] = str(
            RAW_POOL_ROOT / platform.model_id / "layer_files"
        )
    return ServerConfig(
        platform=platform,
        system=variant.name,
        port=port,
        log_path=log,
        trace_path=trace,
        extra_env=environment,
        connector=CSK_CONNECTOR,
        enable_prefix_caching=True,
    )


def _forced_skill_call(
    base_url: str,
    *,
    model: str,
    skill_name: str,
    case_id: str,
) -> str:
    result = nonstream_chat(
        base_url,
        api_key=API_KEY,
        payload={
            "model": model,
            "request_id": f"{case_id}-select",
            "messages": [
                {"role": "user", "content": f"Load the {skill_name} Skill."}
            ],
            "tools": skill_tool(skill_name),
            "tool_choice": {
                "type": "function",
                "function": {"name": "skill"},
            },
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response = result.response or {}
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else {}
    calls = (message or {}).get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "skill":
        raise RuntimeError("forced request did not produce one Skill tool call")
    return str(calls[0]["id"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _wait_csk_ready(profile_path: Path, ticket: str) -> None:
    deadline = time.monotonic() + PREFETCH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for record in _read_jsonl(profile_path):
            if record.get("event") == "csk_host_ready" and ticket in {
                record.get("ticket"),
                record.get("request_id"),
            }:
                return
        time.sleep(0.02)
    raise TimeoutError(f"CSKCache host prefetch did not become ready: {ticket}")


def _timeline_ttft(path: Path, case_id: str) -> tuple[float, int, int]:
    records = [
        record
        for record in _read_jsonl(path)
        if str(record.get("request_id", "")) == case_id
    ]
    starts = [r for r in records if r.get("event") == "api_request_received"]
    ends = [r for r in records if r.get("event") == "first_token_ready"]
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError(
            f"expected one TTFT boundary pair for {case_id}; "
            f"found starts={len(starts)} ends={len(ends)}"
        )
    ttft_ms = (int(ends[0]["monotonic_ns"]) - int(starts[0]["monotonic_ns"])) / 1e6
    if ttft_ms < 0:
        raise RuntimeError("negative server TTFT")
    return (
        ttft_ms,
        int(ends[0].get("prompt_tokens") or 0),
        int(ends[0].get("cached_tokens") or 0),
    )


def _fallback(profile_path: Path, case_id: str) -> tuple[bool, str]:
    reasons = []
    for record in _read_jsonl(profile_path):
        if str(record.get("request_id", "")) != case_id:
            continue
        if "fallback" in str(record.get("event", "")):
            reasons.append(str(record.get("reason") or record.get("event")))
    return (bool(reasons), ";".join(reasons))


def prepare_request_pair(
    server,
    *,
    variant: SystemVariant,
    skill_name: str,
    skill_text: str,
    task_prompt: str,
    case_id: str,
    source_skill_text: str | None = None,
    wait_for_prefetch: bool = True,
) -> PreparedRequest:
    """Run excluded setup work without starting the measured request B."""

    tool_call_id = _forced_skill_call(
        server.base_url,
        model=server.config.platform.served_model,
        skill_name=skill_name,
        case_id=case_id,
    )
    profile_value = server.config.extra_env.get(
        "CSKCACHE_PROFILE_TRACE_PATH", ""
    )
    profile_path = Path(profile_value) if profile_value else Path()
    if variant.family == "cskcache" and wait_for_prefetch:
        _wait_csk_ready(profile_path, tool_call_id)
    messages = tuple(
        target_messages(
            skill_name=skill_name,
            skill_text=skill_text,
            task_prompt=task_prompt,
            tool_call_id=tool_call_id,
        )
    )
    if variant.family == "cacheblend":
        source_messages = target_messages(
            skill_name=skill_name,
            skill_text=(
                skill_text if source_skill_text is None else source_skill_text
            ),
            task_prompt=f"Cache this source context.\n{BLEND_SEPARATOR}",
            tool_call_id=tool_call_id,
        )
        nonstream_chat(
            server.base_url,
            api_key=API_KEY,
            payload={
                "model": server.config.platform.served_model,
                "request_id": f"{case_id}-cache-source",
                "messages": source_messages,
                "tools": skill_tool(skill_name),
                "tool_choice": "none",
                "temperature": 0,
                "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    return PreparedRequest(messages, tool_call_id, profile_path)


def execute_prepared_request(
    server,
    *,
    variant: SystemVariant,
    prepared: PreparedRequest,
    skill_name: str,
    case_id: str,
    max_tokens: int,
    stream: bool,
    reset_prefix_cache: bool = True,
) -> RequestResult:
    """Execute measured request B from a prepared, excluded setup state."""

    if reset_prefix_cache:
        server.reset_prefix_cache()
    client_request_id = _benchmark_request_id(case_id)
    server_request_id = f"chatcmpl-{client_request_id}"
    payload = {
        "model": server.config.platform.served_model,
        "request_id": client_request_id,
        "messages": list(prepared.messages),
        "tools": skill_tool(skill_name),
        "tool_choice": "none",
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    completion = (
        stream_chat(
            server.base_url,
            api_key=API_KEY,
            payload=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if stream
        else nonstream_chat(
            server.base_url,
            api_key=API_KEY,
            payload=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    )
    ttft_ms, prompt_tokens, cached_tokens = _timeline_ttft(
        server.config.trace_path, server_request_id
    )
    fallback, reason = (
        _fallback(prepared.profile_path, server_request_id)
        if variant.family == "cskcache"
        else (False, "")
    )
    return RequestResult(
        completion=completion,
        server_ttft_ms=ttft_ms,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        tool_call_id=prepared.tool_call_id,
        fallback=fallback,
        fallback_reason=reason,
    )


def run_request_pair(
    server,
    *,
    variant: SystemVariant,
    skill_name: str,
    skill_text: str,
    task_prompt: str,
    case_id: str,
    max_tokens: int,
    stream: bool,
    source_skill_text: str | None = None,
    wait_for_prefetch: bool = True,
    reset_prefix_cache: bool = True,
) -> RequestResult:
    """Run excluded request A, then measure request B on the real server."""

    prepared = prepare_request_pair(
        server,
        variant=variant,
        skill_name=skill_name,
        skill_text=skill_text,
        task_prompt=task_prompt,
        case_id=case_id,
        source_skill_text=source_skill_text,
        wait_for_prefetch=wait_for_prefetch,
    )
    return execute_prepared_request(
        server,
        variant=variant,
        prepared=prepared,
        skill_name=skill_name,
        case_id=case_id,
        max_tokens=max_tokens,
        stream=stream,
        reset_prefix_cache=reset_prefix_cache,
    )


def evaluate_rules(
    response: Mapping[str, Any] | None,
    text: str,
    rules: Sequence[Mapping[str, Any]],
) -> tuple[int, int, list[dict[str, Any]]]:
    message: Mapping[str, Any] = {}
    if response:
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
    results: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        kind = str(rule.get("type", ""))
        expected = rule.get("value")
        if kind == "contains":
            passed = str(expected).lower() in text.lower()
        elif kind == "not_contains":
            passed = str(expected).lower() not in text.lower()
        elif kind == "regex":
            passed = re.search(str(expected), text, flags=re.IGNORECASE) is not None
        elif kind == "tool_name":
            calls = message.get("tool_calls") or []
            names = [call.get("function", {}).get("name") for call in calls]
            passed = expected in names
        elif kind == "json_valid":
            try:
                json.loads(text)
                passed = True
            except json.JSONDecodeError:
                passed = False
        else:
            raise ValueError(f"unsupported rule type: {kind}")
        results.append(
            {
                "rule_index": index,
                "type": kind,
                "value": expected,
                "passed": passed,
            }
        )
    return sum(1 for result in results if result["passed"]), len(results), results
