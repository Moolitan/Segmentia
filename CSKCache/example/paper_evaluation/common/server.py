"""vLLM lifecycle and CSKCache/CacheBlend environment construction."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from paper_evaluation.config import (
    API_KEY,
    CSKCACHE_ROOT,
    HOST,
    LMCACHE_ROOT,
    REQUEST_TIMEOUT_SECONDS,
    SERVER_START_TIMEOUT_SECONDS,
    VLLM_ROOT,
    Platform,
)
from .http_client import request_json


CSK_CONNECTOR = {
    "kv_connector": "CSKCacheConnectorV1",
    "kv_connector_module_path": "cskcache.integrations.vllm.connector",
    "kv_role": "kv_both",
}
CACHEBLEND_CONNECTOR = {
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both",
}


@dataclass(frozen=True)
class ServerConfig:
    platform: Platform
    system: str
    port: int
    log_path: Path
    trace_path: Path
    extra_env: Mapping[str, str]
    connector: Mapping[str, Any] | None
    enable_prefix_caching: bool


def base_environment(platform: Platform) -> dict[str, str]:
    pythonpath = ":".join(
        (str(VLLM_ROOT), str(LMCACHE_ROOT), str(CSKCACHE_ROOT))
    )
    existing = os.getenv("PYTHONPATH")
    if existing:
        pythonpath += f":{existing}"
    return {
        "PYTHONPATH": pythonpath,
        "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in platform.gpu_ids),
        "PYTHONHASHSEED": "0",
        "VLLM_SERVER_DEV_MODE": "1",
        "CSKCACHE_DISABLE_VISUALIZER": "1",
    }


def cacheblend_environment(
    *, ratio: float, chunk_tokens: int, storage_root: Path
) -> dict[str, str]:
    return {
        "LMCACHE_CHUNK_SIZE": str(chunk_tokens),
        "LMCACHE_ENABLE_BLENDING": "True",
        "LMCACHE_BLEND_SPECIAL_STR": " # # ",
        "LMCACHE_BLEND_CHECK_LAYERS": "1",
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": str(ratio),
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_LOCAL_CPU": "False",
        "LMCACHE_LOCAL_DISK": f"file://{storage_root}",
        "LMCACHE_MAX_LOCAL_DISK_SIZE": "1000",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
        "LMCACHE_FORCE_SKIP_SAVE": "0",
    }


def cskcache_environment(
    extra_config: Mapping[str, Any], *, host_page_tokens: int | None = None
) -> dict[str, str]:
    logical_chunk_tokens = int(extra_config["csk_chunk_size_tokens"])
    physical_page_tokens = (
        logical_chunk_tokens if host_page_tokens is None else host_page_tokens
    )
    if physical_page_tokens <= 0:
        raise ValueError("host_page_tokens must be positive")
    if physical_page_tokens < logical_chunk_tokens:
        raise ValueError(
            "host_page_tokens must be at least csk_chunk_size_tokens"
        )
    return {
        "LMCACHE_CHUNK_SIZE": str(physical_page_tokens),
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_FORCE_SKIP_SAVE": "1",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
        "LMCACHE_MAX_LOCAL_DISK_SIZE": "1000",
        "VLLM_CSK_T0_PREFETCH": "1",
        "CSKCACHE_PROFILE": "1",
        "LMCACHE_EXTRA_CONFIG": json.dumps(
            dict(extra_config), separators=(",", ":")
        ),
    }


class VLLMServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.config.port}"

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("vLLM server already started")
        cfg = self.config
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.trace_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.trace_path.write_text("", encoding="utf-8")
        self._log = cfg.log_path.open("wb")
        command = [
            "vllm",
            "serve",
            str(cfg.platform.model_path),
            "--served-model-name",
            cfg.platform.served_model,
            "--api-key",
            API_KEY,
            "--host",
            HOST,
            "--port",
            str(cfg.port),
            "--dtype",
            cfg.platform.dtype,
            "--max-model-len",
            str(cfg.platform.max_model_len),
            "--gpu-memory-utilization",
            str(cfg.platform.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(cfg.platform.tensor_parallel_size),
            "--enforce-eager",
            "--no-enable-log-requests",
            "--no-async-scheduling",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
        ]
        if cfg.platform.reasoning_parser:
            command.extend(("--reasoning-parser", cfg.platform.reasoning_parser))
        command.append(
            "--enable-prefix-caching"
            if cfg.enable_prefix_caching
            else "--no-enable-prefix-caching"
        )
        if cfg.platform.quantization:
            command.extend(("--quantization", cfg.platform.quantization))
        if cfg.connector is not None:
            command.extend(
                (
                    "--kv-transfer-config",
                    json.dumps(dict(cfg.connector), separators=(",", ":")),
                )
            )
        environment = os.environ.copy()
        environment.update(base_environment(cfg.platform))
        environment["VLLM_REQUEST_TIMELINE_PATH"] = str(cfg.trace_path)
        environment.update(cfg.extra_env)
        for proxy in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "all_proxy",
        ):
            environment.pop(proxy, None)
        self.process = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup; see {self.config.log_path}"
                )
            try:
                request_json(
                    f"{self.base_url}/v1/models",
                    api_key=API_KEY,
                    payload=None,
                    method="GET",
                    timeout=5,
                )
                return
            except (OSError, TimeoutError, RuntimeError):
                time.sleep(2)
        raise TimeoutError(f"vLLM readiness timeout: {self.config.log_path}")

    def reset_prefix_cache(self) -> None:
        for _ in range(20):
            try:
                response = request_json(
                    f"{self.base_url}/reset_prefix_cache?reset_external=false",
                    api_key=API_KEY,
                    payload=None,
                    method="POST",
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.get("success") is True:
                    return
            except (OSError, TimeoutError, RuntimeError):
                pass
            time.sleep(0.2)
        raise RuntimeError("vLLM prefix cache reset did not succeed")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
