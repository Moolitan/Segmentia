from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# PYTHONPATH=/home/wsh/openhands_code_research/CSKCache:/home/wsh/vllm:$PYTHONPATH

REPO_ROOT = Path(__file__).resolve().parents[3]
CSKCACHE_ROOT = REPO_ROOT / "CSKCache"
VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/home/wsh/vllm"))
DEFAULT_MODEL_PATH = (
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8012")))
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))


def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def pid_file(port: int) -> Path:
    return Path(os.environ.get("CSKCACHE_VLLM_PID_FILE", f"/tmp/cskcache_vllm_{port}.pid"))


def default_log_file(port: int) -> Path:
    log_dir = Path(os.environ.get("CSKCACHE_VLLM_LOG_DIR", str(REPO_ROOT / "log")))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"cskcache_vllm_{port}.log"


def request_json(
    method: str,
    url: str,
    *,
    api_key: str = "EMPTY",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return {} if not body else json.loads(body)


def wait_health(url: str, *, timeout_s: int, api_key: str = "EMPTY") -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            request_json("GET", f"{url.rstrip('/')}/health", api_key=api_key, timeout=5.0)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"vLLM health check timed out after {timeout_s}s: {last_error}")


def wait_health_for_process(
    proc: subprocess.Popen,
    url: str,
    *,
    timeout_s: int,
    api_key: str,
    log_file: Path,
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = "".join(log_file.read_text(errors="replace").splitlines(keepends=True)[-120:])
            raise RuntimeError(f"vLLM exited with code {proc.returncode}\n{tail}")
        try:
            request_json("GET", f"{url.rstrip('/')}/health", api_key=api_key, timeout=5.0)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"vLLM health check timed out after {timeout_s}s: {last_error}")


def stop_server(port: int) -> None:
    path = pid_file(port)
    if not path.exists():
        print(f"No pid file at {path}; nothing stopped.")
        return
    pid = int(path.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        print(f"Process {pid} already exited.")
        return
    for _ in range(30):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            print(f"Stopped vLLM PID {pid}.")
            return
        time.sleep(1)
    os.kill(pid, signal.SIGKILL)
    path.unlink(missing_ok=True)
    print(f"SIGKILLed vLLM PID {pid}.")


def start_vllm_server(
    *,
    host: str,
    port: int,
    model_path: str,
    served_name: str,
    api_key: str,
    log_file: Path,
    kv_dir: Path | None = None,
    probe_enabled: bool = False,
    max_model_len: int = 4096,
    gpu_util: float = 0.85,
    wait_timeout_s: int = 900,
) -> subprocess.Popen:
    if pid_file(port).exists():
        raise RuntimeError(
            f"pid file already exists: {pid_file(port)}. "
            "Stop it first or choose another VLLM_PORT."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{CSKCACHE_ROOT}:{VLLM_ROOT}:"
        + env.get("PYTHONPATH", "")
    )
    conda_prefix = env.get("CONDA_PREFIX") or str(Path(sys.executable).parents[1])
    env["LD_LIBRARY_PATH"] = (
        f"{conda_prefix}/lib"
        + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    )

    extra_config: dict[str, Any] = {}
    if kv_dir is not None:
        extra_config["cskcache.kv_dir"] = str(kv_dir)
    if probe_enabled:
        extra_config.update(
            {
                "cskcache.probe_enabled": True,
                "cskcache.probe_tokens": int(os.environ.get("CSKCACHE_PROBE_TOKENS", "4")),
                "cskcache.anchor_tokens": int(os.environ.get("CSKCACHE_ANCHOR_TOKENS", "32")),
                "cskcache.probe_tau": float(os.environ.get("CSKCACHE_PROBE_TAU", "2.1")),
                "cskcache.gate_metric": os.environ.get("CSKCACHE_GATE_METRIC", "max"),
            }
        )

    kv_transfer_config = {
        "kv_connector": "CSKCacheConnectorV1",
        "kv_connector_module_path": "cskcache.integration.vllm.v1_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra_config,
    }

    # vLLM 不认识 CSKCache
    # -> kv_transfer_config 告诉它模块路径和类名
    # -> PYTHONPATH 让 Python 找得到 cskcache 包
    # -> factory 动态 import 并创建 connector
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--model",
        model_path,
        "--served-model-name",
        served_name,
        "--dtype",
        "auto",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--no-enable-log-requests",
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config, separators=(",", ":")),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(VLLM_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pid_file(port).write_text(str(proc.pid))
    print(f"Started vLLM PID {proc.pid} on {base_url(host, port)}")
    print(f"Log: {log_file}")
    try:
        wait_health_for_process(
            proc,
            base_url(host, port),
            timeout_s=wait_timeout_s,
            api_key=api_key,
            log_file=log_file,
        )
    except Exception:
        pid_file(port).unlink(missing_ok=True)
        raise
    print("vLLM health check passed.")
    return proc
