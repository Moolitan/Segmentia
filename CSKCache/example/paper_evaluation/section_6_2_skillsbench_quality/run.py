"""Run end-to-end SkillsBench quality evaluation through OpenHands."""

from __future__ import annotations

import json
import os
import queue
import signal
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import config as local
from common.driver import make_server_config
from common.run_state import RunContext, utc_now
from common.schema import SCHEMA_VERSION, append_csv, append_jsonl
from common.server import VLLMServer
from paper_evaluation import config as suite
from paper_evaluation.config import API_KEY, BASE_PORT, OUTPUT_ROOT, PLATFORMS


SECTION = "section_6_2_skillsbench_quality"
HOST_NETWORK_COMPOSE = "services:\n  main:\n    network_mode: host\n"
QUALITY_SAMPLE_COLUMNS = (
    "schema_version",
    "run_id",
    "section",
    "case_id",
    "status",
    "hostname",
    "platform_id",
    "gpu_name",
    "model_id",
    "model_path",
    "tensor_parallel_size",
    "system",
    "prefill_mode",
    "task_id",
    "task_digest",
    "skillsbench_commit",
    "agent",
    "agent_model",
    "sandbox",
    "skill_mode",
    "required_skills",
    "repetition",
    "reward",
    "task_success",
    "pipeline_healthy",
    "n_tool_calls",
    "n_skill_invocations",
    "required_skill_invocations",
    "include_task_skills",
    "skill_source",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "usage_source",
    "vllm_request_count",
    "vllm_prompt_tokens",
    "vllm_cached_tokens",
    "vllm_evidence_source",
    "wall_time_seconds",
    "agent_error",
    "verifier_error",
    "benchflow_result_path",
    "jobs_dir",
    "input_fingerprint",
    "started_utc",
    "completed_utc",
)


def load_workloads() -> list[dict[str, Any]]:
    payload = json.loads(local.WORKLOAD_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("workloads.json must contain a non-empty list")
    workloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("each workload must be an object")
        task_id = str(item.get("task_id", "")).strip()
        skills = item.get("required_skills")
        if not task_id or task_id in seen:
            raise ValueError(f"invalid or duplicate task_id: {task_id!r}")
        if not isinstance(skills, list) or not skills:
            raise ValueError(f"required_skills must be non-empty: {task_id}")
        task_dir = local.SKILLSBENCH_ROOT / "tasks" / task_id
        required = tuple(str(value).strip() for value in skills)
        for skill_name in required:
            skill_file = task_dir / "environment/skills" / skill_name / "SKILL.md"
            if not skill_file.is_file():
                raise FileNotFoundError(f"required Skill does not exist: {skill_file}")
        for relative in ("task.md", "environment/Dockerfile", "verifier/test.sh"):
            if not (task_dir / relative).is_file():
                raise FileNotFoundError(f"incomplete SkillsBench task: {task_dir / relative}")
        seen.add(task_id)
        workloads.append(
            {"task_id": task_id, "required_skills": required, "task_dir": task_dir}
        )
    return workloads


def _skillsbench_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(local.SKILLSBENCH_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _task_input_paths(workloads: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for workload in workloads:
        task_dir = Path(workload["task_dir"])
        paths.extend(path for path in sorted(task_dir.rglob("*")) if path.is_file())
    return paths


def _stage_workloads(
    run_dir: Path, workloads: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    staged_workloads: list[dict[str, Any]] = []
    staging_root = run_dir / "staged_tasks"
    staging_root.mkdir(parents=True, exist_ok=True)
    for workload in workloads:
        source = Path(workload["task_dir"])
        destination = staging_root / str(workload["task_id"])
        if not destination.exists():
            shutil.copytree(source, destination)
        compose_path = destination / "environment/docker-compose.yaml"
        if compose_path.exists():
            if compose_path.read_text(encoding="utf-8") != HOST_NETWORK_COMPOSE:
                raise RuntimeError(
                    f"unexpected staged Docker Compose override: {compose_path}"
                )
        else:
            compose_path.write_text(HOST_NETWORK_COMPOSE, encoding="utf-8")
        staged = dict(workload)
        staged["source_task_dir"] = source
        staged["task_dir"] = destination
        staged_workloads.append(staged)
    return staged_workloads


def _next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("attempt-*"):
        try:
            indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = root / f"attempt-{max(indices, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[command] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(environment) if environment is not None else None,
        start_new_session=True,
    )
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = (
        time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    )
    stream_closed = False
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            while process.poll() is None or not stream_closed or not lines.empty():
                if deadline is not None and time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    raise TimeoutError(
                        f"command exceeded {timeout_seconds}s; see {log_path}"
                    )
                try:
                    line = lines.get(timeout=1)
                except queue.Empty:
                    continue
                if line is None:
                    stream_closed = True
                    continue
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command exited with code {return_code}; see {log_path}"
        )


def _bench_command(*arguments: str) -> list[str]:
    if not local.BENCH_EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"BenchFlow is not installed at {local.BENCH_EXECUTABLE}; "
            f"run: uv sync --project {local.SKILLSBENCH_ROOT} --locked --no-dev"
        )
    return [str(local.BENCH_EXECUTABLE), *arguments]


def _find_result(jobs_dir: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(jobs_dir.rglob("result.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one rollout result below {jobs_dir}, found {len(paths)}"
        )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"BenchFlow result is not an object: {paths[0]}")
    return paths[0], payload


def _reward(result: Mapping[str, Any]) -> float | None:
    rewards = result.get("rewards")
    if not isinstance(rewards, Mapping):
        return None
    value = rewards.get("reward")
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise TypeError(f"rewards.reward is not numeric: {value!r}")
    return float(value)


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} is not numeric: {value!r}")
    return int(value)


def _required_skill_invocations(
    result_path: Path, required_skills: Sequence[str]
) -> int:
    trajectory = result_path.parent / "trajectory/acp_trajectory.jsonl"
    if not trajectory.is_file():
        return 0
    required = set(required_skills)
    count = 0
    for line in trajectory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
        if not any(skill_name in serialized for skill_name in required):
            continue
        stack = [event]
        is_skill_call = False
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                kind = value.get("kind", value.get("tool_kind"))
                if kind == "skill":
                    is_skill_call = True
                    break
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        if is_skill_call:
            count += 1
    return count


def _vllm_totals(trace_path: Path, log_path: Path) -> dict[str, int | str]:
    request_count = 0
    prompt_tokens = 0
    cached_tokens = 0
    if not trace_path.is_file():
        lines: list[str] = []
    else:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.strip():
            event = json.loads(line)
            if event.get("event") == "first_token_ready":
                request_count += 1
                prompt_tokens += int(event.get("prompt_tokens") or 0)
                cached_tokens += int(event.get("cached_tokens") or 0)
    evidence_source = "request_timeline" if request_count else "unavailable"
    if request_count == 0 and log_path.is_file():
        access_marker = 'POST /v1/chat/completions HTTP/1.1" 200 OK'
        request_count = sum(
            access_marker in line
            for line in log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        )
        if request_count:
            evidence_source = "vllm_access_log"
    return {
        "vllm_request_count": request_count,
        "vllm_prompt_tokens": prompt_tokens,
        "vllm_cached_tokens": cached_tokens,
        "vllm_evidence_source": evidence_source,
    }


def _sample_is_healthy(sample: Mapping[str, Any]) -> bool:
    return all(
        (
            sample.get("reward") not in (None, ""),
            not sample.get("agent_error"),
            not sample.get("verifier_error"),
            int(sample.get("n_tool_calls") or 0) > 0,
            int(sample.get("n_skill_invocations") or 0) > 0,
            int(sample.get("required_skill_invocations") or 0) > 0,
            sample.get("include_task_skills") is True,
            int(sample.get("total_tokens") or 0) > 0,
            int(sample.get("vllm_request_count") or 0) > 0,
        )
    )


def _prepare_docker_config(run_dir: Path) -> Path:
    plugin = local.DOCKER_COMPOSE_PLUGIN
    if not plugin.is_file() or not os.access(plugin, os.X_OK):
        raise FileNotFoundError(
            "an executable Docker Compose plugin is required at " f"{plugin}"
        )
    config_dir = run_dir / "docker_config"
    plugin_dir = config_dir / "cli-plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    link = plugin_dir / "docker-compose"
    if link.is_symlink():
        if link.resolve() != plugin.resolve():
            raise RuntimeError(f"unexpected Docker Compose symlink target: {link}")
    elif link.exists():
        raise RuntimeError(f"Docker Compose plugin path is not a symlink: {link}")
    else:
        link.symlink_to(plugin)
    proxy_config = {
        "httpProxy": os.getenv("http_proxy") or os.getenv("HTTP_PROXY") or "",
        "httpsProxy": os.getenv("https_proxy")
        or os.getenv("HTTPS_PROXY")
        or "",
        "noProxy": os.getenv("no_proxy") or os.getenv("NO_PROXY") or "",
    }
    if not proxy_config["httpProxy"] or not proxy_config["httpsProxy"]:
        raise RuntimeError(
            "host-network Docker setup requires http_proxy and https_proxy"
        )
    config_path = config_dir / "config.json"
    expected = json.dumps(
        {"proxies": {"default": proxy_config}}, indent=2, sort_keys=True
    ) + "\n"
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"unexpected run-scoped Docker config: {config_path}")
    else:
        config_path.write_text(expected, encoding="utf-8")
    return config_dir


def _validate_oracle(
    run_dir: Path, workload: Mapping[str, Any], docker_config: Path
) -> None:
    task_id = str(workload["task_id"])
    task_dir = Path(workload["task_dir"])
    root = run_dir / "preflight" / task_id
    marker = root / "oracle_passed.json"
    if marker.is_file():
        return
    attempt = _next_attempt(root)
    _run_logged(
        _bench_command("tasks", "check", str(task_dir)),
        log_path=attempt / "task_check.log",
        environment=_benchflow_environment(docker_config=docker_config),
        timeout_seconds=300,
    )
    jobs_dir = attempt / "oracle_jobs"
    _run_logged(
        _bench_command(
            "eval",
            "run",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "oracle",
            "--sandbox",
            local.SANDBOX,
            "--concurrency",
            "1",
            "--jobs-dir",
            str(jobs_dir),
        ),
        log_path=attempt / "oracle.log",
        environment=_benchflow_environment(docker_config=docker_config),
        timeout_seconds=local.BENCHFLOW_WALL_TIMEOUT_SECONDS,
    )
    result_path, result = _find_result(jobs_dir)
    reward = _reward(result)
    if reward != 1.0 or result.get("error") or result.get("verifier_error"):
        raise RuntimeError(
            f"oracle preflight failed: reward={reward}, result={result_path}"
        )
    marker.write_text(
        json.dumps(
            {"task_id": task_id, "reward": reward, "result": str(result_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _benchflow_environment(
    *, docker_config: Path, server: VLLMServer | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("DEBUG", None)
    environment.pop("ALL_PROXY", None)
    environment.pop("all_proxy", None)
    environment["LITELLM_LOG"] = "INFO"
    environment["DOCKER_CONFIG"] = str(docker_config)
    if server is not None:
        environment.update(
            {
                "BENCHFLOW_PROVIDER_BASE_URL": f"{server.base_url}/v1",
                "BENCHFLOW_PROVIDER_API_KEY": API_KEY,
                "OPENAI_API_KEY": API_KEY,
            }
        )
    bypass = environment.get("NO_PROXY", environment.get("no_proxy", ""))
    hosts = [value for value in bypass.split(",") if value]
    for host in ("127.0.0.1", "localhost"):
        if host not in hosts:
            hosts.append(host)
    environment["NO_PROXY"] = ",".join(hosts)
    environment["no_proxy"] = environment["NO_PROXY"]
    return environment


def _run_case(
    *,
    run: RunContext,
    platform_id: str,
    platform_index: int,
    workload: Mapping[str, Any],
    repetition: int,
    skillsbench_commit: str,
    docker_config: Path,
) -> None:
    platform = PLATFORMS[platform_id]
    variant = local.SYSTEMS[0]
    task_id = str(workload["task_id"])
    task_dir = Path(workload["task_dir"])
    case_id = f"{platform_id}__{variant.name}__{task_id}__r{repetition}"
    if run.completed(case_id):
        return
    case_root = run.run_dir / "cases" / case_id
    attempt = _next_attempt(case_root)
    run.mark(case_id, "running", attempt_dir=str(attempt))
    started_utc = utc_now()
    started = time.monotonic()
    server_cfg = make_server_config(
        platform=platform,
        variant=variant,
        port=BASE_PORT + platform_index,
        case_root=attempt / "server",
        chunk_tokens=256,
    )
    try:
        with VLLMServer(server_cfg) as server:
            jobs_dir = attempt / "benchflow_jobs"
            agent_model = f"{local.MODEL_PROVIDER}/{platform.served_model}"
            command = _bench_command(
                "eval",
                "run",
                "--tasks-dir",
                str(task_dir),
                "--agent",
                local.AGENT,
                "--model",
                agent_model,
                "--agent-env",
                f"BENCHFLOW_PROVIDER_BASE_URL={server.base_url}/v1",
                "--agent-env",
                f"BENCHFLOW_PROVIDER_API_KEY={API_KEY}",
                "--sandbox",
                local.SANDBOX,
                "--skill-mode",
                local.SKILL_MODE,
                "--skills-dir",
                str(task_dir / "environment/skills"),
                "--concurrency",
                str(local.CONCURRENCY),
                "--usage-tracking",
                local.USAGE_TRACKING,
                "--agent-idle-timeout",
                local.AGENT_IDLE_TIMEOUT,
                "--sandbox-setup-timeout",
                str(local.SANDBOX_SETUP_TIMEOUT_SECONDS),
                "--jobs-dir",
                str(jobs_dir),
            )
            _run_logged(
                command,
                log_path=attempt / "benchflow.log",
                environment=_benchflow_environment(
                    docker_config=docker_config, server=server
                ),
                timeout_seconds=local.BENCHFLOW_WALL_TIMEOUT_SECONDS,
            )
            result_path, result = _find_result(jobs_dir)
            vllm = _vllm_totals(server_cfg.trace_path, server_cfg.log_path)

        reward = _reward(result)
        agent_result = result.get("agent_result")
        final_metrics = result.get("final_metrics")
        if not isinstance(agent_result, Mapping):
            agent_result = {}
        if not isinstance(final_metrics, Mapping):
            final_metrics = {}
        n_tool_calls = _integer(result, "n_tool_calls")
        n_skill_invocations = _integer(result, "n_skill_invocations")
        required_skill_invocations = _required_skill_invocations(
            result_path, workload["required_skills"]
        )
        include_task_skills = result.get("include_task_skills") is True
        total_tokens = _integer(agent_result, "total_tokens")
        sample = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run.run_id,
            "section": SECTION,
            "case_id": case_id,
            "status": "invalid",
            "hostname": socket.gethostname(),
            "platform_id": platform_id,
            "gpu_name": platform.gpu_name,
            "model_id": platform.model_id,
            "model_path": str(platform.model_path),
            "tensor_parallel_size": platform.tensor_parallel_size,
            "system": variant.name,
            "prefill_mode": "full_recompute_no_external_kv",
            "task_id": task_id,
            "task_digest": result.get("task_digest", ""),
            "skillsbench_commit": skillsbench_commit,
            "agent": local.AGENT,
            "agent_model": agent_model,
            "sandbox": local.SANDBOX,
            "skill_mode": local.SKILL_MODE,
            "required_skills": list(workload["required_skills"]),
            "repetition": repetition,
            "reward": "" if reward is None else reward,
            "task_success": reward == 1.0,
            "pipeline_healthy": False,
            "n_tool_calls": n_tool_calls,
            "n_skill_invocations": n_skill_invocations,
            "required_skill_invocations": required_skill_invocations,
            "include_task_skills": include_task_skills,
            "skill_source": result.get("skill_source", ""),
            "total_tokens": total_tokens,
            "prompt_tokens": _integer(final_metrics, "total_prompt_tokens"),
            "completion_tokens": _integer(
                final_metrics, "total_completion_tokens"
            ),
            "usage_source": agent_result.get("usage_source", ""),
            **vllm,
            "wall_time_seconds": time.monotonic() - started,
            "agent_error": result.get("error") or "",
            "verifier_error": result.get("verifier_error") or "",
            "benchflow_result_path": str(result_path),
            "jobs_dir": str(jobs_dir),
            "input_fingerprint": run.fingerprint,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
        }
        healthy = _sample_is_healthy(sample)
        sample["pipeline_healthy"] = healthy
        sample["status"] = "completed" if healthy else "invalid"
        (attempt / "parsed_result.json").write_text(
            json.dumps(sample, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not healthy:
            raise RuntimeError(
                "BenchFlow completed but the rollout failed the real-run health gate; "
                f"see {attempt / 'parsed_result.json'}"
            )
        append_jsonl(run.samples_jsonl, sample)
        append_csv(run.samples_csv, sample, QUALITY_SAMPLE_COLUMNS)
        run.mark(case_id, "completed", attempt_dir=str(attempt))
    except Exception as exc:
        failure = {
            "case_id": case_id,
            "error": f"{type(exc).__name__}: {exc}",
            "attempt_dir": str(attempt),
            "failed_utc": utc_now(),
        }
        (attempt / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.mark(
            case_id,
            "failed",
            error=failure["error"],
            attempt_dir=failure["attempt_dir"],
            failed_utc=failure["failed_utc"],
        )
        raise


def main() -> None:
    if not local.SKILLSBENCH_ROOT.is_dir():
        raise FileNotFoundError(
            f"SkillsBench checkout does not exist: {local.SKILLSBENCH_ROOT}"
        )
    workloads = load_workloads()
    skillsbench_commit = _skillsbench_commit()
    values = {
        "skillsbench_root": str(local.SKILLSBENCH_ROOT),
        "skillsbench_commit": skillsbench_commit,
        "platform_ids": list(local.PLATFORM_IDS),
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "workloads": [
            {
                "task_id": item["task_id"],
                "required_skills": list(item["required_skills"]),
            }
            for item in workloads
        ],
        "agent": local.AGENT,
        "model_provider": local.MODEL_PROVIDER,
        "sandbox": local.SANDBOX,
        "skill_mode": local.SKILL_MODE,
        "usage_tracking": local.USAGE_TRACKING,
        "docker_network_mode": "host",
        "docker_compose_plugin": str(local.DOCKER_COMPOSE_PLUGIN),
        "repetitions": local.REPETITIONS,
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(
            Path(__file__),
            Path(local.__file__),
            Path(suite.__file__),
            local.WORKLOAD_FILE,
            *_task_input_paths(workloads),
        ),
        config_values=values,
    )
    docker_config = _prepare_docker_config(run.run_dir)
    workloads = _stage_workloads(run.run_dir, workloads)
    if local.RUN_ORACLE_PREFLIGHT:
        for workload in workloads:
            _validate_oracle(run.run_dir, workload, docker_config)
    for platform_index, platform_id in enumerate(local.PLATFORM_IDS):
        platform = PLATFORMS[platform_id]
        if not platform.model_path.is_dir():
            raise FileNotFoundError(f"model does not exist: {platform.model_path}")
        for workload in workloads:
            for repetition in range(local.REPETITIONS):
                _run_case(
                    run=run,
                    platform_id=platform_id,
                    platform_index=platform_index,
                    workload=workload,
                    repetition=repetition,
                    skillsbench_commit=skillsbench_commit,
                    docker_config=docker_config,
                )
    from analyze import analyze

    analyze(run.run_dir)
    run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
