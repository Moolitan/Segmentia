#!/usr/bin/env python3
"""Diagnose the T0--T1 Agent path with three independent Mermaid runs.

The script starts a fresh vLLM and OpenHands conversation for each repetition.
It does not mix prefixes or scheduler state across runs.  Raw traces stay under
the configured external output root; only a compact CSV and figure are copied
to ``results/``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import signal
import statistics
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/segmentia-mpl")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))
from cskcache import SOURCE_ARTIFACT_TYPE

from common import (
    atomic_write_json,
    atomic_write_text,
    discover_manifest_layers,
    load_config,
    raw_run_dir,
    resolve_skill_manifest,
    result_dir,
)


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = Path(__file__).resolve()
TRACE_FILES = {
    "skill_actions": "segmentia_skill_action_ready.jsonl",
    "locator": "segmentia_skill_locator.jsonl",
    "scheduler": "segmentia_scheduler_admission.jsonl",
    "execution": "segmentia_skill_execution.jsonl",
    "agent": "segmentia_agent_timeline.json",
    "transport": "segmentia_transport_events.json",
}
INTERVALS = (
    ("t0_o1_ms", "vLLM response and HTTP return", "#264653", "white"),
    ("o1_o2_ms", "SDK response to ActionEvent", "#2A9D8F", "white"),
    ("o2_o3_ms", "ActionEvent to SkillTool start", "#287271", "white"),
    ("o3_o4_ms", "SkillTool execution", "#3A86A8", "white"),
    ("o4_o5_ms", "Observation construction", "#70A9A1", "black"),
    ("o5_o6_ms", "Observation emission", "#A8DADC", "black"),
    ("o6_o7_ms", "Next-request assembly", "#E9C46A", "black"),
    ("o7_o8_ms", "Segmentia metadata injection", "#F4A261", "black"),
    ("o8_t1_ms", "HTTP send to vLLM arrival", "#E76F51", "white"),
    ("t1_t2_ms", "vLLM template and tokenization", "#8E6C8A", "white"),
    ("t2_t3_ms", "Post-tokenization to scheduler", "#6D597A", "white"),
)
CSV_FIELDS = (
    "repetition",
    "task",
    "skill",
    "skill_tokens",
    *(field for field, _, _, _ in INTERVALS),
    "t0_t3_ms",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"trace file is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object: {path}")
        records.append(value)
    if not records:
        raise ValueError(f"trace file has no records: {path}")
    return records


def unique_record(
    records: list[dict[str, Any]], predicate: Any, description: str
) -> dict[str, Any]:
    matched = [record for record in records if predicate(record)]
    if len(matched) != 1:
        raise ValueError(f"expected one {description}, found {len(matched)}")
    return matched[0]


def resolve_skill_cache(
    pool_dir: Path, skill: str, expected_layers: int
) -> dict[str, Any]:
    manifest_path = resolve_skill_manifest(pool_dir, skill)
    manifest, layers = discover_manifest_layers(manifest_path, expected_layers)
    token_count = manifest.get("token_count")
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
    ):
        raise ValueError(f"manifest token_count is invalid: {manifest_path}")
    cache_bytes = sum(layer.size_bytes for layer in layers)
    if cache_bytes <= 0:
        raise ValueError(f"Skill cache has no bytes: {manifest_path}")
    return {
        "token_count": token_count,
        "cache_bytes": cache_bytes,
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def validate_locator_manifest(cache_info: dict[str, Any]) -> None:
    manifest_path = Path(cache_info["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise ValueError(
            "Mermaid direct-reuse timeline requires a CSKCache source object: "
            f"{manifest_path}"
        )
    if not manifest.get("start_marker_token_ids"):
        raise ValueError(f"unsupported Skill locator manifest: {manifest_path}")


def stream_process(
    command: list[str], environment: dict[str, str], log: Path
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                # Drain the child pipe continuously and retain the complete log,
                # but do not synchronously mirror the large Skill observation to
                # the terminal.  Terminal rendering/backpressure is outside the
                # prefetch-window mechanism being measured.
                handle.write(line)
                handle.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise


def read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"trace file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"trace must be a JSON array of objects: {path}")
    return value


def timestamp(record: dict[str, Any], field: str = "monotonic_ns") -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid {field} in trace record: {record}")
    return value


def analyze_workspace(
    workspace: Path,
    *,
    repetition: int,
    task: str,
    skill: str,
    skill_tokens: int,
    fingerprint: str,
) -> dict[str, Any]:
    actions = read_jsonl(workspace / TRACE_FILES["skill_actions"])
    locators = read_jsonl(workspace / TRACE_FILES["locator"])
    scheduler = read_jsonl(workspace / TRACE_FILES["scheduler"])
    execution = read_jsonl(workspace / TRACE_FILES["execution"])
    agent = read_json_array(workspace / TRACE_FILES["agent"])
    transport = read_json_array(workspace / TRACE_FILES["transport"])

    located = unique_record(
        locators,
        lambda row: row.get("skill_name") == skill and row.get("status") == "ok",
        "successful request-B locator",
    )
    tool_call_id = str(located.get("source_tool_call_id", ""))
    if not tool_call_id:
        raise ValueError("request-B locator has no source_tool_call_id")
    action = unique_record(
        actions,
        lambda row: row.get("boundary") == "structured_skill_action_ready"
        and row.get("skill_name") == skill
        and row.get("tool_call_id") == tool_call_id,
        "request-A SkillAction",
    )
    request_a_id = str(action.get("request_id", ""))
    request_b_id = str(located.get("request_id", ""))
    admission = unique_record(
        scheduler,
        lambda row: row.get("request_id") == request_b_id
        and row.get("source_tool_call_id") == tool_call_id
        and row.get("boundary") == "immediately_before_scheduler_add_request",
        "request-B scheduler admission",
    )
    response_a = unique_record(
        transport,
        lambda row: row.get("request_id") == request_a_id
        and row.get("boundary") == "client_transport_response_received",
        "request-A client response",
    )
    transport_b = unique_record(
        transport,
        lambda row: row.get("request_id") == request_b_id
        and row.get("boundary") == "client_transport_response_received",
        "request-B client transport",
    )

    def by_tool(records: list[dict[str, Any]], boundary: str) -> dict[str, Any]:
        return unique_record(
            records,
            lambda row: row.get("tool_call_id") == tool_call_id
            and row.get("boundary") == boundary,
            boundary,
        )

    action_callback = by_tool(agent, "skill_action_event_callback")
    tool_start = by_tool(execution, "skill_tool_execution_start")
    tool_returned = by_tool(execution, "skill_tool_execution_returned")
    observation_created = by_tool(execution, "skill_observation_event_created")
    observation_callback = by_tool(agent, "skill_observation_event_callback")

    points = {
        "t0": timestamp(action, "skill_action_ready_monotonic_ns"),
        "o1": timestamp(response_a, "client_response_received_monotonic_ns"),
        "o2": timestamp(action_callback),
        "o3": timestamp(tool_start),
        "o4": timestamp(tool_returned),
        "o5": timestamp(observation_created),
        "o6": timestamp(observation_callback),
        "o7": timestamp(transport_b, "request_wrapper_enter_monotonic_ns"),
        "o8": timestamp(transport_b, "client_transport_handoff_monotonic_ns"),
        "t1": timestamp(located, "request_received_monotonic_ns"),
        "t2": timestamp(located, "tokenization_completed_monotonic_ns"),
        "t3": timestamp(admission, "scheduler_admission_monotonic_ns"),
    }
    ordered_names = tuple(points)
    ordered_values = [points[name] for name in ordered_names]
    if ordered_values != sorted(ordered_values):
        raise ValueError(f"timeline is not monotonic: {points}")

    boot_ids = {
        row.get("boot_id")
        for row in (
            action,
            response_a,
            action_callback,
            tool_start,
            tool_returned,
            observation_created,
            observation_callback,
            transport_b,
            located,
            admission,
        )
    }
    if len(boot_ids) != 1 or not next(iter(boot_ids)):
        raise ValueError(f"timeline boot IDs disagree: {boot_ids}")

    adjacent = zip(ordered_names, ordered_names[1:])
    intervals = {
        f"{left}_{right}_ms": (points[right] - points[left]) / 1e6
        for left, right in adjacent
    }
    expected_fields = {field for field, _, _, _ in INTERVALS}
    if set(intervals) != expected_fields:
        raise AssertionError(f"interval fields disagree: {set(intervals)}")
    total_ms = (points["t3"] - points["t0"]) / 1e6
    if abs(sum(intervals.values()) - total_ms) > 0.001:
        raise ValueError("fine-grained intervals do not sum to T0--T3")
    return {
        "schema_version": 1,
        "status": "completed",
        "fingerprint": fingerprint,
        "repetition": repetition,
        "task": task,
        "skill": skill,
        "skill_tokens": skill_tokens,
        "request_a_id": request_a_id,
        "request_b_id": request_b_id,
        "source_tool_call_id": tool_call_id,
        "boot_id": next(iter(boot_ids)),
        "points_monotonic_ns": points,
        **intervals,
        "t0_t3_ms": total_ms,
    }


def fingerprint(paths: tuple[Path, ...], cache_manifest: Path, settings: dict[str, Any]) -> str:
    payload = {
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cache_manifest": hashlib.sha256(cache_manifest.read_bytes()).hexdigest(),
        "settings": settings,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)
    atomic_write_text(path, buffer.getvalue())


def plot(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.2, 4.6))
    left = [0.0] * len(rows)
    for field, label, color, text_color in INTERVALS:
        values = [float(row[field]) for row in rows]
        bars = axis.barh(
            range(len(rows)), values, left=left, label=label, color=color,
            edgecolor="black", linewidth=0.55,
        )
        for bar, offset, value in zip(bars, left, values):
            if value >= 4.0:
                axis.text(
                    offset + value / 2, bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}", ha="center", va="center", color=text_color,
                    fontsize=7.5,
                )
        left = [offset + value for offset, value in zip(left, values)]
    axis.set_yticks(range(len(rows)), [f"Run {row['repetition']}" for row in rows])
    axis.set_xlabel("Latency from parsed SkillAction to scheduler admission (ms)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(left) * 1.03)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
        frameon=False, fontsize=8,
    )
    figure.tight_layout()
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"mermaid_skill_t0_t3_fine_timeline.{suffix}",
            dpi=dpi, bbox_inches="tight",
        )
    plt.close(figure)


def update_manifest(path: Path, batch_root: Path) -> None:
    rows = (
        list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        if path.is_file()
        else []
    )
    artifacts = {
        "figures/mermaid_skill_t0_t3_fine_timeline.pdf",
        "figures/mermaid_skill_t0_t3_fine_timeline.png",
        "tables/mermaid_skill_t0_t3_fine_timeline.csv",
        "data/mermaid_skill_timeline_run_pointer.txt",
    }
    rows = [row for row in rows if row.get("artifact") not in artifacts]
    rows.extend(
        {"artifact": item, "source": str(batch_root)}
        for item in sorted(artifacts)
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("artifact", "source"))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def main() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "opencode":
        raise RuntimeError("activate the opencode conda environment first")
    config = load_config()
    section = config["mermaid_timeline"]
    repetitions = int(section["repetitions"])
    if repetitions != 3:
        raise ValueError("mermaid_timeline.repetitions must be 3")
    task = str(section["task"])
    skill = str(section["skill"])
    max_iterations = int(section["max_iterations"])
    prompt = Path(config["agent_schedule"]["prompt_dir"]) / f"{task}.txt"
    launcher = Path(config["agent_schedule"]["launcher"]).resolve()
    pool_dir = Path(config["skill_cache"]["pool_dir"]).resolve()
    cache = resolve_skill_cache(
        pool_dir, skill, int(config["skill_cache"]["expected_layers"])
    )
    validate_locator_manifest(cache)
    batch_root = (
        raw_run_dir(config)
        / "10_mermaid_skill_timeline"
        / str(section["batch_id"])
    )
    batch_root.mkdir(parents=True, exist_ok=True)
    settings = {
        "task": task, "skill": skill, "max_iterations": max_iterations,
        "repetitions": repetitions, "mode": "direct_reuse",
    }
    implementation = (
        SCRIPT_PATH,
        ROOT / "scripts/08_lmcache_mp/paper_motivation/3.1/interactive_agent.py",
        ROOT / "scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh",
        ROOT / "software-agent-sdk/openhands-sdk/openhands/sdk/agent/agent.py",
        ROOT / "vllm/vllm/entrypoints/openai/chat_completion/serving.py",
        ROOT / "vllm/vllm/v1/engine/core.py",
        prompt,
    )
    current_fingerprint = fingerprint(
        implementation, Path(cache["manifest_path"]), settings
    )

    completed: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        case_dir = batch_root / f"run_{repetition:02d}"
        result_path = case_dir / "case_result.json"
        if result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "completed"
                and previous.get("fingerprint") == current_fingerprint
            ):
                print(f"[skip] run {repetition} already completed")
                completed.append(previous)
                continue
        workspace = case_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        command = [
            "bash", str(launcher), "--skill", skill, "--max-iterations",
            str(max_iterations), "--prompt-file", str(prompt),
        ]
        environment = os.environ.copy()
        environment["SEGMENTIA_MODE"] = "direct_reuse"
        environment["SEGMENTIA_FINE_TIMELINE"] = "1"
        environment["SEGMENTIA_DISABLE_VISUALIZER"] = "1"
        environment["OPENHANDS_WORKSPACE"] = str(workspace)
        print(f"[run] {repetition}/{repetitions} task={task}")
        return_code = stream_process(
            command, environment, case_dir / "launcher.log"
        )
        if return_code != 0:
            raise RuntimeError(f"run {repetition} launcher exited with code {return_code}")
        result = analyze_workspace(
            workspace, repetition=repetition, task=task,
            skill=skill, skill_tokens=int(cache["token_count"]),
            fingerprint=current_fingerprint,
        )
        atomic_write_json(result_path, result)
        completed.append(result)
        print(f"[captured] run={repetition} T0-T3={result['t0_t3_ms']:.3f}ms")

    completed.sort(key=lambda row: int(row["repetition"]))
    summary = {
        "schema_version": 1,
        "batch_id": str(section["batch_id"]),
        "task": task,
        "skill": skill,
        "completed": len(completed),
        "t0_t3_ms": {
            "minimum": min(float(row["t0_t3_ms"]) for row in completed),
            "mean": statistics.fmean(float(row["t0_t3_ms"]) for row in completed),
            "median": statistics.median(float(row["t0_t3_ms"]) for row in completed),
            "maximum": max(float(row["t0_t3_ms"]) for row in completed),
        },
    }
    atomic_write_json(batch_root / "batch_summary.json", summary)

    output = result_dir(config)
    figure_dir, table_dir, data_dir = output / "figures", output / "tables", output / "data"
    for directory in (figure_dir, table_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / "mermaid_skill_t0_t3_fine_timeline.csv", completed)
    plot(completed, figure_dir)
    atomic_write_text(data_dir / "mermaid_skill_timeline_run_pointer.txt", str(batch_root) + "\n")
    update_manifest(output / "source_manifest.csv", batch_root)
    print(f"[completed] batch={batch_root} repetitions={len(completed)}")


if __name__ == "__main__":
    main()
