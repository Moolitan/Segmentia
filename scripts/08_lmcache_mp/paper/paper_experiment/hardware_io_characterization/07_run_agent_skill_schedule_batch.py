#!/usr/bin/env python3
"""Run every configured task prompt and plot its full Skill prefetch window.

This is a long-running vLLM experiment. Run it from the ``opencode`` conda
environment. Each task gets a fresh vLLM process and fresh Conversation. Raw
logs stay under the external output root; only the CSV and PDF/PNG summary are
published in ``results/``.
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
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/segmentia-mpl")
import matplotlib.pyplot as plt

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
    "skill_actions": "normal_prefill_skill_action_ready.jsonl",
    "agent_events": "normal_prefill_schedule_events.json",
    "transport": "normal_prefill_transport_events.json",
    "scheduler": "normal_prefill_scheduler_admission.jsonl",
}
WINDOW_CSV_FIELDS = (
    "task",
    "skill",
    "skill_tokens",
    "prefetch_window_ms",
)
BREAKDOWN_CSV_FIELDS = (
    "task",
    "skill",
    "skill_tokens",
    "cache_bytes",
    "t0_t1_ms",
    "t1_t2_ms",
    "t2_t3_ms",
    "t3_t4_ms",
    "t4_t5_ms",
    "prefetch_window_ms",
)
ESTIMATE_CSV_FIELDS = (
    "task",
    "skill",
    "skill_tokens",
    "cache_bytes",
    "prefetch_window_ms",
    "estimated_pinned_h2d_ms",
    "estimated_warm_serial_ms",
    "estimated_cold_serial_ms",
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    section = config["agent_schedule"]
    prompt_dir = Path(section["prompt_dir"]).resolve()
    values = section.get("cases")
    if not isinstance(values, list) or not values:
        raise ValueError("agent_schedule.cases must be a non-empty list")
    cases: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("every agent_schedule case must be a mapping")
        task = str(value.get("task", "")).strip()
        skill = str(value.get("skill", "")).strip()
        exposed = value.get("exposed_skills")
        if not task or "/" in task or task in seen_tasks:
            raise ValueError(f"invalid or duplicate task name: {task!r}")
        if not skill or not isinstance(exposed, list) or skill not in exposed:
            raise ValueError(f"case {task} must expose its target Skill {skill!r}")
        exposed_skills = [str(item).strip() for item in exposed]
        if any(not item for item in exposed_skills) or len(set(exposed_skills)) != len(
            exposed_skills
        ):
            raise ValueError(f"case {task} has invalid exposed_skills")
        prompt_path = prompt_dir / f"{task}.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"task prompt is missing: {prompt_path}")
        cases.append(
            {
                "task": task,
                "skill": skill,
                "exposed_skills": exposed_skills,
                "prompt_path": prompt_path,
            }
        )
        seen_tasks.add(task)
    actual = {path.stem for path in prompt_dir.glob("*.txt")}
    configured = {case["task"] for case in cases}
    if actual != configured:
        raise ValueError(
            "configured task prompts disagree with src/task_prompt: "
            f"missing={sorted(actual - configured)}, "
            f"extra={sorted(configured - actual)}"
        )
    return cases


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
        "manifest_sha256": sha256_path(manifest_path),
    }


def resolve_skill_tokens(pool_dir: Path, skill: str) -> int:
    """Compatibility helper used by the existing static tests."""
    return int(resolve_skill_cache(pool_dir, skill, expected_layers=40)["token_count"])


def case_fingerprint(
    case: dict[str, Any],
    max_iterations: int,
    implementation_hashes: dict[str, str],
    cache_info: dict[str, Any],
) -> str:
    payload = {
        "task": case["task"],
        "skill": case["skill"],
        "exposed_skills": case["exposed_skills"],
        "prompt_sha256": sha256_path(case["prompt_path"]),
        "max_iterations": max_iterations,
        "implementation_sha256": implementation_hashes,
        "skill_cache": {
            "token_count": cache_info["token_count"],
            "cache_bytes": cache_info["cache_bytes"],
            "manifest_sha256": cache_info["manifest_sha256"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def analyze_case(
    case: dict[str, Any],
    workspace: Path,
    cache_info: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    skill_actions = read_jsonl(workspace / TRACE_FILES["skill_actions"])
    scheduler_records = read_jsonl(workspace / TRACE_FILES["scheduler"])
    agent_path = workspace / TRACE_FILES["agent_events"]
    if not agent_path.is_file():
        raise FileNotFoundError(f"Agent event file is missing: {agent_path}")
    agent_events = json.loads(agent_path.read_text(encoding="utf-8"))
    if not isinstance(agent_events, list):
        raise ValueError(f"Agent event file must contain a list: {agent_path}")
    transport_path = workspace / TRACE_FILES["transport"]
    if not transport_path.is_file():
        raise FileNotFoundError(f"transport event file is missing: {transport_path}")
    transport_events = json.loads(transport_path.read_text(encoding="utf-8"))
    if not isinstance(transport_events, list):
        raise ValueError(f"transport event file must contain a list: {transport_path}")

    target_actions = [
        row
        for row in skill_actions
        if row.get("boundary") == "structured_skill_action_ready"
        and row.get("skill_name") == case["skill"]
    ]
    if len(target_actions) != 1:
        raise ValueError(
            f"expected one structured {case['skill']} action, "
            f"found {len(target_actions)}"
        )
    action = target_actions[0]
    request_a_id = str(action.get("request_id", ""))
    transport_by_id: dict[str, dict[str, Any]] = {}
    for record in transport_events:
        if not isinstance(record, dict):
            raise ValueError(f"transport event is not an object: {record!r}")
        request_id = str(record.get("request_id", ""))
        if not request_id or request_id in transport_by_id:
            raise ValueError(
                f"invalid or duplicate transport request_id: {request_id!r}"
            )
        if record.get("boundary") != "client_transport_response_received":
            raise ValueError(f"transport event has the wrong boundary: {request_id}")
        transport_by_id[request_id] = record
    response_a = transport_by_id.get(request_a_id)
    if response_a is None:
        raise ValueError(f"request A has no client response record: {request_a_id}")
    target_events = [
        row
        for row in agent_events
        if isinstance(row, dict)
        and row.get("execution_mode") == "normal_prefill"
        and row.get("skill") == case["skill"]
        and isinstance(row.get("schedule_timing"), dict)
        and row["schedule_timing"].get("tool_call_id") == action.get("tool_call_id")
    ]
    if len(target_events) != 1:
        raise ValueError(
            "expected one post-Skill request linked by tool_call_id, found "
            f"{len(target_events)}"
        )
    event = target_events[0]
    timing = event["schedule_timing"]
    request_b_id = str(event.get("request_id", ""))
    scheduler_by_id: dict[str, dict[str, Any]] = {}
    for record in scheduler_records:
        request_id = str(record.get("request_id", ""))
        if not request_id or request_id in scheduler_by_id:
            raise ValueError(
                f"invalid or duplicate scheduler request_id: {request_id!r}"
            )
        scheduler_by_id[request_id] = record
    admission = scheduler_by_id.get(request_b_id)
    if admission is None:
        raise ValueError(f"request B has no scheduler record: {request_b_id}")
    if admission.get("boundary") != "immediately_before_scheduler_add_request":
        raise ValueError(f"request B has the wrong T5 boundary: {request_b_id}")

    t0 = int(action["skill_action_ready_unix_ns"])
    t1 = int(response_a["client_response_received_unix_ns"])
    t2 = int(timing["observation_callback_unix_ns"])
    t3 = int(timing["request_wrapper_enter_unix_ns"])
    t4 = int(timing["client_transport_handoff_unix_ns"])
    t5 = int(admission["scheduler_admission_unix_ns"])
    if not t0 <= t1 <= t2 <= t3 <= t4 <= t5:
        raise ValueError(
            "timestamps must satisfy T0 <= T1 <= T2 <= T3 <= T4 <= T5: "
            f"{(t0, t1, t2, t3, t4, t5)}"
        )
    window_ms = (t5 - t0) / 1e6
    if window_ms <= 0:
        raise ValueError(f"prefetch window must be positive: {window_ms}")
    intervals_ms = {
        "t0_t1_ms": (t1 - t0) / 1e6,
        "t1_t2_ms": (t2 - t1) / 1e6,
        "t2_t3_ms": (t3 - t2) / 1e6,
        "t3_t4_ms": (t4 - t3) / 1e6,
        "t4_t5_ms": (t5 - t4) / 1e6,
    }
    if abs(sum(intervals_ms.values()) - window_ms) > 0.001:
        raise ValueError("T0-T5 interval sum disagrees with the complete window")
    return {
        "schema_version": 2,
        "status": "completed",
        "fingerprint": fingerprint,
        "task": case["task"],
        "skill": case["skill"],
        "skill_tokens": cache_info["token_count"],
        "cache_bytes": cache_info["cache_bytes"],
        "skill_cache_manifest": str(cache_info["manifest_path"]),
        "prompt_path": str(case["prompt_path"]),
        "prompt_sha256": sha256_path(case["prompt_path"]),
        "request_a_id": request_a_id,
        "request_b_id": request_b_id,
        "tool_call_id": action["tool_call_id"],
        "t0_unix_ns": t0,
        "t1_unix_ns": t1,
        "t2_unix_ns": t2,
        "t3_unix_ns": t3,
        "t4_unix_ns": t4,
        "t5_unix_ns": t5,
        **intervals_ms,
        "prefetch_window_ms": window_ms,
        "measurement_definition": {
            "t0": "vLLM has produced a complete structured Skill tool call",
            "t1": "OpenHands client has received request A's structured response",
            "t2": "the target Skill Observation callback has completed",
            "t3": "request B has entered the LiteLLM transport wrapper",
            "t4": "request B has been handed to the original LiteLLM transport",
            "t5": "immediately before request B calls scheduler.add_request",
            "metric": "prefetch_window_ms = (T5 - T0) / 1e6",
        },
    }


def stream_process(command: list[str], environment: dict[str, str], log: Path) -> int:
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
                sys.stdout.write(line)
                handle.write(line)
                handle.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise


def run_case(
    case: dict[str, Any],
    case_dir: Path,
    launcher: Path,
    max_iterations: int,
    cache_info: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    result_path = case_dir / "case_result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "completed"
            and previous.get("fingerprint") == fingerprint
        ):
            (case_dir / "case_failure.json").unlink(missing_ok=True)
            print(f"[skip] {case['task']} already completed")
            return previous

    workspace = case_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(launcher)]
    for skill in case["exposed_skills"]:
        command.extend(("--skill", skill))
    command.extend(
        (
            "--max-iterations",
            str(max_iterations),
            "--prompt-file",
            str(case["prompt_path"]),
        )
    )
    environment = os.environ.copy()
    environment["SEGMENTIA_MODE"] = "no_reuse"
    environment["OPENHANDS_WORKSPACE"] = str(workspace)
    print(f"[case] {case['task']} skill={case['skill']}")
    return_code = stream_process(command, environment, case_dir / "launcher.log")
    if return_code != 0:
        raise RuntimeError(f"launcher exited with code {return_code}")
    result = analyze_case(case, workspace, cache_info, fingerprint)
    atomic_write_json(result_path, result)
    (case_dir / "case_failure.json").unlink(missing_ok=True)
    print(
        f"[captured] {case['task']} "
        f"window={result['prefetch_window_ms']:.3f}ms"
    )
    return result


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]
) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fieldnames})
    atomic_write_text(path, buffer.getvalue())


def load_path_baseline(config: dict[str, Any]) -> dict[str, Any]:
    path = raw_run_dir(config) / "05_skill_kv_path.json"
    if not path.is_file():
        raise FileNotFoundError(f"Test 05 baseline is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    skill_cache = payload.get("skill_cache")
    measurements = payload.get("measurements")
    if not isinstance(skill_cache, dict) or not isinstance(measurements, list):
        raise ValueError(f"Test 05 baseline has an invalid schema: {path}")
    base_bytes = skill_cache.get("total_bytes")
    if (
        isinstance(base_bytes, bool)
        or not isinstance(base_bytes, int)
        or base_bytes <= 0
    ):
        raise ValueError(f"Test 05 total_bytes is invalid: {path}")
    cold = [row for row in measurements if row.get("state") == "cold"]
    warm = [row for row in measurements if row.get("state") == "warm"]
    if not cold or not warm:
        raise ValueError(f"Test 05 must contain cold and warm samples: {path}")

    def median(rows: list[dict[str, Any]], field: str) -> float:
        values = [float(row[field]) for row in rows]
        if any(value <= 0 for value in values):
            raise ValueError(f"Test 05 {field} contains a non-positive value")
        return float(statistics.median(values))

    return {
        "source": str(path),
        "source_sha256": sha256_path(path),
        "base_cache_bytes": base_bytes,
        "pinned_h2d_ms": median(warm, "h2d_wall_ms"),
        "warm_serial_ms": median(warm, "total_ms"),
        "cold_serial_ms": median(cold, "total_ms"),
        "estimate_method": "capacity_scaled_from_test_05_medians",
    }


def add_loading_estimates(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    base_bytes = float(baseline["base_cache_bytes"])
    for row in rows:
        ratio = float(row["cache_bytes"]) / base_bytes
        enriched.append(
            {
                **row,
                "estimated_pinned_h2d_ms": ratio
                * float(baseline["pinned_h2d_ms"]),
                "estimated_warm_serial_ms": ratio
                * float(baseline["warm_serial_ms"]),
                "estimated_cold_serial_ms": ratio
                * float(baseline["cold_serial_ms"]),
            }
        )
    return enriched


def plot_windows(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["prefetch_window_ms"]))
    labels = [f"{row['task']}  [{row['skill']}]" for row in ordered]
    values = [float(row["prefetch_window_ms"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(11.2, 7.0))
    bars = axis.barh(
        range(len(values)),
        values,
        color="#4C956C",
        edgecolor="black",
        linewidth=1.2,
    )
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Prefetch window T5 - T0 (ms)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    padding = max(values) * 0.015
    for bar, value in zip(bars, values):
        axis.text(
            value + padding,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, max(values) * 1.13)
    figure.tight_layout()
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"agent_skill_prefetch_windows.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_window_breakdown(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["prefetch_window_ms"]))
    labels = [str(row["skill"]) for row in ordered]
    segments = (
        (
            "t0_t1_ms",
            "T0-T1: vLLM returns the parsed SkillAction to OpenHands",
            "#264653",
            "white",
        ),
        (
            "t1_t2_ms",
            "T1-T2: OpenHands executes SkillTool and loads the Skill content",
            "#2A9D8F",
            "white",
        ),
        (
            "t2_t3_ms",
            "T2-T3: OpenHands merges tool results and assembles the next request",
            "#E9C46A",
            "black",
        ),
        (
            "t3_t4_ms",
            "T3-T4: OpenHands attaches request identification and timing metadata",
            "#F4A261",
            "black",
        ),
        (
            "t4_t5_ms",
            "T4-T5: OpenHands sends the request; vLLM templates, tokenizes, and "
            "prepares it for scheduling",
            "#E76F51",
            "white",
        ),
    )
    figure, axis = plt.subplots(figsize=(12.2, 7.2))
    left = [0.0] * len(ordered)
    for field, label, color, text_color in segments:
        values = [float(row[field]) for row in ordered]
        bars = axis.barh(
            range(len(ordered)),
            values,
            left=left,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.65,
        )
        for bar, start, value in zip(bars, left, values):
            narrow = value < 8.0
            axis.text(
                start + value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}" if narrow else f"{value:.1f}",
                ha="center",
                va="center",
                rotation=90 if narrow else 0,
                color=text_color,
                fontsize=6.5 if narrow else 8.0,
            )
        left = [start + value for start, value in zip(left, values)]
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Latency (ms)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    padding = max(left) * 0.012
    for index, value in enumerate(left):
        axis.text(value + padding, index, f"{value:.1f}", va="center", fontsize=9)
    axis.set_xlim(0, max(left) * 1.12)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.4,
    )
    figure.tight_layout()
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"agent_skill_prefetch_window_breakdown.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_loading_estimates(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["prefetch_window_ms"]))
    labels = [f"{row['task']}  [{row['skill']}]" for row in ordered]
    panels = (
        ("estimated_pinned_h2d_ms", "Pinned CPU to GPU"),
        ("estimated_warm_serial_ms", "Page-cache warm to GPU"),
        ("estimated_cold_serial_ms", "SSD cold to GPU (Gen2 x4)"),
    )
    positions = list(range(len(ordered)))
    height = 0.36
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 7.2),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    legend_handles = None
    for axis, (field, title) in zip(axes, panels):
        windows = [float(row["prefetch_window_ms"]) for row in ordered]
        estimates = [float(row[field]) for row in ordered]
        first = axis.barh(
            [position - height / 2 for position in positions],
            windows,
            height=height,
            label="Available window",
            color="#457B9D",
            edgecolor="black",
            linewidth=0.7,
        )
        second = axis.barh(
            [position + height / 2 for position in positions],
            estimates,
            height=height,
            label="Estimated KV load",
            color="#E9C46A",
            edgecolor="black",
            linewidth=0.7,
        )
        if legend_handles is None:
            legend_handles = (first[0], second[0])
        axis.set_title(title)
        axis.set_xlabel("Latency (ms)")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.set_xlim(0, max(windows + estimates) * 1.08)
    axes[0].set_yticks(positions, labels)
    if legend_handles is None:
        raise ValueError("loading estimate plot has no rows")
    figure.legend(
        legend_handles,
        ("Available window", "Estimated KV load"),
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(left=0.25, right=0.99, top=0.91, bottom=0.12)
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"agent_skill_kv_loading_estimates.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def update_source_manifest(
    path: Path, batch_root: Path, baseline_path: Path
) -> None:
    rows: list[dict[str, str]] = []
    if path.is_file():
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    artifacts = {
        "figures/agent_skill_kv_loading_estimates.pdf",
        "figures/agent_skill_kv_loading_estimates.png",
        "figures/agent_skill_prefetch_window_breakdown.pdf",
        "figures/agent_skill_prefetch_window_breakdown.png",
        "figures/agent_skill_prefetch_windows.pdf",
        "figures/agent_skill_prefetch_windows.png",
        "tables/agent_skill_kv_loading_estimates.csv",
        "tables/agent_skill_prefetch_window_breakdown.csv",
        "tables/agent_skill_prefetch_windows.csv",
        "data/agent_skill_schedule_batch_run_pointer.txt",
    }
    rows = [row for row in rows if row.get("artifact") not in artifacts]
    for artifact in sorted(artifacts):
        source = str(batch_root)
        if "kv_loading_estimates" in artifact:
            source = f"{batch_root};{baseline_path}"
        rows.append({"artifact": artifact, "source": source})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("artifact", "source"))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def main() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "opencode":
        raise RuntimeError("activate the opencode conda environment first")
    config = load_config()
    section = config["agent_schedule"]
    cases = load_cases(config)
    launcher = Path(section["launcher"]).resolve()
    if not launcher.is_file():
        raise FileNotFoundError(f"Agent launcher is missing: {launcher}")
    max_iterations = int(section["max_iterations"])
    if max_iterations != 2:
        raise ValueError("this experiment requires agent_schedule.max_iterations=2")
    batch_id = str(section["batch_id"]).strip()
    if not batch_id or "/" in batch_id:
        raise ValueError("agent_schedule.batch_id must be path-safe")
    batch_root = raw_run_dir(config) / "07_agent_skill_schedule_batch" / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)

    implementation_paths = (
        SCRIPT_PATH,
        ROOT
        / "scripts/08_lmcache_mp/paper_motivation/3.1/interactive_agent_no_reuse.py",
        launcher,
        ROOT / "vllm/vllm/entrypoints/openai/chat_completion/serving.py",
        ROOT / "vllm/vllm/v1/engine/core.py",
    )
    implementation_hashes = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in implementation_paths
    }
    pool_dir = Path(config["skill_cache"]["pool_dir"]).resolve()
    expected_layers = int(config["skill_cache"]["expected_layers"])
    path_baseline = load_path_baseline(config)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        cache_info = resolve_skill_cache(pool_dir, case["skill"], expected_layers)
        fingerprint = case_fingerprint(
            case, max_iterations, implementation_hashes, cache_info
        )
        case_dir = batch_root / case["task"]
        try:
            completed.append(
                run_case(
                    case,
                    case_dir,
                    launcher,
                    max_iterations,
                    cache_info,
                    fingerprint,
                )
            )
        except Exception as error:
            failure = {
                "task": case["task"],
                "skill": case["skill"],
                "error": str(error),
            }
            failures.append(failure)
            atomic_write_json(
                case_dir / "case_failure.json", {"status": "failed", **failure}
            )
            print(f"[failed] {case['task']}: {error}", file=sys.stderr)

    estimated_rows = add_loading_estimates(completed, path_baseline)
    atomic_write_json(
        batch_root / "batch_result.json",
        {
            "schema_version": 2,
            "batch_id": batch_id,
            "measurement": "prefetch_window_ms = (T5 - T0) / 1e6",
            "boundaries": {
                "t0": "structured SkillAction is ready in vLLM",
                "t1": "request A response is received by the OpenHands client",
                "t2": "target Skill Observation callback completes",
                "t3": "request B enters the LiteLLM transport wrapper",
                "t4": "request B is handed to the original LiteLLM transport",
                "t5": "immediately before scheduler.add_request(request B)",
            },
            "loading_estimate_baseline": path_baseline,
            "loading_estimate_semantics": (
                "capacity-scaled opportunity estimate from Test 05; no prefetch "
                "was executed in this batch"
            ),
            "completed": estimated_rows,
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} case(s) failed; rerun the same command to resume"
        )
    if len(completed) != len(cases):
        raise RuntimeError("batch completed count disagrees with configured cases")

    output = result_dir(config)
    figure_dir = output / "figures"
    table_dir = output / "tables"
    data_dir = output / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        table_dir / "agent_skill_prefetch_windows.csv",
        estimated_rows,
        WINDOW_CSV_FIELDS,
    )
    write_csv(
        table_dir / "agent_skill_prefetch_window_breakdown.csv",
        estimated_rows,
        BREAKDOWN_CSV_FIELDS,
    )
    write_csv(
        table_dir / "agent_skill_kv_loading_estimates.csv",
        estimated_rows,
        ESTIMATE_CSV_FIELDS,
    )
    plot_windows(estimated_rows, figure_dir)
    plot_window_breakdown(estimated_rows, figure_dir)
    plot_loading_estimates(estimated_rows, figure_dir)
    atomic_write_text(
        data_dir / "agent_skill_schedule_batch_run_pointer.txt",
        f"{batch_root}\n",
    )
    update_source_manifest(
        output / "source_manifest.csv",
        batch_root,
        Path(path_baseline["source"]),
    )
    print(f"[completed] batch={batch_root}")
    print(f"[plotted] {figure_dir / 'agent_skill_prefetch_windows.pdf'}")
    print(
        f"[plotted] {figure_dir / 'agent_skill_prefetch_window_breakdown.pdf'}"
    )
    print(f"[plotted] {figure_dir / 'agent_skill_kv_loading_estimates.pdf'}")


if __name__ == "__main__":
    main()
