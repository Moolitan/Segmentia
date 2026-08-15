#!/usr/bin/env python3
"""Measure vLLM's post-tokenization Skill-span locator on every task prompt.

This is a long-running experiment. Each task starts an independent vLLM server
through ``run_interactive_agent.sh`` and runs exactly two Agent iterations:
request A produces SkillAction, then request B carries the loaded Skill and uses
direct KV reuse. All measured boundaries are recorded inside vLLM and joined by
the source tool-call ID: T0 is parsed SkillAction, T1 is request-B handler entry,
T2 is completed chat tokenization, and T3 is scheduler admission.
"""
from __future__ import annotations

import argparse
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
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
}
CSV_FIELDS = (
    "task",
    "skill",
    "skill_tokens",
    "t0_t1_ms",
    "t1_t2_ms",
    "t2_t3_ms",
    "prefetch_window_ms",
)
TYPICAL_EXCLUDED_SKILLS = frozenset({"docx", "writing-systems-papers"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or reanalyze the configured Agent Skill locator batch."
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "reanalyze the configured batch's existing trace files without "
            "starting vLLM or OpenHands"
        ),
    )
    return parser.parse_args(argv)


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
        if any(not item for item in exposed_skills) or len(
            set(exposed_skills)
        ) != len(exposed_skills):
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
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
                # Keep draining stdout and preserve the complete launcher log,
                # but do not mirror large Agent observations to the terminal.
                # Terminal rendering/backpressure is not part of the scheduling
                # window that this batch is intended to measure.
                handle.write(line)
                handle.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise


def unique_record(
    records: list[dict[str, Any]],
    predicate: Any,
    description: str,
) -> dict[str, Any]:
    matched = [record for record in records if predicate(record)]
    if len(matched) != 1:
        raise ValueError(f"expected one {description}, found {len(matched)}")
    return matched[0]


def validate_locator_manifest(cache_info: dict[str, Any]) -> None:
    manifest_path = Path(cache_info["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise ValueError(
            f"direct-reuse locator experiment requires a CSKCache source object: "
            f"{manifest_path}"
        )
    if not manifest.get("start_marker_token_ids"):
        raise ValueError(f"unsupported Skill locator manifest: {manifest_path}")


def analyze_case(
    case: dict[str, Any],
    workspace: Path,
    cache_info: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    actions = read_jsonl(workspace / TRACE_FILES["skill_actions"])
    locators = read_jsonl(workspace / TRACE_FILES["locator"])
    scheduler = read_jsonl(workspace / TRACE_FILES["scheduler"])

    locator = unique_record(
        locators,
        lambda row: row.get("skill_name") == case["skill"]
        and row.get("status") == "ok",
        "successful request B Skill locator trace",
    )
    source_tool_call_id = locator.get("source_tool_call_id")
    if not isinstance(source_tool_call_id, str) or not source_tool_call_id:
        raise ValueError("request B locator has no source_tool_call_id")
    action = unique_record(
        actions,
        lambda row: row.get("boundary") == "structured_skill_action_ready"
        and row.get("skill_name") == case["skill"]
        and row.get("tool_call_id") == source_tool_call_id,
        f"executed structured {case['skill']} action",
    )
    request_a_id = str(action.get("request_id", ""))
    request_b_id = str(locator.get("request_id", ""))
    admission = unique_record(
        scheduler,
        lambda row: row.get("request_id") == request_b_id
        and row.get("source_tool_call_id") == source_tool_call_id
        and row.get("boundary") == "immediately_before_scheduler_add_request",
        "request B scheduler admission",
    )

    boot_ids = {action.get("boot_id"), locator.get("boot_id"), admission.get("boot_id")}
    if len(boot_ids) != 1 or not next(iter(boot_ids)):
        raise ValueError(f"T0--T3 boot IDs disagree: {boot_ids}")

    t0 = int(action["skill_action_ready_monotonic_ns"])
    t1 = int(locator["request_received_monotonic_ns"])
    t2 = int(locator["tokenization_completed_monotonic_ns"])
    t3 = int(admission["scheduler_admission_monotonic_ns"])
    if not t0 <= t1 <= t2 <= t3:
        raise ValueError(
            "timestamps must satisfy T0 <= T1 <= T2 <= T3: "
            f"{(t0, t1, t2, t3)}"
        )
    intervals = {
        "t0_t1_ms": (t1 - t0) / 1e6,
        "t1_t2_ms": (t2 - t1) / 1e6,
        "t2_t3_ms": (t3 - t2) / 1e6,
    }
    window_ms = (t3 - t0) / 1e6
    if abs(sum(intervals.values()) - window_ms) > 0.001:
        raise ValueError("T0--T3 interval sum disagrees with the complete window")
    return {
        "schema_version": 2,
        "status": "completed",
        "fingerprint": fingerprint,
        "task": case["task"],
        "skill": case["skill"],
        "skill_tokens": cache_info["token_count"],
        "cache_bytes": cache_info["cache_bytes"],
        "skill_cache_manifest": str(cache_info["manifest_path"]),
        "request_a_id": request_a_id,
        "request_b_id": request_b_id,
        "source_tool_call_id": source_tool_call_id,
        "boot_id": next(iter(boot_ids)),
        "segment_start": int(locator["segment_start"]),
        "segment_end": int(locator["segment_end"]),
        "t0_monotonic_ns": t0,
        "t1_monotonic_ns": t1,
        "t2_monotonic_ns": t2,
        "t3_monotonic_ns": t3,
        **intervals,
        "prefetch_window_ms": window_ms,
        "measurement_definition": {
            "t0": "vLLM parsed request A's complete Skill tool call",
            "t1": "vLLM chat handler received request B",
            "t2": "vLLM completed request B chat template and tokenization",
            "t3": "vLLM EngineCore immediately before scheduler.add_request",
        },
    }


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
        if previous.get("status") == "completed" and previous.get(
            "fingerprint"
        ) == fingerprint:
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
    environment["SEGMENTIA_MODE"] = "direct_reuse"
    environment["SEGMENTIA_DISABLE_VISUALIZER"] = "1"
    environment["OPENHANDS_WORKSPACE"] = str(workspace)
    print(f"[case] {case['task']} skill={case['skill']}")
    return_code = stream_process(command, environment, case_dir / "launcher.log")
    if return_code != 0:
        raise RuntimeError(f"launcher exited with code {return_code}")
    result = analyze_case(case, workspace, cache_info, fingerprint)
    atomic_write_json(result_path, result)
    print(
        f"[captured] {case['task']} "
        f"T0-T3={result['prefetch_window_ms']:.3f}ms"
    )
    return result


def analyze_existing_case(
    case: dict[str, Any],
    case_dir: Path,
    cache_info: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Rebuild one result from an existing workspace without model execution."""
    workspace = case_dir / "workspace"
    if not workspace.is_dir():
        raise FileNotFoundError(f"existing case workspace is missing: {workspace}")
    result = analyze_case(case, workspace, cache_info, fingerprint)
    atomic_write_json(case_dir / "case_result.json", result)
    failure_path = case_dir / "case_failure.json"
    if failure_path.is_file():
        failure_path.unlink()
    print(
        f"[reanalyzed] {case['task']} "
        f"T0-T3={result['prefetch_window_ms']:.3f}ms"
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    atomic_write_text(path, buffer.getvalue())


def save_figure(figure: Any, figure_dir: Path, stem: str) -> None:
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)


def plot_breakdown(
    rows: list[dict[str, Any]], figure_dir: Path, output_stem: str
) -> None:
    ordered = sorted(rows, key=lambda row: float(row["prefetch_window_ms"]))
    labels = [str(row["skill"]) for row in ordered]
    segments = (
        (
            "t0_t1_ms",
            "Parsed SkillAction to request-B arrival",
            "#264653",
            "white",
        ),
        ("t1_t2_ms", "vLLM chat template and tokenization", "#2A9D8F", "white"),
        ("t2_t3_ms", "vLLM post-tokenization to scheduler", "#E9C46A", "black"),
    )
    figure, axis = plt.subplots(figsize=(12.6, 7.2))
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
            linewidth=0.55,
        )
        for bar, offset, value in zip(bars, left, values):
            if value < 1.0:
                continue
            axis.text(
                offset + value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.5,
            )
        left = [offset + value for offset, value in zip(left, values)]
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Latency (ms)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    axis.set_xlim(0, max(left) * 1.04)
    figure.tight_layout()
    save_figure(figure, figure_dir, output_stem)


def update_source_manifest(path: Path, batch_root: Path) -> None:
    rows: list[dict[str, str]] = []
    if path.is_file():
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    artifacts = {
        "figures/agent_skill_prefetch_window_vllm_boundaries.pdf",
        "figures/agent_skill_prefetch_window_vllm_boundaries.png",
        "figures/agent_skill_prefetch_window_vllm_boundaries_typical.pdf",
        "figures/agent_skill_prefetch_window_vllm_boundaries_typical.png",
        "tables/agent_skill_prefetch_window_vllm_boundaries.csv",
        "tables/agent_skill_prefetch_window_vllm_boundaries_typical.csv",
        "data/agent_skill_locator_batch_run_pointer.txt",
    }
    retired_artifacts = {
        "figures/agent_skill_span_location_latency.pdf",
        "figures/agent_skill_span_location_latency.png",
        "figures/agent_skill_prefetch_window_with_locator.pdf",
        "figures/agent_skill_prefetch_window_with_locator.png",
        "tables/agent_skill_span_location.csv",
    }
    rows = [
        row
        for row in rows
        if row.get("artifact") not in artifacts | retired_artifacts
    ]
    rows.extend(
        {"artifact": artifact, "source": str(batch_root)}
        for artifact in sorted(artifacts)
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("artifact", "source"))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if os.environ.get("CONDA_DEFAULT_ENV") != "opencode":
        raise RuntimeError("activate the opencode conda environment first")
    config = load_config()
    section = config["agent_locator"]
    cases = load_cases(config)
    launcher = Path(config["agent_schedule"]["launcher"]).resolve()
    max_iterations = int(section["max_iterations"])
    if max_iterations != 2:
        raise ValueError("agent_locator.max_iterations must be 2")
    batch_id = str(section["batch_id"]).strip()
    if not batch_id or "/" in batch_id:
        raise ValueError("agent_locator.batch_id must be path-safe")
    batch_root = raw_run_dir(config) / "09_agent_skill_locator_batch" / batch_id
    if args.analysis_only:
        if not batch_root.is_dir():
            raise FileNotFoundError(
                f"existing locator batch is missing: {batch_root}"
            )
    else:
        batch_root.mkdir(parents=True, exist_ok=True)

    implementation_paths = (
        SCRIPT_PATH,
        ROOT / "scripts/08_lmcache_mp/paper_motivation/3.1/interactive_agent.py",
        launcher,
        ROOT / "vllm/vllm/entrypoints/openai/chat_completion/serving.py",
        ROOT / "vllm/vllm/v1/engine/core.py",
    )
    implementation_hashes = {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in implementation_paths
    }
    pool_dir = Path(config["skill_cache"]["pool_dir"]).resolve()
    expected_layers = int(config["skill_cache"]["expected_layers"])
    cache_by_task: dict[str, dict[str, Any]] = {}
    for case in cases:
        cache_info = resolve_skill_cache(
            pool_dir, case["skill"], expected_layers
        )
        validate_locator_manifest(cache_info)
        cache_by_task[case["task"]] = cache_info

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        cache_info = cache_by_task[case["task"]]
        fingerprint = case_fingerprint(
            case, max_iterations, implementation_hashes, cache_info
        )
        case_dir = batch_root / case["task"]
        try:
            if args.analysis_only:
                result = analyze_existing_case(
                    case,
                    case_dir,
                    cache_info,
                    fingerprint,
                )
            else:
                result = run_case(
                    case,
                    case_dir,
                    launcher,
                    max_iterations,
                    cache_info,
                    fingerprint,
                )
            completed.append(result)
        except Exception as error:
            failure = {
                "task": case["task"],
                "skill": case["skill"],
                "error": str(error),
            }
            failures.append(failure)
            atomic_write_json(case_dir / "case_failure.json", failure)
            print(f"[failed] {case['task']}: {error}", file=sys.stderr)

    summary = {
        "schema_version": 2,
        "batch_id": batch_id,
        "completed": len(completed),
        "failures": failures,
    }
    if completed:
        values = [float(row["prefetch_window_ms"]) for row in completed]
        summary["prefetch_window_ms"] = {
            "minimum": min(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }
    atomic_write_json(batch_root / "batch_summary.json", summary)
    if failures or len(completed) != len(cases):
        raise RuntimeError(
            f"batch incomplete: completed={len(completed)} failures={len(failures)}"
        )

    output_dir = result_dir(config)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    for directory in (figure_dir, table_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        table_dir / "agent_skill_prefetch_window_vllm_boundaries.csv",
        completed,
    )
    plot_breakdown(
        completed,
        figure_dir,
        "agent_skill_prefetch_window_vllm_boundaries",
    )
    typical = [
        row for row in completed if row["skill"] not in TYPICAL_EXCLUDED_SKILLS
    ]
    excluded = {row["skill"] for row in completed} - {
        row["skill"] for row in typical
    }
    if excluded != TYPICAL_EXCLUDED_SKILLS or len(typical) != 11:
        raise ValueError(
            "typical-case filter must exclude exactly docx and "
            f"writing-systems-papers and retain 11 rows; excluded={excluded}, "
            f"retained={len(typical)}"
        )
    write_csv(
        table_dir / "agent_skill_prefetch_window_vllm_boundaries_typical.csv",
        typical,
    )
    plot_breakdown(
        typical,
        figure_dir,
        "agent_skill_prefetch_window_vllm_boundaries_typical",
    )
    atomic_write_text(
        data_dir / "agent_skill_locator_batch_run_pointer.txt",
        str(batch_root) + "\n",
    )
    update_source_manifest(output_dir / "source_manifest.csv", batch_root)
    print(f"[completed] batch={batch_root} cases={len(completed)}")


if __name__ == "__main__":
    main()
