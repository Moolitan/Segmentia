#!/usr/bin/env python3
"""Validate and summarize real-Agent Recompute versus CSKCache latency."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from config import MEASURE_CASES, MODES, REPLICAS, RESULT_DIR, WARMUP_CASES


ESSENTIAL_VLLM_EVENTS = (
    "api_request_received",
    "render_tokenize_start",
    "render_tokenize_complete",
    "scheduler_add_request",
    "first_token_ready",
    "api_response_ready",
)
ESSENTIAL_CSK_EVENTS = (
    "csk_t0_prefetch_begin",
    "csk_t0_prefetch_submit",
    "csk_host_read_start",
    "csk_host_read_complete",
    "csk_host_ready",
    "csk_request_bind",
    "csk_reuse_plan",
    "csk_reuse_boundary_ready",
    "csk_reuse_scheduler_activate",
    "cskcache_h2d_breakdown",
    "csk_reuse_gpu_breakdown",
    "csk_worker_load_complete",
)
MAX_PAIRED_ADDED_TOKEN_DELTA = 8
EXPECTED_CSK_LAYERS = 40


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exactly_one(
    records: Iterable[dict[str, Any]], event: str
) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == event]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {event}, found {len(matches)}")
    return matches[0]


def belongs_to_request(record_request_id: Any, request_id: str) -> bool:
    """Match one API request and its vLLM-internal derived request IDs.

    The OpenAI-facing request keeps ``request_id`` exactly.  EngineCore appends
    a hyphen and an internal suffix.  Matching the full case ID is incorrect:
    request A and request B intentionally share that case ID.
    """
    candidate = str(record_request_id or "")
    return candidate == request_id or candidate.startswith(f"{request_id}-")


def paired_prompt_deltas(
    recompute: dict[str, Any], cskcache: dict[str, Any]
) -> tuple[int, int]:
    """Validate comparable request-B growth while exposing static path drift."""
    request_a_delta = int(cskcache["request_a_prompt_tokens"]) - int(
        recompute["request_a_prompt_tokens"]
    )
    request_b_added_delta = int(cskcache["request_b_added_tokens"]) - int(
        recompute["request_b_added_tokens"]
    )
    if abs(request_b_added_delta) > MAX_PAIRED_ADDED_TOKEN_DELTA:
        raise ValueError(
            "paired request-B content differs by "
            f"{request_b_added_delta} tokens; allowed absolute delta is "
            f"{MAX_PAIRED_ADDED_TOKEN_DELTA}"
        )
    return request_a_delta, request_b_added_delta


def delta_ms(end: dict[str, Any], start: dict[str, Any]) -> float:
    if end.get("boot_id") != start.get("boot_id"):
        raise ValueError("timing records came from different boot clock domains")
    value = (int(end["monotonic_ns"]) - int(start["monotonic_ns"])) / 1e6
    if value < 0:
        raise ValueError("timing boundary order is negative")
    return value


def same_host_delta_ms(end: dict[str, Any], start: dict[str, Any]) -> float:
    """Compare same-host monotonic timestamps when an old trace lacks boot_id."""
    end_boot = end.get("boot_id")
    start_boot = start.get("boot_id")
    if end_boot is not None and start_boot is not None and end_boot != start_boot:
        raise ValueError("timing records came from different boot clock domains")
    value = (int(end["monotonic_ns"]) - int(start["monotonic_ns"])) / 1e6
    if value < 0:
        raise ValueError("timing boundary order is negative")
    return value


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _validated_layer_intervals(
    values: Iterable[dict[str, Any]],
    *,
    label: str,
    expected_layers: int,
) -> list[dict[str, float | int | str]]:
    """Validate one complete, uniquely indexed set of CUDA intervals."""
    by_layer: dict[int, dict[str, float | int | str]] = {}
    for value in values:
        layer = int(value["layer"])
        if layer in by_layer:
            raise ValueError(f"duplicate {label} interval for layer {layer}")
        start = float(value["start_ms"])
        end = float(value["end_ms"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(
                f"invalid {label} CUDA interval for layer {layer}: "
                f"[{start}, {end}]"
            )
        by_layer[layer] = {
            "kind": label,
            "layer": layer,
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start,
        }
    expected = set(range(expected_layers))
    actual = set(by_layer)
    if actual != expected:
        raise ValueError(
            f"{label} CUDA timeline layers differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return [by_layer[layer] for layer in range(expected_layers)]


def _interval_union_ms(
    intervals: Iterable[dict[str, float | int | str]],
) -> float:
    ordered = sorted(
        (float(item["start_ms"]), float(item["end_ms"]))
        for item in intervals
    )
    if not ordered:
        return 0.0
    union = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union += current_end - current_start
            current_start, current_end = start, end
    return union + current_end - current_start


def parse_gpu_pipeline(
    related: list[dict[str, Any]],
    gpu_breakdown: dict[str, Any],
    worker_complete: dict[str, Any],
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    """Build one request's strict shared-clock layerwise GPU timeline."""
    if gpu_breakdown.get("shared_cuda_timeline") is not True:
        raise ValueError(
            "CSKCache trace lacks the shared CUDA timeline; rerun the latency "
            "experiment with the current profiler"
        )
    expected_layers = int(worker_complete.get("layers", 0))
    if expected_layers != EXPECTED_CSK_LAYERS:
        raise ValueError(
            f"expected {EXPECTED_CSK_LAYERS} CSKCache layers, "
            f"found {expected_layers}"
        )
    h2d = _validated_layer_intervals(
        (
            {
                "layer": record["layer"],
                "start_ms": record["start_ms"],
                "end_ms": record["end_ms"],
            }
            for record in related
            if record.get("event") == "cskcache_h2d_layer"
        ),
        label="H2D",
        expected_layers=expected_layers,
    )
    correction = _validated_layer_intervals(
        gpu_breakdown.get("correction_per_layer") or (),
        label="K correction",
        expected_layers=expected_layers,
    )
    commit = _validated_layer_intervals(
        gpu_breakdown.get("commit_per_layer") or (),
        label="KV commit",
        expected_layers=expected_layers,
    )

    pair_overlap_ms = 0.0
    overlap_capacity_ms = 0.0
    for layer in range(expected_layers - 1):
        next_h2d = h2d[layer + 1]
        current_compute = (correction[layer], commit[layer])
        compute_duration = sum(
            float(item["duration_ms"]) for item in current_compute
        )
        overlap_capacity_ms += min(
            float(next_h2d["duration_ms"]), compute_duration
        )
        for compute in current_compute:
            pair_overlap_ms += max(
                0.0,
                min(float(next_h2d["end_ms"]), float(compute["end_ms"]))
                - max(
                    float(next_h2d["start_ms"]),
                    float(compute["start_ms"]),
                ),
            )

    intervals = [*h2d, *correction, *commit]
    serialized_ms = sum(float(item["duration_ms"]) for item in intervals)
    active_union_ms = _interval_union_ms(intervals)
    pipeline_start = min(float(item["start_ms"]) for item in intervals)
    pipeline_end = max(float(item["end_ms"]) for item in intervals)
    pipeline_span_ms = pipeline_end - pipeline_start
    metrics = {
        "gpu_pair_overlap_ms": pair_overlap_ms,
        "gpu_overlap_capacity_ms": overlap_capacity_ms,
        "gpu_overlap_ratio": (
            pair_overlap_ms / overlap_capacity_ms
            if overlap_capacity_ms > 0
            else 0.0
        ),
        "gpu_serialized_component_ms": serialized_ms,
        "gpu_component_union_ms": active_union_ms,
        "gpu_concurrency_saving_ms": serialized_ms - active_union_ms,
        "gpu_pipeline_span_ms": pipeline_span_ms,
        "gpu_pipeline_vs_serial_saving_ms": serialized_ms - pipeline_span_ms,
    }
    return intervals, metrics


def collect_case(
    case_dir: Path,
    mode: str,
    vllm_records: list[dict[str, Any]],
    csk_records: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    case_id = metadata["case_id"]
    agent_records = read_jsonl(case_dir / "agent_timeline.jsonl")
    starts = [
        record
        for record in agent_records
        if record.get("event") == "client_request_start"
        and record.get("post_skill") is True
    ]
    if len(starts) != 1:
        raise ValueError(f"{case_id}: expected one post-Skill request B")
    client_start = starts[0]
    request_id = str(client_start["request_id"])
    request_a_starts = [
        record
        for record in agent_records
        if record.get("event") == "client_request_start"
        and record.get("post_skill") is False
    ]
    if len(request_a_starts) != 1:
        raise ValueError(f"{case_id}: expected one pre-Skill request A")
    request_a_id = str(request_a_starts[0]["request_id"])
    responses = [
        record
        for record in agent_records
        if record.get("event") == "client_response_received"
        and record.get("request_id") == request_id
    ]
    if len(responses) != 1:
        raise ValueError(f"{case_id}: missing client response boundary")
    client_response = responses[0]
    request_records = [
        record
        for record in vllm_records
        if belongs_to_request(record.get("request_id"), request_id)
    ]
    events = {
        event: exactly_one(request_records, event) for event in ESSENTIAL_VLLM_EVENTS
    }
    request_a_records = [
        record
        for record in vllm_records
        if belongs_to_request(record.get("request_id"), request_a_id)
    ]
    request_a_tokenize = exactly_one(
        request_a_records, "render_tokenize_complete"
    )
    request_a_prompt_tokens = int(request_a_tokenize["prompt_tokens"])
    prompt_tokens = int(events["render_tokenize_complete"]["prompt_tokens"])
    request_b_added_tokens = prompt_tokens - request_a_prompt_tokens
    if request_b_added_tokens <= 0:
        raise ValueError(f"{case_id}: request B did not extend request A")
    if int(events["first_token_ready"]["prompt_tokens"]) != prompt_tokens:
        raise ValueError(f"{case_id}: prompt length changed inside vLLM")

    observations = client_start.get("skill_observations") or []
    if len(observations) != 1:
        raise ValueError(f"{case_id}: request B must follow exactly one Skill result")
    ticket = str(observations[0]["tool_call_id"])
    row: dict[str, Any] = {
        **metadata,
        "request_id": request_id,
        "request_a_id": request_a_id,
        "ticket": ticket,
        "request_a_prompt_tokens": request_a_prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "request_b_added_tokens": request_b_added_tokens,
        "cached_tokens": int(events["first_token_ready"].get("cached_tokens", 0)),
        "client_to_first_token_ms": delta_ms(
            events["first_token_ready"], client_start
        ),
        "api_ttft_ms": delta_ms(
            events["first_token_ready"], events["api_request_received"]
        ),
        "client_roundtrip_ms": delta_ms(client_response, client_start),
        "request_delivery_ms": delta_ms(
            events["api_request_received"], client_start
        ),
        "render_tokenize_ms": delta_ms(
            events["render_tokenize_complete"], events["render_tokenize_start"]
        ),
        "post_tokenize_to_scheduler_ms": delta_ms(
            events["scheduler_add_request"], events["render_tokenize_complete"]
        ),
        "scheduler_to_first_token_ms": delta_ms(
            events["first_token_ready"], events["scheduler_add_request"]
        ),
        "response_after_first_token_ms": delta_ms(
            events["api_response_ready"], events["first_token_ready"]
        ),
        "response_delivery_ms": delta_ms(
            client_response, events["api_response_ready"]
        ),
    }

    if mode == "cskcache":
        related = [
            record
            for record in csk_records
            if record.get("request_id") == ticket
            or record.get("ticket") == ticket
            or belongs_to_request(record.get("request_id"), request_id)
        ]
        csk = {event: exactly_one(related, event) for event in ESSENTIAL_CSK_EVENTS}
        if csk["csk_t0_prefetch_submit"].get("accepted") is not True:
            raise ValueError(f"{case_id}: T0 prefetch was not accepted")
        if csk["csk_reuse_plan"].get("accepted") is not True:
            raise ValueError(f"{case_id}: reuse plan was not accepted")
        if int(csk["csk_reuse_scheduler_activate"].get("external_tokens", 0)) <= 0:
            raise ValueError(f"{case_id}: no external Skill KV was activated")
        try:
            gpu_intervals, gpu_metrics = parse_gpu_pipeline(
                related,
                csk["csk_reuse_gpu_breakdown"],
                csk["csk_worker_load_complete"],
            )
        except ValueError as error:
            raise ValueError(f"{case_id}: {error}") from error
        row.update(
            {
                "host_buffer_acquire_ms": delta_ms(
                    exactly_one(related, "csk_host_buffer_acquire_complete"),
                    exactly_one(related, "csk_host_buffer_acquire_start"),
                ),
                "ssd_to_pinned_ms": delta_ms(
                    csk["csk_host_read_complete"], csk["csk_host_read_start"]
                ),
                "t0_to_host_ready_ms": delta_ms(
                    csk["csk_host_ready"], csk["csk_t0_prefetch_begin"]
                ),
                "pure_h2d_gpu_ms": float(
                    csk["cskcache_h2d_breakdown"]["pure_h2d_gpu_ms"]
                ),
                "correction_gpu_ms": float(
                    csk["csk_reuse_gpu_breakdown"]["correction_gpu_ms"]
                ),
                "commit_gpu_ms": float(
                    csk["csk_reuse_gpu_breakdown"]["commit_gpu_ms"]
                ),
                "external_tokens": int(
                    csk["csk_reuse_scheduler_activate"]["external_tokens"]
                ),
                # All offsets below share C0 as their origin.  Keeping the
                # original boundaries in the per-request table makes the
                # overlap result auditable instead of reconstructing a
                # synthetic timeline from independently aggregated medians.
                "t0_to_host_read_start_ms": delta_ms(
                    csk["csk_host_read_start"], csk["csk_t0_prefetch_begin"]
                ),
                "t0_to_client_request_start_ms": same_host_delta_ms(
                    client_start, csk["csk_t0_prefetch_begin"]
                ),
                "t0_to_api_receive_ms": same_host_delta_ms(
                    events["api_request_received"], csk["csk_t0_prefetch_begin"]
                ),
                "t0_to_host_ready_offset_ms": delta_ms(
                    csk["csk_host_ready"], csk["csk_t0_prefetch_begin"]
                ),
                "t0_to_render_tokenize_complete_ms": same_host_delta_ms(
                    events["render_tokenize_complete"],
                    csk["csk_t0_prefetch_begin"],
                ),
                "t0_to_request_bind_ms": delta_ms(
                    csk["csk_request_bind"], csk["csk_t0_prefetch_begin"]
                ),
                "t0_to_scheduler_add_ms": same_host_delta_ms(
                    events["scheduler_add_request"],
                    csk["csk_t0_prefetch_begin"],
                ),
                "t0_to_reuse_boundary_ms": delta_ms(
                    csk["csk_reuse_boundary_ready"],
                    csk["csk_t0_prefetch_begin"],
                ),
                "t0_to_reuse_activate_ms": delta_ms(
                    csk["csk_reuse_scheduler_activate"],
                    csk["csk_t0_prefetch_begin"],
                ),
                "t0_to_worker_complete_ms": delta_ms(
                    csk["csk_worker_load_complete"],
                    csk["csk_t0_prefetch_begin"],
                ),
                "t0_to_first_token_ms": same_host_delta_ms(
                    events["first_token_ready"], csk["csk_t0_prefetch_begin"]
                ),
                **gpu_metrics,
                "_gpu_pipeline_intervals": gpu_intervals,
            }
        )
        read_start = row["t0_to_host_read_start_ms"]
        host_ready = row["t0_to_host_ready_offset_ms"]
        agent_end = row["t0_to_client_request_start_ms"]
        overlap_ms = max(0.0, min(host_ready, agent_end) - read_start)
        row["ssd_agent_overlap_ms"] = overlap_ms
        row["ssd_agent_overlap_ratio"] = overlap_ms / row["ssd_to_pinned_ms"]
        row["host_ready_after_client_send_ms"] = host_ready - agent_end
        row["host_ready_before_reuse_boundary_ms"] = (
            row["t0_to_reuse_boundary_ms"] - host_ready
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(
        {key for row in rows for key in row if not key.startswith("_")}
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fields}
            for row in rows
        )


def make_figure(
    summary_rows: list[dict[str, Any]], output: Path, sample_kind: str
) -> None:
    import matplotlib.pyplot as plt

    labels = ["Recompute", "CSKCache"]
    medians = [
        next(row["median_api_ttft_ms"] for row in summary_rows if row["mode"] == mode)
        for mode in MODES
    ]
    p95s = [
        next(row["p95_api_ttft_ms"] for row in summary_rows if row["mode"] == mode)
        for mode in MODES
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = range(len(labels))
    bars = ax.bar(x, medians, color=["#9aa0a6", "#2a9d8f"], edgecolor="black", linewidth=1.5)
    ax.errorbar(x, medians, yerr=[[0, 0], [p95s[i] - medians[i] for i in x]], fmt="none", color="black", capsize=5)
    for bar, value in zip(bars, medians, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} ms", ha="center", va="bottom")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Request-B TTFT (ms)")
    if sample_kind == "warmup":
        ax.set_title("First request after each vLLM restart")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_key_timeline_figure(
    rows: list[dict[str, Any]], output: Path, sample_kind: str
) -> None:
    """Plot the few boundaries needed to explain host-load hiding."""
    import matplotlib.pyplot as plt

    csk_rows = [row for row in rows if row["mode"] == "cskcache"]
    median = lambda field: statistics.median(  # noqa: E731
        float(row[field]) for row in csk_rows
    )

    t0 = 0.0
    read_start = median("t0_to_host_read_start_ms")
    client_send = median("t0_to_client_request_start_ms")
    api_receive = median("t0_to_api_receive_ms")
    host_ready = median("t0_to_host_ready_offset_ms")
    prepared = median("t0_to_render_tokenize_complete_ms")
    bound = median("t0_to_request_bind_ms")
    scheduled = median("t0_to_scheduler_add_ms")
    reuse_boundary = median("t0_to_reuse_boundary_ms")
    reuse_activate = median("t0_to_reuse_activate_ms")
    worker_complete = median("t0_to_worker_complete_ms")
    first_token = median("t0_to_first_token_ms")

    fig, (timeline_ax, overlap_ax) = plt.subplots(
        1, 2, figsize=(12.6, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    lanes = [
        ("SSD to pinned CPU", read_start, host_ready, "#4c78a8"),
        ("OpenHands processing", t0, client_send, "#72b7b2"),
        ("Request delivery", client_send, api_receive, "#f2cf5b"),
        ("vLLM request preparation", api_receive, max(prepared, bound), "#f28e2b"),
        ("Admission and prefix Prefill", scheduled, reuse_boundary, "#b6992d"),
        ("KV load, correction, and commit", reuse_activate, worker_complete, "#59a14f"),
        ("Remaining Prefill to first token", worker_complete, first_token, "#af7aa1"),
    ]
    for index, (label, start, end, color) in enumerate(lanes):
        timeline_ax.barh(
            index,
            end - start,
            left=start,
            height=0.58,
            color=color,
            edgecolor="black",
            linewidth=1.1,
        )
        timeline_ax.text(
            (start + end) / 2,
            index,
            f"{end - start:.1f}",
            ha="center",
            va="center",
            fontsize=8.5,
        )
    timeline_ax.set_yticks(range(len(lanes)), [lane[0] for lane in lanes])
    timeline_ax.invert_yaxis()
    timeline_ax.set_xlabel("Time since SkillAction parsing, T0 (ms)")
    timeline_kind = "non-steady-state " if sample_kind == "warmup" else ""
    timeline_ax.set_title(f"(a) Median {timeline_kind}CSKCache request timeline")
    timeline_ax.grid(axis="x", alpha=0.25)

    ordered = sorted(csk_rows, key=lambda row: float(row["ssd_agent_overlap_ratio"]))
    y = list(range(len(ordered)))
    agent_ends = [float(row["t0_to_client_request_start_ms"]) for row in ordered]
    read_starts = [float(row["t0_to_host_read_start_ms"]) for row in ordered]
    read_ends = [float(row["t0_to_host_ready_offset_ms"]) for row in ordered]
    overlap_ax.barh(
        y,
        agent_ends,
        height=0.60,
        color="#72b7b2",
        edgecolor="black",
        linewidth=0.8,
        label="OpenHands processing",
    )
    overlap_ax.barh(
        y,
        [end - start for start, end in zip(read_starts, read_ends, strict=True)],
        left=read_starts,
        height=0.30,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.8,
        label="SSD to pinned CPU",
    )
    overlap_ax.set_yticks(y, [f"Request {index + 1}" for index in y])
    overlap_ax.set_xlabel("Time since SkillAction parsing, T0 (ms)")
    overlap_ax.set_title(f"(b) Host-load overlap across {len(csk_rows)} requests")
    overlap_ax.grid(axis="x", alpha=0.25)
    overlap_ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_gpu_breakdown_figure(
    rows: list[dict[str, Any]], output: Path
) -> None:
    """Show aggregate measured GPU work and remaining reuse-stage wall time."""
    import matplotlib.pyplot as plt

    csk_rows = [row for row in rows if row["mode"] == "cskcache"]
    median = lambda field: statistics.median(  # noqa: E731
        float(row[field]) for row in csk_rows
    )
    h2d = median("pure_h2d_gpu_ms")
    correction = median("correction_gpu_ms")
    commit = median("commit_gpu_ms")
    other = statistics.median(
        max(
            0.0,
            float(row["t0_to_worker_complete_ms"])
            - float(row["t0_to_reuse_activate_ms"])
            - float(row["pure_h2d_gpu_ms"])
            - float(row["correction_gpu_ms"])
            - float(row["commit_gpu_ms"]),
        )
        for row in csk_rows
    )
    components = [
        ("H2D", h2d, "#4c78a8"),
        ("K correction", correction, "#59a14f"),
        ("KV commit", commit, "#f2cf5b"),
        ("Synchronization and other", other, "#bab0ac"),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    left = 0.0
    for label, value, color in components:
        ax.barh(
            [0], [value], left=left, height=0.48, label=label,
            color=color, edgecolor="black", linewidth=1.1,
        )
        ax.text(
            left + value / 2,
            0,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=8.5,
        )
        left += value
    ax.set_yticks([0], ["Reuse stage"])
    ax.set_xlabel("Median latency (ms)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=9)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_gpu_pipeline_figure(
    rows: list[dict[str, Any]], output: Path
) -> dict[str, Any]:
    """Plot one real request nearest the median measured GPU pipeline span."""
    import matplotlib.pyplot as plt

    csk_rows = [row for row in rows if row["mode"] == "cskcache"]
    median_span = statistics.median(
        float(row["gpu_pipeline_span_ms"]) for row in csk_rows
    )
    representative = min(
        csk_rows,
        key=lambda row: abs(float(row["gpu_pipeline_span_ms"]) - median_span),
    )
    intervals = representative["_gpu_pipeline_intervals"]
    origin = min(float(item["start_ms"]) for item in intervals)
    lane_y = {"H2D": 2, "K correction": 1, "KV commit": 0}
    colors = {
        "H2D": "#4c78a8",
        "K correction": "#59a14f",
        "KV commit": "#f2cf5b",
    }

    fig, ax = plt.subplots(figsize=(13.2, 3.6))
    for item in intervals:
        kind = str(item["kind"])
        start = float(item["start_ms"]) - origin
        width = float(item["duration_ms"])
        y = lane_y[kind]
        ax.barh(
            y,
            width,
            left=start,
            height=0.58,
            color=colors[kind],
            edgecolor="black",
            linewidth=0.55,
        )
        ax.text(
            start + width / 2,
            y,
            str(int(item["layer"])),
            ha="center",
            va="center",
            fontsize=4.6,
            clip_on=True,
        )

    h2d_by_layer = {
        int(item["layer"]): item for item in intervals if item["kind"] == "H2D"
    }
    correction_by_layer = {
        int(item["layer"]): item
        for item in intervals
        if item["kind"] == "K correction"
    }
    commit_by_layer = {
        int(item["layer"]): item
        for item in intervals
        if item["kind"] == "KV commit"
    }
    overlap_label_used = False
    for layer in range(EXPECTED_CSK_LAYERS - 1):
        h2d = h2d_by_layer[layer + 1]
        for compute in (correction_by_layer[layer], commit_by_layer[layer]):
            start = max(float(h2d["start_ms"]), float(compute["start_ms"]))
            end = min(float(h2d["end_ms"]), float(compute["end_ms"]))
            if end <= start:
                continue
            ax.axvspan(
                start - origin,
                end - origin,
                color="#8fcf65",
                alpha=0.18,
                linewidth=0,
                label="Measured overlap" if not overlap_label_used else None,
            )
            overlap_label_used = True

    ax.set_yticks([0, 1, 2], ["KV commit", "K correction", "H2D"])
    ax.set_xlabel("GPU time since layerwise reuse begins (ms)")
    ax.set_title(
        "CSKCache layerwise H2D and correction pipeline "
        f"(request {representative['case_id']})"
    )
    ax.grid(axis="x", alpha=0.25)
    if overlap_label_used:
        ax.legend(loc="upper right", fontsize=9)
    ax.text(
        1.0,
        -0.23,
        "Numbers inside bars are model-layer IDs.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return representative


def main() -> None:
    raw = os.getenv("CSKCACHE_LATENCY_RUN_DIR")
    if not raw:
        raise RuntimeError("CSKCACHE_LATENCY_RUN_DIR is required")
    run_dir = Path(raw).resolve()
    rows: list[dict[str, Any]] = []
    for replica in range(REPLICAS):
        for mode in MODES:
            leaf = run_dir / f"replica_{replica}" / mode
            vllm_records = read_jsonl(leaf / "vllm_request_timeline.jsonl")
            csk_records = read_jsonl(leaf / "cskcache_profile.jsonl")
            for case_dir in sorted((leaf / "cases").iterdir()):
                rows.append(collect_case(case_dir, mode, vllm_records, csk_records))

    measured = [row for row in rows if row["kind"] == "measure"]
    warmups = [row for row in rows if row["kind"] == "warmup"]
    if measured:
        analysis_rows = measured
        sample_kind = "measure"
        expected_per_mode = REPLICAS * MEASURE_CASES
        output_dir = RESULT_DIR
        sample_description = "steady-state measured requests"
    else:
        analysis_rows = warmups
        sample_kind = "warmup"
        expected_per_mode = REPLICAS * WARMUP_CASES
        output_dir = RESULT_DIR / "non_steady_state"
        sample_description = "first request after each vLLM restart"
    for mode in MODES:
        count = sum(row["mode"] == mode for row in analysis_rows)
        if count != expected_per_mode:
            raise ValueError(
                f"{mode}: expected {expected_per_mode} {sample_kind} cases, "
                f"found {count}"
            )
    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in analysis_rows:
        paired[(int(row["replica"]), int(row["ordinal"]))][row["mode"]] = row
    for key, modes in paired.items():
        if set(modes) != set(MODES):
            raise ValueError(f"unpaired sample {key}")
        try:
            request_a_delta, request_b_added_delta = paired_prompt_deltas(
                modes["recompute"], modes["cskcache"]
            )
        except ValueError as error:
            raise ValueError(f"workload mismatch for sample {key}: {error}") from error
        for row in modes.values():
            row["paired_request_a_prompt_delta"] = request_a_delta
            row["paired_request_b_added_delta"] = request_b_added_delta

    summary_rows: list[dict[str, Any]] = []
    for mode in MODES:
        values = [
            float(row["api_ttft_ms"])
            for row in analysis_rows
            if row["mode"] == mode
        ]
        summary_rows.append(
            {
                "mode": mode,
                "samples": len(values),
                "median_api_ttft_ms": statistics.median(values),
                "p95_api_ttft_ms": percentile(values, 0.95),
                "mean_api_ttft_ms": statistics.mean(values),
            }
        )
    recompute = next(row for row in summary_rows if row["mode"] == "recompute")
    cskcache = next(row for row in summary_rows if row["mode"] == "cskcache")
    reduction = 1 - cskcache["median_api_ttft_ms"] / recompute["median_api_ttft_ms"]
    speedup = recompute["median_api_ttft_ms"] / cskcache["median_api_ttft_ms"]
    max_added_delta = max(
        abs(int(row["paired_request_b_added_delta"])) for row in analysis_rows
    )
    csk_measured = [
        row for row in analysis_rows if row["mode"] == "cskcache"
    ]
    median_ssd_agent_overlap = statistics.median(
        float(row["ssd_agent_overlap_ms"]) for row in csk_measured
    )
    median_ssd_agent_overlap_ratio = statistics.median(
        float(row["ssd_agent_overlap_ratio"]) for row in csk_measured
    )
    host_ready_before_boundary = sum(
        float(row["host_ready_before_reuse_boundary_ms"]) >= 0
        for row in csk_measured
    )
    median_gpu_overlap_ms = statistics.median(
        float(row["gpu_pair_overlap_ms"]) for row in csk_measured
    )
    median_gpu_overlap_ratio = statistics.median(
        float(row["gpu_overlap_ratio"]) for row in csk_measured
    )
    median_gpu_concurrency_saving_ms = statistics.median(
        float(row["gpu_concurrency_saving_ms"]) for row in csk_measured
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_request_latency.csv", analysis_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    make_figure(summary_rows, output_dir / "latency_comparison", sample_kind)
    make_key_timeline_figure(
        analysis_rows, output_dir / "cskcache_key_timeline", sample_kind
    )
    make_gpu_breakdown_figure(
        analysis_rows, output_dir / "cskcache_gpu_reuse_breakdown"
    )
    representative = make_gpu_pipeline_figure(
        analysis_rows, output_dir / "cskcache_layerwise_gpu_pipeline"
    )
    gpu_overlap_rows = [
        {
            "case_id": row["case_id"],
            "replica": row["replica"],
            "ordinal": row["ordinal"],
            "kind": row["kind"],
            "gpu_pair_overlap_ms": row["gpu_pair_overlap_ms"],
            "gpu_overlap_capacity_ms": row["gpu_overlap_capacity_ms"],
            "gpu_overlap_ratio": row["gpu_overlap_ratio"],
            "gpu_serialized_component_ms": row[
                "gpu_serialized_component_ms"
            ],
            "gpu_component_union_ms": row["gpu_component_union_ms"],
            "gpu_concurrency_saving_ms": row["gpu_concurrency_saving_ms"],
            "gpu_pipeline_span_ms": row["gpu_pipeline_span_ms"],
            "gpu_pipeline_vs_serial_saving_ms": row[
                "gpu_pipeline_vs_serial_saving_ms"
            ],
            "representative": row is representative,
        }
        for row in csk_measured
    ]
    write_csv(output_dir / "cskcache_layerwise_gpu_overlap.csv", gpu_overlap_rows)
    (output_dir / "summary.md").write_text(
        "# CSKCache end-to-end latency\n\n"
        f"Data source: `{run_dir}`.\n\n"
        f"Sample kind: `{sample_kind}` ({sample_description}).  \n"
        f"Samples per mode: {expected_per_mode}.  \n\n"
        f"Recompute median request-B TTFT: {recompute['median_api_ttft_ms']:.3f} ms.  \n"
        f"CSKCache median request-B TTFT: {cskcache['median_api_ttft_ms']:.3f} ms.  \n"
        f"Median reduction: {reduction * 100:.2f}%; speedup: {speedup:.3f}×.\n\n"
        f"Maximum paired request-B added-token difference: {max_added_delta} tokens "
        f"(gate: ≤ {MAX_PAIRED_ADDED_TOKEN_DELTA}).  \n"
        "Request-A prompt length is reported separately because per-case workspace "
        "paths are embedded in the real Agent system prompt and are already covered "
        "by the natural prefix-cache hit.\n\n"
        "TTFT is measured from vLLM receiving request B to the first output token. "
        + (
            "This non-steady-state result is diagnostic rather than the formal "
            "steady-state publication result. "
            if sample_kind == "warmup"
            else "This run is eligible for steady-state publication. "
        )
        + "Every analyzed CSKCache sample passed T0, "
        "host-read, authentication, activation, H2D, correction and commit gates.\n\n"
        f"Median SSD/OpenHands overlap: {median_ssd_agent_overlap:.3f} ms "
        f"({median_ssd_agent_overlap_ratio * 100:.2f}% of SSD-to-pinned time).  \n"
        f"HOST_READY preceded the reuse boundary in {host_ready_before_boundary}/"
        f"{len(csk_measured)} analyzed requests.  \n"
        f"Median adjacent-layer H2D/compute overlap: {median_gpu_overlap_ms:.3f} ms; "
        f"median overlap ratio: {median_gpu_overlap_ratio * 100:.2f}%.  \n"
        f"Median concurrency saving over the union of measured GPU operations: "
        f"{median_gpu_concurrency_saving_ms:.3f} ms.  \n"
        "The layerwise pipeline figure uses the real request whose measured GPU "
        "pipeline span is nearest the sample median. H2D, K correction and KV commit "
        "share one CUDA Event origin; the highlighted intersections therefore measure "
        "actual GPU-time overlap rather than inferring it from non-blocking calls.\n",
        encoding="utf-8",
    )
    (output_dir / "source_manifest.csv").write_text(
        "artifact,source\n"
        f"per_request_latency.csv,{run_dir}\n"
        f"summary.csv,{run_dir}\n"
        f"latency_comparison.pdf,{run_dir}\n"
        f"latency_comparison.png,{run_dir}\n",
        encoding="utf-8",
    )
    with (output_dir / "source_manifest.csv").open("a", encoding="utf-8") as file:
        file.write(
            f"cskcache_key_timeline.pdf,{run_dir}\n"
            f"cskcache_key_timeline.png,{run_dir}\n"
            f"cskcache_gpu_reuse_breakdown.pdf,{run_dir}\n"
            f"cskcache_gpu_reuse_breakdown.png,{run_dir}\n"
            f"cskcache_layerwise_gpu_pipeline.pdf,{run_dir}\n"
            f"cskcache_layerwise_gpu_pipeline.png,{run_dir}\n"
            f"cskcache_layerwise_gpu_overlap.csv,{run_dir}\n"
        )
    print(
        f"[summary] recompute={recompute['median_api_ttft_ms']:.3f}ms "
        f"cskcache={cskcache['median_api_ttft_ms']:.3f}ms "
        f"reduction={reduction * 100:.2f}% speedup={speedup:.3f}x"
    )
    print(f"[published] kind={sample_kind} output={output_dir}")


if __name__ == "__main__":
    main()
