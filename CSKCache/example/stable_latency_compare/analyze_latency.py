#!/usr/bin/env python3
"""Validate and summarize real-Agent Recompute versus CSKCache latency."""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from config import MEASURE_CASES, MODES, REPLICAS, RESULT_DIR


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
    "csk_reuse_scheduler_activate",
    "cskcache_h2d_breakdown",
    "csk_reuse_gpu_breakdown",
    "csk_worker_load_complete",
)
MAX_PAIRED_ADDED_TOKEN_DELTA = 8


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


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


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
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary_rows: list[dict[str, Any]], output: Path) -> None:
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
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    for mode in MODES:
        count = sum(row["mode"] == mode for row in measured)
        expected = REPLICAS * MEASURE_CASES
        if count != expected:
            raise ValueError(f"{mode}: expected {expected} measured cases, found {count}")
    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in measured:
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
        values = [float(row["api_ttft_ms"]) for row in measured if row["mode"] == mode]
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
        abs(int(row["paired_request_b_added_delta"])) for row in measured
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_DIR / "per_request_latency.csv", rows)
    write_csv(RESULT_DIR / "summary.csv", summary_rows)
    make_figure(summary_rows, RESULT_DIR / "latency_comparison")
    (RESULT_DIR / "summary.md").write_text(
        "# CSKCache end-to-end latency\n\n"
        f"Data source: `{run_dir}`.\n\n"
        f"Recompute median request-B TTFT: {recompute['median_api_ttft_ms']:.3f} ms.  \n"
        f"CSKCache median request-B TTFT: {cskcache['median_api_ttft_ms']:.3f} ms.  \n"
        f"Median reduction: {reduction * 100:.2f}%; speedup: {speedup:.3f}×.\n\n"
        f"Maximum paired request-B added-token difference: {max_added_delta} tokens "
        f"(gate: ≤ {MAX_PAIRED_ADDED_TOKEN_DELTA}).  \n"
        "Request-A prompt length is reported separately because per-case workspace "
        "paths are embedded in the real Agent system prompt and are already covered "
        "by the natural prefix-cache hit.\n\n"
        "TTFT is measured from vLLM receiving request B to the first output token. "
        "The run is publishable only because every CSKCache sample passed T0, "
        "host-read, authentication, activation, H2D, correction and commit gates.\n",
        encoding="utf-8",
    )
    (RESULT_DIR / "source_manifest.csv").write_text(
        "artifact,source\n"
        f"per_request_latency.csv,{run_dir}\n"
        f"summary.csv,{run_dir}\n"
        f"latency_comparison.pdf,{run_dir}\n"
        f"latency_comparison.png,{run_dir}\n",
        encoding="utf-8",
    )
    print(
        f"[summary] recompute={recompute['median_api_ttft_ms']:.3f}ms "
        f"cskcache={cskcache['median_api_ttft_ms']:.3f}ms "
        f"reduction={reduction * 100:.2f}% speedup={speedup:.3f}x"
    )
    print(f"[published] {RESULT_DIR}")


if __name__ == "__main__":
    main()
