#!/usr/bin/env python3
"""Plot CSKCache request, probe, and bulk-preload time decomposition.

The script reads profiler JSONL directly. It never embeds measured values in
code, and it writes the normalized long-form data used by the figure
alongside the output. Multiple reuse occurrences in one run become separate
stacked groups.

The figure is a single shared millisecond timeline per occurrence: row (a) is
the reuse critical path, and rows (b)/(c) are drill-downs of the "Probe
roundtrip" and "Bulk preload" segments, drawn at their true x-position inside
row (a) and linked to it with a shaded connector. This avoids the earlier
version's independent per-panel axes, which stretched a 155 ms bar and a
1062 ms bar to the same visual width.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cskcache-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cskcache-xdg-cache")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from paper_plot_style import apply_publication_style, save_figure


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path(
    os.environ.get(
        "CSKCACHE_AGENT_RUN_ROOT",
        "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/real_agent_runs",
    )
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "problem_exploration"
    / "cskcache_time_breakdown"
)

# Fixed reading order + one hex per leaf stage. Stages shared between the
# probe and bulk-preload drill-downs (Key H2D, Value H2D, RoPE, Storage lookup)
# keep the same color in both rows. "Probe roundtrip" and "Bulk preload" are
# containers, not leaves: they are rendered as a hatched placeholder in row
# (a) because they are decomposed in rows (b)/(c), so they never compete with
# the legend below.
STAGE_ORDER = [
    "Scheduler lookup",
    "Gap prefill",
    "Recompute prefill",
    "Load dispatch",
    "Disk deserialize",
    "Prefetch wait",
    "Storage lookup",
    "Key H2D",
    "Value H2D",
    "RoPE",
    "Probe gather",
    "Residual",
    "Scatter",
    "Other/control",
]
STAGE_COLORS = {
    "Scheduler lookup": "#4477AA",
    "Gap prefill": "#DDCC77",
    "Recompute prefill": "#AA4499",
    "Load dispatch": "#117733",
    "Disk deserialize": "#882255",
    # Background disk prefetch (cskcache.v1.async_load.disk_prefetch): time
    # capture_probes() actually blocked on handle.result(), as opposed to
    # "Disk deserialize" which is the old fully-synchronous storage.get().
    # The two are mutually exclusive per occurrence -- whichever path ran.
    "Prefetch wait": "#D95F02",
    "Storage lookup": "#CC6677",
    "Key H2D": "#88CCEE",
    "Value H2D": "#999933",
    "RoPE": "#EE7733",
    "Probe gather": "#BBCC33",
    "Residual": "#6699CC",
    "Scatter": "#44AA99",
    "Other/control": "#999999",
}
# "Bulk preload" replaces the old "Tail KV load" name: since the redesign
# (整体先搬，探针/重算用 vLLM 真前向覆盖，只确认不重搬), the whole span's KV
# is scattered right after Gap prefill, not after Probe/Anchor -- it is no
# longer a "tail" in the timeline, it is the second thing that happens.
CONTAINER_STAGES = {"Probe roundtrip", "Bulk preload"}
CONTAINER_FACE = "#E5E5E2"
CONTAINER_EDGE = "#8A8A85"
BAR_HEIGHT = 0.62
CONNECTOR_ALPHA = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        help="Profiler JSONL. If omitted, use the newest file under the run root.",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="cskcache_time_breakdown")
    return parser.parse_args()


def newest_profile(run_root: Path) -> Path:
    candidates = list(run_root.glob("*/profile_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No profile JSONL found under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "kind" not in record:
                raise ValueError(f"Invalid profiler record at {path}:{line_number}")
            records.append(record)
    return records


def one_match(
    records: Iterable[dict[str, Any]],
    *,
    kind: str,
    req_id: str,
    cache_id: str,
    target_start: int | None = None,
    target_end: int | None = None,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record.get("kind") == kind
        and record.get("req_id") == req_id
        and record.get("cache_id") == cache_id
        and (target_start is None or record.get("target_start") == target_start)
        and (target_end is None or record.get("target_end") == target_end)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {kind} match for req={req_id} cache={cache_id} "
            f"target=[{target_start},{target_end}), found {len(matches)}"
        )
    return matches[0]


def stage_value(record: dict[str, Any], name: str) -> float:
    return float(record.get("host_stage_ms", {}).get(name, 0.0) or 0.0)


# Mirrors cskcache/profiling/trace.py's TimelineTrace._DURATION_PAIRS.
# Recomputed here from the raw `events` list instead of trusting the
# timeline record's own pre-baked "stage_ms" field, because a profile
# JSONL is a historical artifact: it may have been captured before
# _DURATION_PAIRS last changed, in which case its "stage_ms" reflects
# whatever (possibly stale) event pairing was live when it ran.
_STAGE_DURATION_PAIRS = {
    "gap_prefill": ("gap_scheduled", "bulk_preload_dispatched"),
    "bulk_preload_wait": ("bulk_preload_dispatched", "gap_completed"),
    "probe_roundtrip": ("probe_dispatched", "probe_decision_received"),
    "recompute_prefill": ("recompute_scheduled", "recompute_completed"),
    "load_dispatch": ("load_planned", "load_dispatched"),
}


def recompute_stage_ms(timeline: dict[str, Any]) -> dict[str, float]:
    """Recompute named stage durations for one occurrence from its raw
    events, taking the *last* occurrence of each event name --
    "load_dispatched" fires twice (bulk-preload plan, then confirm plan),
    and load_dispatch's pairing only makes sense against the later one.
    """
    offsets: dict[str, float] = {}
    for item in timeline.get("events", []):
        offsets[str(item["event"])] = float(item["offset_ms"])  # last wins
    return {
        name: offsets[end] - offsets[start]
        for name, (start, end) in _STAGE_DURATION_PAIRS.items()
        if start in offsets and end in offsets
    }


def last_event_wall_time_ns(timeline: dict[str, Any], event: str) -> int:
    """Wall-clock ns of the last occurrence of `event` in a timeline record.

    "load_dispatched" fires twice per probe-gated occurrence now (once for
    the bulk-preload plan, once for the confirm plan), so callers that want
    "when did this occurrence's own work actually finish" must take the
    last match, not the first.
    """
    matches = [
        item["wall_time_ns"] for item in timeline.get("events", []) if item.get("event") == event
    ]
    if not matches:
        raise ValueError(f"No '{event}' event in timeline for {timeline.get('req_id')}")
    return int(matches[-1])


def nonnegative_other(total: float, components: dict[str, float]) -> float:
    remainder = total - sum(components.values())
    if remainder < -1.0:
        raise ValueError(
            f"Stage decomposition exceeds total by {-remainder:.3f} ms: {components}"
        )
    return max(remainder, 0.0)


def build_occurrences(
    records: list[dict[str, Any]], source: Path
) -> list[dict[str, Any]]:
    timelines = [record for record in records if record.get("kind") == "request_timeline"]
    timelines.sort(key=lambda record: int(record["started_at_ns"]))
    occurrences: list[dict[str, Any]] = []
    for index, timeline in enumerate(timelines, start=1):
        req_id = str(timeline["req_id"])
        cache_id = str(timeline["cache_id"])
        skill_start = int(timeline["target_start"])
        skill_end = int(timeline["target_end"])
        lookup = one_match(
            records,
            kind="scheduler_lookup",
            req_id=req_id,
            cache_id=cache_id,
            target_start=skill_start,
            target_end=skill_end,
        )
        probe = one_match(
            records,
            kind="worker_probe_capture",
            req_id=req_id,
            cache_id=cache_id,
            target_start=skill_start,
        )
        load = one_match(
            records,
            kind="worker_load",
            req_id=req_id,
            cache_id=cache_id,
            target_end=skill_end,
        )
        if int(probe.get("captured_layers", -1)) != int(
            probe.get("expected_layers", -2)
        ):
            raise ValueError(f"Incomplete probe layers for {req_id} occurrence {index}")
        if int(load.get("scattered_layers", -1)) != int(
            load.get("expected_layers", -2)
        ) or int(load.get("skipped_layers", -1)) != 0:
            raise ValueError(f"Incomplete load layers for {req_id} occurrence {index}")

        # Chronological order matches the redesigned reuse flow: the whole
        # span is bulk-preloaded right after Gap prefill (not after Probe/
        # Anchor as before), so "Bulk preload" (the old "Tail KV load") now
        # sits right after "Gap prefill", well before "Probe roundtrip".
        stage_ms = recompute_stage_ms(timeline)
        request_parts = {
            "Scheduler lookup": float(lookup["total_ms"]),
            "Gap prefill": stage_ms.get("gap_prefill", 0.0),
            "Bulk preload": float(load["total_ms"]),
            "Probe roundtrip": stage_ms.get("probe_roundtrip", 0.0),
            "Recompute prefill": stage_ms.get("recompute_prefill", 0.0),
            "Load dispatch": stage_ms.get("load_dispatch", 0.0),
        }
        # request_timeline's own "total_ms"/"request_finished" spans the
        # whole vLLM request (set once, on_finished()), which can run well
        # past this occurrence's own span if there are trailing tokens or a
        # second reuse entry -- not what we want here. "reuse_confirmed"
        # fires exactly once, right when this occurrence's own work is
        # done, so anchor end-to-end on that instead.
        end_to_end_ms = (
            last_event_wall_time_ns(timeline, "reuse_confirmed")
            - int(lookup["started_at_ns"])
        ) / 1_000_000.0
        request_parts["Other/control"] = nonnegative_other(
            end_to_end_ms, request_parts
        )

        probe_parts: dict[str, float] = {}
        disk_ms = stage_value(probe, "disk_deserialize")
        prefetch_wait_ms = stage_value(probe, "prefetch_wait")
        if disk_ms:
            # Fully-synchronous storage.get(): no prefetch hint was consumed
            # for this occurrence (e.g. profiling run predates async_load, or
            # the hint never arrived in time).
            probe_parts["Disk deserialize"] = disk_ms
        elif prefetch_wait_ms:
            # A background disk_prefetch handle existed and capture_probes()
            # blocked on it via handle.result(); this is how much of the
            # read was NOT hidden behind gap/probe compute.
            probe_parts["Prefetch wait"] = prefetch_wait_ms
        else:
            # Neither: the entry was already resident (CPU-tier hit), so
            # there was nothing to wait on either way.
            probe_parts["Storage lookup"] = stage_value(probe, "storage_get")
        probe_parts.update(
            {
                "Key H2D": stage_value(probe, "key_h2d_host"),
                "Value H2D": stage_value(probe, "value_h2d_host"),
                "RoPE": stage_value(probe, "rope_host"),
                "Probe gather": stage_value(probe, "probe_gather_host"),
                "Residual": stage_value(probe, "residual_host"),
            }
        )
        probe_parts["Other/control"] = nonnegative_other(
            float(probe["total_ms"]), probe_parts
        )

        load_parts = {
            "Storage lookup": stage_value(load, "storage_get"),
            "Key H2D": stage_value(load, "key_h2d_host"),
            "Value H2D": stage_value(load, "value_h2d_host"),
            "RoPE": stage_value(load, "rope_host"),
            "Scatter": stage_value(load, "scatter_span_host"),
        }
        load_parts["Other/control"] = nonnegative_other(
            float(load["total_ms"]), load_parts
        )
        occurrences.append(
            {
                "index": index,
                "label": f"Occ. {index}",
                "req_id": req_id,
                "cache_id": cache_id,
                "skill_start": skill_start,
                "skill_end": skill_end,
                "source": str(source),
                "request_total_ms": end_to_end_ms,
                "request": request_parts,
                "probe_total_ms": float(probe["total_ms"]),
                "probe": probe_parts,
                "load_total_ms": float(load["total_ms"]),
                "load": load_parts,
            }
        )
    if not occurrences:
        raise ValueError(f"No request_timeline records in {source}")
    return occurrences


def write_long_csv(occurrences: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "occurrence",
                "req_id",
                "cache_id",
                "skill_start",
                "skill_end",
                "panel",
                "stage",
                "milliseconds",
                "source_jsonl",
            ),
        )
        writer.writeheader()
        for occurrence in occurrences:
            for panel in ("request", "probe", "load"):
                for stage, milliseconds in occurrence[panel].items():
                    writer.writerow(
                        {
                            "occurrence": occurrence["index"],
                            "req_id": occurrence["req_id"],
                            "cache_id": occurrence["cache_id"],
                            "skill_start": occurrence["skill_start"],
                            "skill_end": occurrence["skill_end"],
                            "panel": panel,
                            "stage": stage,
                            "milliseconds": f"{milliseconds:.6f}",
                            "source_jsonl": occurrence["source"],
                        }
                    )


def stage_color(name: str) -> str:
    if name in CONTAINER_STAGES:
        return CONTAINER_FACE
    return STAGE_COLORS[name]


def draw_row(
    ax: plt.Axes,
    y_center: float,
    stages: dict[str, float],
    *,
    x0: float,
    pad_ms: float,
    label_min_frac: float = 0.08,
) -> list[tuple[str, float, float]]:
    """Draw one stacked horizontal row starting at x0. Returns (name, left, width)."""
    total = sum(stages.values())
    spans: list[tuple[str, float, float]] = []
    left = x0
    for name, value in stages.items():
        spans.append((name, left, value))
        left += value
    for name, seg_left, value in spans:
        is_container = name in CONTAINER_STAGES
        ax.barh(
            y_center,
            value,
            left=seg_left,
            height=BAR_HEIGHT,
            color=stage_color(name),
            edgecolor="white" if not is_container else CONTAINER_EDGE,
            linewidth=1.3 if not is_container else 1.0,
            hatch="///" if is_container else None,
            zorder=3,
        )
        # Skip inline labels on slivers: text needs both a minimum share of
        # the row and a minimum absolute width, or it collides with its
        # neighbors (this is what produced garbled overlapping numbers on
        # the sub-100 ms rows before this threshold was added).
        if is_container or (value >= label_min_frac * total and value >= 32.0):
            text = f"{name}\n{value:.1f} ms" if is_container else f"{value:.1f}"
            ax.text(
                seg_left + value / 2,
                y_center,
                text,
                ha="center",
                va="center",
                fontsize=6.2,
                color="#333333" if is_container else "white",
                zorder=4,
                linespacing=1.15,
            )
    ax.text(
        left + pad_ms,
        y_center,
        f"{total:.1f} ms",
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        zorder=4,
    )
    return spans


def connector(ax: plt.Axes, parent_span: tuple[float, float], child_span: tuple[float, float], y_top: float, y_bottom: float) -> None:
    (p_left, p_width), (c_left, c_width) = parent_span, child_span
    verts = [
        (p_left, y_top),
        (p_left + p_width, y_top),
        (c_left + c_width, y_bottom),
        (c_left, y_bottom),
    ]
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor="#888888",
            alpha=CONNECTOR_ALPHA,
            edgecolor="none",
            zorder=0.5,
        )
    )


def draw_occurrence(ax: plt.Axes, occurrence: dict[str, Any]) -> None:
    y_a, y_b, y_c = 2.0, 1.0, 0.0
    total_a = sum(occurrence["request"].values())
    pad_ms = max(15.0, 0.02 * total_a)
    spans_a = draw_row(ax, y_a, occurrence["request"], x0=0.0, pad_ms=pad_ms)
    probe_left, probe_width = next(
        (left, width) for name, left, width in spans_a if name == "Probe roundtrip"
    )
    load_left, load_width = next(
        (left, width) for name, left, width in spans_a if name == "Bulk preload"
    )

    spans_b = draw_row(ax, y_b, occurrence["probe"], x0=probe_left, pad_ms=pad_ms)
    b_total = sum(occurrence["probe"].values())
    connector(
        ax,
        parent_span=(probe_left, probe_width),
        child_span=(probe_left, b_total),
        y_top=y_a - BAR_HEIGHT / 2,
        y_bottom=y_b + BAR_HEIGHT / 2,
    )

    draw_row(ax, y_c, occurrence["load"], x0=load_left, pad_ms=pad_ms)
    c_total = sum(occurrence["load"].values())
    connector(
        ax,
        parent_span=(load_left, load_width),
        child_span=(load_left, c_total),
        y_top=y_a - BAR_HEIGHT / 2,
        y_bottom=y_c + BAR_HEIGHT / 2,
    )

    ax.set_xlim(0, max(total_a, probe_left + b_total) * 1.14)
    ax.set_ylim(y_c - BAR_HEIGHT, y_a + BAR_HEIGHT)
    ax.set_yticks([y_a, y_b, y_c])
    ax.set_yticklabels(
        [
            f"(a) Reuse critical path\n{total_a:.0f} ms total",
            f"(b) Probe host path\n{b_total:.0f} ms",
            f"(c) Bulk-preload host path\n{c_total:.0f} ms",
        ],
        fontsize=7.5,
    )
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#E4E4E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Time since scheduler-lookup start (ms)")
    ax.text(
        0.0,
        1.10,
        f"{occurrence['req_id']}  ·  skill=\"{occurrence['cache_id']}\"  ·  "
        f"span=[{occurrence['skill_start']}, {occurrence['skill_end']})",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#666666",
        family="monospace",
    )


def draw_legend(ax: plt.Axes, occurrence: dict[str, Any]) -> None:
    """Attach a legend directly under one occurrence's own axes.

    Scoped to only the stages that occurrence actually uses (e.g. "Prefetch
    wait" vs "Disk deserialize" vs "Storage lookup" are mutually exclusive
    per occurrence), and placed via bbox_to_anchor in that axes' own
    coordinate system so it stays attached to this subplot instead of
    collecting at the bottom of the whole multi-occurrence figure.
    """
    present: list[str] = []
    for panel in ("request", "probe", "load"):
        for name in occurrence[panel]:
            if name not in present and name not in CONTAINER_STAGES:
                present.append(name)
    ordered = [name for name in STAGE_ORDER if name in present]
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STAGE_COLORS[name], edgecolor="white", linewidth=0.8)
        for name in ordered
    ]
    handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=CONTAINER_FACE, edgecolor=CONTAINER_EDGE, hatch="///", linewidth=0.8
        )
    )
    labels = ordered + ["Decomposed below (see child row)"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=6.6,
        columnspacing=1.0,
        handlelength=1.2,
    )


def plot(occurrences: list[dict[str, Any]], output_stem: Path) -> None:
    apply_publication_style()
    n = len(occurrences)
    # Extra height per occurrence (vs. the old shared-legend layout) to fit
    # each one's own legend row right below it.
    fig, axes = plt.subplots(
        n, 1, figsize=(7.2, 4.2 * n), squeeze=False, constrained_layout=False
    )
    for ax, occurrence in zip(axes[:, 0], occurrences):
        draw_occurrence(ax, occurrence)
        draw_legend(ax, occurrence)
    fig.tight_layout(h_pad=4.5)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(output_stem))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    profile_jsonl = (
        args.profile_jsonl.resolve()
        if args.profile_jsonl is not None
        else newest_profile(args.run_root).resolve()
    )
    records = read_records(profile_jsonl)
    occurrences = build_occurrences(records, profile_jsonl)
    figure_dir = args.output_dir / "figures"
    data_path = args.output_dir / "data" / f"{args.name}.csv"
    write_long_csv(occurrences, data_path)
    output_stem = figure_dir / args.name
    plot(occurrences, output_stem)
    print(f"profile_jsonl={profile_jsonl}")
    print(f"occurrences={len(occurrences)}")
    print(f"data_csv={data_path}")
    print(f"figure_pdf={output_stem}.pdf")
    print(f"figure_png={output_stem}.png")


if __name__ == "__main__":
    main()
