"""Plot E1 — does the resume-from-layer-N repair close the output gap of reuse?

Reads results/.../DP3/repair_impact.csv (long format: one row per reuse point x
condition, condition in {reuse, resume@4, resume@8, resume@12}).

Two panels:
  (left)  Deliverable semantic similarity (embed cosine) per reuse point,
          4 bars per point (reuse + each resume@N), grouped by task. Shows
          directly whether repair lifts the points that reuse drags down.
  (right) Quality vs compute saved on the skill rows: x = fraction of the skill
          segment's per-layer computation skipped (reuse = 100%, resume@N =
          N/40), y = deliverable cosine (all points + median). The quality-cost
          curve of the repair knob N.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DEF_CSV = PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "DP3" / "repair_impact.csv"
DEF_OUT = PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "plot" / "repair_impact.png"

TASK_ORDER = [
    "internal_comms_incident_update",
    "doc_coauthoring_design_doc",
    "mcp_server_and_spec",
    "web_artifact_with_theme",
    "launch_poster_page_pack",
    "slack_launch_pack",
]
TASK_LABEL = {
    "internal_comms_incident_update": "incident-\nupdate",
    "doc_coauthoring_design_doc":     "doc-\ncoauth",
    "mcp_server_and_spec":            "mcp-\nspec",
    "web_artifact_with_theme":        "web-\ntheme",
    "launch_poster_page_pack":        "poster-\npack",
    "slack_launch_pack":              "slack-\npack",
}
SKILL_SHORT = {
    "internal-comms": "int-comms", "doc-coauthoring": "doc-coauth",
    "mcp-builder": "mcp-bld", "web-artifacts-builder": "web-artif",
    "theme-factory": "theme", "canvas-design": "canvas",
    "slack-gif-creator": "gif-crt", "brand-guidelines": "brand-gl",
}
BAND_COLORS = ["#f0f4f8", "#ffffff"]
N_LAYERS = 40  # Qwen3-14B

COND_ORDER = ["reuse", "resume@12", "resume@8", "resume@4"]
COND_COLOR = {"reuse": "#c0504d", "resume@12": "#f4b183",
              "resume@8": "#6baed6", "resume@4": "#2c7fb8"}
# fraction of the skill rows' per-layer computation that is skipped
COND_SAVED = {"reuse": 1.0, "resume@4": 4 / N_LAYERS,
              "resume@8": 8 / N_LAYERS, "resume@12": 12 / N_LAYERS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEF_CSV))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    conds = [c for c in COND_ORDER if any(r["condition"] == c for r in rows)]

    # group rows by reuse point, ordered by task then skill/occ
    by_point: dict[tuple, dict[str, float | None]] = defaultdict(dict)
    for r in rows:
        key = (r["task"], r["skill"], int(r["occ"]))
        by_point[key][r["condition"]] = (
            float(r["deliv_cos"]) if r["deliv_cos"] not in ("", None) else None)
    points = []
    for task in TASK_ORDER:
        points.extend(sorted((k for k in by_point if k[0] == task),
                             key=lambda k: (k[1], k[2])))

    task_spans = []
    i = 0
    for task in TASK_ORDER:
        cnt = sum(1 for k in points if k[0] == task)
        if cnt:
            task_spans.append((task, i, i + cnt))
            i += cnt

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(18, 6),
        gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True,
    )

    # ── left: deliverable cosine per point, one bar per condition ────────────
    x = np.arange(len(points))
    width = 0.8 / len(conds)
    for bi, (task, s, e) in enumerate(task_spans):
        axL.axvspan(s - 0.5, e - 0.5, color=BAND_COLORS[bi % 2], zorder=0)
    for ci, cond in enumerate(conds):
        vals = [by_point[k].get(cond) for k in points]
        xs = x - 0.4 + width * (ci + 0.5)
        axL.bar(xs, [v if v is not None else 0 for v in vals], width=width,
                color=COND_COLOR[cond], edgecolor="white", linewidth=0.3,
                zorder=3, label=cond)
    axL.axhline(0.99, color="#4a9e5f", linestyle=":", linewidth=1, zorder=2)
    axL.axhline(0.95, color="#c0504d", linestyle=":", linewidth=1, zorder=2)

    for task, s, e in task_spans:
        axL.text((s + e - 1) / 2, -0.135, TASK_LABEL[task], ha="center", va="top",
                 fontsize=8, color="#555", transform=axL.get_xaxis_transform())
        if s > 0:
            axL.axvline(s - 0.5, color="#cccccc", linewidth=0.8, zorder=1)

    axL.set_xticks(x)
    axL.set_xticklabels([f"{SKILL_SHORT[k[1]]}\nocc{k[2]}" for k in points], fontsize=7)
    axL.set_ylim(0.4, 1.02)
    axL.set_ylabel("Deliverable semantic similarity (vs truth)", fontsize=11)
    axL.set_title("Deliverable embedding cosine per reuse point — no repair (reuse) vs resume@N repair",
                  fontsize=10)
    axL.spines[["top", "right"]].set_visible(False)
    axL.grid(axis="y", color="#e3e8ee", zorder=0)
    axL.legend(frameon=False, fontsize=8, loc="lower left", ncol=len(conds))

    # ── right: quality vs compute saved on the skill rows ────────────────────
    rng = np.random.default_rng(0)
    med_xy = []
    for cond in conds:
        vals = [by_point[k].get(cond) for k in points]
        vals = [v for v in vals if v is not None]
        sx = COND_SAVED[cond] * 100
        jitter = rng.uniform(-1.5, 1.5, len(vals))
        axR.scatter(sx + jitter, vals, color=COND_COLOR[cond], s=30,
                    edgecolor="white", linewidth=0.5, zorder=3, alpha=0.85)
        med_xy.append((sx, float(np.median(vals))))
    med_xy.sort()
    axR.plot([p[0] for p in med_xy], [p[1] for p in med_xy],
             color="#444", linewidth=1.2, marker="D", markersize=5, zorder=4,
             label="median")
    axR.axhline(0.99, color="#4a9e5f", linestyle=":", linewidth=1)
    axR.axhline(0.95, color="#c0504d", linestyle=":", linewidth=1)
    axR.set_xlabel("Skill-segment computation saved (%)\n"
                   "(resume@N saves N/40 layers; reuse saves 100%)", fontsize=10)
    axR.set_ylabel("Deliverable embedding cosine", fontsize=11)
    axR.set_ylim(0.4, 1.02)
    axR.set_title("Quality vs compute saved — the repair-depth knob", fontsize=10)
    axR.spines[["top", "right"]].set_visible(False)
    axR.grid(color="#e3e8ee", zorder=0)
    axR.legend(frameon=False, fontsize=8, loc="lower left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out}")

    # console summary per condition
    print(f"\n{'condition':>10} {'n':>3} {'median':>7} {'min':>7} {'>0.99':>6} {'<0.95':>6}")
    for cond in conds:
        vals = [by_point[k].get(cond) for k in points]
        vals = [v for v in vals if v is not None]
        med = float(np.median(vals))
        print(f"{cond:>10} {len(vals):>3} {med:>7.4f} {min(vals):>7.4f} "
              f"{sum(v > 0.99 for v in vals):>6} {sum(v < 0.95 for v in vals):>6}")


if __name__ == "__main__":
    main()
