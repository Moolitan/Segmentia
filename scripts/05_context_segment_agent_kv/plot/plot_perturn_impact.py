"""Plot per-use reuse impact — position-fixed, think vs deliverable split.

Reads results/.../DP3/perturn_invocation_impact.csv with columns
  {think,deliv,full}_{bleu4,rouge_l,cos}.

Two panels:
  (left)  Deliverable semantic similarity (embed cosine) per reuse instance,
          grouped by task. 14/24 are near-lossless (>0.99); a few drop low —
          those are where the agent picks a DIFFERENT next action under reuse.
  (right) Scatter deliv_bleu4 vs deliv_cos for all instances, with the divergent
          points annotated. Most cluster at high-cos (benign); the low-cos tail
          is next-action divergence, not garbled text.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DEF_CSV = PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "DP3" / "perturn_invocation_impact.csv"
DEF_OUT = PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "plot" / "perturn_impact.png"

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
LOW = 0.95  # below this = deliverable genuinely diverged (next-action flip)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEF_CSV))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    ordered = []
    for task in TASK_ORDER:
        ordered.extend(sorted((r for r in rows if r["task"] == task),
                              key=lambda r: (r["skill"], int(r["occ"]))))

    dcos = [float(r["deliv_cos"]) for r in ordered]
    dbleu = [float(r["deliv_bleu4"]) for r in ordered]
    xlabels = [f"{SKILL_SHORT[r['skill']]}\nocc{r['occ']}" for r in ordered]

    task_spans = []
    i = 0
    for task in TASK_ORDER:
        cnt = sum(1 for r in ordered if r["task"] == task)
        if cnt:
            task_spans.append((task, i, i + cnt))
            i += cnt

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True,
    )

    # ── left: deliverable cosine per instance, grouped by task ───────────────
    x = np.arange(len(ordered))
    for bi, (task, s, e) in enumerate(task_spans):
        axL.axvspan(s - 0.5, e - 0.5, color=BAND_COLORS[bi % 2], zorder=0)
    colors = ["#c0504d" if c < LOW else ("#4a9e5f" if c > 0.99 else "#9aa0a6") for c in dcos]
    axL.bar(x, dcos, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    axL.axhline(0.99, color="#4a9e5f", linestyle=":", linewidth=1, zorder=2)
    axL.axhline(LOW, color="#c0504d", linestyle=":", linewidth=1, zorder=2)

    for xi, r, c in zip(x, ordered, dcos):
        if c < LOW:
            axL.text(xi, c - 0.015, f"{SKILL_SHORT[r['skill']]}\nocc{r['occ']}",
                     ha="center", va="top", fontsize=6.5, color="#8a2622")

    for task, s, e in task_spans:
        axL.text((s + e - 1) / 2, -0.135, TASK_LABEL[task], ha="center", va="top",
                 fontsize=8, color="#555", transform=axL.get_xaxis_transform())
        if s > 0:
            axL.axvline(s - 0.5, color="#cccccc", linewidth=0.8, zorder=1)

    axL.set_xticks(x)
    axL.set_xticklabels(xlabels, fontsize=7)
    axL.set_ylim(0.4, 1.02)
    axL.set_ylabel("Deliverable semantic similarity (reuse vs truth)", fontsize=11)
    n_clean = sum(c > 0.99 for c in dcos)
    n_low = sum(c < LOW for c in dcos)
    axL.set_title(f"Deliverable embedding cosine — {n_clean}/{len(dcos)} near-lossless (>0.99), "
                  f"{n_low}/{len(dcos)} diverge (<{LOW})\n"
                  "low-cos points = agent takes a different (still valid) next action",
                  fontsize=10)
    axL.spines[["top", "right"]].set_visible(False)
    axL.grid(axis="y", color="#e3e8ee", zorder=0)

    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(color="#4a9e5f", label="near-lossless (>0.99)"),
                        Patch(color="#9aa0a6", label="minor (0.95–0.99)"),
                        Patch(color="#c0504d", label="diverged (<0.95)")],
               frameon=False, fontsize=8, loc="lower left")

    # ── right: deliv BLEU-4 vs deliv cosine scatter ──────────────────────────
    for r, b, c in zip(ordered, dbleu, dcos):
        col = "#c0504d" if c < LOW else ("#4a9e5f" if c > 0.99 else "#9aa0a6")
        axR.scatter(b, c, color=col, s=55, edgecolor="white", linewidth=0.6, zorder=3)
        if c < LOW:
            axR.annotate(f"{SKILL_SHORT[r['skill']]} occ{r['occ']}", (b, c),
                         textcoords="offset points", xytext=(7, 0), fontsize=7, color="#8a2622")
    axR.axhline(0.99, color="#4a9e5f", linestyle=":", linewidth=1)
    axR.axhline(LOW, color="#c0504d", linestyle=":", linewidth=1)
    axR.set_xlim(-0.05, 1.05)
    axR.set_ylim(0.4, 1.02)
    axR.set_xlabel("Deliverable BLEU-4 (surface wording)", fontsize=11)
    axR.set_ylabel("Deliverable embedding cosine (semantics)", fontsize=11)
    axR.set_title("High-cos cluster = benign; low-cos tail = next-action flips\n"
                  "(BLEU-4≈0 there means a different tool/length, not garbled text)",
                  fontsize=10)
    axR.spines[["top", "right"]].set_visible(False)
    axR.grid(color="#e3e8ee", zorder=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
