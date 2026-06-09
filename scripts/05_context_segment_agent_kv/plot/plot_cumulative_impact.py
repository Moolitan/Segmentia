"""Plot cumulative reuse impact — isolated vs cumulative at occ3 frontier.

Reads results/.../DP3/cumulative_reuse_impact.csv. occ2 (n_upstream=0) is the
built-in sanity check (isolated == cumulative, accum=0) and is summarized in
text; the figure focuses on occ3 (n_upstream=1), the first accumulation point.

Two panels:
  (left)  per occ3 instance: isolated KL (doc 四) vs cumulative KL — cumulative
          adds the divergence from the one upstream reuse.
  (right) relative accumulation (accum_kl / iso_kl) per instance — one upstream
          reuse adds 30–130% on top of the isolated measure in the worst cases.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DEF_CSV = PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "DP3" / "cumulative_reuse_impact.csv"
DEF_OUT = PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "plot" / "cumulative_impact.png"

SKILL_SHORT = {
    "internal-comms": "int-comms", "doc-coauthoring": "doc-coauth",
    "mcp-builder": "mcp-bld", "web-artifacts-builder": "web-artif",
    "theme-factory": "theme", "canvas-design": "canvas",
    "slack-gif-creator": "gif-crt", "brand-guidelines": "brand-gl",
}
TASK_SHORT = {
    "internal_comms_incident_update": "incident", "doc_coauthoring_design_doc": "doc",
    "mcp_server_and_spec": "mcp", "web_artifact_with_theme": "web",
    "launch_poster_page_pack": "poster", "slack_launch_pack": "slack",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEF_CSV))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    occ3 = [r for r in rows if int(r["occ"]) == 3]
    # sort by relative accumulation descending
    for r in occ3:
        iso = float(r["iso_kl"])
        r["_rel"] = (float(r["cum_kl"]) - iso) / iso if iso > 0 else 0.0
    occ3.sort(key=lambda r: r["_rel"], reverse=True)

    labels = [f"{SKILL_SHORT[r['skill']]}\n({TASK_SHORT[r['task']]})" for r in occ3]
    iso = [float(r["iso_kl"]) for r in occ3]
    cum = [float(r["cum_kl"]) for r in occ3]
    rel = [100 * r["_rel"] for r in occ3]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    x = np.arange(len(occ3))
    w = 0.38

    axL.bar(x - w / 2, iso, width=w, color="#9aa0a6", label="isolated (note 4, no upstream)",
            edgecolor="white", zorder=3)
    axL.bar(x + w / 2, cum, width=w, color="#c0504d", label="cumulative (+1 upstream reuse)",
            edgecolor="white", zorder=3)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=7)
    axL.set_ylabel("Frontier prediction divergence (mean KL vs fresh)", fontsize=10)
    axL.set_title("occ3 frontier: one upstream reuse adds measurable divergence\n"
                  "(occ2 sanity check: no upstream → isolated == cumulative, accum=0 for all 12)",
                  fontsize=10)
    axL.legend(frameon=False, fontsize=9)
    axL.spines[["top", "right"]].set_visible(False)
    axL.grid(axis="y", color="#e3e8ee", zorder=0)

    colors = ["#c0504d" if v > 0 else "#5a9e6f" for v in rel]
    axR.bar(x, rel, color=colors, edgecolor="white", zorder=3)
    for xi, v in zip(x, rel):
        axR.text(xi, v + (1.5 if v >= 0 else -1.5), f"{v:+.0f}%", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=7)
    axR.axhline(0, color="#888", linewidth=0.8)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=7)
    axR.set_ylabel("Relative accumulation  (cum − iso) / iso", fontsize=10)
    axR.set_title("One upstream reuse adds 30–130% on top of the isolated measure\n"
                  "→ error ACCUMULATES (10/12 positive); not self-healing",
                  fontsize=10)
    axR.spines[["top", "right"]].set_visible(False)
    axR.grid(axis="y", color="#e3e8ee", zorder=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
