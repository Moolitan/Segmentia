"""Plot the quality-vs-cost Pareto for KV-reuse repair strategies.

Reads the CSV produced by replay/pareto_recompute_14b.py and draws a scatter+line
figure: x = recompute cost (fraction of full skill forward), y = deep-layer VALUE
cosine vs full-recompute truth. Each repair family (depth / token / tokenOracle /
2D) is a separate series; full-reuse and full-recompute are anchor points.

Usage:
    python plot_pareto_recompute.py
    python plot_pareto_recompute.py --csv <path> --out <png>
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
DEF_CSV = (PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "Pareto"
           / "pareto_internal_comms_incident_update_internal-comms_occ1-2.csv")
DEF_OUT = (PKG_ROOT.parents[1] / "results" / "05_context_segment_agent_kv" / "plot"
           / "pareto_recompute.png")

STYLE = {  # family -> (color, marker, label)
    "depth": ("#2f6f9f", "o", "depth-partial (residual checkpoint)"),
    "token": ("#c65f3b", "s", "token-selective (shallow-signal pick)"),
    "tokenOracle": ("#e0a030", "^", "token-selective (oracle pick)"),
    "2D": ("#5a9e6f", "D", "2D (token x depth)"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEF_CSV))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--title", default="Repairing reused KV: quality vs compute (internal-comms, occ1->occ2)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    fam = {}
    anchors = {}
    for r in rows:
        x, y = float(r["cost"]) * 100, float(r["deepV_cos"])
        if r["family"] == "anchor":
            anchors[r["strategy"]] = (x, y)
        else:
            fam.setdefault(r["family"], []).append((x, y, r["strategy"]))

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)

    for family, pts in fam.items():
        pts = sorted(pts)
        color, marker, label = STYLE.get(family, ("#888", "x", family))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                color=color, linewidth=2, markersize=8, label=label)

    # anchors
    if "full-reuse" in anchors:
        x, y = anchors["full-reuse"]
        ax.scatter([x], [y], color="#444", marker="*", s=240, zorder=5, label="full-reuse (free)")
        ax.annotate("full-reuse\n(0%, 0.53)", (x, y), textcoords="offset points",
                    xytext=(12, -4), fontsize=9)
    if "full-recompute" in anchors:
        x, y = anchors["full-recompute"]
        ax.scatter([x], [y], color="#444", marker="P", s=200, zorder=5, label="full-recompute")
        ax.annotate("full-recompute\n(100%, 1.0)", (x, y), textcoords="offset points",
                    xytext=(-110, -6), fontsize=9)

    ax.axhline(0.8, color="#bbb", linestyle="--", linewidth=1)
    ax.annotate("quality 0.80", (2, 0.805), color="#999", fontsize=8)

    ax.set_xlabel("recompute cost  (% of full skill forward FLOPs)", fontsize=12)
    ax.set_ylabel("deep-layer VALUE cosine vs truth  (1.0 = lossless)", fontsize=12)
    ax.set_title(args.title, fontsize=12.5)
    ax.set_xlim(-3, 105)
    ax.set_ylim(0.45, 1.03)
    ax.grid(color="#e3e8ee", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
