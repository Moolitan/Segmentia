from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.cksim_plot import plot_trace_cksim  # noqa: E402
from core.config import ROOT  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    default_result_dir = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim"
    default_plot_dir = ROOT / "results" / "05_context_segment_agent_kv" / "plot"
    ap = argparse.ArgumentParser(description="Plot trace replay CKSim results.")
    ap.add_argument(
        "--summary",
        default=str(default_result_dir / "trace_reuse_cksim_summary.json"),
        help="trace_reuse_cksim_summary.json path",
    )
    ap.add_argument(
        "--csv",
        default=str(default_result_dir / "trace_reuse_cksim.csv"),
        help="trace_reuse_cksim.csv path",
    )
    ap.add_argument(
        "--output-dir",
        default=str(default_plot_dir),
        help="directory for output PNG files",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = plot_trace_cksim(args.summary, args.csv, args.output_dir)
    for output in outputs:
        print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
