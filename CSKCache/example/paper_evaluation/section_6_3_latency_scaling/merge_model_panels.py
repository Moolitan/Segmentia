#!/usr/bin/env python3
"""Merge three completed model summaries into the final 1x3 figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_evaluation.common.schema import read_csv, write_csv

from . import config as local
from .analyze import _complete_platform, plot_model_panels
from .schema import SUMMARY_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries: list[dict[str, object]] = []
    for run_dir in args.run_dir:
        path = run_dir / "summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"summary is missing: {path}")
        summaries.extend(read_csv(path))
    missing = [
        platform_id
        for platform_id in local.MODEL_PANEL_ORDER
        if not _complete_platform(summaries, platform_id)
    ]
    if missing:
        raise RuntimeError(
            "refusing to create placeholder panels; incomplete platforms: "
            + ", ".join(missing)
        )
    output = args.output_dir.resolve()
    write_csv(output / "summary_all_models.csv", summaries, SUMMARY_COLUMNS)
    plot_model_panels(
        output,
        summaries,
        local.MODEL_PANEL_ORDER,
        stem="ttft_model_comparison",
    )
    print(f"[merged] output={output / 'ttft_model_comparison.pdf'}")


if __name__ == "__main__":
    main()
