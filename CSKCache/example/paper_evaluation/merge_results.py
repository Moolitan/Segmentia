"""Merge completed subsection CSV files from all configured platform roots."""

from __future__ import annotations

import csv
from pathlib import Path

from config import MERGED_OUTPUT_DIR, MERGE_INPUT_ROOTS
from common.schema import SAMPLE_COLUMNS, SUMMARY_COLUMNS, read_csv, write_csv


def _collect(filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in MERGE_INPUT_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob(filename)):
            resolved = path.resolve()
            if resolved in seen or MERGED_OUTPUT_DIR.resolve() in resolved.parents:
                continue
            seen.add(resolved)
            rows.extend(read_csv(path))
    deduplicated: list[dict[str, str]] = []
    row_keys: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in row_keys:
            continue
        row_keys.add(key)
        deduplicated.append(row)
    return deduplicated


def main() -> None:
    samples = _collect("samples.csv")
    summaries = _collect("summary.csv")
    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(MERGED_OUTPUT_DIR / "combined_samples.csv", samples, SAMPLE_COLUMNS)
    write_csv(
        MERGED_OUTPUT_DIR / "combined_summary.csv", summaries, SUMMARY_COLUMNS
    )
    print(f"merged samples={len(samples)} summaries={len(summaries)}")
    print(f"output={MERGED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
