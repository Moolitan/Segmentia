#!/usr/bin/env python3
"""CPU-only smoke tests for H2D benchmark aggregation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def case_records(case_id: str, profiling: str, repetitions: int) -> list[dict]:
    case = {
        "record_type": "case",
        "case_id": case_id,
        "profiling": profiling,
        "memory": "pageable",
        "position_shift": 0,
        "warmup": 1,
        "repetitions": repetitions,
        "cache_id": "skill",
        "tokens": 4,
        "bytes": 64,
        "layers": 1,
        "disk_load_ms": 1.0,
        "pin_setup_ms": 0.0,
        "gpu_name": "synthetic",
    }
    records = [case]
    for iteration in range(repetitions):
        cuda_stages = (
            {
                "key_h2d": 1.0,
                "value_h2d": 1.1,
                "rope": 0.2,
                "scatter_span": 0.5,
            }
            if profiling == "on"
            else {}
        )
        records.append(
            {
                "record_type": "iteration",
                "case_id": case_id,
                "iteration": iteration,
                "operation_wall_ms": 4.0 + iteration,
                "end_to_end_wall_ms": 4.2 + iteration,
                "outer_cuda_ms": 3.0 + iteration,
                "path_gbps": 2.0,
                "expected_layers": 1,
                "scattered_layers": 1,
                "skipped_layers": 0,
                "cuda_stage_ms": cuda_stages,
                "host_stage_ms": {},
            }
        )
    return records


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        cases = root / "cases"
        output = root / "output"
        cases.mkdir()
        for index, profiling in enumerate(("off", "on"), start=1):
            path = cases / f"case-{index}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in case_records(f"case-{index}", profiling, 3)
                )
            )
        subprocess.run(
            [
                sys.executable,
                str(HERE / "summarize.py"),
                "--case-dir",
                str(cases),
                "--output-dir",
                str(output),
                "--expected-cases",
                "2",
                "--expected-repetitions",
                "3",
            ],
            check=True,
        )
        assert (output / "config.json").is_file()
        assert (output / "summary.csv").read_text().count("\n") == 3
        assert (output / "comparisons.csv").read_text().count("\n") == 2
        assert (output / "raw_iterations.jsonl").read_text().count("\n") == 6
    print("H2D microbenchmark summarizer tests passed")


if __name__ == "__main__":
    main()
