"""Fail before a long run when the configured local GPU is occupied."""

from __future__ import annotations

import subprocess

from paper_evaluation.config import PLATFORMS

from . import config as local


def parse_gpu_rows(output: str) -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        index, used, total = (int(field) for field in fields)
        rows[index] = (used, total)
    return rows


def main() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = parse_gpu_rows(result.stdout)
    required = sorted(
        {
            gpu_id
            for platform_id in local.PLATFORM_IDS
            for gpu_id in PLATFORMS[platform_id].gpu_ids
        }
    )
    for gpu_id in required:
        if gpu_id not in rows:
            raise RuntimeError(f"configured GPU {gpu_id} is not visible")
        used, total = rows[gpu_id]
        if used >= 500:
            raise RuntimeError(
                f"configured GPU {gpu_id} is occupied: {used}/{total} MiB"
            )
        print(f"[gpu-ready] index={gpu_id} used={used}MiB total={total}MiB")


if __name__ == "__main__":
    main()
