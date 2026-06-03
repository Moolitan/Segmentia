from __future__ import annotations

import glob
import os
import re

from core.constants import BENCH_ROOT
from core.schema import SequenceTemplate, TaskSpec


_TURN_RE = re.compile(r"turn_(\d+)\.txt$")


def _turn_index(path: str) -> int:
    m = _TURN_RE.search(path)
    if not m:
        raise ValueError(f"无法解析 turn 序号: {path}")
    return int(m.group(1))


def load_benchmark_sequence(
    repo_name: str, bench_root: str | None = None
) -> SequenceTemplate:
    """Build a sequence from <bench_root>/<repo_name>/turns/turn_N.txt."""
    root = bench_root if bench_root else BENCH_ROOT
    repo_dir = os.path.join(root, repo_name)
    if not os.path.isdir(repo_dir):
        raise ValueError(f"benchmark repo 不存在: {repo_dir}")
    turns_dir = os.path.join(repo_dir, "turns")

    turn_files = sorted(
        glob.glob(os.path.join(turns_dir, "turn_*.txt")),
        key=_turn_index,
    )
    turns = []
    for turn_file in turn_files:
        i = _turn_index(turn_file)
        with open(turn_file, encoding="utf-8") as f:
            message = f.read().strip()
        turns.append(
            TaskSpec(
                task_id=f"{repo_name}_t{i}",
                message=message,
                description=f"Turn {i}",
            )
        )

    if not turns:
        raise ValueError(f"未找到任何 turn_*.txt: {turns_dir}")

    return SequenceTemplate(
        template_id=f"bench_{repo_name}",
        description=f"Benchmark: {repo_name}",
        turns=turns,
    )
