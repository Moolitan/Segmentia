"""Load saved agent-trace invocation JSONs from src/traces/."""
from __future__ import annotations

import json

from .config import TRACES_DIR


def load_invocations(task: str) -> list[dict]:
    """Load all turn_<N>_inv_<M>.json for a task, sorted by (turn, invocation)."""
    task_dir = TRACES_DIR / task
    files = sorted(
        task_dir.glob("turn_*_inv_*.json"),
        key=lambda p: (
            int(p.stem.split("turn_")[1].split("_")[0]),
            int(p.stem.split("inv_")[1]),
        ),
    )
    return [json.loads(f.read_text()) for f in files]
