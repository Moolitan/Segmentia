from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    """Per-turn task loaded from a benchmark repo."""

    task_id: str
    message: str
    description: str


@dataclass
class SequenceTemplate:
    """A benchmark sequence containing all turns for one repo."""

    template_id: str
    description: str
    turns: list[TaskSpec] = field(default_factory=list)
