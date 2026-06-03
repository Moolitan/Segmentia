from __future__ import annotations

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import (
    SequenceTemplate,
    TaskSpec,
    attach_llm_request_attempt_collector,
    create_agent_and_llm,
    flatten_request_messages,
    load_benchmark_sequence,
    load_skill_doc,
    parse_tool_activity,
    resolve_skills_dir,
    run_sequence,
)

__all__ = [
    "SequenceTemplate",
    "TaskSpec",
    "attach_llm_request_attempt_collector",
    "create_agent_and_llm",
    "flatten_request_messages",
    "load_benchmark_sequence",
    "load_skill_doc",
    "parse_tool_activity",
    "resolve_skills_dir",
    "run_sequence",
]


if __name__ == "__main__":
    from core.cli import main

    main()
