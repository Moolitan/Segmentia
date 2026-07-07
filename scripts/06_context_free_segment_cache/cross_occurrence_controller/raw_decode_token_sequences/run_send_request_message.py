"""Dump Qwen3 chat-template text for every replay case without inference."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parents[1]
MODULE_DIR = PACKAGE_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import DEFAULT_MODEL_PATH, DEFAULT_TASKS  # noqa: E402
from replay import selected_cases  # noqa: E402
from trace_utils import (  # noqa: E402
    convert_messages,
    load_invocations,
    load_system_prompt,
    load_tools,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "problem_exploration"
    / "raw_decode_token_sequences"
    / "rendered_prompts"
)
MODES = ("recompute", "rope")
OCCURRENCES = (1, 2, 3)


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def prompt_filename(case: dict[str, Any]) -> str:
    return (
        f"inv{int(case['invocation_index']):03d}--"
        f"{safe_component(str(case['task']))}--"
        f"{safe_component(str(case['skill']))}--"
        f"occ{int(case['occurrence'])}.txt"
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    system_prompt = load_system_prompt()
    tools = load_tools()
    invocations_by_task = {
        task: load_invocations(task)
        for task in DEFAULT_TASKS
    }
    cases = selected_cases(
        list(DEFAULT_TASKS),
        list(OCCURRENCES),
        include_first_occurrence=True,
    )

    rendered_by_filename: dict[str, str] = {}
    for case in cases:
        invocation_index = int(case["invocation_index"])
        invocation = invocations_by_task[str(case["task"])][invocation_index - 1]
        messages, _ = convert_messages(
            invocation["messages"],
            system_prompt,
        )
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if not isinstance(rendered, str):
            raise TypeError(
                f"Expected rendered text for case={case}, got {type(rendered)}"
            )

        filename = prompt_filename(case)
        if filename in rendered_by_filename:
            raise ValueError(f"Duplicate rendered prompt filename: {filename}")
        rendered_by_filename[filename] = rendered

        for mode in MODES:
            output_path = args.output_dir / mode / filename
            atomic_write_text(output_path, rendered)
            print(
                f"[saved] mode={mode} task={case['task']} "
                f"skill={case['skill']} occ={case['occurrence']} "
                f"inv={invocation_index} chars={len(rendered)} "
                f"path={output_path}",
                flush=True,
            )

    expected_files = len(cases) * len(MODES)
    print(
        f"[done] cases={len(cases)} modes={len(MODES)} "
        f"txt_files={expected_files} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
