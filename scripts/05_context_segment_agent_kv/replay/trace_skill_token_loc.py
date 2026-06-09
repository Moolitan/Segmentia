from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from pathlib import Path
from typing import Any

# Put the package root (.../05_context_segment_agent_kv) on sys.path so the
# local `core` package resolves (distinct from the repo-root `core`).
PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import DEFAULT_TASKS, ROOT, TRACES_DIR  # noqa: E402
from core.message_convert import convert_messages, convert_tools  # noqa: E402
from core.segments import find_skill_segments, span_token_offsets  # noqa: E402

# bash scripts/05_context_segment_agent_kv/replay/run_trace_skill_token_loc.sh
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Collect every skill occurrence token span for each task from the "
            "task's final cumulative trace JSON."
        )
    )
    ap.add_argument("--tasks", default="all", help="comma list of task names, or 'all'")
    ap.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    ap.add_argument("--base-url", default=None, help="override vLLM base URL")
    ap.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    ap.add_argument(
        "--output",
        default=str(
            ROOT
            / "results"
            / "05_context_segment_agent_kv"
            / "replay"
            / "skill_token_locations.json"
        ),
    )
    ap.add_argument(
        "--system-prefix",
        default="",
        help=(
            "Optional text prepended to _system_prompt.txt before tokenizing. "
            "Use this if the replay pass adds a nonce and the spans must match it."
        ),
    )
    ap.add_argument(
        "--char-only",
        action="store_true",
        help="Only collect character spans; do not call vLLM /tokenize.",
    )
    return ap


def parse_tasks(raw: str) -> list[str]:
    return DEFAULT_TASKS if raw == "all" else [t.strip() for t in raw.split(",") if t.strip()]


def invocation_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    return (
        int(stem.split("turn_")[1].split("_")[0]),
        int(stem.split("inv_")[1]),
    )


def last_invocation_path(task: str) -> Path:
    task_dir = TRACES_DIR / task
    files = sorted(task_dir.glob("turn_*_inv_*.json"), key=invocation_sort_key)
    if not files:
        raise FileNotFoundError(f"No invocation JSON files found for task: {task}")
    return files[-1]


def collect_skill_first_invocations(task: str, system_prompt: str) -> dict[str, list[int]]:
    """Walk all invocation files for a task in chronological order.

    Returns {skill_name: [inv_file_idx_occ1, inv_file_idx_occ2, ...]}, where
    each index is 1-based (the Nth file in sorted order).  An index is appended
    each time a new copy of the skill appears in the cumulative message history
    (i.e. the skill's occurrence count in that request exceeds the previous max).
    """
    files = sorted((TRACES_DIR / task).glob("turn_*_inv_*.json"), key=invocation_sort_key)
    seen: dict[str, int] = {}      # skill -> max occurrence count seen so far
    result: dict[str, list[int]] = {}
    for idx, f in enumerate(files, start=1):
        inv = json.loads(f.read_text(encoding="utf-8"))
        msgs, _ = convert_messages(inv["messages"], system_prompt)
        segs = find_skill_segments(msgs)
        counts: dict[str, int] = {}
        for name, _, _, _ in segs:
            counts[name] = counts.get(name, 0) + 1
        for skill, c in counts.items():
            if c > seen.get(skill, 0):
                seen[skill] = c
                result.setdefault(skill, []).append(idx)
    return result


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_task_skill_locations(
    task: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    tools: list[dict],
    char_only: bool,
) -> dict[str, Any]:
    trace_path = last_invocation_path(task)
    invocation = json.loads(trace_path.read_text(encoding="utf-8"))
    openai_msgs, _ = convert_messages(invocation["messages"], system_prompt)
    segments = find_skill_segments(openai_msgs)

    skills: dict[str, dict[str, Any]] = {}
    occurrence_counts: dict[str, int] = {}
    flat_occurrences: list[dict[str, Any]] = []

    for name, msg_idx, char_start, char_end in segments:
        occurrence_counts[name] = occurrence_counts.get(name, 0) + 1
        occurrence_index = occurrence_counts[name]
        token_span = None
        token_count = None
        if not char_only:
            token_start, token_end = span_token_offsets(
                base_url,
                model,
                openai_msgs,
                tools,
                api_key,
                msg_idx,
                char_start,
                char_end,
            )
            token_span = [token_start, token_end]
            token_count = token_end - token_start

        occurrence = {
            "skill": name,
            "occurrence": occurrence_index,
            "message_index": msg_idx,
            "char_span": [char_start, char_end],
            "char_count": char_end - char_start,
            "token_span": token_span,
            "tokens": token_count,
        }
        flat_occurrences.append(occurrence)
        skill_record = skills.setdefault(name, {"count": 0, "occurrences": []})
        skill_record["count"] += 1
        skill_record["occurrences"].append(
            {k: v for k, v in occurrence.items() if k != "skill"}
        )

    # Walk all invocation files to record which file index each skill occ first appears in.
    inv_indices = collect_skill_first_invocations(task, system_prompt)
    for skill_name in skills:
        skills[skill_name]["invocation_indices"] = inv_indices.get(skill_name, [])

    return {
        "task": task,
        "trace_file": relative_to_root(trace_path),
        "turn": invocation.get("turn"),
        "invocation": invocation.get("invocation"),
        "num_messages": len(openai_msgs),
        "num_skill_occurrences": len(flat_occurrences),
        "num_unique_skills": len(skills),
        "skills": skills,
        "occurrences": flat_occurrences,
    }


def build_cross_task_index(task_records: list[dict[str, Any]]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for task_record in task_records:
        task = task_record["task"]
        for skill, skill_record in task_record["skills"].items():
            entry = index.setdefault(skill, {"task_count": 0, "occurrence_count": 0, "tasks": []})
            entry["task_count"] += 1
            entry["occurrence_count"] += skill_record["count"]
            entry["tasks"].append(
                {
                    "task": task,
                    "count": skill_record["count"],
                    "token_spans": [
                        occ["token_span"] for occ in skill_record["occurrences"]
                    ],
                    "invocation_indices": skill_record.get("invocation_indices", []),
                }
            )
    return index


def format_skill_token_locations(task_records: list[dict[str, Any]]) -> str:
    """Generate Python source code for the SKILL_TOKEN_LOCATIONS constant."""
    lines = ["SKILL_TOKEN_LOCATIONS: dict[str, dict] = {"]
    for record in task_records:
        task = record["task"]
        lines.append(f'    "{task}": {{')
        lines.append('        "skills": {')
        for skill_name, skill_data in record["skills"].items():
            occs = skill_data["occurrences"]
            tokens = occs[0]["tokens"] if occs else None
            msg_indices = [o["message_index"] for o in occs]
            token_spans = [o["token_span"] for o in occs]
            inv_indices = skill_data.get("invocation_indices", [])
            lines.append(f'            "{skill_name}": {{')
            lines.append(f'                "tokens": {tokens},')
            lines.append(f'                "message_indices": {msg_indices},')
            lines.append(f'                "token_spans": {token_spans},')
            lines.append(f'                "invocation_indices": {inv_indices},')
            lines.append('            },')
        lines.append('        },')
        lines.append('    },')
    lines.append("}")
    return "\n".join(lines)


def update_config_py(task_records: list[dict[str, Any]]) -> None:
    """Rewrite the SKILL_TOKEN_LOCATIONS block in core/config.py in-place."""
    config_path = PKG_ROOT / "core" / "config.py"
    content = config_path.read_text(encoding="utf-8")

    start_marker = "SKILL_TOKEN_LOCATIONS: dict[str, dict] = {"
    start_idx = content.index(start_marker)

    # Find the matching closing brace at depth 0.
    depth, end_idx = 0, start_idx
    for i, ch in enumerate(content[start_idx:], start=start_idx):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    new_block = format_skill_token_locations(task_records)
    config_path.write_text(
        content[:start_idx] + new_block + content[end_idx:], encoding="utf-8"
    )
    print(f"[config] updated SKILL_TOKEN_LOCATIONS in {config_path}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    tasks = parse_tasks(args.tasks)

    system_prompt = args.system_prefix + (TRACES_DIR / "_system_prompt.txt").read_text(
        encoding="utf-8"
    )
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8")))

    task_records = []
    for task in tasks:
        print(f"[collect] task={task}", flush=True)
        try:
            task_records.append(
                collect_task_skill_locations(
                    task,
                    base_url=base_url,
                    model=args.model,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    tools=tools,
                    char_only=args.char_only,
                )
            )
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Failed to call vLLM /tokenize at {base_url}: {exc}. "
                "Start vLLM or pass --char-only to collect only character spans."
            ) from exc

    out = {
        "model": args.model,
        "base_url": None if args.char_only else base_url,
        "char_only": args.char_only,
        "tasks": tasks,
        "summary": [
            {
                "task": r["task"],
                "trace_file": r["trace_file"],
                "num_skill_occurrences": r["num_skill_occurrences"],
                "num_unique_skills": r["num_unique_skills"],
                "skills": {name: sk["count"] for name, sk in r["skills"].items()},
            }
            for r in task_records
        ],
        "cross_task_index": build_cross_task_index(task_records),
        "task_records": task_records,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {output}", flush=True)

    # Rewrite SKILL_TOKEN_LOCATIONS in config.py when we have token span data.
    if not args.char_only and task_records:
        update_config_py(task_records)


if __name__ == "__main__":
    main()
