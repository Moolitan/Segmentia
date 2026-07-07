"""Small cross-task contextual occurrence-1 KV reuse comparison.

Cross-reuse plan (all targets are occurrence 1):

1. doc_coauthoring_design_doc/doc-coauthoring
   -> mcp_server_and_spec/doc-coauthoring
2. internal_comms_incident_update/internal-comms
   -> slack_launch_pack/internal-comms
3. web_artifact_with_theme/web-artifacts-builder
   -> launch_poster_page_pack/web-artifacts-builder

Each source is the task-specific occurrence-1 KV produced by
``prefill_contextual_skill_kv.py``. The comparison has two modes:

- recompute: target skill KV is computed normally in the target task.
- cross_contextual_rope: source-task contextual KV is moved to the target
  occurrence-1 span with RoPE key correction; values are reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parents[1]
MODULE_DIR = PACKAGE_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import (  # noqa: E402
    DEFAULT_CONTEXTUAL_KV_DIR,
    DEFAULT_SERVED_MODEL,
    DEFAULT_VLLM_PORT,
    RESULTS_DIR,
    SKILL_TOKEN_LOCATIONS,
)
from trace_utils import (  # noqa: E402
    convert_messages,
    load_invocations,
    load_system_prompt,
    load_tools,
)
from vllm_client import chat_completion  # noqa: E402


MODES = ("recompute", "cross_contextual_rope")
CROSS_REUSE_CASES: tuple[dict[str, str], ...] = (
    {
        "source_task": "doc_coauthoring_design_doc",
        "target_task": "mcp_server_and_spec",
        "skill": "doc-coauthoring",
    },
    {
        "source_task": "internal_comms_incident_update",
        "target_task": "slack_launch_pack",
        "skill": "internal-comms",
    },
    {
        "source_task": "web_artifact_with_theme",
        "target_task": "launch_poster_page_pack",
        "skill": "web-artifacts-builder",
    },
)
TARGET_TASKS = tuple(case["target_task"] for case in CROSS_REUSE_CASES)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "cross_task_contextual_occ1_reuse"


def contextual_cache_id(task: str, skill: str) -> str:
    return f"contextual-occ1-skill-{task}-{skill}"


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def case_filename(case: dict[str, Any]) -> str:
    return (
        f"inv{int(case['invocation_index']):03d}--"
        f"source-{safe_component(str(case['source_task']))}--"
        f"target-{safe_component(str(case['target_task']))}--"
        f"{safe_component(str(case['skill']))}--occ1.txt"
    )


def case_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row["source_task"]),
        str(row["target_task"]),
        str(row["skill"]),
        int(row["occurrence"]),
        str(row["mode"]),
    )


def load_manifest(path: Path) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = case_key(row)
            if key in rows:
                raise ValueError(
                    f"Duplicate manifest key at line {line_number}: {key}"
                )
            rows[key] = row
    return rows


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def raw_generated_tokens(response: dict[str, Any]) -> list[str]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one choice, got {len(choices)}")
    entries = (choices[0].get("logprobs") or {}).get("content")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Response is missing non-empty choices[0].logprobs.content")
    tokens: list[str] = []
    for token_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("token") is None:
            raise ValueError(f"Missing raw token at token_index={token_index}")
        tokens.append(str(entry["token"]))
    return tokens


def prepare_case(target_task: str, kv_dir: Path) -> dict[str, Any]:
    matches = [
        case for case in CROSS_REUSE_CASES if case["target_task"] == target_task
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cross-reuse case for target_task={target_task}, got {matches}"
        )
    plan = matches[0]
    source_task = plan["source_task"]
    skill = plan["skill"]
    source_record = SKILL_TOKEN_LOCATIONS[source_task]["skills"][skill]
    target_record = SKILL_TOKEN_LOCATIONS[target_task]["skills"][skill]
    source_start, source_end = map(int, source_record["token_spans"][0])
    target_start, target_end = map(int, target_record["token_spans"][0])
    if source_end - source_start != target_end - target_start:
        raise RuntimeError(
            f"Span length mismatch for skill={skill}: "
            f"source=[{source_start},{source_end}), "
            f"target=[{target_start},{target_end})"
        )
    if target_start < source_start:
        raise RuntimeError(
            f"RoPE reuse cannot move skill={skill} backwards: "
            f"source_start={source_start}, target_start={target_start}"
        )

    cache_id = contextual_cache_id(source_task, skill)
    cache_path = kv_dir / f"{cache_id}.pt"
    invocation_index = int(target_record["invocation_indices"][0])
    return {
        **plan,
        "occurrence": 1,
        "invocation_index": invocation_index,
        "source_start": source_start,
        "source_end": source_end,
        "target_start": target_start,
        "target_end": target_end,
        "cache_id": cache_id,
        "cache_path": cache_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-task", choices=TARGET_TASKS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kv-dir", type=Path, default=DEFAULT_CONTEXTUAL_KV_DIR)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip an existing case only after validating its file, hash, and sampling parameters.",
    )
    args = parser.parse_args()

    case = prepare_case(args.target_task, args.kv_dir)
    if args.mode == "cross_contextual_rope" and not case["cache_path"].exists():
        raise FileNotFoundError(f"Missing contextual source KV: {case['cache_path']}")

    key = (
        case["source_task"],
        case["target_task"],
        case["skill"],
        case["occurrence"],
        args.mode,
    )
    existing = load_manifest(args.manifest)
    if key in existing:
        if not args.resume:
            raise FileExistsError(
                f"Manifest already contains case={key}; pass --resume or use a "
                "new output directory"
            )
        old = existing[key]
        old_path = Path(str(old["sequence_path"]))
        if not old_path.exists():
            raise FileNotFoundError(
                f"Resume manifest points to missing sequence: {old_path}"
            )
        old_digest = hashlib.sha256(old_path.read_text(encoding="utf-8").encode("utf-8"))
        if old_digest.hexdigest() != str(old.get("sha256")):
            raise RuntimeError(f"Resume sequence hash mismatch: {old_path}")
        expected_sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        }
        mismatched_sampling = {
            name: {"manifest": old.get(name), "current": value}
            for name, value in expected_sampling.items()
            if old.get(name) != value
        }
        if mismatched_sampling:
            raise RuntimeError(
                f"Resume sampling parameters changed for case={key}: "
                f"{mismatched_sampling}"
            )
        print(
            f"[skip-valid] mode={args.mode} source={case['source_task']} "
            f"target={case['target_task']} skill={case['skill']} path={old_path}",
            flush=True,
        )
        return

    invocation = load_invocations(case["target_task"])[case["invocation_index"] - 1]
    system_prompt = load_system_prompt()
    tools = load_tools()
    messages, _ = convert_messages(invocation["messages"], system_prompt)
    cfg = None
    if args.mode == "cross_contextual_rope":
        cfg = {
            "targets": [
                {
                    "cache_id": case["cache_id"],
                    "mode": "rope",
                    "target_start": case["target_start"],
                    "target_end": case["target_end"],
                }
            ]
        }

    request_id = (
        f"cross-occ1-{args.mode}-{case['source_task']}-to-"
        f"{case['target_task']}-{case['skill']}"
    )
    response, elapsed = chat_completion(
        args.base_url or f"http://127.0.0.1:{args.vllm_port}",
        args.model,
        messages,
        tools,
        args.api_key,
        max_tokens=args.max_tokens,
        request_id=request_id,
        context_segment_cache=cfg,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        logprobs=True,
    )
    tokens = raw_generated_tokens(response)
    raw_text = "".join(tokens)
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    output_path = args.sequence_dir / case_filename(case)
    if output_path.exists():
        raise FileExistsError(f"Sequence output already exists: {output_path}")
    atomic_write_text(output_path, raw_text)

    choice = (response.get("choices") or [{}])[0]
    row = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in case.items()
    }
    row.update(
        {
            "mode": args.mode,
            "request_id": request_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "token_count": len(tokens),
            "character_count": len(raw_text),
            "sha256": digest,
            "finish_reason": choice.get("finish_reason"),
            "sequence_path": str(output_path),
            "elapsed_sec": round(float(elapsed), 4),
        }
    )
    append_manifest(args.manifest, row)
    print(
        f"[saved] mode={args.mode} source={case['source_task']} "
        f"target={case['target_task']} skill={case['skill']} "
        f"target_span=[{case['target_start']},{case['target_end']}) "
        f"tokens={len(tokens)} path={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
