"""Decode recompute/direct/rope outputs at configured skill reuse points."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_CKSIM_KV_DIR,
    DEFAULT_KV_DIR,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_SERVED_MODEL,
    DEFAULT_VLLM_PORT,
    SKILL_TOKEN_LOCATIONS,
    cache_id_for_skill,
    get_skill_token_span,
    parse_tasks,
)
from trace_utils import convert_messages, load_invocations, load_system_prompt, load_tools  # noqa: E402
from vllm_client import chat_completion, extract_response  # noqa: E402


def repair_cache_id(arm: str, task: str, skill: str, occurrence: int) -> str:
    """Per-case cache id for the value-side repair arms (see build_repair_arms_kv.py)."""
    return f"cf-{arm}-{task}-{skill}-occ{occurrence}"


# An "arm" is one experimental condition. Each maps to (a) the vLLM injection
# mode applied at the target span and (b) how its cache id is resolved.
#   - recompute: no injection (ground-truth reference).
#   - direct/rope: inject the per-skill context-free KV, with/without RoPE key
#     correction. These read the offline_skill_kv directory.
#   - vrep/krep/oracle: the value-side repair 2x2 ablation. They read prebaked
#     per-case mixed KV from repair_arms_kv (built by build_repair_arms_kv.py):
#       vrep   = rope-corrected skill key + recompute (oracle) value
#       krep   = oracle key (already at target position) + skill value
#       oracle = oracle key + oracle value  (should reproduce recompute)
#     vrep uses ROPE so the worker re-rotates the skill key from its offline
#     source position; krep/oracle use direct because the oracle key was dumped
#     at the in-context target position and needs no rotation.
ARMS: dict[str, dict[str, Any]] = {
    "recompute": {"inject": None, "cache_id": None},
    "direct": {"inject": "direct", "cache_id": "skill"},
    "rope": {"inject": "rope", "cache_id": "skill"},
    "vrep": {"inject": "rope", "cache_id": "repair"},
    "krep": {"inject": "direct", "cache_id": "repair"},
    "oracle": {"inject": "direct", "cache_id": "repair"},
}


def resolve_cache_id(arm: str, case: dict[str, Any]) -> str | None:
    kind = ARMS[arm]["cache_id"]
    if kind is None:
        return None
    if kind == "skill":
        return cache_id_for_skill(str(case["skill"]))
    return repair_cache_id(
        arm, str(case["task"]), str(case["skill"]), int(case["occurrence"])
    )


def selected_cases(tasks: list[str], occurrences: list[int]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task in tasks:
        invocations = load_invocations(task)
        task_cases: list[dict[str, Any]] = []
        for skill, record in SKILL_TOKEN_LOCATIONS[task]["skills"].items():
            for occurrence in occurrences:
                if occurrence < 1 or occurrence > len(record["invocation_indices"]):
                    continue
                if occurrence == 1:
                    continue
                inv_idx = int(record["invocation_indices"][occurrence - 1])
                start, end = get_skill_token_span(task, skill, occurrence)
                invocation = invocations[inv_idx - 1]
                task_cases.append(
                    {
                        "task": task,
                        "skill": skill,
                        "occurrence": occurrence,
                        "invocation_index": inv_idx,
                        "turn": invocation["turn"],
                        "invocation": invocation["invocation"],
                        "target_start": start,
                        "target_end": end,
                    }
                )
        # Sort by invocation_index to replay skills in the order they actually
        # appear in the trace. This keeps the vLLM prefix cache state consistent
        # with the real agent execution order and prevents later requests from
        # seeing spurious prefix cache hits that would not exist mid-trace.
        task_cases.sort(key=lambda c: c["invocation_index"])
        cases.extend(task_cases)
    return cases


def cksim_dump_cache_id(mode: str, task: str, skill: str, occurrence: int) -> str:
    return f"cf-{mode}-{task}-{skill}-occ{occurrence}"


def context_config_for_case(
    arm: str,
    case: dict[str, Any],
    *,
    dump_kv_for_cksim: bool,
) -> dict[str, Any] | None:
    start = int(case["target_start"])
    end = int(case["target_end"])
    skill = str(case["skill"])
    task = str(case["task"])
    occurrence = int(case["occurrence"])

    sources: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    spec = ARMS[arm]
    if spec["inject"] is not None:
        targets.append(
            {
                "cache_id": resolve_cache_id(arm, case),
                "mode": spec["inject"],
                "target_start": start,
                "target_end": end,
            }
        )
    if dump_kv_for_cksim:
        # +1 forces registration after an injected target has been spliced.
        sources.append(
            {
                "cache_id": cksim_dump_cache_id(arm, task, skill, occurrence),
                "source_start": start,
                "source_end": end + 1,
            }
        )
    if not sources and not targets:
        return None
    return {"sources": sources, "targets": targets}


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--modes", default="recompute,direct,rope")
    parser.add_argument("--occurrences", default="2,3")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--cksim-kv-dir", default=str(DEFAULT_CKSIM_KV_DIR))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of decode samples per (case, mode). Use >1 with a non-zero "
            "temperature to measure whether trajectory divergences are a stable "
            "systematic effect or just sampling noise."
        ),
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="If set, sample i of each case uses seed = seed-base + i for reproducibility.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "If set, run only this single sample index instead of looping over "
            "range(repeats). Used by run_decode_compare.sh to restart vLLM between "
            "repeats: each repeat must see a fresh prefix cache, since a repeated "
            "identical injection request would otherwise find its own prior pass's "
            "prompt already fully cached and re-trigger the WAITING-queue "
            "ContextSegmentKV/Prometheus bug."
        ),
    )
    parser.add_argument("--dump-kv-for-cksim", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid_modes = sorted(set(modes) - set(ARMS))
    if invalid_modes:
        raise ValueError(
            f"Unsupported modes: {invalid_modes}. Known arms: {sorted(ARMS)}"
        )
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.sample_index is not None and args.repeats != 1:
        raise ValueError("--sample-index requires --repeats 1 (one sample per invocation)")
    sample_indices = (
        [args.sample_index] if args.sample_index is not None else list(range(args.repeats))
    )
    occurrences = [int(x) for x in args.occurrences.split(",") if x.strip()]
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    output = Path(args.output)
    if output.exists() and not args.append:
        output.unlink()
    system_prompt = load_system_prompt()
    tools = load_tools()
    cases = selected_cases(tasks, occurrences)

    existing: set[tuple[str, str, int, str, int]] = set()
    if args.skip_existing and output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error"):
                    # Error rows are incomplete attempts, not completed work.
                    # Leave them in the JSONL for auditability, but retry the
                    # same case/sample on resumed runs.
                    continue
                existing.add(
                    (
                        row["task"],
                        row["skill"],
                        int(row["occurrence"]),
                        row["mode"],
                        int(row.get("sample_index", 0)),
                    )
                )

    rows: list[dict[str, Any]] = []
    for case in cases:
        invocation = load_invocations(case["task"])[case["invocation_index"] - 1]
        messages, _ = convert_messages(invocation["messages"], system_prompt)
        for mode in modes:
            cfg = context_config_for_case(
                mode,
                case,
                dump_kv_for_cksim=args.dump_kv_for_cksim,
            )
            for sample_index in sample_indices:
                existing_key = (
                    case["task"],
                    case["skill"],
                    int(case["occurrence"]),
                    mode,
                    sample_index,
                )
                if existing_key in existing:
                    print(
                        f"[skip-existing] {mode:9s} {case['task']} {case['skill']} "
                        f"occ{case['occurrence']} s{sample_index}",
                        flush=True,
                    )
                    continue
                seed = (
                    None if args.seed_base is None else args.seed_base + sample_index
                )
                request_id = (
                    f"cf-decode-{mode}-{case['task']}-{case['skill']}"
                    f"-occ{case['occurrence']}-s{sample_index}"
                )
                try:
                    response, elapsed = chat_completion(
                        base_url,
                        args.model,
                        messages,
                        tools,
                        args.api_key,
                        max_tokens=args.max_tokens,
                        request_id=request_id,
                        context_segment_cache=cfg,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=seed,
                    )
                    parts = extract_response(response)
                    error = None
                except RuntimeError as exc:
                    response = {}
                    elapsed = 0.0
                    parts = {
                        "text": "",
                        "content": "",
                        "reasoning": "",
                        "tool_calls": [],
                        "finish_reason": None,
                    }
                    error = str(exc)
                usage = response.get("usage", {})
                row = {
                    **case,
                    "mode": mode,
                    "sample_index": sample_index,
                    "temperature": args.temperature,
                    "seed": seed,
                    "cache_id": resolve_cache_id(mode, case),
                    "cksim_cache_id": (
                        cksim_dump_cache_id(
                            mode,
                            case["task"],
                            case["skill"],
                            int(case["occurrence"]),
                        )
                        if args.dump_kv_for_cksim
                        else None
                    ),
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "latency_s": round(elapsed, 4),
                    "usage": usage,
                    "error": error,
                    "text": parts["text"],
                    "content": parts["content"],
                    "reasoning": parts["reasoning"],
                    "tool_calls": parts["tool_calls"],
                    "finish_reason": parts["finish_reason"],
                }
                rows.append(row)
                append_jsonl(output, row)
                status = "error" if error else "ok"
                print(
                    f"[{status}] {mode:9s} {case['task']} {case['skill']} "
                    f"occ{case['occurrence']} s{sample_index} "
                    f"chars={len(parts['text'])} think={len(parts['reasoning'])}",
                    flush=True,
                )

    metadata = {
        "model": args.model,
        "base_url": base_url,
        "tasks": tasks,
        "modes": modes,
        "occurrences": occurrences,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed_base": args.seed_base,
        "output": str(output),
        "offline_kv_dir_expected_by_server": str(Path(args.kv_dir)),
        "cksim_save_dir_expected_by_server": str(Path(args.cksim_kv_dir)),
        "note": (
            "For direct/rope to actually inject, start vLLM with "
            "VLLM_CONTEXT_SEGMENT_KV_DIR pointing at offline_kv_dir. "
            "For CKSim dumps, pass --dump-kv-for-cksim and start vLLM with "
            "VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR pointing at cksim_save_dir. "
            "CKSim dumping keeps collected tensors in the vLLM registry and can "
            "OOM on long prompts, so it is disabled by default for decode runs."
        ),
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] outputs: {output}")
    print(f"[done] metadata: {meta_path}")


if __name__ == "__main__":
    main()
