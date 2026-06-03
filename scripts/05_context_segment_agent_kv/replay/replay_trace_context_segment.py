from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Put the package root (.../05_context_segment_agent_kv) on sys.path so the
# local `core` package resolves (distinct from the repo-root `core`).
PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import DEFAULT_TASKS, ROOT, TRACES_DIR  # noqa: E402
from core.message_convert import convert_tools  # noqa: E402
from core.replay import replay_task  # noqa: E402
from core.vllm_client import chat_completion  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Replay agent traces against vLLM with ContextSegmentKV skill reuse."
    )
    ap.add_argument("--tasks", default="all", help="comma list of task names, or 'all'")
    ap.add_argument("--mode", choices=["recompute", "reuse", "all"], default="all")
    ap.add_argument("--reuse-scope", choices=["per-task", "cross-task"], default="per-task")
    ap.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    ap.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    ap.add_argument(
        "--output",
        default=str(ROOT / "results" / "05_context_segment_agent_kv" / "replay" / "replay_results.json"),
    )
    return ap


def summarize(all_runs: list[dict]) -> list[dict]:
    """Per-task recompute-vs-reuse diff summary."""
    by_task_mode = {(r["task"], r["mode"]): r for r in all_runs}
    tasks_seen: list[str] = []
    for r in all_runs:
        if r["task"] not in tasks_seen:
            tasks_seen.append(r["task"])

    summary = []
    for task in tasks_seen:
        r = by_task_mode.get((task, "recompute"))
        u = by_task_mode.get((task, "reuse"))
        row: dict = {"task": task}
        if r:
            row["recompute_latency_s"] = r["total_latency_s"]
            row["recompute_prompt_tokens"] = r["total_prompt_tokens"]
        if u:
            row["reuse_latency_s"] = u["total_latency_s"]
            row["reuse_prompt_tokens"] = u["total_prompt_tokens"]
            row["segkv_source_tokens"] = u["segkv_source_tokens"]
            row["segkv_target_tokens"] = u["segkv_target_tokens"]
        if r and u:
            row["latency_delta_s"] = round(u["total_latency_s"] - r["total_latency_s"], 4)
            row["latency_speedup"] = (
                round(r["total_latency_s"] / u["total_latency_s"], 3)
                if u["total_latency_s"] else None
            )
        summary.append(row)
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()

    base_url = f"http://127.0.0.1:{args.vllm_port}"
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    tasks = DEFAULT_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]

    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text()
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text()))

    modes = {"recompute": [False], "reuse": [True], "all": [False, True]}[args.mode]
    run_id = f"{int(time.time())}-{os.getpid()}"

    all_runs: list[dict] = []
    for use_reuse in modes:
        label = "reuse" if use_reuse else "recompute"
        print(f"\n===== mode={label} reuse_scope={args.reuse_scope} =====")
        # Skill token locations now come from core.config.SKILL_TOKEN_LOCATIONS.
        # Those absolute spans were generated with the raw trace system prompt,
        # so do not prepend a replay nonce here.
        pass_system_prompt = system_prompt
        # GPU/kernel warmup so the first timed request isn't penalized.
        chat_completion(
            base_url, args.model,
            [{"role": "system", "content": pass_system_prompt},
             {"role": "user", "content": "warmup"}],
            tools, api_key, max_tokens=1, request_id=f"warmup-{label}",
        )

        collected: set[str] = set()
        for task in tasks:
            # seen_occurrences tracks within-conversation appearances → always
            # reset per task. collected (saved source KVs) follows reuse_scope.
            seen_occurrences: set[tuple[str, int]] = set()
            if args.reuse_scope == "per-task":
                collected = set()
                cache_namespace = f"seg-{run_id}-{label}-{task}"
            else:
                cache_namespace = f"seg-{run_id}-{label}"
            print(f"\n--- task={task} ---")
            run = replay_task(
                task,
                base_url=base_url, model=args.model, api_key=api_key,
                system_prompt=pass_system_prompt, tools=tools,
                use_reuse=use_reuse, collected=collected,
                seen_occurrences=seen_occurrences,
                cache_namespace=cache_namespace, reuse_scope=args.reuse_scope,
            )
            all_runs.append(run)

    summary = summarize(all_runs)
    out = {
        "run_id": run_id,
        "reuse_scope": args.reuse_scope,
        "mode": args.mode,
        "model": args.model,
        "tasks": tasks,
        "summary": summary,
        "runs": all_runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[done] results: {output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
