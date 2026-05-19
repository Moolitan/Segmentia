"""
Supplement experiment 1, no-query variant: multi-prompt statistics.

This mirrors run_supplement_exp1_multi_prompt.py, but follows the no-query
layout from run_kv_reuse2.py:

  [USER_CONTEXT] + [REFERENCE_v1] + [MIDDLE_HISTORY_WITH_NEXT_TASK]
  + [REFERENCE_v2] + [ASSISTANT_PREFIX]

The second task request is moved before the repeated reference block. After
reference_v2, only an assistant continuation prefix remains.

Run:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock/run_supplement_exp1_multi_prompt_no_query.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR_ROOT = SCRIPT_DIR.parent
COMMON_DIR = BASE_DIR_ROOT / "kvreused_naturalreferenceblock"
if (COMMON_DIR / "kv_reuse_common.py").exists():
    sys.path.append(str(COMMON_DIR))

from kv_reuse_common import KVReuseRunner, PromptSpec, ensure_dir
from run_supplement_exp1_multi_prompt import (
    BASE_DIR,
    DEFAULT_MODEL,
    MultiPromptCase,
    built_in_cases,
    dump_a_focused_attention_map,
    first_answer,
    parse_layer_list,
    reference_card,
    should_dump_attention_for_case,
    summarize_results,
    write_plots,
)


DEFAULT_OUT_DIR = (
    BASE_DIR
    / "results"
    / "kv_reuse_natural_reference_block"
    / "supplement_exp1_multi_prompt_no_query"
)
SCENARIO = "KVReuse_NaturalReferenceBlock_MultiPrompt_NoQuery"

ASSISTANT_PREFIX = """
[assistant]
"""


def no_query_middle_history(case: MultiPromptCase) -> str:
    return f"""
[assistant]
Understood. I will use the reference card for the next request.

[user]
Using that format, write an update about {case.first_topic}.

Facts:
- {case.first_facts[0]}
- {case.first_facts[1]}
- {case.first_facts[2]}

[assistant]
{first_answer(case)}

[user]
Before the next writing task, continue the launch-pack cleanup in the workspace. Read the brand notes, check the current announcement, make
one focused edit if needed, and run the local validation script.

[assistant]
Agent Action
Summary: Read the brand notes before changing the launch materials

Action: ReadFileAction
Arguments:
  file_path: "seed_files/brand_notes.md"

Observation
Tool: read_file
Result:
  1  # Brand Notes
  2  Style requirements:
  3  - simple
  4  - clear
  5  - trustworthy
  6  - technical-team tone
  7  - not flashy
  8
  9  Visual direction:
 10  - calm colors
 11  - simple shapes
 12  - avoid excessive exclamation marks or hype language

Agent Action
Summary: Inspect the current announcement before editing

Action: ReadFileAction
Arguments:
  file_path: "internal_announcement.md"

Observation
Tool: read_file
Result:
  1  ## Feature Launch: {case.first_topic.title()}
  2
  3  We're pleased to announce an update for internal users.
  4
  5  ### Key Benefits
  6  - Faster follow-up for operators
  7  - Better visibility into current status
  8  - Less manual coordination

Agent Action
Summary: Try to tighten the announcement language

Action: EditAction
Arguments:
  file_path: "internal_announcement.md"
  old_string: "We're pleased to announce an update for internal users."
  new_string: "The update is now ready for internal review and staged rollout."

Observation
Tool: edit
Result:
  Error: Could not find the string to replace. 0 occurrences found. Use read_file to verify the exact text.

Agent Action
Summary: Re-read the file and apply the edit against the exact current text

Action: ReadFileAction
Arguments:
  file_path: "internal_announcement.md"

Observation
Tool: read_file
Result:
  1  ## Feature Launch: {case.first_topic.title()}
  2
  3  The update is ready for internal users, with rollout checks still active.
  4
  5  ### Key Benefits
  6  - Faster follow-up for operators
  7  - Better visibility into current status
  8  - Less manual coordination

Action: EditAction
Arguments:
  file_path: "internal_announcement.md"
  old_string: "The update is ready for internal users, with rollout checks still active."
  new_string: "The update is ready for internal review, with rollout checks still active."

Observation
Tool: edit
Result:
  Edited internal_announcement.md with 1 replacement.

Agent Action
Summary: Run the local content check

$ python scripts/validate_launch_pack.py

Observation
Tool: terminal
Result:
  Working directory: workspace/launch_pack
  announcement.md: ok
  retry_gif.gif: ok
  Exit code: 0

[assistant]
Launch-pack cleanup is complete. I tightened the wording against the brand notes, kept the asset set unchanged, and validation passed.

[user]
That format works. I have another update request about {case.second_topic}.

Use these facts when you continue immediately after the repeated reference card below:
- {case.second_facts[0]}
- {case.second_facts[1]}
- {case.second_facts[2]}

Keep it concise and operational. I am pasting the same reference card again below so you can use
the exact same format. As soon as the reference card ends, continue directly with the new update
without waiting for another user turn.
"""


def make_prompt_spec(case: MultiPromptCase) -> PromptSpec:
    ref = reference_card(case)
    user_context = f"""\
[system]
You are {case.assistant_role}. When the user provides a reference block, follow it closely.
Do not mention the reference block unless the user asks.

[user]
First, read the reference card below and keep it in mind for later writing tasks.

[assistant]
"""
    return PromptSpec(
        name=case.name,
        reference_text=ref,
        user_context=user_context,
        middle_history=no_query_middle_history(case),
        query_text=ASSISTANT_PREFIX,
        note=(
            "multi-prompt no-query statistics; second task request is before "
            "reference_v2 and decode starts after assistant prefix"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--free-max-tokens", type=int, default=256)
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Run only the first N built-in cases; 0 means all cases.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-dump-a-attention",
        action="store_true",
        help="Disable the default focused attention dump.",
    )
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="Only dump A/B1/B2 focused attention and skip KV-reuse metrics.",
    )
    parser.add_argument(
        "--attention-cases",
        default="first",
        help=(
            "Which prompt cases to dump focused attention for: first, all, none, "
            "or a comma-separated list of case names."
        ),
    )
    parser.add_argument(
        "--attention-layers",
        default="5,15,25",
        help="Comma-separated layer ids for focused attention heatmaps.",
    )
    parser.add_argument(
        "--attention-decode-steps",
        type=int,
        default=64,
        help="Number of greedy decoded tokens to include as heatmap rows per variant.",
    )
    parser.add_argument(
        "--attention-chunk",
        type=int,
        default=512,
        help="Query-row chunk size used by the patched eager attention dump.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    case_dir = ensure_dir(out_dir / "cases")
    fig_dir = ensure_dir(out_dir / "figures")
    attention_layers = parse_layer_list(args.attention_layers)
    if args.attention_decode_steps <= 0:
        raise ValueError("--attention-decode-steps must be positive.")
    if args.attention_chunk <= 0:
        raise ValueError("--attention-chunk must be positive.")

    cases = built_in_cases()
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    specs = [make_prompt_spec(case) for case in cases]
    case_names = [spec.name for spec in specs]

    runner = KVReuseRunner(
        args.model_path,
        t_steps=args.t_steps,
        free_max_tokens=args.free_max_tokens,
        dtype=torch.bfloat16,
        device=args.device,
    )

    results = []
    attention_summaries = []
    for spec in specs:
        if (
            not args.no_dump_a_attention
            and should_dump_attention_for_case(args.attention_cases, spec.name, case_names)
        ):
            attention_summary = dump_a_focused_attention_map(
                runner,
                spec,
                out_dir=out_dir,
                target_layers=attention_layers,
                decode_steps=args.attention_decode_steps,
                chunk_size=args.attention_chunk,
            )
            attention_summaries.append(attention_summary)

        if args.attention_only:
            continue

        result = runner.run_prompt(spec, scenario=SCENARIO)
        results.append(result)
        with open(case_dir / f"{spec.name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    summary = summarize_results(results) if results else {}
    final = {
        "scenario": SCENARIO,
        "model": args.model_path,
        "T_steps": args.t_steps,
        "free_max_tokens": args.free_max_tokens,
        "case_names": [r["case_name"] for r in results] if results else case_names,
        "attention": {
            "enabled": not args.no_dump_a_attention,
            "attention_only": bool(args.attention_only),
            "attention_cases": args.attention_cases,
            "attention_layers": attention_layers,
            "attention_decode_steps": args.attention_decode_steps,
            "attention_chunk": args.attention_chunk,
            "summaries": attention_summaries,
        },
        "summary": summary,
        "results": results,
    }
    with open(out_dir / f"{SCENARIO}_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    if results:
        write_plots(results, summary, fig_dir)
    print(f"\nWrote summary: {out_dir / f'{SCENARIO}_summary.json'}")
    print(f"Wrote per-case JSON: {case_dir}")
    if results:
        print(f"Wrote figures: {fig_dir}")
    if attention_summaries:
        print(f"Wrote attention outputs: {out_dir / 'attention'}")


if __name__ == "__main__":
    main()
