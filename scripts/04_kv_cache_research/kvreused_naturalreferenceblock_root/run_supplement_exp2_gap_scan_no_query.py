"""
Supplement experiment 2, no-query variant: gap-size scan.

This mirrors run_supplement_exp2_gap_scan.py, but follows the no-query layout
from run_kv_reuse2.py:

  [USER_CONTEXT] + [REFERENCE_v1] + [MIDDLE_HISTORY_WITH_NEXT_TASK]
  + [REFERENCE_v2] + [ASSISTANT_PREFIX]

The second task request is moved before the repeated reference block. After
reference_v2, only an assistant continuation prefix remains.

Run:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock/run_supplement_exp2_gap_scan_no_query.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR_ROOT = SCRIPT_DIR.parent
COMMON_DIR = BASE_DIR_ROOT / "kvreused_naturalreferenceblock"
if (COMMON_DIR / "kv_reuse_common.py").exists():
    sys.path.append(str(COMMON_DIR))

from kv_reuse_common import KVReuseRunner, PromptSpec, ensure_dir
from run_supplement_exp2_gap_scan import (
    BASE_DIR,
    DEFAULT_MODEL,
    FILLER_UNITS,
    MIDDLE_BASE,
    REFERENCE_TEXT,
    USER_CONTEXT,
    dump_a_focused_attention_map,
    parse_gap_list,
    parse_layer_list,
    should_dump_attention_for_gap,
    write_plots,
)


DEFAULT_OUT_DIR = (
    BASE_DIR
    / "results"
    / "kv_reuse_natural_reference_block"
    / "supplement_exp2_gap_scan_no_query"
)
SCENARIO = "KVReuse_NaturalReferenceBlock_GapScan_NoQuery"

NO_QUERY_REQUEST = """
[user]
That format works. I have another update request about failed-task retry support in the
experiment dashboard.

Use these facts when you continue immediately after the repeated reference card below:
- Progress: the retry endpoint is deployed; the UI retry button is live for 30% of users; failure logs now label retryable and non-retryable cases separately.
- Plans: expand to 100% next week; add retry success rate to the weekly ops review; publish a short usage note for researchers.
- Problems: older tasks still show generic failure states; retry-related alerts are noisier than expected.

Keep it concise and operational. I am pasting the same reference card again below so you can use
the exact same format. As soon as the reference card ends, continue directly with the new update
without waiting for another user turn.
"""

ASSISTANT_PREFIX = """
[assistant]
"""


def make_middle_for_target(runner: KVReuseRunner, target_middle_tokens: int) -> str:
    middle = MIDDLE_BASE
    unit_idx = 0
    while runner.token_len(middle + NO_QUERY_REQUEST) < target_middle_tokens:
        middle += FILLER_UNITS[unit_idx % len(FILLER_UNITS)]
        unit_idx += 1
    return middle + NO_QUERY_REQUEST


def make_prompt_spec(
    runner: KVReuseRunner,
    target_middle_tokens: int,
) -> tuple[PromptSpec, int]:
    middle = make_middle_for_target(runner, target_middle_tokens)
    actual_middle_tokens = runner.token_len(middle)
    spec = PromptSpec(
        name=f"middle_gap_{target_middle_tokens:04d}_actual_{actual_middle_tokens:04d}",
        reference_text=REFERENCE_TEXT,
        user_context=USER_CONTEXT,
        middle_history=middle,
        query_text=ASSISTANT_PREFIX,
        note=(
            "no-query gap scan; second task request is inside middle-history "
            "before reference_v2, and decode starts after assistant prefix"
        ),
    )
    return spec, actual_middle_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--free-max-tokens", type=int, default=256)
    parser.add_argument(
        "--middle-gaps",
        default="50,100,200,400,800,1600,2000",
        help="Comma-separated target middle-history token counts.",
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
        "--attention-gaps",
        default="first",
        help=(
            "Which requested middle gaps to dump focused attention for: first, all, none, "
            "or a comma-separated list such as 50,400."
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
    case_dir = ensure_dir(out_dir / "gaps")
    fig_dir = ensure_dir(out_dir / "figures")
    target_gaps = parse_gap_list(args.middle_gaps)
    attention_layers = parse_layer_list(args.attention_layers)
    if args.attention_decode_steps <= 0:
        raise ValueError("--attention-decode-steps must be positive.")
    if args.attention_chunk <= 0:
        raise ValueError("--attention-chunk must be positive.")
    runner = KVReuseRunner(
        args.model_path,
        t_steps=args.t_steps,
        free_max_tokens=args.free_max_tokens,
        dtype=torch.bfloat16,
        device=args.device,
    )

    results: list[dict[str, Any]] = []
    requested_to_actual = []
    attention_summaries = []
    for target_middle in target_gaps:
        spec, actual_middle = make_prompt_spec(runner, target_middle)
        if (
            not args.no_dump_a_attention
            and should_dump_attention_for_gap(args.attention_gaps, target_middle, target_gaps)
        ):
            attention_summary = dump_a_focused_attention_map(
                runner,
                spec,
                out_dir=out_dir,
                target_layers=attention_layers,
                decode_steps=args.attention_decode_steps,
                chunk_size=args.attention_chunk,
            )
            attention_summary["requested_middle_gap_tokens"] = target_middle
            attention_summary["actual_middle_gap_tokens"] = actual_middle
            attention_summaries.append(attention_summary)

        if args.attention_only:
            requested_to_actual.append(
                {
                    "requested_middle_gap_tokens": target_middle,
                    "actual_middle_gap_tokens": actual_middle,
                }
            )
            continue

        result = runner.run_prompt(spec, scenario=SCENARIO)
        result["requested_middle_gap_tokens"] = target_middle
        result["actual_middle_gap_tokens"] = actual_middle
        results.append(result)
        requested_to_actual.append(
            {
                "requested_middle_gap_tokens": target_middle,
                "actual_middle_gap_tokens": actual_middle,
                "reference_start_position_gap": result["sequence_info"][
                    "reference_start_position_gap"
                ],
            }
        )
        with open(case_dir / f"{spec.name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    final = {
        "scenario": SCENARIO,
        "model": args.model_path,
        "T_steps": args.t_steps,
        "free_max_tokens": args.free_max_tokens,
        "gap_definition": (
            "Requested values are target total middle-history token counts between "
            "repeated reference blocks. In the no-query layout, the second task "
            "request must live inside middle-history, so small requested values may "
            "map to the fixed minimum actual middle length. "
            "sequence_info.reference_start_position_gap records the absolute "
            "reference start-position difference."
        ),
        "attention": {
            "enabled": not args.no_dump_a_attention,
            "attention_only": bool(args.attention_only),
            "attention_gaps": args.attention_gaps,
            "attention_layers": attention_layers,
            "attention_decode_steps": args.attention_decode_steps,
            "attention_chunk": args.attention_chunk,
            "summaries": attention_summaries,
        },
        "requested_to_actual": requested_to_actual,
        "results": results,
    }
    with open(out_dir / f"{SCENARIO}_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    if results:
        write_plots(results, fig_dir)
    print(f"\nWrote summary: {out_dir / f'{SCENARIO}_summary.json'}")
    print(f"Wrote per-gap JSON: {case_dir}")
    if results:
        print(f"Wrote figures: {fig_dir}")
    if attention_summaries:
        print(f"Wrote attention outputs: {out_dir / 'attention'}")


if __name__ == "__main__":
    main()
