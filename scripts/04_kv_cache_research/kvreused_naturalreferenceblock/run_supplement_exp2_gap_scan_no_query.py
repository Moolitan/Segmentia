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
from pathlib import Path
from typing import Any

import torch

from kv_reuse_common import KVReuseRunner, PromptSpec, ensure_dir
from run_supplement_exp2_gap_scan import (
    BASE_DIR,
    DEFAULT_MODEL,
    FILLER_UNITS,
    MIDDLE_BASE,
    REFERENCE_TEXT,
    USER_CONTEXT,
    parse_gap_list,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    case_dir = ensure_dir(out_dir / "gaps")
    fig_dir = ensure_dir(out_dir / "figures")
    target_gaps = parse_gap_list(args.middle_gaps)
    runner = KVReuseRunner(
        args.model_path,
        t_steps=args.t_steps,
        free_max_tokens=args.free_max_tokens,
        dtype=torch.bfloat16,
        device=args.device,
    )

    results: list[dict[str, Any]] = []
    requested_to_actual = []
    for target_middle in target_gaps:
        spec, actual_middle = make_prompt_spec(runner, target_middle)
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
        "requested_to_actual": requested_to_actual,
        "results": results,
    }
    with open(out_dir / f"{SCENARIO}_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    write_plots(results, fig_dir)
    print(f"\nWrote summary: {out_dir / f'{SCENARIO}_summary.json'}")
    print(f"Wrote per-gap JSON: {case_dir}")
    print(f"Wrote figures: {fig_dir}")


if __name__ == "__main__":
    main()
