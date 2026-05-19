"""
Supplement experiment 2: gap-size scan.

The report defines the original "gap" as the start-position difference between
reference_v1 and reference_v2. Because that value is at least the reference
block length, this script scans the controllable middle-history length and also
records the resulting absolute reference start-position gap.

Run:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock/run_supplement_exp2_gap_scan.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from kv_reuse_common import KVReuseRunner, PromptSpec, ensure_dir


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
DEFAULT_OUT_DIR = (
    BASE_DIR
    / "results"
    / "kv_reuse_natural_reference_block"
    / "supplement_exp2_gap_scan"
)
SCENARIO = "KVReuse_NaturalReferenceBlock_GapScan"


REFERENCE_TEXT = """\
Reference Card: Platform Update Format

1. Start with a one-line title in the form "Platform Update - <topic>".
2. Use exactly three sections: Progress, Plans, Problems.
3. Under each section, write 2 short bullet points.
4. Keep the tone operational, calm, and specific.
5. Prefer concrete facts over vague claims.
6. Mention rollout status or the next checkpoint when relevant.
7. Keep the total length between 120 and 180 words.
8. End with one sentence stating the immediate next checkpoint.
"""

USER_CONTEXT = """\
[system]
You are a concise workplace writing assistant. When the user provides a reference block, follow it
closely. Do not mention the reference block unless the user asks.

[user]
I need help drafting internal updates for the platform engineering team. First, please read the
reference card below and keep it in mind for later requests.

[assistant]
"""

QUERY_TEXT = """
[user]
Now use that same reference card to write a new update about failed-task retry support in the
experiment dashboard.

Facts:
- Progress: the retry endpoint is deployed; the UI retry button is live for 30% of users; failure logs now label retryable and non-retryable cases separately.
- Plans: expand to 100% next week; add retry success rate to the weekly ops review; publish a short usage note for researchers.
- Problems: older tasks still show generic failure states; retry-related alerts are noisier than expected.

Keep it concise and operational.

[assistant]

"""

MIDDLE_BASE = """
[assistant]
Understood.
"""

FILLER_UNITS = [
    """
[user]
Before the next update, continue the launch-pack cleanup in the workspace. Start by reading the brand notes.

[assistant]
Agent Action
Summary: Read brand notes before editing launch materials

Action: ReadFileAction
Arguments:
  file_path: "seed_files/brand_notes.md"
""",
    """
[assistant]
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
""",
    """
[user]
Now inspect the current announcement before changing the copy.

[assistant]
Agent Action
Summary: Read the current announcement

Action: ReadFileAction
Arguments:
  file_path: "internal_announcement.md"
""",
    """
[assistant]
Observation
Tool: read_file
Result:
  1  ## Feature Launch: Experiment Task Dashboard Retry Support
  2
  3  The experiment dashboard now includes retry support for failed tasks.
  4
  5  ### Key Benefits
  6  - Retry failed tasks with one click
  7  - Clearer failure reasons
  8  - Less manual restart work
""",
    """
[user]
Make the announcement and GIF feel like one launch pack. Keep the tone technical and remove anything too flashy.

[assistant]
Agent Action
Summary: Replace the failure color with the brand accent

Action: EditAction
Arguments:
  file_path: "retry_animation.py"
  old_string: "fill='red'"
  new_string: "fill='#d97757'"
  expected_replacements: 2
""",
    """
[assistant]
Observation
Tool: edit
Result:
  Edited retry_animation.py with 2 replacements.
""",
    """
[assistant]
Agent Action
Summary: Tighten the announcement sentence

Action: EditAction
Arguments:
  file_path: "internal_announcement.md"
  old_string: "We're pleased to announce a new feature in the experiment task dashboard."
  new_string: "The experiment task dashboard now includes retry support for failed tasks."
  expected_replacements: 1
""",
    """
[assistant]
Observation
Tool: edit
Result:
  Error: Could not find the string to replace. 0 occurrences found. Use read_file to verify the exact text.
""",
    """
[user]
The edit missed the exact text. Re-read the file, then patch only the sentence that needs tightening.

[assistant]
Agent Action
Summary: Re-read the file after the failed edit

Action: ReadFileAction
Arguments:
  file_path: "internal_announcement.md"
""",
    """
[assistant]
Observation
Tool: read_file
Result:
  1  ## Feature Launch: Experiment Task Dashboard Retry Support
  2
  3  We're pleased to announce retry support for failed tasks in the experiment task dashboard.
  4
  5  ### Key Benefits
  6  - Auto-retry failed tasks with one click
  7  - Improved visibility into task failure reasons
  8  - Reduced manual intervention requirements
""",
    """
[assistant]
Agent Action
Summary: Apply the edit against the exact sentence

Action: EditAction
Arguments:
  file_path: "internal_announcement.md"
  old_string: "We're pleased to announce retry support for failed tasks in the experiment task dashboard."
  new_string: "The experiment task dashboard now includes retry support for failed tasks."
""",
    """
[assistant]
Observation
Tool: edit
Result:
  Edited internal_announcement.md with 1 replacement.
""",
    """
[user]
Run the local validation script and summarize what changed.

[assistant]
Agent Action
Summary: Validate the launch pack after the text and color edits

$ python scripts/validate_launch_pack.py
""",
    """
[assistant]
Observation
Tool: terminal
Result:
  Working directory: workspace/03_14B_anthropic_3/slack_launch_pack
  internal_announcement.md: ok
  retry_gif.gif: ok
  brand colors: ok
  Exit code: 0
""",
    """
[assistant]
Validation passed. The announcement now uses direct technical wording, and the GIF uses the brand light background with orange and blue status accents.

[user]
I will paste the same reference card again below before the next task.

[assistant]
Ready.
""",
]


def parse_gap_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Gap values must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one gap value is required.")
    return values


def make_middle_for_target(runner: KVReuseRunner, target_middle_tokens: int) -> str:
    middle = MIDDLE_BASE
    unit_idx = 0
    while runner.token_len(middle) < target_middle_tokens:
        middle += FILLER_UNITS[unit_idx % len(FILLER_UNITS)]
        unit_idx += 1
    return middle


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
        query_text=QUERY_TEXT,
        note=(
            "gap scan; requested gap is middle-history token count between repeated "
            "reference blocks; absolute start-position gap is recorded in sequence_info"
        ),
    )
    return spec, actual_middle_tokens


def write_plots(results: list[dict[str, Any]], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(fig_dir)
    x_middle = [
        r["sequence_info"]["middle_tokens_between_references"]
        for r in results
    ]
    x_start_gap = [
        r["sequence_info"]["reference_start_position_gap"]
        for r in results
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_items = [
        (axes[0, 0], "KL_first", "KL first", "B1", "B2"),
        (axes[0, 1], "KL_mean", "KL mean", "B1", "B2"),
        (axes[1, 0], "argmax_match_rate", "Argmax match", "B1", "B2"),
    ]
    for ax, field, ylabel, side1, side2 in plot_items:
        ax.plot(x_middle, [r[side1][field] for r in results], marker="o", label=side1)
        ax.plot(x_middle, [r[side2][field] for r in results], marker="s", label=side2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    ax = axes[1, 1]
    if results and results[0].get("text_metrics"):
        ax.plot(
            x_middle,
            [r["text_metrics"]["A_vs_B1"]["bleu4"] for r in results],
            marker="o",
            label="B1",
        )
        ax.plot(
            x_middle,
            [r["text_metrics"]["A_vs_B2"]["bleu4"] for r in results],
            marker="s",
            label="B2",
        )
    ax.set_ylabel("BLEU-4")
    ax.grid(alpha=0.3)
    ax.legend()

    for ax in axes[1, :]:
        ax.set_xlabel("Middle-history tokens between reference blocks")
    fig.suptitle("Supplement Exp2: gap scan")
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_scan_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_middle, x_start_gap, marker="o")
    ax.set_xlabel("Middle-history tokens between reference blocks")
    ax.set_ylabel("Reference start-position gap")
    ax.set_title("Middle length vs absolute reference start-position gap")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_scan_actual_position_gap.png", dpi=160)
    plt.close(fig)


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

    results = []
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
            "Requested values are middle-history token counts between repeated "
            "reference blocks. sequence_info.reference_start_position_gap records "
            "the absolute start-position difference used by earlier reports."
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
