"""
Supplement experiment 1: multi-prompt statistics.

This script runs the A/B1/B2 KV reuse comparison on multiple independent
prompt instances, then reports mean/std for distribution and text metrics.

Run:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock/run_supplement_exp1_multi_prompt.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
COMMON_DIR = BASE_DIR / "kvreused_naturalreferenceblock"
if (COMMON_DIR / "kv_reuse_common.py").exists():
    sys.path.append(str(COMMON_DIR))

from kv_reuse_common import KVReuseRunner, PromptSpec, ensure_dir, mean_std
from run_supplement_exp2_gap_scan import (
    dump_a_focused_attention_map,
    parse_layer_list,
)

DEFAULT_MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
DEFAULT_OUT_DIR = (
    BASE_DIR
    / "results"
    / "kv_reuse_natural_reference_block"
    / "supplement_exp1_multi_prompt"
)
SCENARIO = "KVReuse_NaturalReferenceBlock_MultiPrompt"


@dataclass(frozen=True)
class MultiPromptCase:
    name: str
    assistant_role: str
    format_name: str
    title_prefix: str
    sections: tuple[str, str, str]
    tone_rule: str
    first_topic: str
    second_topic: str
    first_facts: tuple[str, str, str]
    second_facts: tuple[str, str, str]


def reference_card(case: MultiPromptCase) -> str:
    s1, s2, s3 = case.sections
    return f"""\
Reference Card: {case.format_name}

1. Start with a one-line title in the form "{case.title_prefix} - <topic>".
2. Use exactly three sections: {s1}, {s2}, {s3}.
3. Under each section, write 2 short bullet points.
4. Keep the tone {case.tone_rule}.
5. Prefer concrete facts over vague claims.
6. Mention rollout status, owner, or next checkpoint when relevant.
7. Keep the total length between 120 and 180 words.
8. End with one sentence stating the immediate next checkpoint.
"""


def first_answer(case: MultiPromptCase) -> str:
    s1, s2, s3 = case.sections
    f1, f2, f3 = case.first_facts
    return f"""\
{case.title_prefix} - {case.first_topic}

{s1}
- {f1}
- The current status is stable enough for the next review checkpoint.

{s2}
- {f2}
- The owner will confirm readiness before expanding the work.

{s3}
- {f3}
- The remaining risk is tracked before the next wider rollout.

Immediate next checkpoint: confirm the owner, status, and rollout decision in the next review.
"""


def realistic_middle_history(case: MultiPromptCase) -> str:
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
That format works. I have another request, and I am pasting the same reference card again below
so we use the exact same format.

[assistant]
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
    middle = realistic_middle_history(case)
    query = f"""
[user]
Now use that same reference card to write a new update about {case.second_topic}.

Facts:
- {case.second_facts[0]}
- {case.second_facts[1]}
- {case.second_facts[2]}

Keep it concise and operational.

[assistant]

"""
    return PromptSpec(
        name=case.name,
        reference_text=ref,
        user_context=user_context,
        middle_history=middle,
        query_text=query,
        note="multi-prompt statistics; decode starts after reference_v2 + query",
    )


def built_in_cases() -> list[MultiPromptCase]:
    return [
        MultiPromptCase(
            "platform_retry",
            "a concise workplace writing assistant",
            "Platform Update Format",
            "Platform Update",
            ("Progress", "Plans", "Problems"),
            "operational, calm, and specific",
            "dataset sync migration",
            "failed-task retry support in the experiment dashboard",
            (
                "the new sync pipeline is running in staging; checksum validation caught two schema mismatches",
                "start a 10% production rollout on Monday; publish rollback steps",
                "one downstream dashboard still reads from the legacy table",
            ),
            (
                "the retry endpoint is deployed; the UI retry button is live for 30% of users",
                "expand to 100% next week; add retry success rate to the weekly ops review",
                "older tasks still show generic failure states; retry alerts are noisier than expected",
            ),
        ),
        MultiPromptCase(
            "data_quality",
            "a data operations update writer",
            "Data Quality Review Format",
            "Data Quality Review",
            ("Signals", "Actions", "Risks"),
            "factual, plain, and non-promotional",
            "warehouse freshness checks",
            "billing-event duplicate detection",
            (
                "freshness checks now run every 15 minutes for the core warehouse tables",
                "add a daily owner summary and route failures to the analytics on-call",
                "two legacy jobs still report late because their source systems batch overnight",
            ),
            (
                "duplicate detection is active for billing events; sampled precision is above 98%",
                "backfill labels for the last 45 days and expose counts in the quality dashboard",
                "one partner feed still sends retries without stable request identifiers",
            ),
        ),
        MultiPromptCase(
            "model_release",
            "a model deployment note writer",
            "Model Release Note Format",
            "Model Release",
            ("Ready", "Next", "Watchouts"),
            "technical, careful, and concrete",
            "ranking model shadow test",
            "intent classifier rollout",
            (
                "the ranking model completed a 7-day shadow test with no latency regression",
                "move to 5% traffic and compare click-through and complaint rate daily",
                "the evaluation set still under-represents long-tail enterprise queries",
            ),
            (
                "the classifier is packaged and offline accuracy improved by 2.1 points",
                "enable canary routing for support traffic and publish rollback thresholds",
                "ambiguous short queries still trigger unstable labels in manual review",
            ),
        ),
        MultiPromptCase(
            "incident_followup",
            "an incident follow-up assistant",
            "Incident Follow-up Format",
            "Incident Follow-up",
            ("Status", "Mitigation", "Follow-up"),
            "direct, accountable, and specific",
            "cache saturation event",
            "queue backlog recovery",
            (
                "cache saturation peaked for 18 minutes and error rates returned to baseline",
                "raise eviction headroom and add a saturation alert before the next release",
                "the load-test profile did not include the largest tenant burst pattern",
            ),
            (
                "the backlog is down from 1.8 million jobs to 140 thousand jobs",
                "run a controlled drain test and document the manual pause procedure",
                "worker autoscaling still reacts too slowly during sharp ingest spikes",
            ),
        ),
        MultiPromptCase(
            "support_ops",
            "a support operations communicator",
            "Support Ops Brief Format",
            "Support Ops Brief",
            ("Resolved", "Pending", "Blockers"),
            "brief, practical, and customer-aware",
            "export ticket backlog",
            "workspace invite failures",
            (
                "the oldest export tickets are cleared and median wait time is below 6 hours",
                "add weekend coverage and publish a status macro for support agents",
                "large CSV exports still time out for accounts with more than 5 million rows",
            ),
            (
                "invite failure volume dropped after the email validation fix reached production",
                "review bounced-domain logs and add self-serve resend guidance",
                "some SSO-backed workspaces still need manual identity-provider checks",
            ),
        ),
        MultiPromptCase(
            "infra_capacity",
            "an infrastructure capacity update writer",
            "Capacity Update Format",
            "Capacity Update",
            ("Capacity", "Changes", "Risks"),
            "measured, operational, and specific",
            "GPU pool expansion",
            "Postgres storage headroom",
            (
                "the new GPU nodes are provisioned and idle capacity is above 22%",
                "shift batch inference to the new pool and retire two older node groups",
                "one region still waits on quota approval from the cloud provider",
            ),
            (
                "storage headroom is back to 31% after vacuum and index cleanup",
                "complete partition pruning and review growth projections every Friday",
                "two audit tables are growing faster than the retention policy assumed",
            ),
        ),
        MultiPromptCase(
            "compliance",
            "a compliance status writer",
            "Compliance Checklist Format",
            "Compliance Checklist",
            ("Completed", "Upcoming", "Gaps"),
            "precise, audit-friendly, and concise",
            "vendor access review",
            "encryption evidence package",
            (
                "access records for 34 vendors are reconciled against the procurement list",
                "send exception owners a sign-off request and archive evidence by Friday",
                "three vendors still lack named business owners in the tracking sheet",
            ),
            (
                "database encryption screenshots are collected for all production clusters",
                "map the evidence to the control matrix and request security review",
                "two staging clusters need updated key-rotation proof before closure",
            ),
        ),
        MultiPromptCase(
            "research_milestone",
            "a research program update assistant",
            "Research Milestone Format",
            "Research Milestone",
            ("Findings", "Next Tests", "Risks"),
            "clear, empirical, and restrained",
            "retrieval ablation run",
            "KV reuse quality scan",
            (
                "the ablation shows a 3.4 point gain when reranking is enabled",
                "rerun with a larger seed set and export per-query failure cases",
                "the current benchmark overweights short factual questions",
            ),
            (
                "initial KV reuse runs keep argmax agreement above 94% on the pilot prompt",
                "add multi-prompt statistics and scan larger position gaps",
                "the current evidence is single-model and may not transfer across RoPE variants",
            ),
        ),
        MultiPromptCase(
            "security_rollout",
            "a security rollout update writer",
            "Security Rollout Format",
            "Security Rollout",
            ("Progress", "Controls", "Open Issues"),
            "specific, low-drama, and action-oriented",
            "admin session hardening",
            "secret scanning enforcement",
            (
                "new admin sessions now require step-up verification after 30 minutes",
                "enable reporting for all admin groups and tune the helpdesk escalation path",
                "two service accounts still rely on long-lived console sessions",
            ),
            (
                "secret scanning is active on protected branches for 82 repositories",
                "turn warnings into blocking checks after the developer notice period",
                "three generated-client repos still produce noisy false positives",
            ),
        ),
        MultiPromptCase(
            "onboarding",
            "an internal enablement update writer",
            "Onboarding Update Format",
            "Onboarding Update",
            ("Done", "Next", "Friction"),
            "helpful, concrete, and concise",
            "new hire lab environment",
            "manager checklist refresh",
            (
                "the lab image now includes the CLI, sample data, and a reset script",
                "pilot with the next onboarding cohort and collect setup-time measurements",
                "Windows users still need a separate credential-helper step",
            ),
            (
                "the manager checklist now includes buddy assignment and first-week goals",
                "publish reminders in the people-ops channel and collect completion rates",
                "some teams still use local spreadsheets instead of the central tracker",
            ),
        ),
        MultiPromptCase(
            "analytics_dashboard",
            "an analytics product update writer",
            "Dashboard Update Format",
            "Dashboard Update",
            ("Changes", "Validation", "Issues"),
            "plain, metric-aware, and specific",
            "revenue dashboard refresh",
            "cohort retention view",
            (
                "the refreshed dashboard now uses the canonical revenue table",
                "compare totals against finance exports and publish validation notes",
                "one regional filter still maps legacy market names incorrectly",
            ),
            (
                "the retention view now supports weekly cohorts and plan-tier filters",
                "validate against notebook calculations and add saved views for PMs",
                "older accounts with merged workspaces still show duplicate cohort rows",
            ),
        ),
        MultiPromptCase(
            "mobile_reliability",
            "a mobile reliability update writer",
            "Mobile Reliability Format",
            "Mobile Reliability",
            ("Progress", "Plans", "Problems"),
            "calm, user-focused, and concrete",
            "crash-free session improvement",
            "offline draft sync",
            (
                "crash-free sessions improved from 99.1% to 99.5% after the image-cache fix",
                "ship the fix to all users and watch memory pressure by device class",
                "one older Android version still shows elevated startup crashes",
            ),
            (
                "offline draft sync is enabled for 20% of beta users with no data-loss reports",
                "expand the beta and add conflict counts to the release dashboard",
                "large drafts still retry more often on slow networks",
            ),
        ),
    ]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    dist_fields = [
        "KL_first",
        "KL_mean",
        "KL_max",
        "TVD_first",
        "TVD_mean",
        "argmax_match_rate",
        "top5_overlap_mean",
    ]
    text_fields = ["bleu4", "rouge_l", "embed_cosine"]
    summary: dict[str, Any] = {
        "num_cases": len(results),
        "distribution": {"B1": {}, "B2": {}},
        "text": {"B1": {}, "B2": {}},
    }

    for side in ("B1", "B2"):
        for field in dist_fields:
            summary["distribution"][side][field] = mean_std(
                [float(r[side][field]) for r in results]
            )

    text_label = {"B1": "A_vs_B1", "B2": "A_vs_B2"}
    for side, label in text_label.items():
        for field in text_fields:
            vals = [
                float(r["text_metrics"][label][field])
                for r in results
                if label in r.get("text_metrics", {})
            ]
            summary["text"][side][field] = mean_std(vals)
    return summary


def write_plots(results: list[dict[str, Any]], summary: dict[str, Any], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(fig_dir)

    has_text_metrics = bool(results and results[0].get("text_metrics"))
    metrics = [
        ("KL_mean", "distribution", "KL mean", False),
        ("argmax_match_rate", "distribution", "Argmax match", True),
    ]
    if has_text_metrics:
        metrics.extend(
            [
                ("bleu4", "text", "BLEU-4", True),
                ("rouge_l", "text", "ROUGE-L", True),
            ]
        )
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, side, color in [(-width / 2, "B1", "tab:red"), (width / 2, "B2", "tab:orange")]:
        means = [summary[group][side][field]["mean"] for field, group, _, _ in metrics]
        stds = [summary[group][side][field]["std"] for field, group, _, _ in metrics]
        ax.bar(x + offset, means, width, yerr=stds, label=side, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label, _ in metrics])
    ax.set_title("Supplement Exp1: multi-prompt mean/std")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_prompt_metric_bars.png", dpi=160)
    plt.close(fig)

    names = [r["case_name"] for r in results]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(names, [r["B1"]["KL_mean"] for r in results], marker="o", label="B1")
    axes[0].plot(names, [r["B2"]["KL_mean"] for r in results], marker="s", label="B2")
    axes[0].set_ylabel("KL mean")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if has_text_metrics:
        axes[1].plot(
            names,
            [r["text_metrics"]["A_vs_B1"]["bleu4"] for r in results],
            marker="o",
            label="B1",
        )
        axes[1].plot(
            names,
            [r["text_metrics"]["A_vs_B2"]["bleu4"] for r in results],
            marker="s",
            label="B2",
        )
        axes[1].set_ylabel("BLEU-4")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Free-run skipped", ha="center", va="center")
    axes[1].set_xlabel("Prompt case")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.xticks(rotation=35, ha="right")
    fig.suptitle("Supplement Exp1: per-case KL and BLEU")
    fig.tight_layout()
    fig.savefig(fig_dir / "multi_prompt_per_case_curves.png", dpi=160)
    plt.close(fig)


def should_dump_attention_for_case(selector: str, case_name: str, case_names: list[str]) -> bool:
    raw = selector.strip()
    lowered = raw.lower()
    if lowered in {"none", "false", "0", "off"}:
        return False
    if lowered == "first":
        return case_name == case_names[0]
    if lowered == "all":
        return True
    selected = {part.strip() for part in raw.split(",") if part.strip()}
    return case_name in selected


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
        "--attention-gaps",
        dest="attention_cases",
        help=argparse.SUPPRESS,
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
