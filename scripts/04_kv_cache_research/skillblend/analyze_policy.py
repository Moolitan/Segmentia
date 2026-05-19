"""Run the SkillBlend policy over existing KV-reuse experiment summaries.

This is a lightweight bridge from the current motivation experiments to the
future LMCache/vLLM prototype. It does not run a model; it uses token lengths
and query-anchor lengths already recorded by the experiment JSON files.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from skillblend.policy import (  # noqa: E402
    ReuseMode,
    SkillBlendConfig,
    SkillCachePolicy,
    SkillRequestContext,
    SkillSegment,
    summarize_plans,
)


DEFAULT_RESULTS_DIR = Path(
    "scripts/04_kv_cache_research/results/kv_reuse_natural_reference_block"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--query-threshold", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--recompute-ratios", default="0.05,0.12,0.18")
    parser.add_argument("--critical-mode", action="store_true")
    parser.add_argument("--load-ms-per-token", type=float, default=0.004)
    parser.add_argument("--prefill-ms-per-token", type=float, default=0.030)
    parser.add_argument("--storage-location", default="LocalCPUBackend")
    return parser.parse_args()


def load_case_files(results_dir: Path) -> list[tuple[str, Path]]:
    groups = [
        ("multi_prompt_with_query", results_dir / "supplement_exp1_multi_prompt" / "cases"),
        (
            "multi_prompt_no_query",
            results_dir / "supplement_exp1_multi_prompt_no_query" / "cases",
        ),
        ("gap_scan_with_query", results_dir / "supplement_exp2_gap_scan" / "gaps"),
        (
            "gap_scan_no_query",
            results_dir / "supplement_exp2_gap_scan_no_query" / "gaps",
        ),
    ]
    files: list[tuple[str, Path]] = []
    for group, directory in groups:
        if directory.exists():
            files.extend((group, path) for path in sorted(directory.glob("*_summary.json")))
    return files


def build_context(
    *,
    group: str,
    data: dict[str, Any],
    storage_location: str,
    load_ms_per_token: float,
    prefill_ms_per_token: float,
) -> SkillRequestContext:
    info = data["sequence_info"]
    case_name = data["case_name"]
    skill_len = int(info.get("L_reference", 0))
    query_anchor_len = int(info.get("query_tokens_after_reference_v2", 0))
    position_gap = int(info.get("reference_start_position_gap", 0))
    token_range_raw = info.get("reference_v2") or info.get("skill_v2")
    if token_range_raw is None:
        token_range = (0, skill_len)
    else:
        token_range = (int(token_range_raw[0]), int(token_range_raw[1]))

    segment = SkillSegment(
        skill_id=case_name,
        version_hash="from_existing_experiment",
        token_hash=f"{group}:{case_name}",
        token_range=token_range,
        skill_type=classify_skill_type(case_name),
        length=skill_len,
    )

    estimated_load_ms = skill_len * load_ms_per_token
    estimated_full_recompute_ms = skill_len * prefill_ms_per_token
    return SkillRequestContext(
        segment=segment,
        cache_hit=True,
        query_anchor_len=query_anchor_len,
        position_gap=position_gap,
        storage_location=storage_location,
        estimated_load_ms=estimated_load_ms,
        estimated_full_recompute_ms=estimated_full_recompute_ms,
        rope_supported=True,
        hit_count=2,
        request_is_critical=is_critical_case(case_name),
        metadata={"group": group},
    )


def classify_skill_type(case_name: str) -> str:
    lowered = case_name.lower()
    if any(word in lowered for word in ("security", "compliance")):
        return "compliance"
    if any(word in lowered for word in ("platform", "infra", "ops", "retry")):
        return "tool"
    if any(word in lowered for word in ("dashboard", "data", "model", "research")):
        return "analysis"
    return "generic"


def is_critical_case(case_name: str) -> bool:
    lowered = case_name.lower()
    return any(word in lowered for word in ("security", "compliance", "incident"))


def plan_to_dict(group: str, path: Path, plan) -> dict[str, Any]:
    row = asdict(plan)
    row["reuse_mode"] = plan.reuse_mode.value
    row["segment"]["token_range"] = list(plan.segment.token_range)
    row["group"] = group
    row["source"] = str(path)
    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "group",
        "case",
        "query",
        "len",
        "mode",
        "ratio",
        "risk",
        "reason",
    )
    print(" | ".join(header))
    print(" | ".join("-" * len(item) for item in header))
    for row in rows:
        seg = row["segment"]
        reason = row["reason"]
        if len(reason) > 58:
            reason = reason[:55] + "..."
        print(
            " | ".join(
                [
                    row["group"],
                    seg["skill_id"],
                    str(row["query_anchor_len"]),
                    str(seg["length"]),
                    row["reuse_mode"],
                    f"{row['recompute_ratio']:.2f}",
                    f"{row['quality_risk']:.2f}",
                    reason,
                ]
            )
        )


def main() -> int:
    args = parse_args()
    ratios = tuple(float(x) for x in args.recompute_ratios.split(","))
    config = SkillBlendConfig(
        skill_min_tokens=args.min_tokens,
        skill_query_anchor_threshold=args.query_threshold,
        skill_recompute_ratios=ratios,  # type: ignore[arg-type]
        skill_critical_mode=args.critical_mode,
    )
    policy = SkillCachePolicy(config)

    rows: list[dict[str, Any]] = []
    plans = []
    for group, path in load_case_files(args.results_dir):
        data = json.loads(path.read_text())
        if "sequence_info" not in data:
            continue
        ctx = build_context(
            group=group,
            data=data,
            storage_location=args.storage_location,
            load_ms_per_token=args.load_ms_per_token,
            prefill_ms_per_token=args.prefill_ms_per_token,
        )
        plan = policy.plan(ctx)
        plans.append(plan)
        rows.append(plan_to_dict(group, path, plan))

    print_table(rows)
    summary = summarize_plans(plans)
    print("\nsummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "plans": rows}, ensure_ascii=False, indent=2)
            + "\n"
        )
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
