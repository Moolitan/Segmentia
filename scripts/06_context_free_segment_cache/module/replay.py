"""Reusable replay helpers for Segmentia skill KV experiments."""
from __future__ import annotations

from typing import Any

from config import (
    SKILL_TOKEN_LOCATIONS,
    cache_id_for_skill,
    get_skill_token_span,
)
from trace_utils import load_invocations


def repair_cache_id(arm: str, task: str, skill: str, occurrence: int) -> str:
    """Per-case cache id for repair/controller arms."""
    return f"cf-{arm}-{task}-{skill}-occ{occurrence}"


# An "arm" is one experimental condition. Each maps to the vLLM injection mode
# and cache-id family used at the target skill span.
ARMS: dict[str, dict[str, Any]] = {
    "recompute": {"inject": None, "cache_id": None},
    "direct": {"inject": "direct", "cache_id": "skill"},
    "rope": {"inject": "rope", "cache_id": "skill"},
    "vrep": {"inject": "rope", "cache_id": "repair"},
    "krep": {"inject": "direct", "cache_id": "repair"},
    "oracle": {"inject": "direct", "cache_id": "repair"},
}


def resolve_cache_id(arm: str, case: dict[str, Any]) -> str | None:
    if arm.startswith("xocc_"):
        return repair_cache_id(
            arm, str(case["task"]), str(case["skill"]), int(case["occurrence"])
        )
    kind = ARMS[arm]["cache_id"]
    if kind is None:
        return None
    if kind == "skill":
        return cache_id_for_skill(str(case["skill"]))
    return repair_cache_id(
        arm, str(case["task"]), str(case["skill"]), int(case["occurrence"])
    )


def selected_cases(
    tasks: list[str],
    occurrences: list[int],
    *,
    include_first_occurrence: bool = False,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task in tasks:
        invocations = load_invocations(task)
        task_cases: list[dict[str, Any]] = []
        for skill, record in SKILL_TOKEN_LOCATIONS[task]["skills"].items():
            for occurrence in occurrences:
                if occurrence < 1 or occurrence > len(record["invocation_indices"]):
                    continue
                if occurrence == 1 and not include_first_occurrence:
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
    spec = (
        {"inject": "rope", "cache_id": "repair"}
        if arm.startswith("xocc_")
        else ARMS[arm]
    )
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
