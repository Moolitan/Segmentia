#!/usr/bin/env python3
"""Resolve explicit cross-task Skill pairs to first-use trace requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json, first_skill_request, sha256_text


ROOT = Path(__file__).resolve().parents[3]


def prepare_endpoint(
    *, traces_dir: Path, task: str, skill: str
) -> dict[str, Any]:
    path, invocation, tool_call_id = first_skill_request(traces_dir, task, skill)
    return {
        "task": task,
        "trace_path": str(path.resolve()),
        "turn": int(invocation["turn"]),
        "invocation": int(invocation["invocation"]),
        "target_tool_call_id": tool_call_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--traces-dir", type=Path, default=ROOT / "src" / "traces")
    parser.add_argument("--skills-dir", type=Path, default=ROOT / "skills")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_cases = config.get("cases")
    if config.get("schema_version") != 1 or not isinstance(raw_cases, list):
        raise ValueError("cases config must have schema_version=1 and a cases list")
    wanted = set(args.case_id)
    available = {str(case.get("case_id")) for case in raw_cases}
    missing = wanted - available
    if missing:
        raise ValueError(f"Unknown --case-id values: {sorted(missing)}")

    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case in raw_cases:
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Missing or duplicate case_id: {case_id!r}")
        seen_ids.add(case_id)
        if wanted and case_id not in wanted:
            continue
        skill = str(case["skill"])
        source_task = str(case["source_task"])
        target_task = str(case["target_task"])
        if source_task == target_task:
            raise ValueError(f"Case {case_id} is not cross-task")
        skill_path = args.skills_dir / skill / "SKILL.md"
        skill_content = skill_path.read_text(encoding="utf-8")
        if not skill_content.strip():
            raise ValueError(f"Skill file is empty: {skill_path}")
        prepared.append(
            {
                "case_id": case_id,
                "skill": skill,
                "skill_path": str(skill_path.resolve()),
                "skill_sha256": sha256_text(skill_content),
                "source": prepare_endpoint(
                    traces_dir=args.traces_dir, task=source_task, skill=skill
                ),
                "target": prepare_endpoint(
                    traces_dir=args.traces_dir, task=target_task, skill=skill
                ),
            }
        )
    prepared.sort(
        key=lambda row: (
            row["skill"],
            row["source"]["task"],
            row["target"]["task"],
        )
    )
    if not prepared:
        raise ValueError("No capture cases selected")
    output = {
        "schema_version": 1,
        "config_path": str(args.config.resolve()),
        "traces_dir": str(args.traces_dir.resolve()),
        "skills_dir": str(args.skills_dir.resolve()),
        "cases": prepared,
    }
    atomic_write_json(args.output, output)
    for case in prepared:
        print(
            f"{case['case_id']}: skill={case['skill']} "
            f"source={case['source']['task']}:t{case['source']['turn']}i{case['source']['invocation']} "
            f"target={case['target']['task']}:t{case['target']['turn']}i{case['target']['invocation']}"
        )


if __name__ == "__main__":
    main()
