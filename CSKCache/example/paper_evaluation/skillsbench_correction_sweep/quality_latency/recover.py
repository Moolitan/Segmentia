"""Recover derived fields from completed request-pair artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_evaluation.common.driver import (
    _benchmark_request_id,
    _fallback,
)
from paper_evaluation.common.run_state import utc_now

from . import config as local
from .analyze import analyze
from .profile import parse_csk_profile
from .schema import atomic_json, collect_samples


SECTION = "skillsbench_correction_quality_latency"


def reparse_sample(run_dir: Path, sample: dict[str, Any]) -> dict[str, Any]:
    if sample.get("system_family") != "cskcache":
        return dict(sample)
    updated = dict(sample)
    attempt = run_dir / str(sample["attempt_dir"])
    profile_path = attempt / "server/cskcache_profile.jsonl"
    request_id = f"chatcmpl-{_benchmark_request_id(str(sample['case_id']))}"
    fallback, fallback_reason = _fallback(profile_path, request_id)
    invalid_reasons = []
    thinking_path = run_dir / str(sample["thinking_path"])
    if not thinking_path.read_text(encoding="utf-8").strip():
        invalid_reasons.append("thinking_empty")
    if fallback:
        invalid_reasons.append(f"fallback:{fallback_reason}")
    try:
        profile_values = parse_csk_profile(
            profile_path,
            request_id=request_id,
            profile_layer=local.PROFILE_LAYER,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        profile_values = {}
        invalid_reasons.append(f"profile:{type(exc).__name__}: {exc}")
    if profile_values:
        if int(profile_values["matched_tokens"]) != int(sample["skill_tokens"]):
            invalid_reasons.append("matched_tokens_mismatch")
        if int(profile_values["actual_calibration_tokens"]) != int(
            sample["expected_calibration_tokens"]
        ):
            invalid_reasons.append("calibration_tokens_mismatch")
        if int(profile_values["reused_tokens"]) < local.MINIMUM_REUSE_TOKENS:
            invalid_reasons.append("reuse_interval_too_short")
        updated.update(profile_values)
        updated["actual_calibration_ratio"] = (
            int(profile_values["actual_calibration_tokens"])
            / int(sample["skill_tokens"])
        )
    updated.update(
        {
            "status": "valid" if not invalid_reasons else "invalid",
            "invalid_reason": ";".join(invalid_reasons),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
        }
    )
    return updated


def recover(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("section") != SECTION:
        raise RuntimeError(f"not a {SECTION} run: {run_dir}")
    recovered = []
    for sample in collect_samples(run_dir):
        updated = reparse_sample(run_dir, sample)
        sample_path = run_dir / str(updated["attempt_dir"]) / "sample.json"
        atomic_json(sample_path, updated)
        recovered.append(
            {
                "case_id": updated["case_id"],
                "status_before": sample["status"],
                "status_after": updated["status"],
                "invalid_reason_after": updated["invalid_reason"],
            }
        )

    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    current = {sample["case_id"]: sample for sample in collect_samples(run_dir)}
    for case_id, case_state in state.get("cases", {}).items():
        sample = current.get(case_id)
        if sample is not None:
            case_state["sample_status"] = sample["status"]
            case_state["recovered_utc"] = utc_now()
    atomic_json(state_path, state)
    analyze(run_dir)
    atomic_json(
        run_dir / "recovery.json",
        {
            "reason": (
                "vLLM appends an eight-hex engine suffix to CSK profile request "
                "IDs; reparse completed artifacts without model rerun"
            ),
            "recovered_utc": utc_now(),
            "cases": recovered,
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args()
    recover(arguments.run_dir.resolve())


if __name__ == "__main__":
    main()
