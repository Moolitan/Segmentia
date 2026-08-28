"""Recover completed rollouts after a derived-metric parser fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.run_state import RunContext, utc_now
from common.schema import write_csv

from analyze import analyze
from run import (
    QUALITY_SAMPLE_COLUMNS,
    _sample_is_healthy,
    _vllm_totals,
)


def recover(run_dir: Path) -> None:
    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    active = json.loads(
        (run_dir.parent / "active_run.json").read_text(encoding="utf-8")
    )
    if Path(str(active.get("run_dir", ""))).resolve() != run_dir:
        raise RuntimeError("recovery is restricted to the section's active run")
    samples = []
    recovered_attempts = []
    attempts_by_case = {}
    for case_root in sorted((run_dir / "cases").iterdir()):
        parsed_paths = sorted(case_root.glob("attempt-*/parsed_result.json"))
        if not parsed_paths:
            continue
        parsed_path = parsed_paths[-1]
        attempt = parsed_path.parent
        sample = json.loads(parsed_path.read_text(encoding="utf-8"))
        sample.update(
            _vllm_totals(
                attempt / "server/vllm_timeline.jsonl",
                attempt / "server/vllm.log",
            )
        )
        healthy = _sample_is_healthy(sample)
        sample["pipeline_healthy"] = healthy
        sample["status"] = "completed" if healthy else "invalid"
        if not healthy:
            raise RuntimeError(f"rollout remains unhealthy after recovery: {parsed_path}")
        parsed_path.write_text(
            json.dumps(sample, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        samples.append(sample)
        recovered_attempts.append(str(attempt))
        attempts_by_case[str(sample["case_id"])] = str(attempt)

    if not samples:
        raise RuntimeError(f"no parsed rollout results found below {run_dir}")
    write_csv(run_dir / "samples.csv", samples, QUALITY_SAMPLE_COLUMNS)
    (run_dir / "samples.jsonl").write_text(
        "".join(
            json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    context = RunContext(
        section=str(state["section"]),
        run_id=str(state["run_id"]),
        run_dir=run_dir,
        fingerprint=str(state["input_fingerprint"]),
        state=state,
    )
    for sample in samples:
        context.mark(
            str(sample["case_id"]),
            "completed",
            attempt_dir=attempts_by_case[str(sample["case_id"])],
            recovered_utc=utc_now(),
        )
    analyze(run_dir)
    (run_dir / "recovery.json").write_text(
        json.dumps(
            {
                "reason": (
                    "OpenHands requests are not tagged with the cskcache-latency "
                    "marker; successful vLLM access-log requests are the fallback "
                    "execution evidence."
                ),
                "recovered_attempts": recovered_attempts,
                "recovered_utc": utc_now(),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    context.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    recover(args.run_dir.resolve())


if __name__ == "__main__":
    main()
