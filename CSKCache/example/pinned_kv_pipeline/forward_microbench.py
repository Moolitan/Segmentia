"""Run and summarize the calibration-forward microbenchmark."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from forward_microbench_config import (
    CALIBRATION_TOKENS,
    CHUNK_SIZE_TOKENS,
    CSK_CASE_CONFIG_ENV,
    EXECUTION_ORDER,
    HOST_LAYOUT,
    NATIVE_CASE_CONFIG_ENV,
    NUM_LAYERS,
    PROFILE_LAYER_IDS,
    REPETITIONS,
    RUN_ROOT,
    SKILL_TOKENS,
    STABLE_LAYER_START,
    STABLE_LAYER_STOP,
    STORAGE_LAYOUT,
    WARMUP_REQUESTS,
)


HERE = Path(__file__).resolve().parent
CSK_RUNNER = HERE / "run.py"
NATIVE_RUNNER = HERE / "native_forward_case.py"
CASE_SPECS = RUN_ROOT / "case_specs"
LOG_DIR = RUN_ROOT / "logs"

BREAKDOWN_FIELDS = (
    "input_norm_ms",
    "qkv_projection_ms",
    "q_norm_ms",
    "k_norm_ms",
    "position_build_ms",
    "rope_ms",
    "prefix_paged_kv_ms",
    "kv_concat_ms",
    "attention_ms",
    "output_projection_ms",
    "post_attention_norm_ms",
    "mlp_ms",
    "module_span_ms",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _run_child(
    runner: Path,
    config_env: str,
    spec_path: Path,
    log_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> None:
    environment = os.environ.copy()
    environment[config_env] = str(spec_path)
    if extra_env:
        environment.update(extra_env)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(runner)],
            cwd=HERE,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _archive_incomplete(path: Path) -> None:
    if not path.exists():
        return
    archived = path.with_name(
        f"{path.name}.incomplete-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    path.rename(archived)


def _run_csk_cases() -> None:
    total = len(CALIBRATION_TOKENS) * REPETITIONS
    position = 0
    for calibration_tokens in CALIBRATION_TOKENS:
        for repetition in range(REPETITIONS):
            position += 1
            case_id = f"csk_p{calibration_tokens:03d}_r{repetition}"
            run_dir = RUN_ROOT / "csk_auxiliary" / case_id
            profile_path = run_dir / "cskcache_profile.jsonl"
            if profile_path.is_file() and (run_dir / "summary.json").is_file():
                print(f"[{position}/{total}] reuse {case_id}", flush=True)
                continue
            _archive_incomplete(run_dir)
            spec = {
                "run_dir": str(run_dir),
                "skill_tokens": SKILL_TOKENS,
                "calibration_ratio": calibration_tokens / SKILL_TOKENS,
                "calibration_tokens": calibration_tokens,
                "execution_order": EXECUTION_ORDER,
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "storage_layout": STORAGE_LAYOUT,
                "host_layout": HOST_LAYOUT,
                "warmup_requests": WARMUP_REQUESTS,
            }
            spec_path = CASE_SPECS / f"{case_id}.json"
            spec_path.write_text(
                json.dumps(spec, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[{position}/{total}] run {case_id}", flush=True)
            _run_child(
                CSK_RUNNER,
                CSK_CASE_CONFIG_ENV,
                spec_path,
                LOG_DIR / f"{case_id}.log",
                extra_env={
                    "CSKCACHE_FORWARD_PROFILE_LAYERS": ",".join(
                        str(layer_id) for layer_id in PROFILE_LAYER_IDS
                    )
                },
            )


def _run_native_cases() -> None:
    for repetition in range(REPETITIONS):
        case_id = f"native_r{repetition}"
        output_path = RUN_ROOT / "native_vllm" / f"{case_id}.json"
        if output_path.is_file():
            print(f"[{repetition + 1}/{REPETITIONS}] reuse {case_id}", flush=True)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "repetition": repetition,
            "calibration_tokens": list(CALIBRATION_TOKENS),
            "output_path": str(output_path),
        }
        spec_path = CASE_SPECS / f"{case_id}.json"
        spec_path.write_text(
            json.dumps(spec, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{repetition + 1}/{REPETITIONS}] run {case_id}", flush=True)
        _run_child(
            NATIVE_RUNNER,
            NATIVE_CASE_CONFIG_ENV,
            spec_path,
            LOG_DIR / f"{case_id}.log",
        )


def _csk_row(calibration_tokens: int, repetition: int) -> dict[str, Any]:
    case_id = f"csk_p{calibration_tokens:03d}_r{repetition}"
    records = _profile_records(
        RUN_ROOT / "csk_auxiliary" / case_id / "cskcache_profile.jsonl"
    )
    compute = next(
        record
        for record in records
        if record.get("event") == "cskcache_layer_compute"
    )
    breakdown = next(
        record
        for record in records
        if record.get("event")
        == "cskcache_calibration_forward_breakdown"
    )
    per_layer = {
        int(row["layer"]): row
        for row in compute["calibration_correct_install"]
    }
    stable_layers = [
        layer_id
        for layer_id in range(STABLE_LAYER_START, STABLE_LAYER_STOP)
        if layer_id not in PROFILE_LAYER_IDS
    ]
    row: dict[str, Any] = {
        "calibration_tokens": calibration_tokens,
        "repetition": repetition,
        "auxiliary_forward_per_layer_ms": statistics.median(
            float(per_layer[layer_id]["calibration_forward_ms"])
            for layer_id in stable_layers
        ),
        "auxiliary_prefix_tokens": int(breakdown["prefix_tokens"]),
    }
    for field in BREAKDOWN_FIELDS:
        row[field] = statistics.median(
            float(layer[field]) for layer in breakdown["layers"]
        )
    return row


def _native_rows() -> list[dict[str, Any]]:
    rows = []
    for repetition in range(REPETITIONS):
        payload = _read_json(
            RUN_ROOT / "native_vllm" / f"native_r{repetition}.json"
        )
        for result in payload["results"]:
            forwards = result["forwards"]
            rows.append(
                {
                    "calibration_tokens": int(result["calibration_tokens"]),
                    "repetition": repetition,
                    "native_prefix_tokens": int(result["prefix_tokens"]),
                    "native_cached_tokens": int(result["num_cached_tokens"]),
                    "native_input_tokens": sum(
                        int(record["input_tokens"]) for record in forwards
                    ),
                    "native_forward_calls": len(forwards),
                    "native_forward_total_ms": sum(
                        float(record["gpu_ms"]) for record in forwards
                    ),
                }
            )
    return rows


def _aggregate() -> None:
    csk_rows = [
        _csk_row(calibration_tokens, repetition)
        for calibration_tokens in CALIBRATION_TOKENS
        for repetition in range(REPETITIONS)
    ]
    native_rows = _native_rows()
    raw_rows = []
    for csk in csk_rows:
        native = next(
            row
            for row in native_rows
            if row["calibration_tokens"] == csk["calibration_tokens"]
            and row["repetition"] == csk["repetition"]
        )
        native_per_layer = native["native_forward_total_ms"] / NUM_LAYERS
        raw_rows.append(
            {
                **csk,
                **native,
                "native_forward_per_layer_ms": native_per_layer,
                "auxiliary_over_native": (
                    csk["auxiliary_forward_per_layer_ms"] / native_per_layer
                ),
            }
        )

    summary_rows = []
    numeric_fields = (
        "auxiliary_forward_per_layer_ms",
        "native_forward_total_ms",
        "native_forward_per_layer_ms",
        "auxiliary_over_native",
        *BREAKDOWN_FIELDS,
    )
    for calibration_tokens in CALIBRATION_TOKENS:
        selected = [
            row
            for row in raw_rows
            if row["calibration_tokens"] == calibration_tokens
        ]
        summary = {
            "calibration_tokens": calibration_tokens,
            "auxiliary_prefix_tokens": int(
                statistics.median(row["auxiliary_prefix_tokens"] for row in selected)
            ),
            "native_cached_tokens": int(
                statistics.median(row["native_cached_tokens"] for row in selected)
            ),
            "native_input_tokens": int(
                statistics.median(row["native_input_tokens"] for row in selected)
            ),
            "native_forward_calls": int(
                statistics.median(row["native_forward_calls"] for row in selected)
            ),
        }
        for field in numeric_fields:
            summary[field] = statistics.median(
                float(row[field]) for row in selected
            )
        summary_rows.append(summary)

    with (RUN_ROOT / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    with (RUN_ROOT / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (RUN_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "skill_tokens": SKILL_TOKENS,
                    "calibration_tokens": list(CALIBRATION_TOKENS),
                    "profile_layer_ids": list(PROFILE_LAYER_IDS),
                    "repetitions": REPETITIONS,
                    "warmup_requests": WARMUP_REQUESTS,
                },
                "raw": raw_rows,
                "summary": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nP  auxiliary/layer  native/layer  ratio  prefix+cat  attention  qkv+mlp")
    for row in summary_rows:
        print(
            f"{row['calibration_tokens']:>3}"
            f" {row['auxiliary_forward_per_layer_ms']:>15.3f}"
            f" {row['native_forward_per_layer_ms']:>13.3f}"
            f" {row['auxiliary_over_native']:>6.2f}x"
            f" {row['prefix_paged_kv_ms'] + row['kv_concat_ms']:>11.3f}"
            f" {row['attention_ms']:>10.3f}"
            f" {row['qkv_projection_ms'] + row['mlp_ms']:>8.3f}",
            flush=True,
        )
    print(f"results: {RUN_ROOT}")


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CASE_SPECS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _run_csk_cases()
    _run_native_cases()
    _aggregate()


if __name__ == "__main__":
    main()
