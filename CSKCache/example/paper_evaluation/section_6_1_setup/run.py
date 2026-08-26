"""Validate models, hardware, workloads, and offline Catalogs without vLLM."""

from __future__ import annotations

import csv
import json
import platform as host_platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import (
    ACTIVE_PLATFORMS,
    OUTPUT_ROOT,
    PLATFORMS,
    RAW_POOL_ROOT,
)
from common.csk_config import catalog_path
from common.run_state import RunContext, utc_now


SECTION = "section_6_1_setup"


def _command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _token_count(model_path: Path, skill_path: Path) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text = skill_path.read_text(encoding="utf-8")
    return len(tokenizer.encode(text, add_special_tokens=False))


def main() -> None:
    workload_paths = tuple(
        path
        for _name, skill_path, prompt_path in local.WORKLOADS
        for path in (skill_path, prompt_path)
        if path.is_file()
    )
    config_paths = (
        Path(__file__), Path(local.__file__), Path(suite.__file__), *workload_paths,
    )
    values = {
        "active_platforms": list(ACTIVE_PLATFORMS),
        "workloads": [name for name, _skill, _prompt in local.WORKLOADS],
        "required_mount": str(local.REQUIRED_MOUNT),
        "expected_device": local.EXPECTED_DEVICE,
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(path.resolve() for path in config_paths),
        config_values=values,
    )
    hardware = {
        "hostname": socket.gethostname(),
        "python": host_platform.python_version(),
        "platform": host_platform.platform(),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "mount": _command(
            [
                "findmnt",
                "-no",
                "SOURCE,TARGET,FSTYPE,OPTIONS",
                str(local.REQUIRED_MOUNT),
            ]
        ),
        "block_device": _command(
            ["lsblk", "-ndo", "NAME,MODEL,SIZE,ROTA,RO", local.EXPECTED_DEVICE]
        ),
    }
    (run.run_dir / "environment.json").write_text(
        json.dumps(hardware, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    workload_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for platform_id in ACTIVE_PLATFORMS:
        model = PLATFORMS[platform_id]
        model_exists = model.model_path.is_dir()
        raw_catalog = catalog_path(RAW_POOL_ROOT, model.model_id, "raw_block")
        catalog_skills: set[str] = set()
        catalog_error = ""
        if raw_catalog.is_file():
            try:
                catalog_payload = json.loads(raw_catalog.read_text(encoding="utf-8"))
                catalog_skills = {
                    str(item.get("skill_name"))
                    for item in catalog_payload.get("objects", [])
                    if item.get("skill_name")
                }
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                catalog_error = f"catalog_invalid:{type(exc).__name__}:{exc}"
        else:
            catalog_error = f"catalog_missing:{raw_catalog}"
        for skill_name, skill_path, prompt_path in local.WORKLOADS:
            case_id = f"{platform_id}__{skill_name}"
            if run.completed(case_id):
                continue
            run.mark(case_id, "running")
            error = None
            tokens = None
            if not model_exists:
                error = f"model_missing:{model.model_path}"
            elif not skill_path.is_file():
                error = f"skill_missing:{skill_path}"
            elif not prompt_path.is_file():
                error = f"prompt_missing:{prompt_path}"
            else:
                try:
                    tokens = _token_count(model.model_path, skill_path)
                except Exception as exc:
                    error = f"tokenizer_error:{type(exc).__name__}:{exc}"
            if error is None and catalog_error:
                error = catalog_error
            if error is None and skill_name not in catalog_skills:
                error = f"catalog_object_missing:{skill_name}"
            catalog_exists = raw_catalog.is_file()
            status = "completed" if error is None else "failed"
            row = {
                "platform_id": platform_id,
                "gpu_name": model.gpu_name,
                "model_id": model.model_id,
                "model_path": str(model.model_path),
                "skill_name": skill_name,
                "skill_tokens": tokens,
                "task_id": prompt_path.stem,
                "status": status,
                "fallback_reason": error or ("" if catalog_exists else "catalog_missing"),
                "started_utc": utc_now(),
                "completed_utc": utc_now(),
            }
            run.record(row)
            workload_rows.append(
                {
                    "platform_id": platform_id,
                    "gpu_name": model.gpu_name,
                    "model_id": model.model_id,
                    "model_path": str(model.model_path),
                    "model_exists": model_exists,
                    "skill_name": skill_name,
                    "skill_path": str(skill_path),
                    "prompt_path": str(prompt_path),
                    "skill_tokens": "" if tokens is None else tokens,
                    "catalog_path": str(raw_catalog),
                    "catalog_exists": catalog_exists,
                    "status": status,
                    "error": error or "",
                }
            )
            run.mark(case_id, status, error=error)
            if error:
                failures.append(f"{case_id}: {error}")

    with (run.run_dir / "workloads.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        columns = tuple(workload_rows[0]) if workload_rows else ()
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(workload_rows)
    run.finish()
    print(f"results={run.run_dir}")
    if failures:
        print("setup has unresolved inputs:")
        for failure in failures:
            print(f"  {failure}")


if __name__ == "__main__":
    main()
