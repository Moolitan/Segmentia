#!/usr/bin/env python3
"""Authenticate and optionally stage the six frozen Skill sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from transformers import AutoTokenizer

import config as cfg


SECTION_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "CSKCache/cskcache").is_dir()
)
sys.path.insert(0, str(REPOSITORY_ROOT / "CSKCache"))

from cskcache import build_skill_token_identity  # noqa: E402


SKILLSBENCH_COMMIT = "9a1f4dd5f7659f75707435da3ce854b6e48321d1"
SKILLSBENCH_ROOT = Path(
    os.environ.get(
        "SKILLSBENCH_ROOT",
        "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench",
    )
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_path(root_name: str, relative: str) -> Path:
    roots = {
        "skillsbench": SKILLSBENCH_ROOT,
        "section": SECTION_ROOT,
        "repository": REPOSITORY_ROOT,
    }
    try:
        root = roots[root_name]
    except KeyError as exc:
        raise ValueError(f"unsupported source root: {root_name}") from exc
    return (root / relative).resolve()


def _checkout_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(SKILLSBENCH_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def build_plan() -> dict[str, object]:
    if not cfg.MODEL_PATH.is_dir():
        raise FileNotFoundError(f"model is missing: {cfg.MODEL_PATH}")
    if _checkout_commit() != SKILLSBENCH_COMMIT:
        raise RuntimeError("SkillsBench checkout differs from the frozen commit")
    frozen_path = SECTION_ROOT / "workloads.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    records: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(frozen["workloads"]):
        task_path = _source_path(item["task_root"], item["task_relative_path"])
        skill_path = _source_path(item["skill_root"], item["skill_relative_path"])
        for label, path, expected in (
            ("task", task_path, item["task_text_sha256"]),
            ("Skill", skill_path, item["skill_text_sha256"]),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"frozen {label} source changed: {path}")
        skill_name = str(item["skill_name"])
        if skill_name in seen_names:
            raise RuntimeError(f"duplicate selected Skill name: {skill_name}")
        seen_names.add(skill_name)
        text = skill_path.read_text(encoding="utf-8")
        identity = build_skill_token_identity(tokenizer, skill_name, text)
        token_count = len(identity.token_ids)
        if token_count != int(item["qwen3_14b_skill_tokens"]):
            raise RuntimeError(f"token count changed for {skill_name}")
        if identity.token_ids_sha256 != item["qwen3_14b_token_ids_sha256"]:
            raise RuntimeError(f"token identity changed for {skill_name}")
        bundle_relative = f"{index:02d}-{item['task_id']}/{skill_name}/SKILL.md"
        records.append(
            {
                "selection_order": index,
                "length_bucket": item["length_bucket"],
                "task_id": item["task_id"],
                "source_type": item["source_type"],
                "task_path": str(task_path),
                "task_text_sha256": item["task_text_sha256"],
                "skill_name": skill_name,
                "skill_path": str(skill_path),
                "relative_skill_path": item["skill_relative_path"],
                "skill_text_sha256": item["skill_text_sha256"],
                "bundle_relative_path": bundle_relative,
                "cache_id": "/".join(Path(bundle_relative).parts[:-1]),
                "skill_version": hashlib.sha256(
                    identity.cache_text.encode("utf-8")
                ).hexdigest(),
                "skill_tokens": token_count,
                "skill_token_ids_sha256": identity.token_ids_sha256,
            }
        )
    if tuple(record["length_bucket"] for record in records) != (
        "<1K", "1K-3K", "3K-5K", "5K-8K", "8K-10K", ">10K"
    ):
        raise RuntimeError("frozen workload bucket order changed")
    return {
        "schema_version": 1,
        "artifact_type": "fixed_length_source_plan",
        "model_id": "Qwen3-14B",
        "model_path": str(cfg.MODEL_PATH.resolve()),
        "skillsbench_commit": SKILLSBENCH_COMMIT,
        "pool_root": str((cfg.POOL_ROOT / cfg.POOL_MODEL_DIR).resolve()),
        "source_bundle": str(cfg.SKILLS_DIR.resolve()),
        "workloads": records,
    }


def prepare_sources(plan: dict[str, object]) -> Path:
    bundle_root = cfg.SKILLS_DIR.resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    for record in plan["workloads"]:
        assert isinstance(record, dict)
        source = Path(str(record["skill_path"])).resolve()
        destination = bundle_root / str(record["bundle_relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"source bundle symlink points elsewhere: {destination}")
        elif destination.exists():
            raise RuntimeError(f"source bundle target already exists: {destination}")
        else:
            destination.symlink_to(source)
    plan_path = (cfg.POOL_ROOT / cfg.POOL_MODEL_DIR / "selection_plan.json").resolve()
    atomic_write_json(plan_path, plan)
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    if args.prepare:
        path = prepare_sources(plan)
        print(f"[prepared] workloads=6 plan={path}")
    elif args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("[plan] workloads=6 tokens=" + ",".join(
            str(row["skill_tokens"]) for row in plan["workloads"]
        ))


if __name__ == "__main__":
    main()
