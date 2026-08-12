#!/usr/bin/env python3
"""Discover SKILL.md files or prefill one clean Skill into LMCache SSD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from skill_cache_tokens import (
    CACHE_OBJECT_TYPE,
    CACHE_SCHEMA_VERSION,
    LOCATOR_KIND,
    context_segment_cache_text,
    context_segment_start_marker_text,
    qwen_context_segment_start_marker_token_ids,
    qwen_context_segment_token_ids,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SKILLS_DIR = ROOT / "skills"
DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)


@dataclass(frozen=True)
class SkillSpec:
    cache_id: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--list", action="store_true", help="Print cache ID and path")
    parser.add_argument("--collection")
    parser.add_argument("--skill")
    parser.add_argument(
        "--exclude-skill",
        action="append",
        default=[],
        help="Exclude one exact cache ID; may be repeated.",
    )
    parser.add_argument("--cache-id")
    parser.add_argument("--skill-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--pool-index", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8013")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-input-tokens", type=int, default=32767)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--save-timeout", type=float, default=120.0)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cache_id_for_path(skills_dir: Path, skill_path: Path) -> str:
    parts = list(skill_path.relative_to(skills_dir).parts[:-1])
    # <bundle>/skills/ is structural. Preserve any hierarchy below it.
    if len(parts) >= 3 and parts[1] == "skills":
        parts.pop(1)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid Skill path: {skill_path}")
    return "/".join(parts)


def discover_skills(
    skills_dir: Path,
    collection: str | None,
    selected_skill: str | None,
    excluded_skills: set[str] | None = None,
) -> list[SkillSpec]:
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"skills directory does not exist: {skills_dir}")
    found: dict[str, Path] = {}
    excluded = excluded_skills or set()
    for source_path in sorted(skills_dir.rglob("SKILL.md")):
        cache_id = cache_id_for_path(skills_dir, source_path)
        if collection and cache_id.split("/", 1)[0] != collection:
            continue
        if selected_skill and selected_skill not in {
            cache_id,
            cache_id.rsplit("/", 1)[-1],
        }:
            continue
        if cache_id in excluded:
            continue
        if cache_id in found:
            raise RuntimeError(
                f"duplicate cache ID {cache_id!r}: {found[cache_id]} and {source_path}"
            )
        found[cache_id] = source_path.resolve()
    if not found:
        raise RuntimeError("no matching SKILL.md files found")
    return [SkillSpec(key, found[key]) for key in sorted(found)]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def token_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def post_completion(
    base_url: str,
    api_key: str,
    timeout: float,
    request_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {body}") from exc


def inspect_layer_group(
    cache_dir: Path,
    sidecars: list[Path],
    expected_tokens: int,
    expected_layers: int,
) -> list[str]:
    if len(sidecars) != expected_layers:
        raise RuntimeError(
            f"expected {expected_layers} LMCache layers, found {len(sidecars)}"
        )

    data_files: list[str] = []
    cache_keys: set[str] = set()
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        positions = metadata.get("cached_positions")
        shape = metadata.get("shape")
        data_file = metadata.get("data_file")
        if positions != {"kind": "range", "start": 0, "length": expected_tokens}:
            raise RuntimeError(f"wrong cached positions in {sidecar}: {positions}")
        if not isinstance(shape, list) or len(shape) < 2 or shape[1] != expected_tokens:
            raise RuntimeError(f"wrong token dimension in {sidecar}: {shape}")
        if not isinstance(data_file, str) or not (cache_dir / data_file).is_file():
            raise RuntimeError(f"missing LMCache data file for {sidecar}")
        cache_key = metadata.get("cache_key")
        if not isinstance(cache_key, str):
            raise RuntimeError(f"missing cache key in {sidecar}")
        cache_keys.add(cache_key.rsplit("@", 1)[0])
        data_files.append(data_file)
    if len(cache_keys) != 1:
        raise RuntimeError(f"expected one clean Skill cache group, found {len(cache_keys)}")
    return data_files


def wait_for_new_group(
    staging_dir: Path,
    previous_sidecars: set[str],
    expected_tokens: int,
    expected_layers: int,
    timeout: float,
) -> list[Path]:
    deadline = time.monotonic() + timeout
    new_sidecars: list[Path] = []
    while time.monotonic() < deadline:
        new_sidecars = [
            path
            for path in sorted(staging_dir.glob("*.pt.meta.json"))
            if path.name not in previous_sidecars
        ]
        if len(new_sidecars) == expected_layers:
            break
        time.sleep(0.25)
    inspect_layer_group(
        staging_dir, new_sidecars, expected_tokens, expected_layers
    )
    return new_sidecars


def link_group(source_dir: Path, sidecars: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        data_file = source_dir / metadata["data_file"]
        os.link(data_file, output_dir / data_file.name)
        os.link(sidecar, output_dir / sidecar.name)


def require_one_skill_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "cache_id",
            "skill_path",
            "cache_dir",
            "staging_dir",
            "pool_index",
            "manifest",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit("missing per-Skill arguments: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    if args.list:
        for spec in discover_skills(
            args.skills_dir,
            args.collection,
            args.skill,
            set(args.exclude_skill),
        ):
            print(f"{spec.cache_id}\t{spec.source_path}")
        return

    require_one_skill_args(args)
    source_path = args.skill_path.resolve()
    expected_cache_id = cache_id_for_path(args.skills_dir.resolve(), source_path)
    if args.cache_id != expected_cache_id:
        raise ValueError(
            f"cache ID/path mismatch: {args.cache_id!r} != {expected_cache_id!r}"
        )

    text = source_path.read_text(encoding="utf-8")
    skill_name = source_path.parent.name
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    raw_skill_token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = qwen_context_segment_token_ids(tokenizer, skill_name, text)
    start_marker_token_ids = qwen_context_segment_start_marker_token_ids(
        tokenizer, skill_name
    )
    if not raw_skill_token_ids:
        raise ValueError(f"empty token sequence: {source_path}")
    if len(token_ids) > args.max_input_tokens:
        raise ValueError(
            f"Skill has {len(token_ids)} tokens; limit is {args.max_input_tokens}"
        )

    record: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "dry_run" if args.dry_run else "saving",
        "cache_id": args.cache_id,
        "skill_name": skill_name,
        "skill_path": str(source_path),
        "cache_dir": str(args.cache_dir.resolve()),
        "staging_dir": str(args.staging_dir.resolve()),
        "model_path": str(args.model_path.resolve()),
        "token_count": len(token_ids),
        "raw_skill_token_count": len(raw_skill_token_ids),
        "saved_span": [0, len(token_ids)],
        "cache_object": CACHE_OBJECT_TYPE,
        "cache_object_bounds": [
            f'<context_segment skill_name="{skill_name}">',
            "</context_segment>\n",
        ],
        "add_special_tokens": False,
        "chat_template": False,
        "separator_added_to_prompt": False,
        "cache_object_text_sha256": hashlib.sha256(
            context_segment_cache_text(skill_name, text).encode("utf-8")
        ).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_ids_sha256": token_hash(token_ids),
        "locator": {
            "kind": LOCATOR_KIND,
            "start_marker_text": context_segment_start_marker_text(skill_name),
            "start_marker_token_ids": start_marker_token_ids,
            "start_marker_token_count": len(start_marker_token_ids),
            "start_marker_token_ids_sha256": token_hash(start_marker_token_ids),
        },
    }
    if args.dry_run:
        print(
            f"[dry_run] {args.cache_id} tokens={len(token_ids)} "
            f"span=[0,{len(token_ids)})"
        )
        return

    started = time.perf_counter()
    pool_index = read_json(args.pool_index)
    entries = pool_index.setdefault("entries", {})
    previous = entries.get(record["token_ids_sha256"])
    source_kv_dir = None
    if isinstance(previous, dict):
        candidate = Path(str(previous.get("kv_dir", "")))
        if candidate.is_dir():
            candidate_sidecars = sorted(candidate.glob("*.pt.meta.json"))
            try:
                inspect_layer_group(
                    candidate,
                    candidate_sidecars,
                    len(token_ids),
                    args.expected_layers,
                )
                source_kv_dir = candidate
            except RuntimeError:
                source_kv_dir = None

    before = {path.name for path in args.staging_dir.glob("*.pt.meta.json")}
    request_id = "skill-prefill-" + hashlib.sha256(
        args.cache_id.encode("utf-8")
    ).hexdigest()[:16]
    response = post_completion(
        args.base_url,
        args.api_key,
        args.request_timeout,
        request_id,
        {
            "model": args.served_model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "kv_transfer_params": {
                "lmcache_segmentia_save": {
                    "segment_start": 0,
                    "segment_end": len(token_ids),
                }
            },
        },
    )
    if source_kv_dir is None:
        source_sidecars = wait_for_new_group(
            args.staging_dir,
            before,
            len(token_ids),
            args.expected_layers,
            args.save_timeout,
        )
        link_group(args.staging_dir, source_sidecars, args.cache_dir)
        reused_identical = None
    else:
        source_sidecars = sorted(source_kv_dir.glob("*.pt.meta.json"))
        link_group(source_kv_dir, source_sidecars, args.cache_dir)
        reused_identical = previous.get("cache_id")
    response_id = response.get("id")

    files = inspect_layer_group(
        args.cache_dir,
        sorted(args.cache_dir.glob("*.pt.meta.json")),
        len(token_ids),
        args.expected_layers,
    )
    record.update(
        status="completed",
        request_id=request_id,
        response_id=response_id,
        reused_identical_skill=reused_identical,
        latency_s=round(time.perf_counter() - started, 4),
        layer_count=len(files),
        data_files=files,
    )
    atomic_json(args.manifest, record)
    entries[record["token_ids_sha256"]] = {
        "cache_id": args.cache_id,
        "kv_dir": str(args.cache_dir.resolve()),
    }
    atomic_json(args.pool_index, pool_index)
    args.manifest.with_name("COMPLETED").write_text("completed\n", encoding="utf-8")
    print(
        f"[completed] {args.cache_id} tokens={len(token_ids)} "
        f"span=[0,{len(token_ids)})"
    )


if __name__ == "__main__":
    main()
