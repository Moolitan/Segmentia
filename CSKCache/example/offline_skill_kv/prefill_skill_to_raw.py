#!/usr/bin/env python3
"""Discover Skills or exact-save one Skill into the selected LMCache backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

import torch
from transformers import AutoConfig, AutoTokenizer

from cskcache import (
    ChunkingSpec,
    MetadataManager,
    build_skill_token_identity,
    fingerprint_full_token_chunks,
    fingerprint_model,
    fingerprint_tokenizer,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.token_database import ChunkedTokenDatabase


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if all(
            (candidate / component).is_dir()
            for component in ("CSKCache/cskcache", "LMCache/lmcache", "vllm/vllm")
        ):
            return candidate
    raise RuntimeError("cannot locate CSKCache, LMCache, and vLLM checkout root")


ROOT = repository_root()
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
    parser.add_argument("--skill", action="append")
    parser.add_argument("--exclude-skill", action="append", default=[])
    parser.add_argument("--cache-id")
    parser.add_argument("--skill-path", type=Path)
    parser.add_argument("--pending-dir", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8013")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-input-tokens", type=int, default=32767)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument(
        "--storage-backend",
        choices=("raw_block", "local_disk"),
        default="raw_block",
    )
    parser.add_argument("--local-disk-root", type=Path)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cache_id_for_path(skills_dir: Path, skill_path: Path) -> str:
    parts = list(skill_path.relative_to(skills_dir).parts[:-1])
    if len(parts) >= 3 and parts[1] == "skills":
        parts.pop(1)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid Skill path: {skill_path}")
    return "/".join(parts)


def discover_skills(
    skills_dir: Path,
    collection: str | None,
    selected_skills: set[str] | None,
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
        if selected_skills and not (
            {cache_id, cache_id.rsplit("/", 1)[-1]} & selected_skills
        ):
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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def require_one_skill_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("cache_id", "skill_path", "pending_dir", "catalog")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit("missing per-Skill arguments: " + ", ".join(missing))


def model_dtype(model_config: Any) -> torch.dtype:
    dtype_name = getattr(model_config, "torch_dtype", None)
    if isinstance(dtype_name, torch.dtype):
        dtype = dtype_name
    else:
        dtype = getattr(torch, str(dtype_name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported model KV dtype: {dtype_name!r}")
    return dtype


def main() -> None:
    args = parse_args()
    if args.list:
        for spec in discover_skills(
            args.skills_dir,
            args.collection,
            set(args.skill or ()),
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
    token_identity = build_skill_token_identity(tokenizer, skill_name, text)
    token_ids = list(token_identity.token_ids)
    chunking = ChunkingSpec(
        chunk_size_tokens=int(os.environ["CSKCACHE_CHUNK_SIZE_TOKENS"])
    )
    chunk_token_ids_sha256 = fingerprint_full_token_chunks(
        token_ids, chunking.chunk_size_tokens
    )
    if len(token_ids) > args.max_input_tokens:
        raise ValueError(
            f"Skill has {len(token_ids)} tokens; limit is {args.max_input_tokens}"
        )
    model_config = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model_layers = int(model_config.num_hidden_layers)
    if model_layers != args.expected_layers:
        raise ValueError(
            f"model has {model_layers} layers; expected {args.expected_layers}"
        )
    head_dim = int(
        getattr(
            model_config,
            "head_dim",
            int(model_config.hidden_size) // int(model_config.num_attention_heads),
        )
    )
    kv_hidden_size = int(model_config.num_key_value_heads) * head_dim
    dtype = model_dtype(model_config)
    layer_bytes = 2 * len(token_ids) * kv_hidden_size * dtype.itemsize
    _, _, exact_hash = ChunkedTokenDatabase().process_exact_tokens(
        token_ids,
        make_key=False,
    )
    if not isinstance(exact_hash, int):
        raise TypeError("LMCache exact-token hash must be an integer")

    model_path = args.model_path.resolve()
    model_digest = fingerprint_model(model_path)
    tokenizer_digest = fingerprint_tokenizer(model_path)
    skill_version = hashlib.sha256(
        token_identity.cache_text.encode("utf-8")
    ).hexdigest()
    if not args.force_recompute and args.catalog.is_file():
        manager = MetadataManager(args.catalog, expected_layers=args.expected_layers)
        try:
            existing = manager.resolve_object(
                skill_name=skill_name,
                model_fingerprint=model_digest,
                tokenizer_fingerprint=tokenizer_digest,
            )
        except (KeyError, ValueError):
            existing = None
        if (
            existing is not None
            and existing.skill_version == skill_version
            and existing.token_ids_sha256 == token_identity.token_ids_sha256
            and existing.token_count == len(token_ids)
        ):
            if (
                existing.chunking != chunking
                or existing.chunk_token_ids_sha256
                != chunk_token_ids_sha256
            ):
                manager.configure_chunk_authentication(
                    existing.object_id,
                    token_ids_sha256=token_identity.token_ids_sha256,
                    chunking=chunking,
                    chunk_token_ids_sha256=chunk_token_ids_sha256,
                )
                print(
                    f"[indexed] {args.cache_id} tokens={len(token_ids)} "
                    f"chunks={len(chunk_token_ids_sha256)} (KV unchanged)"
                )
                return
            print(
                f"[skipped] {args.cache_id} tokens={len(token_ids)} "
                "(identical Catalog object exists)"
            )
            return

    if args.dry_run:
        print(
            f"[dry_run] {args.cache_id} tokens={len(token_ids)} "
            f"span=[0,{len(token_ids)}) bytes_per_layer={layer_bytes}"
        )
        return

    base_key = CacheEngineKey(
        model_name=str(model_path),
        world_size=1,
        worker_id=0,
        chunk_hash=exact_hash,
        dtype=dtype,
        request_configs=None,
    )
    layer_keys = base_key.split_layers(args.expected_layers)
    if args.storage_backend == "local_disk" and args.local_disk_root is None:
        raise ValueError("local_disk exact save requires --local-disk-root")
    request_id = "skill-prefill-" + hashlib.sha256(
        args.cache_id.encode("utf-8")
    ).hexdigest()[:16]
    started = time.perf_counter()
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
                "lmcache.exact_save_span": {"start": 0, "end": len(token_ids)}
            },
        },
    )
    pending_name = hashlib.sha256(args.cache_id.encode("utf-8")).hexdigest() + ".json"
    artifact_type = (
        "cskcache_direct_raw_pending"
        if args.storage_backend == "raw_block"
        else "cskcache_local_disk_pending"
    )
    layers = []
    for layer_id, key in enumerate(layer_keys):
        record = {
            "layer_id": layer_id,
            "backend_key": key.to_string(),
            "length_bytes": layer_bytes,
            "dtype": str(dtype).removeprefix("torch."),
            "shape": [2, len(token_ids), kv_hidden_size],
            "memory_layout": "KV_2TD",
        }
        if args.storage_backend == "raw_block":
            record["lookup_key"] = key.to_string()
        else:
            assert args.local_disk_root is not None
            record["data_path"] = str(
                (
                    args.local_disk_root.resolve()
                    / (key.to_string().replace("/", "-") + ".pt")
                )
            )
        layers.append(record)
    atomic_write_json(
        args.pending_dir / pending_name,
        {
            "artifact_type": artifact_type,
            "cache_id": args.cache_id,
            "skill_name": skill_name,
            "skill_path": str(source_path),
            "object_id": (
                f"{skill_name}:{token_identity.token_ids_sha256[:16]}:"
                f"{model_digest[:16]}"
            ),
            "skill_version": skill_version,
            "model_fingerprint": model_digest,
            "tokenizer_fingerprint": tokenizer_digest,
            "token_count": len(token_ids),
            "chunking": {
                "chunk_size_tokens": chunking.chunk_size_tokens,
            },
            "storage_layout": os.environ.get(
                "CSKCACHE_STORAGE_LAYOUT", "chunk_single_layer"
            ),
            "source_position_start": 0,
            "token_ids_sha256": token_identity.token_ids_sha256,
            "chunk_token_ids_sha256": list(chunk_token_ids_sha256),
            "start_marker_token_ids": list(token_identity.start_marker_token_ids),
            "request_id": request_id,
            "response_id": response.get("id"),
            "latency_s": round(time.perf_counter() - started, 4),
            "layers": layers,
        },
    )
    print(
        f"[completed] {args.cache_id} tokens={len(token_ids)} "
        f"span=[0,{len(token_ids)}) backend={args.storage_backend} pending=1"
    )


if __name__ == "__main__":
    main()
