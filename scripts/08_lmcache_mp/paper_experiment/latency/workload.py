#!/usr/bin/env python3
"""Build and measure the identical 8K-Skill workload for three cache modes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODES = ("full", "direct", "correction")
KINDS = ("cold", "warmup", "measure")
DEFAULT_CACHE_ID = "Auto-claude-code-research-in-sleep/paper-write"


@dataclass(frozen=True)
class CachedSkill:
    cache_id: str
    name: str
    tokens: tuple[int, ...]
    text_sha256: str
    token_ids_sha256: str
    manifest_path: Path


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def token_sha256(tokens: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def context_segment_text(skill_name: str, skill_text: str) -> str:
    body = skill_text if skill_text.endswith("\n") else f"{skill_text}\n"
    return (
        f'<context_segment skill_name="{skill_name}">\n'
        f"{body}"
        "</context_segment>\n"
    )


def load_cached_skill(pool_dir: Path, cache_id: str) -> CachedSkill:
    pool = pool_dir.resolve()
    manifest_path = (pool / cache_id / "manifest.json").resolve()
    if not manifest_path.is_relative_to(pool):
        raise ValueError("cache_id escapes the configured pool")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"offline Skill is not completed: {manifest_path}")
    if manifest.get("cache_id") != cache_id:
        raise ValueError("offline Skill manifest cache_id mismatch")
    if manifest.get("cache_object") != "qwen_context_segment":
        raise ValueError("offline object is not a qwen_context_segment")

    skill_name = str(manifest["skill_name"])
    skill_path = Path(manifest["skill_path"])
    skill_text = skill_path.read_text(encoding="utf-8")
    raw_sha = hashlib.sha256(skill_text.encode("utf-8")).hexdigest()
    if raw_sha != manifest.get("text_sha256"):
        raise ValueError("SKILL.md text SHA256 disagrees with offline manifest")
    cached_text = context_segment_text(skill_name, skill_text)
    cached_text_sha = hashlib.sha256(cached_text.encode("utf-8")).hexdigest()
    if cached_text_sha != manifest.get("cache_object_text_sha256"):
        raise ValueError("context_segment text SHA256 disagrees with offline manifest")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_path"], local_files_only=True, trust_remote_code=True
    )
    tokens = tuple(tokenizer.encode(cached_text, add_special_tokens=False))
    expected_count = int(manifest["token_count"])
    if len(tokens) != expected_count:
        raise ValueError(
            f"offline Skill token count mismatch: expected={expected_count} actual={len(tokens)}"
        )
    tokens_sha = token_sha256(tokens)
    if tokens_sha != manifest.get("token_ids_sha256"):
        raise ValueError("context_segment token-id SHA256 disagrees with offline manifest")
    if max(tokens) >= len(tokenizer):
        raise ValueError("offline Skill contains an out-of-vocabulary token id")
    return CachedSkill(
        cache_id=cache_id,
        name=skill_name,
        tokens=tokens,
        text_sha256=cached_text_sha,
        token_ids_sha256=tokens_sha,
        manifest_path=manifest_path,
    )


def synthetic_tokens(length: int, *, namespace: int, nonce: int) -> list[int]:
    """Return deterministic Qwen-valid IDs for dynamic non-Skill context."""
    if length < 0:
        raise ValueError("token length must be non-negative")
    base = 1000 + (namespace * 7919 + nonce * 104729) % 120000
    return [
        1000 + ((base + index * 1543 + (index * index) % 997) % 140000)
        for index in range(length)
    ]


def sample_nonce(replica: int, kind: str, ordinal: int) -> int:
    if replica < 0 or kind not in KINDS or ordinal < 0:
        raise ValueError("invalid sample identity")
    kind_offset = {"cold": 0, "warmup": 100, "measure": 1000}[kind]
    return replica * 10_000 + kind_offset + ordinal


def build_prompt(
    skill: CachedSkill,
    *,
    replica: int,
    kind: str,
    ordinal: int,
    prefix_tokens: int,
    suffix_tokens: int,
) -> tuple[list[int], int, int]:
    nonce = sample_nonce(replica, kind, ordinal)
    prefix = synthetic_tokens(
        prefix_tokens, namespace=17 + replica * 31, nonce=nonce
    )
    suffix = synthetic_tokens(
        suffix_tokens, namespace=53 + replica * 31, nonce=nonce
    )
    segment_start = len(prefix)
    segment_end = segment_start + len(skill.tokens)
    return prefix + list(skill.tokens) + suffix, segment_start, segment_end


def lookup_params(mode: str, segment_start: int, segment_end: int) -> dict[str, Any] | None:
    if mode == "full":
        return None
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    lookup: dict[str, Any] = {
        "segment_start": segment_start,
        "segment_end": segment_end,
    }
    if mode == "correction":
        lookup.update(
            {
                "correction_mode": "prefix_k_headwise",
                "cache_end": segment_end,
                "prefix_tokens": 256,
                "calibration_start": 132,
                "calibration_end": 256,
                "minimum_reuse_tokens": 256,
                "correction_alpha": 0.6,
            }
        )
    return lookup


def request_payload(
    *,
    model: str,
    mode: str,
    prompt: list[int],
    segment_start: int,
    segment_end: int,
    request_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "request_id": request_id,
    }
    lookup = lookup_params(mode, segment_start, segment_end)
    if lookup is not None:
        payload["kv_transfer_params"] = {
            "lmcache_segmentia_lookup": lookup,
            "lmcache.skip_save": True,
        }
    return payload


def post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, socket.timeout) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise TypeError("completion endpoint did not return a JSON object")
    return body


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sample_plan(warmups: int, measurements: int) -> list[tuple[str, int]]:
    if warmups < 0 or measurements <= 0:
        raise ValueError("warmups must be non-negative and measurements positive")
    return (
        [("cold", 0)]
        + [("warmup", ordinal) for ordinal in range(warmups)]
        + [("measure", ordinal) for ordinal in range(measurements)]
    )


def build_row(
    *,
    skill: CachedSkill,
    mode: str,
    replica: int,
    kind: str,
    ordinal: int,
    prefix_tokens: int,
    suffix_tokens: int,
    model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt, segment_start, segment_end = build_prompt(
        skill,
        replica=replica,
        kind=kind,
        ordinal=ordinal,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
    )
    request_id = f"segmentia-latency-{mode}-r{replica}-{kind}{ordinal}"
    row = {
        "schema_version": 1,
        "mode": mode,
        "replica": replica,
        "kind": kind,
        "ordinal": ordinal,
        "sample_id": f"r{replica}-{kind}{ordinal}",
        "cache_id": skill.cache_id,
        "skill_name": skill.name,
        "skill_tokens": len(skill.tokens),
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "prompt_tokens": len(prompt),
        "segment_start": segment_start,
        "segment_end": segment_end,
        "prompt_sha256": token_sha256(prompt),
        "prefix_sha256": token_sha256(prompt[:segment_start]),
        "skill_sha256": token_sha256(prompt[segment_start:segment_end]),
        "request_id": request_id,
    }
    payload = request_payload(
        model=model,
        mode=mode,
        prompt=prompt,
        segment_start=segment_start,
        segment_end=segment_end,
        request_id=request_id,
    )
    return row, payload


def measure(args: argparse.Namespace) -> None:
    if args.mode not in MODES or args.replica < 0:
        raise ValueError("invalid mode or replica")
    skill = load_cached_skill(args.pool_dir, args.cache_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings = args.output_dir / "timings.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if timings.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite leaf: {args.output_dir}")

    for kind, ordinal in sample_plan(args.warmups, args.measurements):
        row, payload = build_row(
            skill=skill,
            mode=args.mode,
            replica=args.replica,
            kind=kind,
            ordinal=ordinal,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            model=args.model,
        )
        started = time.perf_counter()
        response = post_json(
            f"{args.base_url.rstrip('/')}/v1/completions", args.api_key, payload
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        choices = response.get("choices") or []
        usage = response.get("usage") or {}
        if len(choices) != 1:
            raise ValueError("completion response must contain exactly one choice")
        if usage.get("prompt_tokens") not in (None, row["prompt_tokens"]):
            raise ValueError("server prompt token count disagrees with submitted IDs")
        if usage.get("completion_tokens") not in (None, 1):
            raise ValueError("request did not generate exactly one token")
        row.update(
            {
                "status": "completed",
                "elapsed_ms": elapsed_ms,
                "response_id": response.get("id"),
                "finish_reason": choices[0].get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        )
        append_jsonl(timings, row)
        print(
            f"[measured] mode={args.mode} replica={args.replica} "
            f"sample={row['sample_id']} elapsed_ms={elapsed_ms:.3f}"
        )

    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "mode": args.mode,
            "replica": args.replica,
            "cache_id": skill.cache_id,
            "skill_name": skill.name,
            "skill_tokens": len(skill.tokens),
            "skill_text_sha256": skill.text_sha256,
            "skill_token_ids_sha256": skill.token_ids_sha256,
            "offline_manifest": str(skill.manifest_path),
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "warmups": args.warmups,
            "measurements": args.measurements,
            "timings": str(timings.resolve()),
        },
    )


def dry_run(args: argparse.Namespace) -> None:
    skill = load_cached_skill(args.pool_dir, args.cache_id)
    per_mode: dict[str, dict[str, str]] = {}
    first_tokens: set[int] = set()
    canonical: dict[str, str] | None = None
    for mode in MODES:
        hashes: dict[str, str] = {}
        for kind, ordinal in sample_plan(args.warmups, args.measurements):
            row, payload = build_row(
                skill=skill,
                mode=mode,
                replica=args.replica,
                kind=kind,
                ordinal=ordinal,
                prefix_tokens=args.prefix_tokens,
                suffix_tokens=args.suffix_tokens,
                model=args.model,
            )
            hashes[row["sample_id"]] = row["prompt_sha256"]
            if mode == "full":
                prompt = payload["prompt"]
                first_tokens.add(int(prompt[0]))
        if canonical is None:
            canonical = hashes
        elif hashes != canonical:
            raise ValueError("same samples differ across modes")
        per_mode[mode] = hashes
    expected_samples = 1 + args.warmups + args.measurements
    if len(first_tokens) != expected_samples:
        raise ValueError("dynamic samples do not have distinct first tokens")
    print(
        json.dumps(
            {
                "status": "valid",
                "cache_id": skill.cache_id,
                "skill_tokens": len(skill.tokens),
                "prefix_tokens": args.prefix_tokens,
                "suffix_tokens": args.suffix_tokens,
                "prompt_tokens": args.prefix_tokens
                + len(skill.tokens)
                + args.suffix_tokens,
                "samples_per_mode": expected_samples,
                "modes": list(per_mode),
                "cross_mode_prompt_hashes_identical": True,
                "distinct_first_tokens": len(first_tokens),
            },
            indent=2,
            sort_keys=True,
        )
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cache-id", default=DEFAULT_CACHE_ID)
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    add_shared_args(measure_parser)
    measure_parser.add_argument("--mode", choices=MODES, required=True)
    measure_parser.add_argument("--output-dir", type=Path, required=True)
    measure_parser.add_argument("--base-url", default="http://127.0.0.1:8120")
    measure_parser.add_argument("--api-key", default="EMPTY")
    dry_parser = subparsers.add_parser("dry-run")
    add_shared_args(dry_parser)
    args = parser.parse_args()
    if args.prefix_tokens <= 0 or args.suffix_tokens < 0:
        raise ValueError("prefix must be positive and suffix non-negative")
    if args.command == "measure":
        measure(args)
    else:
        dry_run(args)


if __name__ == "__main__":
    main()
