"""Prefill top-level skill Markdown files and persist their KV with CSKCache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_DIR = ROOT / "skills"
DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/"
    "offline_skill_kv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8013")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--health-timeout", type=float, default=900.0)
    parser.add_argument("--max-input-tokens", type=int, default=32767)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float,
    request_id: str | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    return json.loads(body) if body else {}


def wait_for_health(base_url: str, api_key: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            request_json(
                "GET",
                f"{base_url.rstrip('/')}/health",
                api_key=api_key,
                timeout=5.0,
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"vLLM health check timed out after {timeout}s: {last_error}")


def discover_skills(skills_dir: Path) -> list[tuple[str, Path]]:
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"Skills directory does not exist: {skills_dir}")
    result = [
        (directory.name, directory / "SKILL.md")
        for directory in skills_dir.iterdir()
        if directory.is_dir() and (directory / "SKILL.md").is_file()
    ]
    result.sort(key=lambda item: item[0])
    if not result:
        raise RuntimeError(f"No skills/*/SKILL.md files found under {skills_dir}")
    return result


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    special_ids = set(tokenizer.all_special_ids)

    # Import after argument parsing so --help does not require CSKCache on PYTHONPATH.
    from cskcache.v1.storage.local_disk_backend import LocalDiskBackend

    disk = LocalDiskBackend(args.output_dir)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(args.model_path),
        "served_model": args.served_model,
        "skills_dir": str(args.skills_dir),
        "output_dir": str(args.output_dir),
        "tokenization": {"add_special_tokens": False, "chat_template": False},
        "dry_run": args.dry_run,
        "records": [],
    }
    failures = 0

    if not args.dry_run:
        wait_for_health(args.base_url, args.api_key, args.health_timeout)

    for cache_id, skill_path in discover_skills(args.skills_dir):
        text = skill_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        special_tokens = [token_id for token_id in token_ids if token_id in special_ids]
        record: dict[str, Any] = {
            "cache_id": cache_id,
            "skill_path": str(skill_path),
            "source_start": 0,
            "source_end": len(token_ids),
            "num_tokens": len(token_ids),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "special_token_ids_in_prompt": sorted(set(special_tokens)),
        }

        if not token_ids:
            record.update(status="failed", error="skill token sequence is empty")
            failures += 1
        elif len(token_ids) > args.max_input_tokens:
            record.update(
                status="failed",
                error=(
                    f"token count {len(token_ids)} exceeds "
                    f"--max-input-tokens={args.max_input_tokens}"
                ),
            )
            failures += 1
        elif disk.contains(cache_id) and not args.overwrite:
            record["status"] = "skipped_existing"
        elif args.dry_run:
            record["status"] = "dry_run"
        else:
            request_id = f"cskcache-offline-save-{cache_id}"
            payload = {
                "model": args.served_model,
                "request_id": request_id,
                "prompt": token_ids,
                "add_special_tokens": False,
                "max_tokens": 1,
                "temperature": 0,
                "kv_transfer_params": {
                    "cskcache": {
                        "operation": "save",
                        "cache_id": cache_id,
                        "source_start": 0,
                        "source_end": len(token_ids),
                        "overwrite": args.overwrite,
                    }
                },
            }
            started = time.perf_counter()
            try:
                response = request_json(
                    "POST",
                    f"{args.base_url.rstrip('/')}/v1/completions",
                    api_key=args.api_key,
                    payload=payload,
                    timeout=args.request_timeout,
                    request_id=request_id,
                )
                # wait_for_save() runs before vLLM returns the completion response.
                disk = LocalDiskBackend(args.output_dir)
                if not disk.contains(cache_id):
                    raise RuntimeError("request completed but disk entry is missing")
                record.update(
                    status="saved",
                    latency_s=round(time.perf_counter() - started, 4),
                    usage=response.get("usage", {}),
                )
            except Exception as exc:
                record.update(
                    status="failed",
                    latency_s=round(time.perf_counter() - started, 4),
                    error=str(exc),
                )
                failures += 1

        manifest["records"].append(record)
        write_manifest(manifest_path, manifest)
        print(
            f"[{record['status']}] {cache_id:24s} tokens={len(token_ids):5d} "
            f"span=[0,{len(token_ids)})",
            flush=True,
        )

    manifest["summary"] = {
        "total": len(manifest["records"]),
        "saved": sum(r["status"] == "saved" for r in manifest["records"]),
        "skipped_existing": sum(
            r["status"] == "skipped_existing" for r in manifest["records"]
        ),
        "dry_run": sum(r["status"] == "dry_run" for r in manifest["records"]),
        "failed": failures,
    }
    write_manifest(manifest_path, manifest)
    print(f"[done] manifest={manifest_path}", flush=True)
    if failures:
        raise SystemExit(f"{failures} skill(s) failed; inspect {manifest_path}")


if __name__ == "__main__":
    main()
