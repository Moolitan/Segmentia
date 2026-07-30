#!/usr/bin/env python3
"""Send one prepared direct-token request and persist its response."""
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json


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
        raise RuntimeError(f"request connection failure: {exc}") from exc
    if not isinstance(body, dict):
        raise TypeError("vLLM response is not a JSON object")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"request output already exists: {args.output}")
    record = json.loads(args.spec.read_text(encoding="utf-8"))
    if record.get("status") != "prepared":
        raise ValueError(f"request spec is not prepared: {args.spec}")
    record["status"] = "sending"
    atomic_write_json(args.output, record)
    started = time.perf_counter()
    response = post_json(
        f"{args.base_url.rstrip('/')}/v1/completions",
        args.api_key,
        record["request"],
    )
    if "error" in response:
        raise RuntimeError(f"vLLM returned an error object: {response['error']}")
    record.update(
        status="completed",
        elapsed_s=round(time.perf_counter() - started, 6),
        response=response,
        response_id=response.get("id"),
    )
    atomic_write_json(args.output, record)
    print(
        f"[completed] role={record['role']} request_id={record['request_id']} "
        f"elapsed_s={record['elapsed_s']}"
    )


if __name__ == "__main__":
    main()
