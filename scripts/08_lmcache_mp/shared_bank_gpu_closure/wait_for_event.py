#!/usr/bin/env python3
"""Wait for one structured Segmentia event in a growing vLLM log."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def request_matches(event_request_id: object, response_id: str) -> bool:
    """Match vLLM's per-prompt ID derived from an OpenAI response ID."""
    return isinstance(event_request_id, str) and (
        event_request_id == response_id
        or event_request_id.startswith(f"{response_id}-")
        or event_request_id.startswith(f"{response_id}:")
    )


def matching_event(path: Path, event: str, response_id: str) -> dict | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "SEGMENTIA_EVENT "
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if payload.get("event") == event and request_matches(
            payload.get("request_id"), response_id
        ):
            return payload
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--response-record", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--poll-s", type=float, default=0.25)
    args = parser.parse_args()
    record = json.loads(args.response_record.read_text(encoding="utf-8"))
    response_id = record.get("response_id")
    if record.get("status") != "completed" or not isinstance(response_id, str):
        raise ValueError(
            f"response record is not completed: {args.response_record}"
        )
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        payload = matching_event(args.log, args.event, response_id)
        if payload is not None:
            print(json.dumps(payload, sort_keys=True))
            return
        time.sleep(args.poll_s)
    raise TimeoutError(
        f"event={args.event!r} response_id_prefix={response_id!r} "
        f"did not appear in {args.log}"
    )


if __name__ == "__main__":
    main()
