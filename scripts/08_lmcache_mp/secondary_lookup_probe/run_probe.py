#!/usr/bin/env python3
"""Drive one deterministic LMCache secondary-lookup probe.

The driver warms one complete skill segment, then submits a probe request as
pre-tokenized input. It never enables external KV application: the request is
required to carry probe_only=true, and the final summary rejects any event that
reports non-zero external_tokens_applied.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


EVENT_MARKER = "SEGMENTIA_SECONDARY_LOOKUP_EVENT"
EXPECTED_EVENT_ORDER = (
    "secondary_lookup_boundary",
    "secondary_lookup_initial_probe",
    "secondary_lookup_forward_complete",
    "secondary_lookup_pinned",
    "secondary_lookup_requeued",
    "secondary_lookup_external_probe",
    "secondary_lookup_local_reattach",
    "secondary_lookup_unpinned",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _repeat_tokens(tokenizer, text: str, minimum: int) -> list[int]:
    unit = tokenizer.encode(text, add_special_tokens=False)
    if not unit:
        raise ValueError(f"text produced no tokens: {text!r}")
    repetitions = (minimum + len(unit) - 1) // len(unit)
    return (unit * repetitions)[:minimum]


def build_token_layout(
    model_path: str,
    run_id: str,
    blend_special_str: str,
    skill_tokens: int,
    unique_tokens: int,
    suffix_tokens: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded_separator = tokenizer.encode(blend_special_str)
    separator = encoded_separator[1:]
    if not separator:
        raise ValueError(
            "LMCache's effective CacheBlend separator is empty after encode(...)[1:]"
        )

    skill = _repeat_tokens(
        tokenizer,
        "A reusable engineering skill explains requirements, invariants, failure "
        "recovery, validation criteria, and precise execution order. ",
        skill_tokens,
    )
    unique_prefix = _repeat_tokens(
        tokenizer,
        f"Unique probe {run_id} contains request-local reasoning and tool state. ",
        unique_tokens,
    )
    suffix = _repeat_tokens(
        tokenizer,
        "Continue locally after the reusable skill and provide one concise result. ",
        suffix_tokens,
    )
    warm_tail = _repeat_tokens(
        tokenizer,
        "This warm-up tail is intentionally not part of the reusable skill. ",
        suffix_tokens,
    )

    warm_prompt = skill + separator + warm_tail
    probe_prompt = unique_prefix + separator + skill + separator + suffix
    segment_start = len(unique_prefix) + len(separator)
    return {
        "encoded_separator_tokens": encoded_separator,
        "effective_separator_tokens": separator,
        "skill_tokens": skill,
        "unique_prefix_tokens": unique_prefix,
        "suffix_tokens": suffix,
        "warm_prompt_tokens": warm_prompt,
        "probe_prompt_tokens": probe_prompt,
        "segment_start": segment_start,
    }


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from vLLM: {body}") from error


def parse_probe_events(
    log_path: Path, engine_request_id_prefix: str
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if EVENT_MARKER not in line:
            continue
        raw_payload = line.split(EVENT_MARKER, 1)[1].lstrip()
        try:
            event, _ = decoder.raw_decode(raw_payload)
        except json.JSONDecodeError:
            continue
        event_request_id = event.get("request_id")
        if event_request_id == engine_request_id_prefix or (
            isinstance(event_request_id, str)
            and event_request_id.startswith(f"{engine_request_id_prefix}-")
        ):
            event["log_line"] = line_number
            events.append(event)
    return events


def summarize_probe(
    events: list[dict[str, Any]], expected_segment_start: int
) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_name.setdefault(event["event"], []).append(event)

    checks: dict[str, bool] = {
        "all_events_present": all(name in by_name for name in EXPECTED_EVENT_ORDER),
    }
    if not checks["all_events_present"]:
        missing = [name for name in EXPECTED_EVENT_ORDER if name not in by_name]
        return {
            "status": "no_go",
            "checks": checks,
            "missing_events": missing,
            "event_count": len(events),
        }

    selected = {name: by_name[name][-1] for name in EXPECTED_EVENT_ORDER}
    boundary = selected["secondary_lookup_boundary"]
    initial = selected["secondary_lookup_initial_probe"]
    forward = selected["secondary_lookup_forward_complete"]
    pinned = selected["secondary_lookup_pinned"]
    requeued = selected["secondary_lookup_requeued"]
    external = selected["secondary_lookup_external_probe"]
    local = selected["secondary_lookup_local_reattach"]
    unpinned = selected["secondary_lookup_unpinned"]
    cursor = boundary["lookup_cursor"]

    checks.update(
        {
            "segment_start_exact": boundary["segment_start"]
            == expected_segment_start,
            "cursor_aligned": cursor % boundary["alignment"] == 0,
            "initial_lookup_missed_unique_prefix": initial["lookup_start"] == 0
            and initial["matched_end"] == 0,
            "forward_finished_at_cursor": forward["num_computed_tokens"] == cursor
            and forward["num_in_flight_tokens"] == 0,
            "blocks_pinned": pinned["pinned_block_count"] > 0,
            "request_requeued_from_zero": requeued["num_computed_tokens"] == 0
            and requeued["num_in_flight_tokens"] == 0,
            "segment_lookup_started_exactly": external["lookup_start"]
            == expected_segment_start,
            "segment_hit_crossed_cursor": isinstance(external["matched_end"], int)
            and external["matched_end"] > cursor,
            "external_kv_not_applied": external["external_tokens_applied"] == 0,
            "local_apc_reattached": local["local_apc_reattached"] is True
            and local["local_apc_hit_tokens"] >= cursor,
            "temporary_pin_released": unpinned["pinned_block_count"] == 0,
            "event_order": [selected[name]["log_line"] for name in EXPECTED_EVENT_ORDER]
            == sorted(selected[name]["log_line"] for name in EXPECTED_EVENT_ORDER),
        }
    )
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "checks": checks,
        "event_count": len(events),
        "segment_start": expected_segment_start,
        "lookup_cursor": cursor,
        "overlap_tokens": cursor - expected_segment_start,
        "selected_events": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="Qwen3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vllm-log", type=Path, required=True)
    parser.add_argument(
        "--blend-special-str", default="<|fim_pad|><|repo_name|>"
    )
    parser.add_argument("--skill-tokens", type=int, default=768)
    parser.add_argument("--unique-tokens", type=int, default=320)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        config = json.loads(
            (args.run_dir / "config.json").read_text(encoding="utf-8")
        )
        probe_response = json.loads(
            (args.run_dir / "probe_response.json").read_text(encoding="utf-8")
        )
        engine_request_id_prefix = probe_response["id"]
        events = parse_probe_events(args.vllm_log, engine_request_id_prefix)
        with (args.run_dir / "events.jsonl").open("w", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, sort_keys=True) + "\n")
        summary = summarize_probe(events, config["segment_start"])
        _write_json(args.run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "go" else 1

    layout = build_token_layout(
        model_path=args.model_path,
        run_id=args.run_id,
        blend_special_str=args.blend_special_str,
        skill_tokens=args.skill_tokens,
        unique_tokens=args.unique_tokens,
        suffix_tokens=args.suffix_tokens,
    )
    warm_request_id = f"secondary-warm-{args.run_id}"
    probe_request_id = f"secondary-probe-{args.run_id}"
    config = {
        "run_id": args.run_id,
        "model_path": args.model_path,
        "served_model": args.served_model,
        "blend_special_str": args.blend_special_str,
        "encoded_separator_tokens": layout["encoded_separator_tokens"],
        "effective_separator_tokens": layout["effective_separator_tokens"],
        "warm_request_id": warm_request_id,
        "probe_request_id": probe_request_id,
        "skill_tokens": len(layout["skill_tokens"]),
        "unique_prefix_tokens": len(layout["unique_prefix_tokens"]),
        "suffix_tokens": len(layout["suffix_tokens"]),
        "segment_start": layout["segment_start"],
        "probe_only": True,
    }
    _write_json(args.run_dir / "config.json", config)
    if args.prepare_only:
        _write_json(args.run_dir / "summary.json", {"status": "prepared"})
        return 0

    url = f"http://{args.host}:{args.port}/v1/completions"
    common = {
        "model": args.served_model,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
    }
    warm_payload = {
        **common,
        "request_id": warm_request_id,
        "prompt": layout["warm_prompt_tokens"],
    }
    try:
        warm_response = _post_json(url, args.api_key, warm_payload)
    except RuntimeError as error:
        _write_json(
            args.run_dir / "summary.json",
            {"status": "no_go", "failure_stage": "warm_http", "error": str(error)},
        )
        raise
    _write_json(args.run_dir / "warm_response.json", warm_response)

    probe_payload = {
        **common,
        "request_id": probe_request_id,
        "prompt": layout["probe_prompt_tokens"],
        "kv_transfer_params": {
            "lmcache_secondary_lookup": {
                "segment_start": layout["segment_start"],
                "probe_only": True,
            }
        },
    }
    _write_json(
        args.run_dir / "requests.json",
        {
            "warm": warm_payload,
            "probe": probe_payload,
        },
    )
    try:
        probe_response = _post_json(url, args.api_key, probe_payload)
    except RuntimeError as error:
        _write_json(
            args.run_dir / "summary.json",
            {"status": "no_go", "failure_stage": "probe_http", "error": str(error)},
        )
        raise
    _write_json(args.run_dir / "probe_response.json", probe_response)

    time.sleep(0.5)
    engine_request_id_prefix = probe_response["id"]
    config["engine_request_id_prefix"] = engine_request_id_prefix
    _write_json(args.run_dir / "config.json", config)
    events = parse_probe_events(args.vllm_log, engine_request_id_prefix)
    with (args.run_dir / "events.jsonl").open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event, sort_keys=True) + "\n")
    summary = summarize_probe(events, layout["segment_start"])
    _write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
