#!/usr/bin/env python3
"""Run a minimal agent-style ContextSegmentKV experiment against vLLM.

This script deliberately models the agent transcript shape:

1. LLM call #1 decides that the agent should read a reusable context segment.
2. The "agent" executes that decision by reading a local document.
3. LLM call #2 includes the prior assistant decision, a tool-result message,
   then the actual context segment after that history. This second call passes
   vLLM xargs so the segment's KV is injected instead of prefilling it again.

It uses /v1/completions instead of /v1/chat/completions so token offsets are
unambiguous and can be computed with /tokenize on the exact prompt string.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_FILE = ROOT / "skills" / "internal-comms" / "SKILL.md"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "05_context_segment_agent_kv"
    / "context_segment_agent_demo.json"
)


def post_json(base_url: str, path: str, payload: dict[str, Any], api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=720) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


def tokenize(base_url: str, model: str, prompt: str, api_key: str) -> list[int]:
    response = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "prompt": prompt,
            "add_special_tokens": False,
        },
        api_key,
    )
    return list(response["tokens"])


def completion(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    *,
    max_tokens: int,
    request_id: str,
    context_segment_cache: dict[str, Any] | None = None,
) -> tuple[dict, float]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "add_special_tokens": False,
        "request_id": request_id,
    }
    if context_segment_cache is not None:
        # vLLM OpenAI protocol currently types vllm_xargs values as scalars.
        # Request.py accepts this JSON string and decodes it back to a dict.
        payload["vllm_xargs"] = {
            "context_segment_cache": json.dumps(
                context_segment_cache, ensure_ascii=False, sort_keys=True
            )
        }
    start = time.time()
    response = post_json(base_url, "/v1/completions", payload, api_key)
    return response, time.time() - start


def extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("text") or ""


def build_segment_block(segment_id: str, segment_text: str) -> str:
    return (
        f"<context_segment id=\"{segment_id}\">\n"
        f"{segment_text.rstrip()}\n"
        f"</context_segment>\n"
    )


def build_agent_decision_prompt(segment_id: str) -> str:
    return (
        "You are an agent controller. The user asks for an internal launch "
        "announcement. You have access to reusable context segments.\n\n"
        f"Available segment: {segment_id} = internal communications writing rules.\n\n"
        "Return exactly one line in this format if the segment is needed:\n"
        f"READ_CONTEXT_SEGMENT {segment_id}\n"
    )


def build_online_prompt(segment_id: str, decision_text: str, segment_block: str) -> tuple[str, str, str]:
    prefix = (
        "system: You are OpenHands agent, a helpful AI assistant that can "
        "interact with a computer to solve tasks.\n\n"
        "user: Draft a concise internal Slack launch announcement for a new "
        "team feature. Use the team's documented communication style if needed.\n\n"
        f"assistant: {decision_text.strip()}\n\n"
        f"tool: read_context_segment({segment_id}) completed. The retrieved "
        "document is appended below.\n\n"
    )
    suffix = (
        "\nassistant: Based on the prior decision and the retrieved context "
        "segment, write the final Slack launch announcement. Keep it concise, "
        "structured, and practical.\n"
    )
    return prefix, segment_block, suffix


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.vllm_port}"
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    segment_path = Path(args.segment_file)
    segment_text = segment_path.read_text(encoding="utf-8")
    segment_block = build_segment_block(args.segment_id, segment_text)

    segment_tokens = tokenize(base_url, args.model, segment_block, api_key)
    if not segment_tokens:
        raise RuntimeError("Segment tokenization returned no tokens")

    offline_cfg = {
        "sources": [
            {
                "cache_id": args.segment_id,
                "source_start": 0,
                "source_end": len(segment_tokens),
            }
        ]
    }
    offline_resp, offline_elapsed = completion(
        base_url,
        args.model,
        segment_block,
        api_key,
        max_tokens=1,
        request_id=f"ctxseg-offline-{args.segment_id}",
        context_segment_cache=offline_cfg,
    )

    decision_prompt = build_agent_decision_prompt(args.segment_id)
    decision_resp, decision_elapsed = completion(
        base_url,
        args.model,
        decision_prompt,
        api_key,
        max_tokens=32,
        request_id=f"ctxseg-agent-decision-{args.segment_id}",
    )
    decision_text = extract_text(decision_resp).strip()
    if args.force_decision or args.segment_id not in decision_text:
        decision_text = f"READ_CONTEXT_SEGMENT {args.segment_id}"

    prefix, segment_block, suffix = build_online_prompt(
        args.segment_id, decision_text, segment_block
    )
    target_start = len(tokenize(base_url, args.model, prefix, api_key))
    target_end = len(tokenize(base_url, args.model, prefix + segment_block, api_key))
    target_len = target_end - target_start
    if target_len != len(segment_tokens):
        raise RuntimeError(
            "Online segment token span length does not match offline source span. "
            f"offline={len(segment_tokens)} online={target_len}. "
            "Use a segment wrapper/boundary that tokenizes identically in both contexts."
        )

    online_prompt = prefix + segment_block + suffix
    online_cfg = {
        "targets": [
            {
                "cache_id": args.segment_id,
                "mode": "rope",
                "target_start": target_start,
                "target_end": target_end,
            }
        ]
    }
    online_resp, online_elapsed = completion(
        base_url,
        args.model,
        online_prompt,
        api_key,
        max_tokens=args.max_tokens,
        request_id=f"ctxseg-online-agent-{args.segment_id}",
        context_segment_cache=online_cfg,
    )

    baseline_resp = None
    baseline_elapsed = None
    if args.run_baseline:
        baseline_resp, baseline_elapsed = completion(
            base_url,
            args.model,
            online_prompt,
            api_key,
            max_tokens=args.max_tokens,
            request_id=f"ctxseg-baseline-agent-{args.segment_id}",
        )

    result = {
        "model": args.model,
        "segment_id": args.segment_id,
        "segment_file": str(segment_path),
        "segment_tokens": len(segment_tokens),
        "target_start": target_start,
        "target_end": target_end,
        "offline": {
            "elapsed_seconds": round(offline_elapsed, 6),
            "text": extract_text(offline_resp),
            "request_context_segment_cache": offline_cfg,
        },
        "agent_decision": {
            "elapsed_seconds": round(decision_elapsed, 6),
            "raw_text": extract_text(decision_resp),
            "used_text": decision_text,
        },
        "online_injected": {
            "elapsed_seconds": round(online_elapsed, 6),
            "text": extract_text(online_resp),
            "request_context_segment_cache": online_cfg,
        },
        "baseline_no_injection": None
        if baseline_resp is None
        else {
            "elapsed_seconds": round(float(baseline_elapsed), 6),
            "text": extract_text(baseline_resp),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an agent-style ContextSegmentKV offline+online demo."
    )
    parser.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--segment-id", default="segment-internal-comms-v1")
    parser.add_argument("--segment-file", default=str(DEFAULT_SEGMENT_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument(
        "--force-decision",
        action="store_true",
        help="Use the expected READ_CONTEXT_SEGMENT decision even if call #1 differs.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also run the same online prompt without ContextSegmentKV injection.",
    )
    args = parser.parse_args()

    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] wrote {output}")
    print(
        "[summary] "
        f"segment_tokens={result['segment_tokens']} "
        f"target=[{result['target_start']}, {result['target_end']}) "
        f"offline={result['offline']['elapsed_seconds']}s "
        f"decision={result['agent_decision']['elapsed_seconds']}s "
        f"online_injected={result['online_injected']['elapsed_seconds']}s"
    )
    if result["baseline_no_injection"] is not None:
        print(
            "[summary] "
            f"baseline_no_injection={result['baseline_no_injection']['elapsed_seconds']}s"
        )


if __name__ == "__main__":
    main()
