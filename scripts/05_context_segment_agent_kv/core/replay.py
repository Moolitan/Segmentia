"""Replay a single task and drive ContextSegmentKV source/target assignment."""
from __future__ import annotations

from .config import get_skill_token_span
from .message_convert import convert_messages
from .segments import find_skill_segments
from .trace_loader import load_invocations
from .vllm_client import chat_completion


def replay_task(
    task: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    tools: list[dict],
    use_reuse: bool,
    collected: set[str],
    seen_occurrences: set[tuple[str, int]],
    cache_namespace: str,
    reuse_scope: str,
) -> dict:
    """Replay one task. If use_reuse, drive ContextSegmentKV source/targets.

    `collected` (which skills have a saved source KV) is shared/reset by the
    caller per reuse_scope; `seen_occurrences` tracks within-conversation
    appearances and is always reset per task by the caller.
    """
    invocations = load_invocations(task)
    per_skill_source_len: dict[str, int] = {}
    request_records: list[dict] = []
    source_tokens_total = 0
    target_tokens_total = 0

    for inv in invocations:
        openai_msgs, _ = convert_messages(inv["messages"], system_prompt)
        segments = find_skill_segments(openai_msgs)

        # Assign a stable occurrence index per skill (read order within task).
        occ_indexed: list[tuple[str, int, int, int, int]] = []
        local_count: dict[str, int] = {}
        for (name, midx, cs, ce) in segments:
            local_count[name] = local_count.get(name, 0) + 1
            occ_indexed.append((name, local_count[name], midx, cs, ce))

        sources: list[dict] = []
        targets: list[dict] = []
        actions: list[dict] = []

        if use_reuse:
            for (name, occ_idx, midx, cs, ce) in occ_indexed:
                key = (name, occ_idx)
                if key in seen_occurrences:
                    continue  # already prefilled in an earlier request
                seen_occurrences.add(key)
                cache_id = f"{cache_namespace}-{name}-v1"
                tstart, tend = get_skill_token_span(task, name, occ_idx)
                if cache_id not in collected:
                    sources.append(
                        {"cache_id": cache_id, "source_start": tstart, "source_end": tend}
                    )
                    per_skill_source_len[cache_id] = tend - tstart
                    actions.append(
                        {"role": "source", "skill": name, "occ": occ_idx,
                         "span": [tstart, tend], "tokens": tend - tstart}
                    )
                    collected.add(cache_id)
                    source_tokens_total += tend - tstart
                else:
                    targets.append(
                        {"cache_id": cache_id, "mode": "rope",
                         "target_start": tstart, "target_end": tend}
                    )
                    actions.append(
                        {"role": "target", "skill": name, "occ": occ_idx,
                         "span": [tstart, tend], "tokens": tend - tstart}
                    )
                    target_tokens_total += tend - tstart

        ctx_cfg = None
        if sources or targets:
            ctx_cfg = {"sources": sources, "targets": targets}

        req_id = (
            f"replay-{task}-t{inv['turn']}i{inv['invocation']}"
            f"-{'reuse' if use_reuse else 'recompute'}"
        )
        try:
            resp, elapsed = chat_completion(
                base_url, model, openai_msgs, tools, api_key,
                max_tokens=1, request_id=req_id, context_segment_cache=ctx_cfg,
            )
        except RuntimeError as exc:
            # Most likely a context-length overflow (request > max-model-len).
            # Record and continue so one oversized request doesn't kill the run.
            print(f"  [SKIP] t{inv['turn']}i{inv['invocation']}: {exc}", flush=True)
            request_records.append(
                {
                    "turn": inv["turn"],
                    "invocation": inv["invocation"],
                    "error": str(exc)[:300],
                    "num_messages": len(openai_msgs),
                    "segments_in_prompt": len(segments),
                    "ctx_actions": actions,
                }
            )
            continue

        usage = resp.get("usage", {})
        request_records.append(
            {
                "turn": inv["turn"],
                "invocation": inv["invocation"],
                "latency_s": round(elapsed, 4),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                ),
                "num_messages": len(openai_msgs),
                "segments_in_prompt": len(segments),
                "ctx_actions": actions,
            }
        )
        tag = "+".join(f"{a['role'][0]}:{a['skill']}#{a['occ']}" for a in actions) or "-"
        print(
            f"  [{ 'reuse' if use_reuse else 'recompute' }] t{inv['turn']}i{inv['invocation']} "
            f"msgs={len(openai_msgs)} prompt_tok={usage.get('prompt_tokens')} "
            f"cached={ (usage.get('prompt_tokens_details') or {}).get('cached_tokens') } "
            f"lat={elapsed:.3f}s ctx={tag}",
            flush=True,
        )

    total_lat = sum(r.get("latency_s") or 0 for r in request_records)
    total_prompt = sum(r.get("prompt_tokens") or 0 for r in request_records)
    total_cached = sum(r.get("cached_tokens") or 0 for r in request_records)
    n_errors = sum(1 for r in request_records if "error" in r)
    return {
        "task": task,
        "mode": "reuse" if use_reuse else "recompute",
        "reuse_scope": reuse_scope,
        "num_requests": len(request_records),
        "num_errors": n_errors,
        "total_latency_s": round(total_lat, 4),
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "segkv_source_tokens": source_tokens_total,
        "segkv_target_tokens": target_tokens_total,
        "requests": request_records,
    }
