from __future__ import annotations

import copy
import time

from core.vllm_metrics import (
    VllmPrefixCacheSample,
    compute_vllm_prefix_cache_delta,
)


def attach_llm_request_attempt_collector(llm, vllm_port: int) -> None:
    """Collect one unified record per actual transport attempt."""
    if getattr(llm, "_request_attempt_patched", False):
        return

    llm._request_attempt_patched = True
    llm._request_attempts: list[dict] = []
    llm._request_attempt_vllm_metrics_supported = True

    original = getattr(llm, "_transport_call", None)
    if not callable(original):
        return

    def _sample_vllm():
        if not llm._request_attempt_vllm_metrics_supported:
            return None
        try:
            return VllmPrefixCacheSample.sample(vllm_port=vllm_port)
        except Exception:
            llm._request_attempt_vllm_metrics_supported = False
            return None

    def wrapped(*args, **kwargs):
        messages = kwargs.get("messages")
        before = _sample_vllm()
        started_at = time.time()
        error = None
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            error = exc
            raise
        finally:
            ended_at = time.time()
            after = _sample_vllm() if before is not None else None
            attempt = {
                "messages": copy.deepcopy(messages),
                "transport_started_at": started_at,
                "transport_ended_at": ended_at,
                "transport_elapsed_seconds": round(ended_at - started_at, 6),
                "error_type": type(error).__name__ if error is not None else None,
                "error_message": str(error) if error is not None else None,
                "vllm_prefix_cache_queries_tokens": None,
                "vllm_prefix_cache_hits_tokens": None,
                "vllm_prefix_cache_hit_rate": None,
                "vllm_request_prefill_time_seconds": None,
                "vllm_time_to_first_token_seconds": None,
            }
            if before is not None and after is not None:
                try:
                    attempt.update(compute_vllm_prefix_cache_delta(before, after))
                except Exception:
                    pass
            llm._request_attempts.append(attempt)

    object.__setattr__(wrapped, "_request_attempt_wrapped", True)
    object.__setattr__(llm, "_transport_call", wrapped)
