from __future__ import annotations

import os
import sys
import time

from core.io import StripAnsiWriter
from core.messages import flatten_request_messages, parse_tool_activity
from core.schema import SequenceTemplate


def run_sequence(
    template: SequenceTemplate,
    agent,
    seq_workspace: str,
    seq_log_path: str,
    max_iteration_per_run: int = 500,
) -> dict:
    from openhands.sdk import Conversation

    os.makedirs(seq_workspace, exist_ok=True)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    os.makedirs(os.path.dirname(seq_log_path), exist_ok=True)
    seq_log_file = open(seq_log_path, "w", encoding="utf-8")
    seq_writer = StripAnsiWriter(seq_log_file)
    sys.stdout = seq_writer
    sys.stderr = seq_writer

    conversation = Conversation(
        agent=agent,
        workspace=seq_workspace,
        max_iteration_per_run=max_iteration_per_run,
        stuck_detection=True,
        delete_on_close=True,
    )

    all_llm_calls: list[dict] = []
    seq_start = time.time()
    try:
        for i, turn_spec in enumerate(template.turns):
            turn_number = i + 1
            print(
                f"\n{'=' * 60}\n"
                f"[TURN {turn_number}/{len(template.turns)}] "
                f"{turn_spec.description}\n"
                f"{'=' * 60}\n",
                flush=True,
            )
            seq_log_file.flush()

            message = f"Working directory: {seq_workspace}\n\n{turn_spec.message}"

            llm = agent.llm
            request_attempts_before = len(getattr(llm, "_request_attempts", []))

            conversation.send_message(message)
            conversation.run()
            seq_log_file.flush()

            turn_attempts = getattr(llm, "_request_attempts", [])[
                request_attempts_before:
            ]

            for k, attempt in enumerate(turn_attempts):
                next_attempt = (
                    turn_attempts[k + 1] if k + 1 < len(turn_attempts) else None
                )
                tool_activity = parse_tool_activity(attempt, next_attempt)
                all_llm_calls.append(
                    {
                        "call_index_in_turn": k,
                        "turn_number": turn_number,
                        "request_prompt_text": flatten_request_messages(
                            attempt.get("messages", [])
                        ),
                        "tool_calls": tool_activity["tool_calls"],
                        "vllm_prefix_cache_queries_tokens": attempt.get(
                            "vllm_prefix_cache_queries_tokens"
                        ),
                        "vllm_prefix_cache_hits_tokens": attempt.get(
                            "vllm_prefix_cache_hits_tokens"
                        ),
                        "vllm_prefix_cache_hit_rate": attempt.get(
                            "vllm_prefix_cache_hit_rate"
                        ),
                        "vllm_request_prefill_time_seconds": attempt.get(
                            "vllm_request_prefill_time_seconds"
                        ),
                        "vllm_time_to_first_token_seconds": attempt.get(
                            "vllm_time_to_first_token_seconds"
                        ),
                    }
                )
        _ = time.time() - seq_start
    finally:
        try:
            conversation.close()
        except Exception:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        seq_log_file.close()

    return {"llm_calls": all_llm_calls}
