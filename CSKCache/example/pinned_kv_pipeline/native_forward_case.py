"""Measure native vLLM suffix prefill after an exact prefix-cache hit."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from forward_microbench_config import (
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    MODEL_PATH,
    NATIVE_CASE_CONFIG_ENV,
    PREFIX_CONTEXT_TOKENS,
    PROMPT_TOKEN_ID,
    SUFFIX_TOKEN_ID,
    WARMUP_REQUESTS,
)


async def _generate(engine: Any, token_ids: list[int], request_id: str) -> Any:
    from vllm import SamplingParams

    final_output = None
    async for output in engine.generate(
        {"prompt_token_ids": token_ids},
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
        request_id=request_id,
    ):
        final_output = output
    return final_output


async def _reset(engine: Any) -> None:
    if not await engine.reset_prefix_cache(reset_running_requests=False):
        raise RuntimeError("native prefix cache reset failed")


async def _run(case: dict[str, Any]) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(MODEL_PATH),
            dtype="auto",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=True,
            enable_prefix_caching=True,
            async_scheduling=False,
            disable_log_stats=True,
            worker_extension_cls=(
                "native_forward_profile.NativeForwardProfileWorkerExtension"
            ),
        )
    )
    results = []
    prefix = [PROMPT_TOKEN_ID] * PREFIX_CONTEXT_TOKENS
    try:
        for calibration_tokens in case["calibration_tokens"]:
            suffix = [SUFFIX_TOKEN_ID] * int(calibration_tokens)
            for warmup in range(WARMUP_REQUESTS):
                await _reset(engine)
                await _generate(
                    engine,
                    prefix,
                    f"native-prefix-p{calibration_tokens}-w{warmup}",
                )
                await _generate(
                    engine,
                    prefix + suffix,
                    f"native-suffix-p{calibration_tokens}-w{warmup}",
                )

            await _reset(engine)
            await _generate(
                engine,
                prefix,
                f"native-prefix-p{calibration_tokens}-measured",
            )
            await engine.collective_rpc("start_native_forward_profile")
            output = await _generate(
                engine,
                prefix + suffix,
                f"native-suffix-p{calibration_tokens}-measured",
            )
            worker_results = await engine.collective_rpc(
                "finish_native_forward_profile"
            )
            if len(worker_results) != 1:
                raise RuntimeError("native microbenchmark requires one worker")
            results.append(
                {
                    "calibration_tokens": int(calibration_tokens),
                    "prefix_tokens": PREFIX_CONTEXT_TOKENS,
                    "num_cached_tokens": int(output.num_cached_tokens),
                    **worker_results[0],
                }
            )
    finally:
        engine.shutdown()

    Path(case["output_path"]).write_text(
        json.dumps(
            {"repetition": int(case["repetition"]), "results": results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    case_path = os.environ.get(NATIVE_CASE_CONFIG_ENV)
    if not case_path:
        raise RuntimeError(f"{NATIVE_CASE_CONFIG_ENV} is not set")
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    asyncio.run(_run(case))


if __name__ == "__main__":
    main()
