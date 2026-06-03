from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


def fetch_vllm_metrics_text(vllm_port: int, timeout_seconds: float = 2.0) -> str:
    url = f"http://127.0.0.1:{vllm_port}/metrics"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
        return resp.read().decode("utf-8", errors="replace")


def sum_prometheus_metric_samples(text: str, metric_name: str) -> float:
    """
    从一次 Prometheus exposition 快照中，找出指定 metric_name 的所有样本行并求和。

    这里的“求和”是把同名 metric 的多条时序（同一 metric_name、不同 label 组合）
    在当前快照里的样本值聚合起来；不是把同一条 counter 沿时间再次累计。

    比如这两行就是两条不同的时序：
    vllm:prefix_cache_queries_total{model_name="qwen",engine="0"} 12345
    vllm:prefix_cache_queries_total{model_name="qwen",engine="1"} 67890

    单机、单实例、没有 data parallel 时：通常只有 engine="0"
    开了 data parallel 时：会有 engine="0", engine="1" 之类，每个 engine 是一个独立的推理引擎实例
    一个 engine 下面可能还会用多张 GPU 做 TP/PP，所以 engine != GPU


    适用于 counter、gauge，以及 histogram 的 `_sum` / `_count` 这类单值样本；
    忽略 HELP/TYPE 注释行。
    """
    total = 0.0
    prefix = metric_name + "{"
    bare = metric_name + " "
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        if line.startswith(prefix):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except Exception:
                pass
        elif line.startswith(bare):
            try:
                total += float(line.split(" ", 1)[1])
            except Exception:
                pass
    return total


def parse_prometheus_gauge(text: str, metric_name: str) -> float | None:
    """
    解析 Prometheus gauge 单值。

    若同名 metric 暴露了多条时序，则返回最后一个匹配到的样本值；否则返回 None。
    """
    prefix = metric_name + "{"
    bare = metric_name + " "
    val = None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        if line.startswith(prefix) or line.startswith(bare):
            try:
                val = float(line.rsplit(" ", 1)[-1])
            except Exception:
                pass
    return val


def parse_cache_config_num_gpu_blocks(text: str) -> int | None:
    """从 vllm:cache_config_info 的 label 中提取 num_gpu_blocks。"""
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            import re
            m = re.search(r'num_gpu_blocks="(\d+)"', line)
            if m:
                return int(m.group(1))
    return None


def parse_cache_config_block_size(text: str) -> int:
    """从 vllm:cache_config_info 的 label 中提取 block_size，默认 16。"""
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            import re
            m = re.search(r'block_size="(\d+)"', line)
            if m:
                return int(m.group(1))
    return 16


@dataclass(frozen=True)
class VllmPrefixCacheSample:
    prefix_cache_queries_total: float
    prefix_cache_hits_total: float
    request_prefill_time_seconds_sum: float
    request_prefill_time_seconds_count: float
    time_to_first_token_seconds_sum: float
    time_to_first_token_seconds_count: float

    @staticmethod
    def from_metrics_text(text: str) -> "VllmPrefixCacheSample":
        return VllmPrefixCacheSample(
            prefix_cache_queries_total=sum_prometheus_metric_samples(
                # 本地前缀缓存被查询的 Token 总数（自服务启动累计）
                # curl拉取的是这样的:vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3"} 0.0
                # 每当一个请求进入 vLLM 的 Prefill 阶段，引擎会拿着这个请求的 每一个 Token（或每一个 Block）
                # 去显存里的前缀缓存哈希表里 “问一遍：这个 Token 之前算过吗？能直接用吗？”

                # prompt_tokens_total：统计的是 vLLM 实际接收并处理的输入 Token 总数。
                # prefix_cache_queries_total：统计的是 vLLM 去缓存哈希表里查找的 Token 总数。
                # 在 纯文本模型 下，这个 counter 的增量（delta）可以近似看作本次请求的 prompt token 数（或 block 数）；
                text, "vllm:prefix_cache_queries_total"
            ),
            prefix_cache_hits_total=sum_prometheus_metric_samples(
                # 本地前缀缓存命中的 Token 总数（自服务启动累计）
                text, "vllm:prefix_cache_hits_total"
            ),
            request_prefill_time_seconds_sum=sum_prometheus_metric_samples(
                text, "vllm:request_prefill_time_seconds_sum"
            ),
            request_prefill_time_seconds_count=sum_prometheus_metric_samples(
                text, "vllm:request_prefill_time_seconds_count"
            ),
            time_to_first_token_seconds_sum=sum_prometheus_metric_samples(
                text, "vllm:time_to_first_token_seconds_sum"
            ),
            time_to_first_token_seconds_count=sum_prometheus_metric_samples(
                text, "vllm:time_to_first_token_seconds_count"
            ),
        )

    @staticmethod
    def sample(vllm_port: int) -> "VllmPrefixCacheSample":
        return VllmPrefixCacheSample.from_metrics_text(
            fetch_vllm_metrics_text(vllm_port=vllm_port)
        )


def compute_vllm_prefix_cache_delta(before: VllmPrefixCacheSample, after: VllmPrefixCacheSample) -> dict:
    dq = after.prefix_cache_queries_total - before.prefix_cache_queries_total
    dh = after.prefix_cache_hits_total - before.prefix_cache_hits_total
    prefill_sum_delta = max(
        0.0,
        after.request_prefill_time_seconds_sum - before.request_prefill_time_seconds_sum,
    )
    prefill_count_delta = max(
        0.0,
        after.request_prefill_time_seconds_count - before.request_prefill_time_seconds_count,
    )
    ttft_sum_delta = max(
        0.0,
        after.time_to_first_token_seconds_sum - before.time_to_first_token_seconds_sum,
    )
    ttft_count_delta = max(
        0.0,
        after.time_to_first_token_seconds_count - before.time_to_first_token_seconds_count,
    )

    result = {
        "vllm_prefix_cache_queries_tokens": max(0.0, dq),
        "vllm_prefix_cache_hits_tokens": max(0.0, dh),
        "vllm_prefix_cache_hit_rate": round((dh / dq), 6) if dq > 0 else None,
        "vllm_time_to_first_token_seconds_sum_delta": ttft_sum_delta,
        "vllm_time_to_first_token_seconds_count_delta": ttft_count_delta,
        "vllm_time_to_first_token_seconds": (
            round(ttft_sum_delta / ttft_count_delta, 6)
            if ttft_count_delta > 0
            else None
        ),
        "vllm_request_prefill_time_seconds_sum_delta": prefill_sum_delta,
        "vllm_request_prefill_time_seconds_count_delta": prefill_count_delta,
        "vllm_request_prefill_time_seconds": (
            round(prefill_sum_delta / prefill_count_delta, 6)
            if prefill_count_delta > 0
            else None
        ),
    }
    return result


def attach_vllm_per_request_metrics(llm: Any, vllm_port: int) -> None:
    """
    给 OpenHands SDK 的 LLM 实例打补丁：
    - 每次实际发起一次模型请求时，在请求前后采样 vLLM /metrics 的 prefix cache counters
    - 用 delta 估计该次请求的命中/查询 token，并按发生顺序记录到 llm._vllm_request_deltas
    """
    if getattr(llm, "_vllm_metrics_patched", False):
        return

    llm._vllm_metrics_patched = True
    llm._vllm_request_deltas: list[dict] = []
    llm._vllm_metrics_supported = True

    def _record_delta(delta: dict, started_at: float, ended_at: float) -> None:
        llm._vllm_request_deltas.append(
            {
                **delta,
                "vllm_metrics_started_at": started_at,
                "vllm_metrics_ended_at": ended_at,
                "vllm_metrics_elapsed_seconds": round(ended_at - started_at, 6),
            }
        )

    def _wrap_sync(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args, **kwargs):
            if not llm._vllm_metrics_supported:
                return fn(*args, **kwargs)
            t0 = time.time()
            try:
                before = VllmPrefixCacheSample.sample(vllm_port=vllm_port)
            except Exception:
                llm._vllm_metrics_supported = False
                return fn(*args, **kwargs)
            try:
                return fn(*args, **kwargs)
            finally:
                t1 = time.time()
                if llm._vllm_metrics_supported:
                    try:
                        after = VllmPrefixCacheSample.sample(vllm_port=vllm_port)
                        _record_delta(compute_vllm_prefix_cache_delta(before, after), t0, t1)
                    except Exception:
                        llm._vllm_metrics_supported = False

        return wrapped

    def _wrap_async(fn: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(*args, **kwargs):
            if not llm._vllm_metrics_supported:
                return await fn(*args, **kwargs)
            t0 = time.time()
            try:
                before = VllmPrefixCacheSample.sample(vllm_port=vllm_port)
            except Exception:
                llm._vllm_metrics_supported = False
                return await fn(*args, **kwargs)
            try:
                return await fn(*args, **kwargs)
            finally:
                t1 = time.time()
                if llm._vllm_metrics_supported:
                    try:
                        after = VllmPrefixCacheSample.sample(vllm_port=vllm_port)
                        _record_delta(compute_vllm_prefix_cache_delta(before, after), t0, t1)
                    except Exception:
                        llm._vllm_metrics_supported = False

        return wrapped

    # Only patch ONE method — the lowest-level transport function.
    # Patching multiple methods (e.g. both `completion` and `_transport_call`)
    # causes double-counting: each actual request generates 2 delta records,
    # and the outer wrapper's elapsed time includes retry/wait overhead.
    candidates_by_priority = [
        # Prefer the lowest-level function first (closest to actual HTTP call)
        "_transport_call",
        # Fallback: try increasingly higher-level methods
        "_litellm_completion",
        "_litellm_acompletion",
        "_completion",
        "_acompletion",
        "completion",
        "acompletion",
        "chat",
        "achat",
        "chat_completion",
        "achat_completion",
        "chat_completions",
        "achat_completions",
        "generate",
        "agenerate",
        "_generate",
        "_agenerate",
        "_call",
        "_acall",
        "_invoke",
        "_ainvoke",
    ]

    patched_any = False
    import inspect

    for name in candidates_by_priority:
        fn = getattr(llm, name, None)
        if not callable(fn):
            continue
        if getattr(fn, "_vllm_wrapped", False):
            continue
        try:
            if inspect.iscoroutinefunction(fn):
                wrapped = _wrap_async(fn)
            else:
                wrapped = _wrap_sync(fn)
            setattr(wrapped, "_vllm_wrapped", True)

            # pydantic BaseModel 可能拦截 __setattr__；这里用 object.__setattr__ 强制写入
            object.__setattr__(llm, name, wrapped)
            patched_any = True
            break  # Only patch ONE method to avoid double-counting
        except Exception:
            continue

    object.__setattr__(llm, "_vllm_metrics_patched_any", patched_any)
