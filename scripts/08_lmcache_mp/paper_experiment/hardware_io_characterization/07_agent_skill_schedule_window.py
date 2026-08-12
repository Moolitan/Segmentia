#!/usr/bin/env python3
"""Legacy single-case diagnostic for the post-Observation tail window.

必须以 SEGMENTIA_MODE=no_reuse 启动交互 Agent，让它完成一次 SkillTool 加载和
紧随其后的正常 Prefill 请求，并在 Agent 中输入 /exit。该运行不会注入缓存 KV，
也不会执行 Segmentia 的额外 /tokenize；Agent 和 vLLM 分别写出请求关联事件与
scheduler admission 时间戳。然后再运行本脚本汇总调度窗口。

本脚本不会启动 Agent、vLLM 或任何 GPU 实验；它只读取已经采集的事件文件。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from common import load_config, percentile, write_test_result


METRIC_FIELDS = (
    "observation_to_wrapper_ms",
    "wrapper_to_transport_handoff_ms",
    "transport_handoff_to_scheduler_admission_ms",
    "observation_to_scheduler_admission_ms",
)


def load_event_payload(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            "正常Prefill的Agent事件文件不存在。请先以SEGMENTIA_MODE=no_reuse"
            "执行run_interactive_agent.sh，完成一次Skill加载和后续LLM请求，"
            f"再输入/exit：{path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Skill事件文件必须是JSON数组：{path}")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Skill事件数组包含非对象元素：{path}")
    return value


def load_scheduler_payload(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            "vLLM scheduler admission文件不存在。请确认正常Prefill服务由修改后的"
            f"run_interactive_agent.sh启动：{path}"
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"scheduler admission第{line_number}行不是JSON对象：{path}"
            )
        records.append(value)
    if not records:
        raise ValueError(f"scheduler admission文件中没有记录：{path}")
    return records


def index_scheduler_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        request_id = str(record.get("request_id", "")).strip()
        if not request_id:
            raise ValueError("scheduler admission记录缺少request_id")
        if request_id in indexed:
            raise ValueError(f"scheduler admission request_id重复：{request_id}")
        if record.get("boundary") != "immediately_before_scheduler_add_request":
            raise ValueError(f"scheduler admission边界不正确：{request_id}")
        indexed[request_id] = record
    return indexed


def extract_samples(
    events: list[dict[str, Any]], scheduler_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scheduler_by_request = index_scheduler_records(scheduler_records)
    samples: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for index, event in enumerate(events):
        skill = str(event.get("skill", "")).strip() or f"event[{index}]"
        timing = event.get("schedule_timing")
        if (
            event.get("execution_mode") != "normal_prefill"
            or not isinstance(timing, dict)
            or timing.get("status") != "awaiting_scheduler_admission"
        ):
            status = (
                timing.get("status", "missing_schedule_timing")
                if isinstance(timing, dict)
                else "missing_schedule_timing"
            )
            incomplete.append(f"{skill}:{status}")
            continue

        request_id = str(event.get("request_id", "")).strip()
        scheduler_record = scheduler_by_request.get(request_id)
        if scheduler_record is None:
            incomplete.append(f"{skill}:missing_scheduler_admission:{request_id}")
            continue
        observation_ns = int(timing["observation_callback_unix_ns"])
        wrapper_ns = int(timing["request_wrapper_enter_unix_ns"])
        handoff_ns = int(timing["client_transport_handoff_unix_ns"])
        admission_ns = int(scheduler_record["scheduler_admission_unix_ns"])
        if not observation_ns <= wrapper_ns <= handoff_ns <= admission_ns:
            raise ValueError(
                "调度时间戳顺序错误，应满足Observation <= wrapper <= transport "
                "handoff <= scheduler admission："
                f"skill={skill}, values="
                f"{(observation_ns, wrapper_ns, handoff_ns, admission_ns)}"
            )

        sample = {
            "skill": skill,
            "execution_mode": "normal_prefill",
            "request_id": request_id,
            "session_id": timing.get("session_id"),
            "observation_event_id": timing.get("observation_event_id"),
            "observation_event_timestamp": timing.get(
                "observation_event_timestamp"
            ),
            "observation_callback_unix_ns": observation_ns,
            "request_wrapper_enter_unix_ns": wrapper_ns,
            "client_transport_handoff_unix_ns": handoff_ns,
            "scheduler_admission_unix_ns": admission_ns,
            "observation_to_wrapper_ms": (wrapper_ns - observation_ns) / 1e6,
            "wrapper_to_transport_handoff_ms": (handoff_ns - wrapper_ns) / 1e6,
            "transport_handoff_to_scheduler_admission_ms": (
                admission_ns - handoff_ns
            )
            / 1e6,
            "observation_to_scheduler_admission_ms": (
                admission_ns - observation_ns
            )
            / 1e6,
        }
        for field in METRIC_FIELDS:
            value = float(sample[field])
            if value < 0:
                raise ValueError(f"调度时间不能为负数：skill={skill}, {field}={value}")

        components = (
            sample["observation_to_wrapper_ms"]
            + sample["wrapper_to_transport_handoff_ms"]
            + sample["transport_handoff_to_scheduler_admission_ms"]
        )
        if (
            abs(
                components
                - sample["observation_to_scheduler_admission_ms"]
            )
            > 0.001
        ):
            raise ValueError(
                "调度窗口分量与总时间不一致："
                f"skill={skill}, components={components}, "
                f"total={sample['observation_to_scheduler_admission_ms']}"
            )
        samples.append(sample)

    if incomplete:
        raise ValueError(
            "事件文件包含未完成的Skill调度样本。请使用修改后的Agent重新运行："
            + ", ".join(incomplete)
        )
    if not samples:
        raise ValueError(
            "没有完整的Skill调度样本。请确认Agent至少完成一次SkillTool调用，"
            "并在得到Skill结果后继续发送了下一轮completion请求。"
        )
    # New instrumentation assigns an X-Request-Id to request A as well as
    # request B. Request A therefore has a scheduler record but no post-Skill
    # Agent event; it is expected and is ignored by this legacy tail analyzer.
    return samples


def describe(samples: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = [float(sample[field]) for sample in samples]
    return {
        "count": len(values),
        "minimum_ms": min(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "maximum_ms": max(values),
    }


def main() -> None:
    config = load_config()
    source_agent = Path(config["agent_schedule"]["source_agent_events"]).resolve()
    source_scheduler = Path(
        config["agent_schedule"]["source_scheduler_events"]
    ).resolve()
    samples = extract_samples(
        load_event_payload(source_agent),
        load_scheduler_payload(source_scheduler),
    )
    summary = {field: describe(samples, field) for field in METRIC_FIELDS}
    path = write_test_result(
        "07_agent_skill_schedule_window",
        config,
        {
            "source_agent_events": str(source_agent),
            "source_scheduler_events": str(source_scheduler),
            "measurement_definition": {
                "t0": (
                    "Skill ObservationEvent进入Conversation回调，Agent已获得Skill正文"
                ),
                "t1": "正常Prefill的下一轮LLM transport wrapper入口",
                "t2": (
                    "仅附加X-Request-Id后，将正常Prefill completion交给原始"
                    "LiteLLM transport"
                ),
                "t3": (
                    "vLLM完成前端解析、正常tokenization和EngineCore预处理后，"
                    "即将调用scheduler.add_request"
                ),
                "main_metric": (
                    "observation_to_scheduler_admission_ms = T3 - T0"
                ),
                "boundary": (
                    "正常Prefill且不注入任何Segmentia KV；主指标包含客户端、"
                    "网络和vLLM入口处理，但不包含scheduler排队、GPU Prefill或decode"
                ),
            },
            "samples": samples,
            "summary": summary,
        },
    )
    main_metric = summary["observation_to_scheduler_admission_ms"]
    print(
        f"[completed] {path} samples={main_metric['count']} "
        f"p50={main_metric['p50_ms']:.3f}ms "
        f"p95={main_metric['p95_ms']:.3f}ms"
    )


if __name__ == "__main__":
    main()
