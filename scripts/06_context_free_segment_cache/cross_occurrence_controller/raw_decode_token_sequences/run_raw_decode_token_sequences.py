from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parents[1]
MODULE_DIR = PACKAGE_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import (  # noqa: E402
    DEFAULT_KV_DIR,
    DEFAULT_SERVED_MODEL,
    DEFAULT_TASKS,
    DEFAULT_VLLM_PORT,
)
from replay import context_config_for_case, selected_cases  # noqa: E402
from trace_utils import (  # noqa: E402
    convert_messages,
    load_invocations,
    load_system_prompt,
    load_tools,
)
from vllm_client import chat_completion  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_DIR = (
    ROOT / "results" / "problem_exploration" / "raw_decode_token_sequences"
)
DEFAULT_SEQUENCE_DIR = DEFAULT_RESULT_DIR / "sequences_without_occ12"
DEFAULT_MANIFEST = DEFAULT_RESULT_DIR / "data" / "sequence_manifest_without_occ12.jsonl"
SUPPORTED_MODES = {"recompute", "rope"}
# OCCURRENCES = (1, 2, 3)
# OCCURRENCES = (2, 3)
OCCURRENCES = (3,)


def safe_component(value: str) -> str:
    """把 task/skill 名压成可预测、可排序的文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sequence_filename(case: dict[str, Any]) -> str:
    """文件名前置 invocation_index，便于直接按 trace 顺序浏览。"""
    return (
        f"inv{int(case['invocation_index']):03d}--"
        f"{safe_component(str(case['task']))}--"
        f"{safe_component(str(case['skill']))}--"
        f"occ{int(case['occurrence'])}.txt"
    )


def case_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["task"]),
        str(row["skill"]),
        int(row["occurrence"]),
        int(row["invocation_index"]),
        str(row["mode"]),
    )


def load_manifest(path: Path) -> dict[tuple[str, str, int, int, str], dict[str, Any]]:
    """读取已有manifest；重复key直接失败，避免断点续跑掩盖脏数据。"""
    rows: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = case_key(row)
            if key in rows:
                raise ValueError(
                    f"Duplicate manifest key at line {line_number}: {key}"
                )
            rows[key] = row
    return rows


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def raw_generated_tokens(response: dict[str, Any]) -> list[str]:
    """
    只从raw logprob stream取实际生成token，不接触message parser字段。
    """
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one choice, got {len(choices)}")
    logprobs = choices[0].get("logprobs") or {}
    entries = logprobs.get("content")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Response is missing non-empty choices[0].logprobs.content")
    # 举例
    # "logprobs": {
    #         "content": [
    #           {
    #             "token": "<think>",
    #             "logprob": -0.001,
    #             "bytes": [60, 116, 104, 105, 110, 107, 62],
    #             "top_logprobs": []
    #           },
    #           {
    #             "token": "\n",
    #             "logprob": -0.01,
    #             "bytes": [10],
    #             "top_logprobs": []
    #           }
    #         ]
    #       },
    tokens: list[str] = []
    for token_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("token") is None:
            raise ValueError(f"Missing raw token at token_index={token_index}")
        tokens.append(str(entry["token"]))
    return tokens


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DEFAULT_TASKS, required=True)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "仍然按顺序重放已有case以重建prefix状态；若输出已存在，要求新旧"
            "raw sequence完全一致且不重复写manifest。"
        ),
    )
    args = parser.parse_args()

    # include_first_occurrence=True保证 occurrence 1不是隐藏warmup，而是和后续
    # occurrence一样保存输出。selected_cases会按invocation_index排序。
    cases = selected_cases(
        [args.task],
        list(OCCURRENCES),
        include_first_occurrence=True,
    )
    if not cases:
        raise ValueError(f"No cases found for task={args.task}")

    # 用于读取已经完成的输出记录，支持防覆盖和断点续跑
    existing = load_manifest(args.manifest)
    current_keys = {
        (
            str(case["task"]),
            str(case["skill"]),
            int(case["occurrence"]),
            int(case["invocation_index"]),
            args.mode,
        )
        for case in cases
    }
    overlapping = sorted(current_keys & set(existing))
    if overlapping and not args.resume:
        raise FileExistsError(
            f"Manifest already contains current task/mode rows: {overlapping}; "
            "use CLEAN_OUTPUT=1 or RESUME=1"
        )

    system_prompt = load_system_prompt()
    tools = load_tools()
    invocations = load_invocations(args.task)
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"

    for case in cases:
        invocation_index = int(case["invocation_index"])
        invocation = invocations[invocation_index - 1]
        messages, _ = convert_messages(invocation["messages"], system_prompt)
        cfg = context_config_for_case(
            args.mode,
            case,
            dump_kv_for_cksim=False,
        )
        request_id = (
            f"cf-raw-sequence-{args.mode}-{case['task']}-{case['skill']}"
            f"-occ{case['occurrence']}-inv{invocation_index}"
        )

        # logprobs=True使服务返回未经message parser拆分的逐token stream。
        # 不请求top_logprobs，避免保存与当前目标无关的候选分布。

        # 封装在vllm_client.py
        response, elapsed = chat_completion(
            base_url,
            args.model,
            messages,
            tools,
            args.api_key,
            max_tokens=args.max_tokens,
            request_id=request_id,
            context_segment_cache=cfg,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            seed=args.seed,
            logprobs=True,
        )
        tokens = raw_generated_tokens(response)
        raw_text = "".join(tokens)
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        # run 级目录（recompute_run1 / rope 等）已由外层 --sequence-dir 指定，
        # run_name 已含 mode 信息，这里不再多套一层 mode 子目录。
        output_path = args.sequence_dir / sequence_filename(case)
        key = (
            str(case["task"]),
            str(case["skill"]),
            int(case["occurrence"]),
            invocation_index,
            args.mode,
        )
        old = existing.get(key)
        if old is not None:
            old_path = ROOT / str(old["sequence_path"])
            if not old_path.exists():
                raise FileNotFoundError(
                    f"Resume manifest points to missing sequence: {old_path}"
                )
            old_text = old_path.read_text(encoding="utf-8")
            if old_text != raw_text or str(old.get("sha256")) != digest:
                raise RuntimeError(
                    f"Resume replay changed raw sequence for key={key}"
                )
            print(
                f"[replayed-identical] mode={args.mode} task={case['task']} "
                f"skill={case['skill']} occ{case['occurrence']} "
                f"inv{invocation_index} tokens={len(tokens)}",
                flush=True,
            )
            continue

        atomic_write_text(output_path, raw_text)
        manifest_row = {
            **case,
            "mode": args.mode,
            "request_id": request_id,
            "token_count": len(tokens),
            "character_count": len(raw_text),
            "sha256": digest,
            "sequence_path": str(output_path.relative_to(ROOT)),
            "elapsed_sec": round(float(elapsed), 4),
        }
        append_manifest(args.manifest, manifest_row)
        existing[key] = manifest_row
        print(
            f"[saved] mode={args.mode} task={case['task']} "
            f"skill={case['skill']} occ{case['occurrence']} "
            f"inv{invocation_index} tokens={len(tokens)} path={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
