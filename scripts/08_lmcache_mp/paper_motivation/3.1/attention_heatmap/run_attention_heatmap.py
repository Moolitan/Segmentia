#!/usr/bin/env python3
"""Capture or plot CacheBlend Figure-4-style attention matrices."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from skill_cache_tokens import qwen_tool_response_token_ids  # noqa: E402


USER_MESSAGE = 'Please explicitly reference and use the "internal-comms" skill for this turn.'
BASE_SYSTEM_MESSAGE = "You are a helpful assistant."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "skill",
            "description": "Load a named skill guide.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
]


def post_json(
    base_url: str,
    path: str,
    api_key: str,
    payload: dict[str, Any],
    request_id: str | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {path}")
    return value


def messages_for(skill_text: str, padding_tokens: int) -> list[dict[str, Any]]:
    system = BASE_SYSTEM_MESSAGE + " padding" * padding_tokens
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_MESSAGE},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "skill",
                        "arguments": '{"name":"internal-comms"}',
                    },
                }
            ],
        },
        {"role": "tool", "name": "skill", "content": skill_text},
    ]


def rendered_ids(
    tokenizer: Any, messages: list[dict[str, Any]]
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    ids = (
        rendered["input_ids"]
        if hasattr(rendered, "keys") and "input_ids" in rendered
        else rendered
    )
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def find_once(haystack: list[int], needle: list[int]) -> int:
    starts = [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]
    if len(starts) != 1:
        raise RuntimeError(f"expected one Skill token span, found {starts}")
    return starts[0]


def aligned_prompt(
    tokenizer: Any, skill_text: str
) -> tuple[list[dict[str, Any]], list[int], int, int, int]:
    skill_ids = qwen_tool_response_token_ids(tokenizer, skill_text)
    for padding_tokens in range(64):
        messages = messages_for(skill_text, padding_tokens)
        prompt_ids = rendered_ids(tokenizer, messages)
        skill_start = find_once(prompt_ids, skill_ids)
        if skill_start % 16 == 0:
            return (
                messages,
                prompt_ids,
                skill_start,
                skill_start + len(skill_ids),
                padding_tokens,
            )
    raise RuntimeError("could not align the Skill span within 64 padding tokens")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def capture(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    skill_text = args.skill_path.read_text(encoding="utf-8")
    messages, prompt_ids, skill_start, skill_end, padding_tokens = aligned_prompt(
        tokenizer, skill_text
    )
    if skill_end >= len(prompt_ids):
        raise RuntimeError("chat template did not append an assistant-generation suffix")

    tokenize_payload = {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    server_tokens = post_json(
        args.base_url, "/tokenize", args.api_key, tokenize_payload
    ).get("tokens")
    if server_tokens != prompt_ids:
        raise RuntimeError("local tokenizer and vLLM /tokenize disagree")

    request_id = f"segmentia-attention-heatmap-{args.mode}"
    mode_dir = args.output_dir / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "request_id": request_id,
        "mode": args.mode,
        "prefix_start": 0,
        "prefix_end": skill_start,
        "skill_start": skill_start,
        "skill_end": skill_end,
        "prompt_end": len(prompt_ids),
    }
    write_json(args.spec_path, spec)
    write_json(
        mode_dir / "prompt_metadata.json",
        {
            **spec,
            "user_messages": [USER_MESSAGE],
            "system_padding_tokens": padding_tokens,
            "prompt_token_count": len(prompt_ids),
            "skill_token_count": skill_end - skill_start,
            "forward_query_token_count": len(prompt_ids) - skill_end,
            "prompt_token_ids": prompt_ids,
        },
    )

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if args.mode == "direct":
        payload["kv_transfer_params"] = {
            "lmcache_segmentia_lookup": {
                "segment_start": skill_start,
                "segment_end": skill_end,
            }
        }
    response = post_json(
        args.base_url,
        "/v1/chat/completions",
        args.api_key,
        payload,
        request_id=request_id,
    )
    write_json(mode_dir / "response.json", response)

    files = sorted(mode_dir.glob(f"{args.mode}_layer_*.npz"))
    if len(files) != 40:
        raise RuntimeError(f"expected 40 captured layers, found {len(files)}")
    for path in files:
        with np.load(path) as layer:
            cross_rows = int(layer["cross"].shape[0])
            forward_rows = int(layer["forward"].shape[0])
        if args.mode == "recompute" and cross_rows != skill_end - skill_start:
            raise RuntimeError(f"incomplete cross-attention capture: {path}")
        if args.mode == "direct" and cross_rows != 0:
            raise RuntimeError(
                "direct reuse unexpectedly computed Skill query rows; "
                f"file={path}, rows={cross_rows}"
            )
        if forward_rows != len(prompt_ids) - skill_end:
            raise RuntimeError(f"incomplete forward-attention capture: {path}")
    print(
        f"[captured] mode={args.mode} prompt={len(prompt_ids)} "
        f"prefix=[0,{skill_start}) skill=[{skill_start},{skill_end}) "
        f"forward=[{skill_end},{len(prompt_ids)}) layers=40",
        flush=True,
    )


def load_layer(output_dir: Path, mode: str, layer: int) -> dict[str, np.ndarray]:
    path = output_dir / mode / f"{mode}_layer_{layer:02d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def draw_matrix(
    ax: Any,
    matrix: np.ndarray,
    title: str,
    vmax: float,
    *,
    missing: bool = False,
) -> None:
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=max(vmax, np.finfo(np.float32).eps),
    )
    ax.set_title(title)
    ax.set_xlabel("Key token position")
    ax.set_ylabel("Query token position")
    if missing:
        ax.text(
            0.5,
            0.5,
            "not computed\n(full KV reuse)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot(args: argparse.Namespace) -> None:
    recompute_metadata = json.loads(
        (args.output_dir / "recompute" / "prompt_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    direct_metadata = json.loads(
        (args.output_dir / "direct" / "prompt_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if recompute_metadata["prompt_token_ids"] != direct_metadata["prompt_token_ids"]:
        raise RuntimeError("recompute and direct prompts are not token-identical")
    skill_start = int(recompute_metadata["skill_start"])
    skill_end = int(recompute_metadata["skill_end"])
    prompt_end = int(recompute_metadata["prompt_end"])
    pdf_path = args.output_dir / "cacheblend_figure4_all_layers.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer_index in range(40):
            recompute = load_layer(args.output_dir, "recompute", layer_index)
            direct = load_layer(args.output_dir, "direct", layer_index)
            recompute_cross = recompute["cross"]
            recompute_forward = recompute["forward"]
            direct_forward = direct["forward"]
            direct_cross = np.zeros_like(recompute_cross)

            cross_vmax = float(np.nanmax(recompute_cross))
            forward_vmax = float(
                max(np.nanmax(recompute_forward), np.nanmax(direct_forward))
            )
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
            draw_matrix(
                axes[0, 0],
                recompute_cross,
                "Cross-attention",
                cross_vmax,
            )
            draw_matrix(
                axes[0, 1],
                recompute_forward,
                "Forward-attention",
                forward_vmax,
            )
            draw_matrix(
                axes[1, 0],
                direct_cross,
                "Cross-attention",
                cross_vmax,
                missing=True,
            )
            draw_matrix(
                axes[1, 1],
                direct_forward,
                "Forward-attention",
                forward_vmax,
            )
            axes[0, 0].set_ylabel("Full KV recompute\nSkill query token")
            axes[0, 1].set_ylabel("Full KV recompute\nSuffix query token")
            axes[1, 0].set_ylabel("Full KV reuse\nSkill query token")
            axes[1, 1].set_ylabel("Full KV reuse\nSuffix query token")
            fig.suptitle(
                f"CacheBlend Figure 4 reproduction — Qwen3-14B layer {layer_index}",
                fontsize=15,
            )
            fig.text(
                0.5,
                0.005,
                f"prefix=[0,{skill_start})  skill=[{skill_start},{skill_end})  "
                f"assistant suffix=[{skill_end},{prompt_end})",
                ha="center",
                fontsize=9,
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[plotted] {pdf_path} pages=40", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument(
        "--mode", choices=("recompute", "direct"), required=True
    )
    capture_parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    capture_parser.add_argument("--api-key", default="EMPTY")
    capture_parser.add_argument("--model", default="Qwen3")
    capture_parser.add_argument("--model-path", type=Path, required=True)
    capture_parser.add_argument("--skill-path", type=Path, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--spec-path", type=Path, required=True)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "capture":
        capture(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
