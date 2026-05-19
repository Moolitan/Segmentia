"""
Supplement experiment 2: A-only region attention scan.

This script mirrors run_supplement_exp2_gap_scan.py's prompt construction, but
does not run B1/B2 KV-reuse comparisons. It only runs normal A greedy decoding
and records how each decoded token attends to four prior regions:

  USER_CONTEXT, reference_v1, MIDDLE, reference_v2

Run:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp2_a_region_attention.py \
    --middle-gaps 50,100,200,400,800,1600,2000 \
    --attention-gaps all --attention-layers 5,10,15,25,30 --attention-decode-steps 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR_ROOT = SCRIPT_DIR.parent
COMMON_DIR = BASE_DIR_ROOT / "kvreused_naturalreferenceblock"
if (COMMON_DIR / "kv_reuse_common.py").exists():
    sys.path.append(str(COMMON_DIR))

from kv_reuse_common import KVReuseRunner, PromptSpec, ensure_dir
from run_supplement_exp2_gap_scan import (
    BASE_DIR,
    DEFAULT_MODEL,
    make_prompt_spec,
    parse_gap_list,
    parse_layer_list,
    should_dump_attention_for_gap,
)


DEFAULT_OUT_DIR = (
    BASE_DIR
    / "results"
    / "kv_reuse_natural_reference_block"
    / "supplement_exp2_a_region_attention"
)
SCENARIO = "KVReuse_NaturalReferenceBlock_ARegionAttention"
REGION_ORDER = ("user_context", "reference_v1", "middle", "reference_v2")


def locate_prompt_regions(
    runner: KVReuseRunner,
    spec: PromptSpec,
) -> tuple[list[int], dict[str, tuple[int, int]]]:
    prompt = spec.full_prompt
    full_ids, ref1_start, ref1_end = runner.locate_reference_tokens(
        prompt, spec.reference_text, "first"
    )
    _, ref2_start, ref2_end = runner.locate_reference_tokens(
        prompt, spec.reference_text, "last"
    )
    if ref1_end > ref2_start:
        raise RuntimeError(f"{spec.name}: repeated reference blocks overlap.")
    if ref2_end >= len(full_ids):
        raise RuntimeError(f"{spec.name}: query/assistant prefix after ref2 is required.")

    regions = {
        "user_context": (0, ref1_start),
        "reference_v1": (ref1_start, ref1_end),
        "middle": (ref1_end, ref2_start),
        "reference_v2": (ref2_start, ref2_end),
    }
    return full_ids, regions


def dump_a_region_attention(
    runner: KVReuseRunner,
    spec: PromptSpec,
    *,
    out_dir: Path,
    target_layers: list[int],
    decode_steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Dump normal-A greedy decoded-token attention over named prompt regions."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3 import modeling_qwen3 as qwen3_modeling

    full_ids, regions = locate_prompt_regions(runner, spec)
    n_full = len(full_ids)
    region_lengths = {name: end - start for name, (start, end) in regions.items()}
    if any(length <= 0 for length in region_lengths.values()):
        raise RuntimeError(f"{spec.name}: empty region detected: {region_lengths}")

    target_layer_set = set(target_layers)
    repeat_kv = qwen3_modeling.repeat_kv
    original_eager_attention_forward = qwen3_modeling.eager_attention_forward
    dumped_regions: dict[int, dict[str, np.ndarray]] = {}
    dumped_maxlogit: dict[int, np.ndarray] = {}
    current_decode_step = -1

    def patched_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        layer_idx = getattr(module, "layer_idx", None)
        dump_this_layer = layer_idx in target_layer_set

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        batch_size, num_heads, n_query, _ = query.shape
        _, _, n_key, value_dim = value_states.shape
        attn_output = torch.empty(
            (batch_size, num_heads, n_query, value_dim),
            dtype=query.dtype,
            device=query.device,
        )

        if dump_this_layer and layer_idx not in dumped_regions:
            dumped_regions[layer_idx] = {
                name: np.zeros((num_heads, decode_steps, length), dtype=np.float16)
                for name, length in region_lengths.items()
            }
            dumped_maxlogit[layer_idx] = np.full(
                (num_heads, decode_steps), -np.inf, dtype=np.float32
            )

        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            q_chunk = query[:, :, start:end, :]
            scores = torch.matmul(q_chunk, key_states.transpose(2, 3)) * scaling

            if attention_mask is not None:
                scores = scores + attention_mask[:, :, start:end, :n_key]

            dump_decode_row = (
                dump_this_layer
                and n_query == 1
                and start == 0
                and 0 <= current_decode_step < decode_steps
            )
            if dump_decode_row:
                dumped_maxlogit[layer_idx][:, current_decode_step] = (
                    scores[0, :, 0, :].float().max(dim=-1).values.detach().cpu().numpy()
                )

            attn_chunk = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

            if dump_decode_row:
                for region_name, (region_start, region_end) in regions.items():
                    dumped_regions[layer_idx][region_name][:, current_decode_step, :] = (
                        attn_chunk[0, :, 0, region_start:region_end]
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )

            attn_chunk = F.dropout(attn_chunk, p=dropout, training=module.training)
            attn_output[:, :, start:end, :] = torch.matmul(attn_chunk, value_states)
            del scores, attn_chunk

        return attn_output.transpose(1, 2).contiguous(), None

    print(
        f"  A region attention dump: layers={target_layers} "
        f"decode_steps={decode_steps} prompt_keys={n_full} chunk={chunk_size}"
    )

    cache = DynamicCache()
    logits, cache = runner.model_forward(full_ids, cache)
    generated_tokens: list[int] = []

    qwen3_modeling.eager_attention_forward = patched_eager_attention_forward
    try:
        for step in range(decode_steps):
            next_token = int(logits.argmax().item())
            generated_tokens.append(next_token)
            current_decode_step = step
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=runner.device)
            logits, cache = runner.model_forward(token_tensor, cache)
            if next_token == runner.tokenizer.eos_token_id:
                break
    finally:
        qwen3_modeling.eager_attention_forward = original_eager_attention_forward

    actual_decode_steps = len(generated_tokens)
    missing_layers = target_layer_set - set(dumped_regions)
    if missing_layers:
        raise RuntimeError(f"{spec.name}: did not dump attention for layers {missing_layers}")

    attention_dir = ensure_dir(out_dir / "a_region_attention")
    fig_dir = ensure_dir(attention_dir / "figures")
    save_kwargs: dict[str, Any] = {
        "decode_tokens": np.array(generated_tokens, dtype=np.int64),
    }
    for name, (start, end) in regions.items():
        save_kwargs[f"{name}_range"] = np.array([start, end], dtype=np.int64)

    summary: dict[str, Any] = {
        "case_name": spec.name,
        "n_tokens": n_full,
        "decode_tokens": generated_tokens,
        "decode_text": runner.tokenizer.decode(generated_tokens),
        "row_group": "A greedy decoded tokens; row i is generated token i used as the query",
        "regions": {name: list(span) for name, span in regions.items()},
        "region_lengths": region_lengths,
        "target_layers": target_layers,
        "requested_decode_steps": decode_steps,
        "actual_decode_steps": actual_decode_steps,
        "chunk_size": chunk_size,
        "layers": {},
    }

    layer_region_means: dict[int, dict[str, np.ndarray]] = {}
    for layer in target_layers:
        layer_region_means[layer] = {}
        summary["layers"][str(layer)] = {}
        for region_name in REGION_ORDER:
            per_head = dumped_regions[layer][region_name][:, :actual_decode_steps, :]
            mean_head = per_head.astype(np.float32).mean(axis=0)
            mass_per_row = mean_head.sum(axis=-1)
            layer_region_means[layer][region_name] = mean_head

            save_kwargs[f"attn_layer{layer}_{region_name}_mean_head"] = mean_head
            save_kwargs[f"attn_layer{layer}_{region_name}_per_head_fp16"] = per_head
            summary["layers"][str(layer)][region_name] = {
                "mass_per_row_mean": float(mass_per_row.mean()),
                "mass_per_row": mass_per_row.tolist(),
            }
        save_kwargs[f"maxlogit_layer{layer}"] = dumped_maxlogit[layer][
            :, :actual_decode_steps
        ]

    npz_path = attention_dir / f"{spec.name}_A_region_attention.npz"
    np.savez_compressed(npz_path, **save_kwargs)
    json_path = attention_dir / f"{spec.name}_A_region_attention_summary.json"

    heatmap_path = fig_dir / f"{spec.name}_A_decode_to_regions_heatmap.png"
    mass_path = fig_dir / f"{spec.name}_A_decode_region_mass.png"
    write_region_heatmap(layer_region_means, target_layers, summary, heatmap_path)
    write_region_mass_plot(target_layers, summary, mass_path)

    summary["npz_path"] = str(npz_path)
    summary["summary_path"] = str(json_path)
    summary["heatmap_path"] = str(heatmap_path)
    summary["mass_path"] = str(mass_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  A region attention heatmap: {heatmap_path}")
    print(f"  A region mass plot: {mass_path}")
    return summary


def write_region_heatmap(
    layer_region_means: dict[int, dict[str, np.ndarray]],
    target_layers: list[int],
    summary: dict[str, Any],
    fig_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_values = np.concatenate(
        [
            mat.reshape(-1)
            for layer in target_layers
            for mat in layer_region_means[layer].values()
        ]
    )
    log_values = np.log10(all_values[np.isfinite(all_values)] + 1e-12)
    vmin = -6.0
    vmax = max(-1.0, float(np.percentile(log_values, 99.5)))

    fig, axes = plt.subplots(
        len(target_layers),
        len(REGION_ORDER),
        figsize=(18, max(3.2, 3.0 * len(target_layers))),
        sharey=True,
        squeeze=False,
    )
    last_im = None
    for row_idx, layer in enumerate(target_layers):
        for col_idx, region_name in enumerate(REGION_ORDER):
            mat = layer_region_means[layer][region_name]
            ax = axes[row_idx][col_idx]
            last_im = ax.imshow(
                np.log10(mat + 1e-12),
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            mass = summary["layers"][str(layer)][region_name]["mass_per_row_mean"]
            ax.set_title(f"L{layer} {region_name} mass/row={mass:.4f}")
            if col_idx == 0:
                ax.set_ylabel("decode token")
            length = summary["region_lengths"][region_name]
            xticks = sorted(set([0, length // 2, length - 1]))
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])
            if row_idx == len(target_layers) - 1:
                ax.set_xlabel(f"token index in {region_name}")

    fig.suptitle(
        "A normal greedy decode: decoded tokens attending to prompt regions\n"
        f"{summary['case_name']}  (head mean, log10 attention probability)"
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    cbar_ax = fig.add_axes([0.94, 0.14, 0.012, 0.72])
    fig.colorbar(last_im, cax=cbar_ax, label="log10(attention probability)")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_region_mass_plot(
    target_layers: list[int],
    summary: dict[str, Any],
    fig_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(target_layers),
        1,
        figsize=(10, max(3.0, 2.4 * len(target_layers))),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(summary["actual_decode_steps"])
    for row_idx, layer in enumerate(target_layers):
        ax = axes[row_idx][0]
        for region_name in REGION_ORDER:
            y = summary["layers"][str(layer)][region_name]["mass_per_row"]
            ax.plot(x, y, marker="o", markersize=2, linewidth=1.2, label=region_name)
        ax.set_ylabel(f"L{layer} mass")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1][0].set_xlabel("decode token")
    fig.suptitle(f"A decoded-token attention mass by prompt region\n{summary['case_name']}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--free-max-tokens", type=int, default=256)
    parser.add_argument(
        "--middle-gaps",
        default="50,100,200,400,800,1600,2000",
        help="Comma-separated target middle-history token counts.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--attention-gaps",
        default="all",
        help=(
            "Which requested middle gaps to dump A attention for: first, all, none, "
            "or a comma-separated list such as 50,400."
        ),
    )
    parser.add_argument(
        "--attention-layers",
        default="5,15,25",
        help="Comma-separated layer ids for A region attention heatmaps.",
    )
    parser.add_argument(
        "--attention-decode-steps",
        type=int,
        default=64,
        help="Number of A greedy decoded tokens to include as heatmap rows.",
    )
    parser.add_argument(
        "--attention-chunk",
        type=int,
        default=512,
        help="Query-row chunk size used by the patched eager attention dump.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    target_gaps = parse_gap_list(args.middle_gaps)
    attention_layers = parse_layer_list(args.attention_layers)
    if args.attention_decode_steps <= 0:
        raise ValueError("--attention-decode-steps must be positive.")
    if args.attention_chunk <= 0:
        raise ValueError("--attention-chunk must be positive.")

    runner = KVReuseRunner(
        args.model_path,
        t_steps=args.t_steps,
        free_max_tokens=args.free_max_tokens,
        dtype=torch.bfloat16,
        device=args.device,
    )

    requested_to_actual = []
    attention_summaries = []
    for target_middle in target_gaps:
        spec, actual_middle = make_prompt_spec(runner, target_middle)
        requested_to_actual.append(
            {
                "requested_middle_gap_tokens": target_middle,
                "actual_middle_gap_tokens": actual_middle,
            }
        )
        if not should_dump_attention_for_gap(args.attention_gaps, target_middle, target_gaps):
            continue

        attention_summary = dump_a_region_attention(
            runner,
            spec,
            out_dir=out_dir,
            target_layers=attention_layers,
            decode_steps=args.attention_decode_steps,
            chunk_size=args.attention_chunk,
        )
        attention_summary["requested_middle_gap_tokens"] = target_middle
        attention_summary["actual_middle_gap_tokens"] = actual_middle
        attention_summaries.append(attention_summary)

    final = {
        "scenario": SCENARIO,
        "model": args.model_path,
        "gap_definition": (
            "Requested values are middle-history token counts between repeated "
            "reference blocks. This script runs only normal A greedy decoding and "
            "records decoded-token attention to user_context, reference_v1, middle, "
            "and reference_v2."
        ),
        "attention_gaps": args.attention_gaps,
        "attention_layers": attention_layers,
        "attention_decode_steps": args.attention_decode_steps,
        "attention_chunk": args.attention_chunk,
        "requested_to_actual": requested_to_actual,
        "summaries": attention_summaries,
    }
    summary_path = out_dir / f"{SCENARIO}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nWrote summary: {summary_path}")
    if attention_summaries:
        print(f"Wrote A region attention outputs: {out_dir / 'a_region_attention'}")


if __name__ == "__main__":
    main()
