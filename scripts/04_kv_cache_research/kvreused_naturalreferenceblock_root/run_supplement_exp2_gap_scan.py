"""
Supplement experiment 2: gap-size scan.

The report defines the original "gap" as the start-position difference between
reference_v1 and reference_v2. Because that value is at least the reference
block length, this script scans the controllable middle-history length and also
records the resulting absolute reference start-position gap.

Run:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp2_gap_scan.py

A/B1/B2 focused attention quick run:
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp2_gap_scan.py \
    --attention-only --middle-gaps 50,100,200,400,800,1600,2000 \
    --attention-gaps all --attention-layers 5,10,15,25,30 --attention-decode-steps 64

In attention-only mode, each heatmap row is one greedy-decoded token looking
back at reference_v1/reference_v2. Separate heatmaps are written for A, B1,
and B2.
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
BASE_DIR = SCRIPT_DIR.parent
COMMON_DIR = BASE_DIR / "kvreused_naturalreferenceblock"
if (COMMON_DIR / "kv_reuse_common.py").exists():
    sys.path.append(str(COMMON_DIR))

from kv_reuse_common import (
    KVReuseRunner,
    PromptSpec,
    clone_cache,
    concat_cache,
    ensure_dir,
    slice_cache,
)

DEFAULT_MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
DEFAULT_OUT_DIR = (
    BASE_DIR
    / "results"
    / "kv_reuse_natural_reference_block"
    / "supplement_exp2_gap_scan"
)
SCENARIO = "KVReuse_NaturalReferenceBlock_GapScan"


REFERENCE_TEXT = """\
Reference Card: Platform Update Format

1. Start with a one-line title in the form "Platform Update - <topic>".
2. Use exactly three sections: Progress, Plans, Problems.
3. Under each section, write 2 short bullet points.
4. Keep the tone operational, calm, and specific.
5. Prefer concrete facts over vague claims.
6. Mention rollout status or the next checkpoint when relevant.
7. Keep the total length between 120 and 180 words.
8. End with one sentence stating the immediate next checkpoint.
"""

USER_CONTEXT = """\
[system]
You are a concise workplace writing assistant. When the user provides a reference block, follow it
closely. Do not mention the reference block unless the user asks.

[user]
I need help drafting internal updates for the platform engineering team. First, please read the
reference card below and keep it in mind for later requests.

[assistant]
"""

QUERY_TEXT = """
[user]
Now use that same reference card to write a new update about failed-task retry support in the
experiment dashboard.

Facts:
- Progress: the retry endpoint is deployed; the UI retry button is live for 30% of users; failure logs now label retryable and non-retryable cases separately.
- Plans: expand to 100% next week; add retry success rate to the weekly ops review; publish a short usage note for researchers.
- Problems: older tasks still show generic failure states; retry-related alerts are noisier than expected.

Keep it concise and operational.

[assistant]

"""

MIDDLE_BASE = """
[assistant]
Understood.
"""

FILLER_UNITS = [
    """
[user]
Before the next update, continue the launch-pack cleanup in the workspace. Start by reading the brand notes.

[assistant]
Agent Action
Summary: Read brand notes before editing launch materials

Action: ReadFileAction
Arguments:
  file_path: "seed_files/brand_notes.md"
""",
    """
[assistant]
Observation
Tool: read_file
Result:
  1  # Brand Notes
  2  Style requirements:
  3  - simple
  4  - clear
  5  - trustworthy
  6  - technical-team tone
  7  - not flashy
""",
    """
[user]
Now inspect the current announcement before changing the copy.

[assistant]
Agent Action
Summary: Read the current announcement

Action: ReadFileAction
Arguments:
  file_path: "internal_announcement.md"
""",
    """
[assistant]
Observation
Tool: read_file
Result:
  1  ## Feature Launch: Experiment Task Dashboard Retry Support
  2
  3  The experiment dashboard now includes retry support for failed tasks.
  4
  5  ### Key Benefits
  6  - Retry failed tasks with one click
  7  - Clearer failure reasons
  8  - Less manual restart work
""",
    """
[user]
Make the announcement and GIF feel like one launch pack. Keep the tone technical and remove anything too flashy.

[assistant]
Agent Action
Summary: Replace the failure color with the brand accent

Action: EditAction
Arguments:
  file_path: "retry_animation.py"
  old_string: "fill='red'"
  new_string: "fill='#d97757'"
  expected_replacements: 2
""",
    """
[assistant]
Observation
Tool: edit
Result:
  Edited retry_animation.py with 2 replacements.
""",
    """
[assistant]
Agent Action
Summary: Tighten the announcement sentence

Action: EditAction
Arguments:
  file_path: "internal_announcement.md"
  old_string: "We're pleased to announce a new feature in the experiment task dashboard."
  new_string: "The experiment task dashboard now includes retry support for failed tasks."
  expected_replacements: 1
""",
    """
[assistant]
Observation
Tool: edit
Result:
  Error: Could not find the string to replace. 0 occurrences found. Use read_file to verify the exact text.
""",
    """
[user]
The edit missed the exact text. Re-read the file, then patch only the sentence that needs tightening.

[assistant]
Agent Action
Summary: Re-read the file after the failed edit

Action: ReadFileAction
Arguments:
  file_path: "internal_announcement.md"
""",
    """
[assistant]
Observation
Tool: read_file
Result:
  1  ## Feature Launch: Experiment Task Dashboard Retry Support
  2
  3  We're pleased to announce retry support for failed tasks in the experiment task dashboard.
  4
  5  ### Key Benefits
  6  - Auto-retry failed tasks with one click
  7  - Improved visibility into task failure reasons
  8  - Reduced manual intervention requirements
""",
    """
[assistant]
Agent Action
Summary: Apply the edit against the exact sentence

Action: EditAction
Arguments:
  file_path: "internal_announcement.md"
  old_string: "We're pleased to announce retry support for failed tasks in the experiment task dashboard."
  new_string: "The experiment task dashboard now includes retry support for failed tasks."
""",
    """
[assistant]
Observation
Tool: edit
Result:
  Edited internal_announcement.md with 1 replacement.
""",
    """
[user]
Run the local validation script and summarize what changed.

[assistant]
Agent Action
Summary: Validate the launch pack after the text and color edits

$ python scripts/validate_launch_pack.py
""",
    """
[assistant]
Observation
Tool: terminal
Result:
  Working directory: workspace/03_14B_anthropic_3/slack_launch_pack
  internal_announcement.md: ok
  retry_gif.gif: ok
  brand colors: ok
  Exit code: 0
""",
    """
[assistant]
Validation passed. The announcement now uses direct technical wording, and the GIF uses the brand light background with orange and blue status accents.

[user]
I will paste the same reference card again below before the next task.

[assistant]
Ready.
""",
]


def parse_gap_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Gap values must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one gap value is required.")
    return values


def parse_layer_list(raw: str) -> list[int]:
    layers: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0:
            raise ValueError(f"Layer ids must be non-negative: {layer}")
        layers.append(layer)
    if not layers:
        raise ValueError("At least one attention layer is required.")
    return sorted(set(layers))


def should_dump_attention_for_gap(
    selector: str,
    target_middle: int,
    target_gaps: list[int],
) -> bool:
    raw = selector.strip().lower()
    if raw in {"none", "false", "0", "off"}:
        return False
    if raw == "first":
        return target_middle == target_gaps[0]
    if raw == "all":
        return True
    return target_middle in set(parse_gap_list(selector))


def make_middle_for_target(runner: KVReuseRunner, target_middle_tokens: int) -> str:
    middle = MIDDLE_BASE
    unit_idx = 0
    while runner.token_len(middle) < target_middle_tokens:
        middle += FILLER_UNITS[unit_idx % len(FILLER_UNITS)]
        unit_idx += 1
    return middle


def make_prompt_spec(
    runner: KVReuseRunner,
    target_middle_tokens: int,
) -> tuple[PromptSpec, int]:
    middle = make_middle_for_target(runner, target_middle_tokens)
    actual_middle_tokens = runner.token_len(middle)
    spec = PromptSpec(
        name=f"middle_gap_{target_middle_tokens:04d}_actual_{actual_middle_tokens:04d}",
        reference_text=REFERENCE_TEXT,
        user_context=USER_CONTEXT,
        middle_history=middle,
        query_text=QUERY_TEXT,
        note=(
            "gap scan; requested gap is middle-history token count between repeated "
            "reference blocks; absolute start-position gap is recorded in sequence_info"
        ),
    )
    return spec, actual_middle_tokens


def dump_a_focused_attention_map(
    runner: KVReuseRunner,
    spec: PromptSpec,
    *,
    out_dir: Path,
    target_layers: list[int],
    decode_steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Dump A/B1/B2 greedy decoded-token attention over ref1/ref2 windows."""
    from transformers.models.qwen3 import modeling_qwen3 as qwen3_modeling
    from transformers.cache_utils import DynamicCache

    prompt = spec.full_prompt
    full_ids, ref1_start, ref1_end = runner.locate_reference_tokens(
        prompt, spec.reference_text, "first"
    )
    _, ref2_start, ref2_end = runner.locate_reference_tokens(
        prompt, spec.reference_text, "last"
    )
    n_full = len(full_ids)
    l_ref = ref1_end - ref1_start
    if l_ref != ref2_end - ref2_start:
        raise RuntimeError(f"{spec.name}: reference token lengths differ.")
    if ref2_end >= n_full:
        raise RuntimeError(f"{spec.name}: query/assistant prefix after ref2 is required.")

    phase1_ids = full_ids[:ref2_start]
    ref2_ids = full_ids[ref2_start:ref2_end]
    query_ids = full_ids[ref2_end:]

    cache_phase1 = DynamicCache()
    _, cache_phase1 = runner.model_forward(phase1_ids, cache_phase1)
    ref_v1_cache = slice_cache(cache_phase1, ref1_start, ref1_end)

    def make_decode_start(variant: str) -> tuple[torch.Tensor, DynamicCache]:
        if variant == "A":
            cache = clone_cache(cache_phase1)
            _, cache = runner.model_forward(ref2_ids, cache)
        elif variant == "B1":
            cache = concat_cache(cache_phase1, ref_v1_cache)
        elif variant == "B2":
            ref_v1_rerot = runner.make_b2_reference_cache(
                ref_v1_cache, ref1_start, ref2_start, l_ref
            )
            cache = concat_cache(cache_phase1, ref_v1_rerot)
        else:
            raise ValueError(f"Unknown attention variant: {variant}")

        if cache.get_seq_length() != ref2_end:
            raise RuntimeError(
                f"{spec.name}/{variant}: pre-query cache length mismatch "
                f"{cache.get_seq_length()} != {ref2_end}"
            )
        logits, cache = runner.model_forward(query_ids, cache)
        if cache.get_seq_length() != n_full:
            raise RuntimeError(
                f"{spec.name}/{variant}: prompt cache length mismatch "
                f"{cache.get_seq_length()} != {n_full}"
            )
        return logits, cache

    target_layer_set = set(target_layers)
    repeat_kv = qwen3_modeling.repeat_kv
    original_eager_attention_forward = qwen3_modeling.eager_attention_forward
    current_decode_step = -1
    dumped_ref1: dict[int, np.ndarray] = {}
    dumped_ref2: dict[int, np.ndarray] = {}
    dumped_maxlogit: dict[int, np.ndarray] = {}

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

        if dump_this_layer and layer_idx not in dumped_ref1:
            dumped_ref1[layer_idx] = np.zeros(
                (num_heads, decode_steps, l_ref), dtype=np.float16
            )
            dumped_ref2[layer_idx] = np.zeros(
                (num_heads, decode_steps, l_ref), dtype=np.float16
            )
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
                dumped_ref1[layer_idx][:, current_decode_step, :] = (
                    attn_chunk[0, :, 0, ref1_start:ref1_end]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                dumped_ref2[layer_idx][:, current_decode_step, :] = (
                    attn_chunk[0, :, 0, ref2_start:ref2_end]
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

    attention_dir = ensure_dir(out_dir / "attention")
    fig_dir = ensure_dir(attention_dir / "figures")
    summary: dict[str, Any] = {
        "case_name": spec.name,
        "n_tokens": n_full,
        "row_group": (
            "Each variant uses its own greedy decoded tokens; row i is generated "
            "token i used as the query."
        ),
        "reference_v1": [ref1_start, ref1_end],
        "reference_v2": [ref2_start, ref2_end],
        "L_reference": l_ref,
        "target_layers": target_layers,
        "requested_decode_steps": decode_steps,
        "chunk_size": chunk_size,
        "variants": {},
    }

    for variant in ("A", "B1", "B2"):
        dumped_ref1 = {}
        dumped_ref2 = {}
        dumped_maxlogit = {}
        current_decode_step = -1

        print(
            f"  {variant} decode attention dump: layers={target_layers} "
            f"decode_steps={decode_steps} prompt_keys={n_full} chunk={chunk_size}"
        )
        logits, cache = make_decode_start(variant)
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
        missing_layers = target_layer_set - set(dumped_ref1)
        if missing_layers:
            raise RuntimeError(
                f"{spec.name}/{variant}: did not dump attention for layers {missing_layers}"
            )

        save_kwargs: dict[str, Any] = {
            "decode_tokens": np.array(generated_tokens, dtype=np.int64),
            "reference_v1_range": np.array([ref1_start, ref1_end], dtype=np.int64),
            "reference_v2_range": np.array([ref2_start, ref2_end], dtype=np.int64),
        }

        variant_summary: dict[str, Any] = {
            "decode_tokens": generated_tokens,
            "decode_text": runner.tokenizer.decode(generated_tokens),
            "actual_decode_steps": actual_decode_steps,
            "layers": {},
        }
        ref1_mats: dict[int, np.ndarray] = {}
        ref2_mats: dict[int, np.ndarray] = {}
        all_plot_values = []
        for layer in target_layers:
            ref1_per_head = dumped_ref1[layer][:, :actual_decode_steps, :]
            ref2_per_head = dumped_ref2[layer][:, :actual_decode_steps, :]
            ref1_mean = ref1_per_head.astype(np.float32).mean(axis=0)
            ref2_mean = ref2_per_head.astype(np.float32).mean(axis=0)
            ref1_mats[layer] = ref1_mean
            ref2_mats[layer] = ref2_mean
            all_plot_values.append(ref1_mean)
            all_plot_values.append(ref2_mean)

            ref1_mass_per_row = ref1_mean.sum(axis=-1)
            ref2_mass_per_row = ref2_mean.sum(axis=-1)
            ref1_mass = float(ref1_mass_per_row.mean())
            ref2_mass = float(ref2_mass_per_row.mean())
            variant_summary["layers"][str(layer)] = {
                "reference_v1_mass_per_row_mean": ref1_mass,
                "reference_v2_mass_per_row_mean": ref2_mass,
                "reference_v2_over_v1_mass_ratio": float(ref2_mass / max(ref1_mass, 1e-12)),
                "reference_v1_mass_per_row": ref1_mass_per_row.tolist(),
                "reference_v2_mass_per_row": ref2_mass_per_row.tolist(),
            }
            save_kwargs[f"attn_layer{layer}_ref1_mean_head"] = ref1_mean
            save_kwargs[f"attn_layer{layer}_ref2_mean_head"] = ref2_mean
            save_kwargs[f"attn_layer{layer}_ref1_per_head_fp16"] = ref1_per_head
            save_kwargs[f"attn_layer{layer}_ref2_per_head_fp16"] = ref2_per_head
            save_kwargs[f"maxlogit_layer{layer}"] = dumped_maxlogit[layer][
                :, :actual_decode_steps
            ]

        npz_path = attention_dir / f"{spec.name}_{variant}_focused_attention.npz"
        np.savez_compressed(npz_path, **save_kwargs)

        finite_values = np.concatenate([m.reshape(-1) for m in all_plot_values])
        finite_values = finite_values[np.isfinite(finite_values)]
        log_values = np.log10(finite_values + 1e-12)
        vmin = -6.0
        vmax = max(-1.0, float(np.percentile(log_values, 99.5)))

        fig, axes = plt_subplots_for_attention(len(target_layers))
        last_im = None
        x_ticks = [0, l_ref // 2, l_ref - 1]
        x_ticklabels = ["0", str(l_ref // 2), str(l_ref - 1)]
        for row_idx, layer in enumerate(target_layers):
            ref1 = np.log10(ref1_mats[layer] + 1e-12)
            ref2 = np.log10(ref2_mats[layer] + 1e-12)
            ax1 = axes[row_idx][0]
            ax2 = axes[row_idx][1]
            last_im = ax1.imshow(
                ref1, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
            )
            ax2.imshow(ref2, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            layer_summary = variant_summary["layers"][str(layer)]
            ax1.set_title(
                f"L{layer} reference_v1 mass/row="
                f"{layer_summary['reference_v1_mass_per_row_mean']:.4f}"
            )
            ax2.set_title(
                f"L{layer} reference_v2 mass/row="
                f"{layer_summary['reference_v2_mass_per_row_mean']:.4f}"
            )
            ax1.set_ylabel("decode token")
            ax1.set_xticks(x_ticks)
            ax2.set_xticks(x_ticks)
            ax1.set_xticklabels(x_ticklabels)
            ax2.set_xticklabels(x_ticklabels)
        axes[-1][0].set_xlabel("token index inside reference_v1")
        axes[-1][1].set_xlabel("token index inside reference_v2")
        fig.suptitle(
            f"{variant} greedy decode: decoded tokens attending to the two reference blocks\n"
            f"{spec.name}  (head mean, log10 attention probability)"
        )
        fig.tight_layout(rect=[0, 0, 0.90, 0.94])
        cbar_ax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
        fig.colorbar(last_im, cax=cbar_ax, label="log10(attention probability)")
        fig_path = fig_dir / f"{spec.name}_{variant}_decode_to_ref1_ref2_focused_heatmap.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        close_attention_figure(fig)

        variant_summary["npz_path"] = str(npz_path)
        variant_summary["figure_path"] = str(fig_path)
        summary["variants"][variant] = variant_summary
        print(f"  {variant} attention figure: {fig_path}")

    json_path = attention_dir / f"{spec.name}_focused_attention_summary.json"
    summary["summary_path"] = str(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def plt_subplots_for_attention(n_layers: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_layers,
        2,
        figsize=(13, max(3.2, 3.0 * n_layers)),
        sharex=True,
        squeeze=False,
    )
    return fig, axes


def close_attention_figure(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def write_plots(results: list[dict[str, Any]], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(fig_dir)
    x_middle = [
        r["sequence_info"]["middle_tokens_between_references"]
        for r in results
    ]
    x_start_gap = [
        r["sequence_info"]["reference_start_position_gap"]
        for r in results
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_items = [
        (axes[0, 0], "KL_first", "KL first", "B1", "B2"),
        (axes[0, 1], "KL_mean", "KL mean", "B1", "B2"),
        (axes[1, 0], "argmax_match_rate", "Argmax match", "B1", "B2"),
    ]
    for ax, field, ylabel, side1, side2 in plot_items:
        ax.plot(x_middle, [r[side1][field] for r in results], marker="o", label=side1)
        ax.plot(x_middle, [r[side2][field] for r in results], marker="s", label=side2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    ax = axes[1, 1]
    if results and results[0].get("text_metrics"):
        ax.plot(
            x_middle,
            [r["text_metrics"]["A_vs_B1"]["bleu4"] for r in results],
            marker="o",
            label="B1",
        )
        ax.plot(
            x_middle,
            [r["text_metrics"]["A_vs_B2"]["bleu4"] for r in results],
            marker="s",
            label="B2",
        )
    ax.set_ylabel("BLEU-4")
    ax.grid(alpha=0.3)
    ax.legend()

    for ax in axes[1, :]:
        ax.set_xlabel("Middle-history tokens between reference blocks")
    fig.suptitle("Supplement Exp2: gap scan")
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_scan_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_middle, x_start_gap, marker="o")
    ax.set_xlabel("Middle-history tokens between reference blocks")
    ax.set_ylabel("Reference start-position gap")
    ax.set_title("Middle length vs absolute reference start-position gap")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_scan_actual_position_gap.png", dpi=160)
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
        "--no-dump-a-attention",
        action="store_true",
        help="Disable the default focused attention dump.",
    )
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="Only dump A/B1/B2 focused attention and skip KV-reuse metrics.",
    )
    parser.add_argument(
        "--attention-gaps",
        default="first",
        help=(
            "Which requested middle gaps to dump focused attention for: first, all, none, "
            "or a comma-separated list such as 50,400."
        ),
    )
    parser.add_argument(
        "--attention-layers",
        default="5,15,25",
        help="Comma-separated layer ids for focused attention heatmaps.",
    )
    parser.add_argument(
        "--attention-decode-steps",
        type=int,
        default=64,
        help="Number of greedy decoded tokens to include as heatmap rows per variant.",
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
    case_dir = ensure_dir(out_dir / "gaps")
    fig_dir = ensure_dir(out_dir / "figures")
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

    results = []
    requested_to_actual = []
    attention_summaries = []
    for target_middle in target_gaps:
        spec, actual_middle = make_prompt_spec(runner, target_middle)

        if (
            not args.no_dump_a_attention
            and should_dump_attention_for_gap(args.attention_gaps, target_middle, target_gaps)
        ):
            attention_summary = dump_a_focused_attention_map(
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

        if args.attention_only:
            requested_to_actual.append(
                {
                    "requested_middle_gap_tokens": target_middle,
                    "actual_middle_gap_tokens": actual_middle,
                }
            )
            continue

        result = runner.run_prompt(spec, scenario=SCENARIO)
        result["requested_middle_gap_tokens"] = target_middle
        result["actual_middle_gap_tokens"] = actual_middle
        results.append(result)
        requested_to_actual.append(
            {
                "requested_middle_gap_tokens": target_middle,
                "actual_middle_gap_tokens": actual_middle,
                "reference_start_position_gap": result["sequence_info"][
                    "reference_start_position_gap"
                ],
            }
        )
        with open(case_dir / f"{spec.name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    final = {
        "scenario": SCENARIO,
        "model": args.model_path,
        "T_steps": args.t_steps,
        "free_max_tokens": args.free_max_tokens,
        "gap_definition": (
            "Requested values are middle-history token counts between repeated "
            "reference blocks. sequence_info.reference_start_position_gap records "
            "the absolute start-position difference used by earlier reports."
        ),
        "attention": {
            "enabled": not args.no_dump_a_attention,
            "attention_only": bool(args.attention_only),
            "attention_gaps": args.attention_gaps,
            "attention_layers": attention_layers,
            "attention_decode_steps": args.attention_decode_steps,
            "attention_chunk": args.attention_chunk,
            "summaries": attention_summaries,
        },
        "requested_to_actual": requested_to_actual,
        "results": results,
    }
    with open(out_dir / f"{SCENARIO}_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    if results:
        write_plots(results, fig_dir)
    print(f"\nWrote summary: {out_dir / f'{SCENARIO}_summary.json'}")
    print(f"Wrote per-gap JSON: {case_dir}")
    if results:
        print(f"Wrote figures: {fig_dir}")
    if attention_summaries:
        print(f"Wrote attention outputs: {out_dir / 'attention'}")


if __name__ == "__main__":
    main()
