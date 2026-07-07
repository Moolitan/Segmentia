"""Compare dumped skill KV: with task context vs. without context.

For every skill that has both a *contextual* occurrence-1 KV dump and a
*context-free* (clean) KV dump, this script measures how far apart the two
tensors are, layer by layer.

Two dumps enter the comparison for the same skill:

- contextual: the skill span KV as it was prefilled *inside a real task*
  (``contextual-occ1-skill-<task>-<skill>.pt``). The skill tokens attended to
  all preceding task context, and the span sits at a large absolute position.
- clean: the skill span KV prefilled *alone* on a minimal scaffold
  (``context-free-skill-<skill>.pt``). The span sits at a small absolute
  position.

Both dumps cover the exact same skill token ids (verified), so the comparison
is token-aligned. We report, per transformer layer:

- ``v_cos``      cosine similarity of the value vectors. Values carry no RoPE,
                 so this is a clean, position-independent measure of how much
                 the surrounding context changed the cached representation.
- ``k_raw_cos``  cosine similarity of the key vectors *as stored*. Keys are
                 stored already RoPE-rotated to their own absolute position, so
                 this number is dominated by the position gap and is kept only
                 as a reference for how large the raw on-cache difference is.
- ``k_cos``      cosine similarity of the key vectors after rotating the clean
                 keys forward to the contextual span's absolute position. This
                 removes the RoPE position gap and isolates the content
                 difference in the keys.

``rel_l2`` columns give the mean relative L2 distance
``||contextual - clean|| / ||clean||`` for V and for the de-rotated K.

All metrics are averaged over tokens and KV heads within a layer.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
MODULE_DIR = PACKAGE_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import (  # noqa: E402
    DEFAULT_CONTEXTUAL_KV_DIR,
    DEFAULT_KV_DIR,
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    SKILL_TOKEN_LOCATIONS,
    cache_id_for_skill,
)

# Qwen3-14B rotary configuration (config.json: rope_theta, full-dim neox rope).
ROPE_THETA = 1_000_000.0
ROTARY_DIM = 128


def contextual_cache_id(task: str, skill: str) -> str:
    return f"contextual-occ1-skill-{task}-{skill}"


def rope_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    return theta ** (
        -torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim
    )


def rotate_keys_forward(keys: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply a neox-style RoPE rotation of ``delta`` positions to stored keys.

    ``keys`` is ``(seq, kv_heads, head_dim)`` already rotated to some source
    position; adding ``delta`` moves it to ``source + delta`` so it can be
    compared against a span stored at that later position.
    """
    if delta == 0:
        return keys.float()
    head_dim = keys.shape[-1]
    angle = (delta * inv_freq).to(torch.float32)  # (head_dim/2,)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    keys = keys.float()
    first = keys[..., : head_dim // 2]
    second = keys[..., head_dim // 2 :]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    return torch.cat([rotated_first, rotated_second], dim=-1)


def layer_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Mean cosine similarity and mean relative L2 over tokens and heads."""
    a = a.float()
    b = b.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-6)
    return float(cos.mean()), float(rel.mean())


def build_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for task, record in SKILL_TOKEN_LOCATIONS.items():
        for skill in record["skills"]:
            ctx = DEFAULT_CONTEXTUAL_KV_DIR / f"{contextual_cache_id(task, skill)}.pt"
            clean = DEFAULT_KV_DIR / f"{cache_id_for_skill(skill)}.pt"
            if ctx.exists() and clean.exists():
                cases.append(
                    {
                        "task": task,
                        "skill": skill,
                        "contextual_path": str(ctx),
                        "clean_path": str(clean),
                    }
                )
    return cases


def analyze_case(case: dict[str, str], inv_freq: torch.Tensor) -> dict:
    contextual = torch.load(case["contextual_path"], map_location="cpu")
    clean = torch.load(case["clean_path"], map_location="cpu")
    if contextual["token_ids"] != clean["token_ids"]:
        raise RuntimeError(
            f"token id mismatch for {case['task']}/{case['skill']}: "
            "contextual and clean dumps do not cover identical tokens"
        )
    delta = int(contextual["source_start"]) - int(clean["source_start"])
    if delta < 0:
        raise RuntimeError(
            f"clean span starts after contextual span for {case['skill']}; "
            "forward rotation assumption violated"
        )
    layers = list(contextual["kv_by_layer"].keys())
    per_layer: list[dict] = []
    for index, layer in enumerate(layers):
        ctx_k, ctx_v = contextual["kv_by_layer"][layer]
        cln_k, cln_v = clean["kv_by_layer"][layer]
        v_cos, v_rel = layer_stats(ctx_v, cln_v)
        k_raw_cos, _ = layer_stats(ctx_k, cln_k)
        cln_k_rot = rotate_keys_forward(cln_k, delta, inv_freq)
        k_cos, k_rel = layer_stats(ctx_k, cln_k_rot)
        per_layer.append(
            {
                "task": case["task"],
                "skill": case["skill"],
                "layer": index,
                "v_cos": v_cos,
                "v_rel_l2": v_rel,
                "k_raw_cos": k_raw_cos,
                "k_cos": k_cos,
                "k_rel_l2": k_rel,
            }
        )
    token_count = int(contextual["source_end"]) - int(contextual["source_start"])
    v_cos_series = [row["v_cos"] for row in per_layer]
    k_cos_series = [row["k_cos"] for row in per_layer]
    worst_index = min(range(len(v_cos_series)), key=lambda i: v_cos_series[i])
    summary = {
        "task": case["task"],
        "skill": case["skill"],
        "token_count": token_count,
        "delta_pos": delta,
        "n_layers": len(layers),
        "v_cos_mean": sum(v_cos_series) / len(v_cos_series),
        "v_cos_min": v_cos_series[worst_index],
        "v_cos_min_layer": worst_index,
        "k_raw_cos_mean": sum(r["k_raw_cos"] for r in per_layer) / len(per_layer),
        "k_cos_mean": sum(k_cos_series) / len(k_cos_series),
        "v_rel_l2_mean": sum(r["v_rel_l2"] for r in per_layer) / len(per_layer),
        "k_rel_l2_mean": sum(r["k_rel_l2"] for r in per_layer) / len(per_layer),
    }
    return {"per_layer": per_layer, "summary": summary}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def round_rows(rows: list[dict], digits: int = 4) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                key: (round(value, digits) if isinstance(value, float) else value)
                for key, value in row.items()
            }
        )
    return out


def plot_by_depth(per_layer_rows: list[dict], summaries: list[dict], path: Path) -> None:
    cases = [(s["task"], s["skill"]) for s in summaries]
    cmap = plt.get_cmap("turbo")
    colors = {case: cmap(i / max(1, len(cases) - 1)) for i, case in enumerate(cases)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for metric, ax, title in (
        ("v_cos", axes[0], "Value cosine (context vs clean)"),
        ("k_cos", axes[1], "Key cosine, position-aligned (context vs clean)"),
    ):
        for task, skill in cases:
            series = [
                row[metric]
                for row in per_layer_rows
                if row["task"] == task and row["skill"] == skill
            ]
            ax.plot(
                range(len(series)),
                series,
                label=f"{skill} @ {task}",
                color=colors[(task, skill)],
                linewidth=1.4,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("transformer layer (depth)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("cosine similarity")
    axes[1].legend(fontsize=7, loc="lower left", ncol=1)
    fig.suptitle(
        "Contextual vs context-free skill KV divergence by depth", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary_bars(summaries: list[dict], path: Path) -> None:
    labels = [f"{s['skill']}\n@{s['task']}" for s in summaries]
    order = sorted(range(len(summaries)), key=lambda i: summaries[i]["v_cos_mean"])
    labels = [labels[i] for i in order]
    v_means = [summaries[i]["v_cos_mean"] for i in order]
    k_means = [summaries[i]["k_cos_mean"] for i in order]
    x = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([i - width / 2 for i in x], v_means, width, label="V cosine (mean over layers)")
    ax.bar([i + width / 2 for i in x], k_means, width, label="K cosine, aligned (mean)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean cosine similarity")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Per-skill contextual vs clean KV similarity (higher = more context-independent)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "contextual_vs_clean_kv_divergence",
    )
    args = parser.parse_args()

    cases = build_cases()
    if not cases:
        raise RuntimeError("No comparable (contextual, clean) skill KV pairs found")
    inv_freq = rope_inv_freq(ROTARY_DIM, ROPE_THETA)

    per_layer_rows: list[dict] = []
    summaries: list[dict] = []
    for case in cases:
        result = analyze_case(case, inv_freq)
        per_layer_rows.extend(result["per_layer"])
        summaries.append(result["summary"])
        s = result["summary"]
        print(
            f"[case] {s['skill']:22s} @ {s['task']:32s} "
            f"tokens={s['token_count']:5d} V_cos={s['v_cos_mean']:.3f} "
            f"K_cos={s['k_cos_mean']:.3f} K_raw={s['k_raw_cos_mean']:.3f}",
            flush=True,
        )

    out = args.output_dir
    write_csv(
        out / "data" / "per_layer_kv_divergence.csv",
        round_rows(per_layer_rows),
        ["task", "skill", "layer", "v_cos", "v_rel_l2", "k_raw_cos", "k_cos", "k_rel_l2"],
    )
    write_csv(
        out / "tables" / "kv_divergence_summary.csv",
        round_rows(summaries),
        [
            "task",
            "skill",
            "token_count",
            "delta_pos",
            "n_layers",
            "v_cos_mean",
            "v_cos_min",
            "v_cos_min_layer",
            "k_cos_mean",
            "k_raw_cos_mean",
            "v_rel_l2_mean",
            "k_rel_l2_mean",
        ],
    )
    plot_by_depth(per_layer_rows, summaries, out / "figures" / "kv_divergence_by_depth.png")
    plot_summary_bars(summaries, out / "figures" / "kv_divergence_summary_bars.png")
    print(f"[done] wrote CSVs and figures under {out}", flush=True)


if __name__ == "__main__":
    main()
