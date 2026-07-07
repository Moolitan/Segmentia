"""Validate and plot recompute prefill-attention dumps."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NUM_LAYERS = 40
SKILL_QUERY_TOKENS = 30
PRE_SKILL_WINDOW = 512


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def read_dump_rows(dump_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(dump_dir.glob("attention_local_windows_*.jsonl"))
    if not paths:
        raise ValueError(f"No attention_local_windows JSONL in {dump_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_values(row: dict[str, Any], expected_length: int) -> np.ndarray:
    values = np.asarray(row["attention_mean"], dtype=np.float64)
    if values.shape != (expected_length,):
        raise ValueError(
            f"{row['case_id']} {row['query_name']} layer={row['layer_index']}: "
            f"expected {expected_length} values, found {values.shape}"
        )
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(
            f"{row['case_id']} {row['query_name']} layer={row['layer_index']}: "
            "attention values must be finite probabilities in [0, 1]"
        )
    return values


def case_rows(
    rows: list[dict[str, Any]], case_id: str
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["case_id"]) == case_id]


def build_skill_matrix(rows: list[dict[str, Any]], case_id: str) -> np.ndarray:
    relevant = [
        row
        for row in case_rows(rows, case_id)
        if str(row.get("query_name", "")).startswith("skill_token_")
        and str(row["window"]) == "pre_skill_context"
    ]
    expected = NUM_LAYERS * SKILL_QUERY_TOKENS
    if len(relevant) != expected:
        raise ValueError(
            f"{case_id}: expected {expected} skill-query rows, "
            f"found {len(relevant)}"
        )

    tensor = np.full(
        (NUM_LAYERS, SKILL_QUERY_TOKENS, PRE_SKILL_WINDOW),
        np.nan,
        dtype=np.float64,
    )
    seen: set[tuple[int, int]] = set()
    for row in relevant:
        layer = int(row["layer_index"])
        query_offset = int(str(row["query_name"]).rsplit("_", 1)[1])
        key = (layer, query_offset)
        if key in seen:
            raise ValueError(f"{case_id}: duplicate skill row {key}")
        seen.add(key)
        if not 0 <= layer < NUM_LAYERS:
            raise ValueError(f"{case_id}: invalid layer {layer}")
        if not 0 <= query_offset < SKILL_QUERY_TOKENS:
            raise ValueError(f"{case_id}: invalid query offset {query_offset}")
        if int(row["window_length"]) != PRE_SKILL_WINDOW:
            raise ValueError(f"{case_id}: invalid pre-skill window length")
        tensor[layer, query_offset] = validate_values(
            row, PRE_SKILL_WINDOW
        )
    if np.isnan(tensor).any():
        raise ValueError(f"{case_id}: incomplete skill-query tensor")
    return tensor.mean(axis=0)


def build_assistant_matrix(
    rows: list[dict[str, Any]], case_id: str, skill_length: int
) -> np.ndarray:
    relevant = [
        row
        for row in case_rows(rows, case_id)
        if str(row.get("query_name", "")) == "final_assistant_prompt_token"
        and str(row["window"]) == "context_and_skill"
    ]
    if len(relevant) != NUM_LAYERS:
        raise ValueError(
            f"{case_id}: expected {NUM_LAYERS} final-assistant rows, "
            f"found {len(relevant)}"
        )

    width = PRE_SKILL_WINDOW + skill_length
    matrix = np.full((NUM_LAYERS, width), np.nan, dtype=np.float64)
    seen: set[int] = set()
    for row in relevant:
        layer = int(row["layer_index"])
        if layer in seen:
            raise ValueError(f"{case_id}: duplicate assistant layer {layer}")
        seen.add(layer)
        if not 0 <= layer < NUM_LAYERS:
            raise ValueError(f"{case_id}: invalid layer {layer}")
        if int(row["window_length"]) != width:
            raise ValueError(
                f"{case_id}: expected assistant window {width}, "
                f"found {row['window_length']}"
            )
        matrix[layer] = validate_values(row, width)
    if np.isnan(matrix).any():
        raise ValueError(f"{case_id}: incomplete assistant tensor")
    return matrix


def add_colorbar(fig: plt.Figure, ax: plt.Axes, image: Any) -> None:
    colorbar = fig.colorbar(image, ax=ax, fraction=0.027, pad=0.02)
    colorbar.set_label("Raw attention probability")


def plot_skill_matrix(matrix: np.ndarray, path_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=max(float(matrix.max()), 1e-12),
        rasterized=True,
    )
    x_ticks = [0, 127, 255, 383, 511]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["−512", "−385", "−257", "−129", "−1"])
    ax.set_yticks([0, 4, 9, 14, 19, 24, 29])
    ax.set_yticklabels(["0", "4", "9", "14", "19", "24", "29"])
    ax.set_xlabel("Key token offset from Skill start")
    ax.set_ylabel("Skill query-token offset")
    add_colorbar(fig, ax, image)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            path_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def plot_assistant_matrix(matrix: np.ndarray, path_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=max(float(matrix.max()), 1e-12),
        rasterized=True,
    )
    width = matrix.shape[1]
    skill_length = width - PRE_SKILL_WINDOW
    candidate_offsets = [-512, -256, 0]
    if skill_length > 1:
        candidate_offsets.extend(
            sorted({skill_length // 2, skill_length - 1})
        )
    tick_positions = [offset + PRE_SKILL_WINDOW for offset in candidate_offsets]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(offset) for offset in candidate_offsets])
    ax.axvline(PRE_SKILL_WINDOW - 0.5, color="white", linewidth=0.9)
    ax.set_xlabel("Key token offset from Skill start")
    ax.set_ylabel("Layer")
    ax.set_yticks([0, 7, 15, 23, 31, 39])
    add_colorbar(fig, ax, image)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            path_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "skill_length",
        "skill_matrix_rows",
        "skill_matrix_columns",
        "skill_attention_window_mass_mean",
        "assistant_matrix_rows",
        "assistant_matrix_columns",
        "assistant_context_and_skill_mass_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_source_manifest(
    path: Path,
    figure_paths: list[Path],
    dump_dir: Path,
    capture_manifest: Path,
) -> None:
    rows = [
        {
            "path": "summary.md",
            "kind": "summary",
            "description": "Scope metric definition and result interpretation",
            "source": "manual specification and validated capture outputs",
        },
        {
            "path": "tables/attention_capture_summary.csv",
            "kind": "table",
            "description": (
                "Per-case matrix dimensions and unnormalized window mass"
            ),
            "source": f"{dump_dir} and {capture_manifest}",
        },
    ]
    rows.extend(
        {
            "path": str(figure_path),
            "kind": "figure",
            "description": (
                "Validated recompute prefill attention heatmap"
            ),
            "source": f"{dump_dir} and {capture_manifest}",
        }
        for figure_path in figure_paths
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "kind", "description", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    if not manifest_rows:
        raise ValueError(f"Empty manifest: {args.manifest}")
    manifest_by_case: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        case_id = str(row["case_id"])
        if case_id in manifest_by_case:
            raise ValueError(f"Duplicate manifest case: {case_id}")
        manifest_by_case[case_id] = row

    dump_rows = read_dump_rows(args.dump_dir)
    dump_cases = {str(row["case_id"]) for row in dump_rows}
    if dump_cases != set(manifest_by_case):
        raise ValueError(
            f"Manifest/dump case mismatch: manifest={sorted(manifest_by_case)} "
            f"dumps={sorted(dump_cases)}"
        )

    configure_style()
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for case_id, manifest in sorted(manifest_by_case.items()):
        target_start = int(manifest["target_start"])
        target_end = int(manifest["target_end"])
        skill_length = target_end - target_start
        if skill_length < SKILL_QUERY_TOKENS:
            raise ValueError(f"{case_id}: skill span shorter than 30 tokens")

        skill_matrix = build_skill_matrix(dump_rows, case_id)
        assistant_matrix = build_assistant_matrix(
            dump_rows, case_id, skill_length
        )
        stem = safe_component(case_id)
        plot_skill_matrix(
            skill_matrix, figures_dir / f"skill_first30_to_pre512_{stem}"
        )
        plot_assistant_matrix(
            assistant_matrix,
            figures_dir / f"final_assistant_to_context_skill_{stem}",
        )
        for prefix in (
            "skill_first30_to_pre512",
            "final_assistant_to_context_skill",
        ):
            for extension in ("png", "pdf"):
                figure_paths.append(
                    Path("figures") / f"{prefix}_{stem}.{extension}"
                )
        summary_rows.append(
            {
                "case_id": case_id,
                "skill_length": skill_length,
                "skill_matrix_rows": skill_matrix.shape[0],
                "skill_matrix_columns": skill_matrix.shape[1],
                "skill_attention_window_mass_mean": float(
                    skill_matrix.sum(axis=1).mean()
                ),
                "assistant_matrix_rows": assistant_matrix.shape[0],
                "assistant_matrix_columns": assistant_matrix.shape[1],
                "assistant_context_and_skill_mass_mean": float(
                    assistant_matrix.sum(axis=1).mean()
                ),
            }
        )
    write_summary(
        args.output_dir / "tables/attention_capture_summary.csv",
        summary_rows,
    )
    write_source_manifest(
        args.output_dir / "source_manifest.csv",
        figure_paths,
        args.dump_dir,
        args.manifest,
    )
    print(
        f"Validated {len(dump_rows)} dump rows for "
        f"{len(manifest_by_case)} cases"
    )
    print(f"Wrote figures and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
