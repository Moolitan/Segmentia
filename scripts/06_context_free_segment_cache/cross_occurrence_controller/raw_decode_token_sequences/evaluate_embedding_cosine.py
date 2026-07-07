"""Compute BGE embedding cosine for paired raw decode sequences.

The metric intentionally reproduces the older analysis convention:
BGE-large-en-v1.5, CLS pooling, L2 normalization, and 512-token truncation.
Thinking and post-thinking action text are evaluated separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_INPUT_DIR = Path(
    "results/problem_exploration/raw_decode_token_sequences/"
    "temp0.6/without_occ12"
)
DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "BAAI/bge-large-en-v1.5/BAAI/bge-large-en-v1.5"
)
DEFAULT_CSV = Path(
    "results/problem_exploration/raw_decode_token_sequences/tables/"
    "temp0.6_without_occ12_embedding_cosine.csv"
)
DEFAULT_SUMMARY = Path(
    "results/problem_exploration/raw_decode_token_sequences/data/"
    "temp0.6_without_occ12_embedding_cosine_summary.json"
)
ARMS = ("recompute_run1", "recompute_run2", "rope")
PAIRINGS = (
    ("recompute_run1", "recompute_run2", "sampling_baseline"),
    ("recompute_run1", "rope", "same_seed_recompute_vs_rope"),
    ("recompute_run2", "rope", "cross_seed_recompute_vs_rope"),
)


def extract_sections(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing <think>...</think> section: {path}")
    thinking = match.group(1).strip()
    action = text[match.end() :].strip()
    action = re.sub(r"<\|im_end\|>.*$", "", action, flags=re.DOTALL).strip()
    if not thinking:
        raise ValueError(f"Empty thinking section: {path}")
    if not action:
        raise ValueError(f"Empty action section: {path}")
    return thinking, action


def load_inputs(input_dir: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    files_by_arm = {
        arm: {path.name: path for path in (input_dir / arm).glob("*.txt")}
        for arm in ARMS
    }
    filename_sets = [set(files) for files in files_by_arm.values()]
    if not filename_sets or any(files != filename_sets[0] for files in filename_sets[1:]):
        counts = {arm: len(files) for arm, files in files_by_arm.items()}
        raise RuntimeError(f"Arm filename sets differ: {counts}")
    filenames = sorted(filename_sets[0])
    if not filenames:
        raise RuntimeError(f"No paired TXT files found under {input_dir}")

    sections: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        thinking: dict[str, str] = {}
        action: dict[str, str] = {}
        for filename in filenames:
            thinking[filename], action[filename] = extract_sections(
                files_by_arm[arm][filename]
            )
        sections[arm] = {"thinking": thinking, "action": action}
    return filenames, sections


def encode(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state[:, 0, :]
        normalized = torch.nn.functional.normalize(hidden, p=2, dim=1)
        batches.append(normalized.cpu().numpy())
    return np.concatenate(batches, axis=0)


def statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    for output in (args.output_csv, args.output_summary):
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Embedding model not found: {args.model_path}")

    filenames, sections = load_inputs(args.input_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    print(f"Loading {args.model_path} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    model.eval().to(device)

    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        embeddings[arm] = {}
        for section in ("thinking", "action"):
            print(f"Encoding arm={arm} section={section}", flush=True)
            texts = [sections[arm][section][filename] for filename in filenames]
            embeddings[arm][section] = encode(
                texts,
                tokenizer,
                model,
                device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

    rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}
    for left, right, comparison in PAIRINGS:
        think_values: list[float] = []
        action_values: list[float] = []
        for index, filename in enumerate(filenames):
            think_cosine = float(
                np.clip(np.dot(
                    embeddings[left]["thinking"][index],
                    embeddings[right]["thinking"][index],
                ), -1.0, 1.0)
            )
            action_cosine = float(
                np.clip(np.dot(
                    embeddings[left]["action"][index],
                    embeddings[right]["action"][index],
                ), -1.0, 1.0)
            )
            think_values.append(think_cosine)
            action_values.append(action_cosine)
            rows.append(
                {
                    "comparison": comparison,
                    "left_arm": left,
                    "right_arm": right,
                    "filename": filename,
                    "thinking_cosine": think_cosine,
                    "action_cosine": action_cosine,
                }
            )
        aggregate[comparison] = {
            "left_arm": left,
            "right_arm": right,
            "case_count": len(filenames),
            "thinking_cosine": statistics(think_values),
            "action_cosine": statistics(action_values),
        }

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        args.output_summary,
        {
            "input_dir": str(args.input_dir),
            "model_path": str(args.model_path),
            "pooling": "CLS",
            "normalization": "L2",
            "max_length": args.max_length,
            "truncation": True,
            "device": str(device),
            "case_count": len(filenames),
            "pairings": aggregate,
            "per_case_csv": str(args.output_csv),
        },
    )

    print(f"Wrote {args.output_csv}", flush=True)
    print(f"Wrote {args.output_summary}", flush=True)
    for comparison, values in aggregate.items():
        print(
            f"{comparison}: "
            f"thinking_mean={values['thinking_cosine']['mean']:.4f} "
            f"action_mean={values['action_cosine']['mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
