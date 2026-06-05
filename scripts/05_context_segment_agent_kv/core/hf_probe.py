"""HF-transformers probe utilities for the real 14B setting (no vLLM).

Shared building blocks for the residual-checkpoint / drift-analysis experiments
that run the real model directly through transformers (not vLLM):

- rebuild the EXACT token sequence the CKSim harness used (same chat template,
  tools, enable_thinking, empty system prefix) so the skill spans recorded in
  ``core.config.SKILL_TOKEN_LOCATIONS`` line up;
- expose per-layer skill-span activations and the per-layer VALUE projection so
  the experiments can measure KV reuse drift and resume-from-checkpoint recovery.

These were previously defined inside replay/residual_checkpoint_probe_14b.py and
imported across sibling scripts; they live here so the replay scripts depend on a
real module instead of importing each other.

Used by:
  replay/residual_checkpoint_probe_14b.py
  replay/end_to_end_quality_14b.py
  replay/token_drift_concentration.py

NOTE: importing this module pulls in torch/transformers, so it is intentionally
kept out of ``core/__init__.py`` (the HTTP/vLLM replay path must stay light).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import TRACES_DIR
from .message_convert import convert_messages, convert_tools

DEFAULT_MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"


def last_invocation_path(task: str) -> Path:
    files = sorted(
        (TRACES_DIR / task).glob("turn_*_inv_*.json"),
        key=lambda p: (int(p.stem.split("turn_")[1].split("_")[0]),
                       int(p.stem.split("inv_")[1])),
    )
    if not files:
        raise FileNotFoundError(f"no invocation files for {task}")
    return files[-1]


def build_full_ids(tok, task: str):
    """Rebuild the full prompt token ids [1, T] for a task's last invocation."""
    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8")))
    inv = json.loads(last_invocation_path(task).read_text(encoding="utf-8"))
    msgs, _ = convert_messages(inv["messages"], system_prompt)
    ids = tok.apply_chat_template(
        msgs, tools=tools, add_generation_prompt=False,
        tokenize=True, return_tensors="pt",
        chat_template_kwargs={"enable_thinking": True},
    )
    return ids  # [1, T]


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


@torch.no_grad()
def value_of(model, layer_idx: int, h_rows: torch.Tensor) -> torch.Tensor:
    layer = model.model.layers[layer_idx]
    return layer.self_attn.v_proj(layer.input_layernorm(h_rows))


@torch.no_grad()
def skill_hidden(model, ids: torch.Tensor, start: int, end: int) -> list[torch.Tensor]:
    """Return per-layer skill-span activation (input to each layer), on CPU."""
    out = model(ids, output_hidden_states=True, use_cache=False)
    hs = [h[0, start:end, :].to("cpu") for h in out.hidden_states]  # len L+1
    del out
    torch.cuda.empty_cache()
    return hs
