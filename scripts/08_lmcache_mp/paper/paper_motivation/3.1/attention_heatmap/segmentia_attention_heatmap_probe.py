"""Minimal Figure-4 attention capture for the Segmentia 3.1 experiment."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SPEC_ENV = "SEGMENTIA_ATTENTION_HEATMAP_SPEC"
OUT_ENV = "SEGMENTIA_ATTENTION_HEATMAP_OUT_DIR"


@dataclass(frozen=True)
class ActiveRows:
    request_id: str
    mode: str
    query_rows: tuple[int, ...]
    query_positions: tuple[int, ...]
    skill_start: int
    skill_end: int
    cross_key_start: int
    cross_key_end: int
    forward_query_start: int
    forward_query_end: int
    forward_key_start: int
    forward_key_end: int
    prompt_end: int


class AttentionHeatmapProbe:
    def __init__(self) -> None:
        spec_path = os.environ.get(SPEC_ENV)
        out_dir = os.environ.get(OUT_ENV)
        self.enabled = bool(spec_path and out_dir)
        self.spec_path = Path(spec_path) if spec_path else None
        self.out_dir = Path(out_dir) if out_dir else None
        self.active: ActiveRows | None = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _read_spec(self) -> dict[str, Any] | None:
        if not self.enabled or self.spec_path is None:
            return None
        try:
            value = json.loads(self.spec_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError("attention heatmap spec must be a JSON object")
        return value

    def begin_step(
        self,
        *,
        req_ids: list[str],
        query_start_loc: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
    ) -> None:
        self.active = None
        spec = self._read_spec()
        if spec is None:
            return
        if len(req_ids) != 1:
            raise RuntimeError("attention heatmap probe requires one active request")
        request_id = str(req_ids[0])
        marker_request_id = str(spec["request_id"])
        if marker_request_id not in request_id:
            return

        computed = int(num_computed_tokens[0])
        scheduled = int(num_scheduled_tokens[0])
        step_end = computed + scheduled
        skill_start = int(spec["skill_start"])
        skill_end = int(spec["skill_end"])
        prompt_end = int(spec["prompt_end"])
        forward_query_start = int(spec["forward_query_start"])
        forward_query_end = int(spec["forward_query_end"])
        wanted = [
            pos
            for pos in (
                *range(skill_start, skill_end),
                *range(forward_query_start, forward_query_end),
            )
            if computed <= pos < step_end
        ]
        if not wanted:
            return
        row_base = int(query_start_loc[0])
        self.active = ActiveRows(
            request_id=request_id,
            mode=str(spec["mode"]),
            query_rows=tuple(row_base + pos - computed for pos in wanted),
            query_positions=tuple(wanted),
            skill_start=skill_start,
            skill_end=skill_end,
            cross_key_start=int(spec["cross_key_start"]),
            cross_key_end=int(spec["cross_key_end"]),
            forward_query_start=forward_query_start,
            forward_query_end=forward_query_end,
            forward_key_start=int(spec["forward_key_start"]),
            forward_key_end=int(spec["forward_key_end"]),
            prompt_end=prompt_end,
        )

    @staticmethod
    def _layer_index(layer: torch.nn.Module) -> int:
        name = str(getattr(layer, "layer_name", ""))
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None:
            raise RuntimeError(f"cannot parse layer index from {name!r}")
        return int(match.group(1))

    @staticmethod
    def _gather_keys(
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        if key_cache.dim() != 4:
            raise ValueError(f"unexpected key-cache shape: {tuple(key_cache.shape)}")
        block_size = int(key_cache.shape[1])
        positions = torch.arange(seq_len, device=key_cache.device)
        block_ids = block_table[0, positions // block_size].long()
        offsets = positions % block_size
        return key_cache[block_ids, offsets]

    @staticmethod
    def _attention_rows(
        query: torch.Tensor,
        keys: torch.Tensor,
        query_rows: tuple[int, ...],
        query_positions: tuple[int, ...],
        scale: float,
    ) -> torch.Tensor:
        selected_q = query[list(query_rows)].detach().float()
        keys = keys.detach().float()
        num_queries, num_query_heads, head_size = selected_q.shape
        num_kv_heads = int(keys.shape[1])
        if num_query_heads % num_kv_heads:
            raise ValueError(
                f"unsupported GQA layout: q_heads={num_query_heads}, "
                f"kv_heads={num_kv_heads}"
            )
        group_size = num_query_heads // num_kv_heads
        selected_q = selected_q.view(
            num_queries, num_kv_heads, group_size, head_size
        )
        seq_len = int(keys.shape[0])
        result = torch.zeros(
            (num_queries, seq_len), dtype=torch.float32, device="cpu"
        )

        # Bound peak memory while preserving the exact causal softmax.
        for start in range(0, num_queries, 16):
            end = min(start + 16, num_queries)
            scores = torch.einsum(
                "qhgd,khd->qhgk", selected_q[start:end], keys
            ) * float(scale)
            absolute_q = torch.tensor(
                query_positions[start:end], device=scores.device
            )
            key_positions = torch.arange(seq_len, device=scores.device)
            scores.masked_fill_(
                key_positions.view(1, 1, 1, -1)
                > absolute_q.view(-1, 1, 1, 1),
                float("-inf"),
            )
            probabilities = torch.softmax(scores, dim=-1)
            result[start:end] = probabilities.mean(dim=(1, 2)).cpu()
        return result

    def capture(
        self,
        *,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        scale: float,
        use_cascade: bool,
    ) -> None:
        active = self.active
        if not self.enabled or active is None:
            return
        if use_cascade:
            raise RuntimeError("attention heatmap probe does not support cascade")
        layer_index = self._layer_index(layer)
        keys = self._gather_keys(key_cache, block_table, active.prompt_end)
        rows = self._attention_rows(
            query,
            keys,
            active.query_rows,
            active.query_positions,
            scale,
        )

        cross_positions = [
            pos
            for pos in active.query_positions
            if active.skill_start <= pos < active.skill_end
        ]
        forward_positions = [
            pos
            for pos in active.query_positions
            if active.forward_query_start <= pos < active.forward_query_end
        ]
        cross_indices = [
            i
            for i, pos in enumerate(active.query_positions)
            if active.skill_start <= pos < active.skill_end
        ]
        forward_indices = [
            i
            for i, pos in enumerate(active.query_positions)
            if active.forward_query_start <= pos < active.forward_query_end
        ]

        cross = (
            rows[
                cross_indices,
                active.cross_key_start : active.cross_key_end,
            ].numpy()
            if cross_indices
            else np.empty(
                (0, active.cross_key_end - active.cross_key_start),
                dtype=np.float32,
            )
        )
        forward = (
            rows[
                forward_indices,
                active.forward_key_start : active.forward_key_end,
            ].numpy()
            if forward_indices
            else np.empty(
                (0, active.forward_key_end - active.forward_key_start),
                dtype=np.float32,
            )
        )
        self._merge_layer(
            layer_index=layer_index,
            mode=active.mode,
            cross=cross,
            cross_positions=np.asarray(cross_positions, dtype=np.int64),
            forward=forward,
            forward_positions=np.asarray(forward_positions, dtype=np.int64),
            skill_start=active.skill_start,
            skill_end=active.skill_end,
            cross_key_start=active.cross_key_start,
            cross_key_end=active.cross_key_end,
            forward_query_start=active.forward_query_start,
            forward_query_end=active.forward_query_end,
            forward_key_start=active.forward_key_start,
            forward_key_end=active.forward_key_end,
            prompt_end=active.prompt_end,
        )

    def _merge_layer(self, *, layer_index: int, mode: str, **values: Any) -> None:
        assert self.out_dir is not None
        path = self.out_dir / f"{mode}_layer_{layer_index:02d}.npz"
        previous: dict[str, Any] = {}
        if path.is_file():
            with np.load(path) as loaded:
                previous = {name: loaded[name] for name in loaded.files}

        for matrix_name, position_name in (
            ("cross", "cross_positions"),
            ("forward", "forward_positions"),
        ):
            matrix = values[matrix_name]
            positions = values[position_name]
            if matrix.shape[0] == 0 and matrix_name in previous:
                values[matrix_name] = previous[matrix_name]
                values[position_name] = previous[position_name]
            elif matrix_name in previous and previous[matrix_name].shape[0] > 0:
                combined = {
                    int(pos): row
                    for pos, row in zip(
                        previous[position_name], previous[matrix_name], strict=True
                    )
                }
                combined.update(
                    {
                        int(pos): row
                        for pos, row in zip(positions, matrix, strict=True)
                    }
                )
                ordered = sorted(combined)
                values[position_name] = np.asarray(ordered, dtype=np.int64)
                values[matrix_name] = np.stack([combined[pos] for pos in ordered])

        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, mode=mode, layer=layer_index, **values)
        temporary.replace(path)


_PROBE: AttentionHeatmapProbe | None = None


def get_attention_heatmap_probe() -> AttentionHeatmapProbe:
    global _PROBE
    if _PROBE is None:
        _PROBE = AttentionHeatmapProbe()
    return _PROBE
