from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CSKLayerResidual:
    layer_name: str
    k_mean: float
    v_mean: float
    kv_mean: float


@dataclass(frozen=True)
class CSKProbeMetrics:
    num_layers: int
    k_mean: float
    v_mean: float
    kv_mean: float
    k_max_layer: float
    v_max_layer: float
    kv_max_layer: float
    gate_value: float
    gate_metric: str
    layers: tuple[CSKLayerResidual, ...]


@dataclass(frozen=True)
class CSKProbeDecision:
    req_id: str
    cache_id: str
    passed: bool
    tau: float
    metrics: CSKProbeMetrics


def _token_residual(reuse: torch.Tensor, recompute: torch.Tensor) -> torch.Tensor:
    if reuse.shape != recompute.shape:
        raise ValueError(
            "CSK probe residual requires equal shapes: "
            f"reuse={tuple(reuse.shape)} recompute={tuple(recompute.shape)}"
        )
    reuse_flat = reuse.to(torch.float32).flatten(start_dim=1)
    recompute_flat = recompute.to(torch.float32).flatten(start_dim=1)
    cos = F.cosine_similarity(reuse_flat, recompute_flat, dim=-1, eps=1e-6)
    return 1.0 - cos


def compute_layer_residual(
    layer_name: str,
    *,
    reuse_key: torch.Tensor,
    reuse_value: torch.Tensor,
    recompute_key: torch.Tensor,
    recompute_value: torch.Tensor,
) -> CSKLayerResidual:
    k_residual = _token_residual(reuse_key, recompute_key)
    v_residual = _token_residual(reuse_value, recompute_value)
    k_mean = float(k_residual.mean().detach().cpu())
    v_mean = float(v_residual.mean().detach().cpu())
    return CSKLayerResidual(
        layer_name=layer_name,
        k_mean=k_mean,
        v_mean=v_mean,
        kv_mean=(k_mean + v_mean) / 2.0,
    )


class CSKProbeAccumulator:
    def __init__(
        self,
        *,
        req_id: str,
        cache_id: str,
        tau: float,
        gate_metric: str,
    ) -> None:
        self.req_id = req_id
        self.cache_id = cache_id
        self.tau = tau
        self.gate_metric = gate_metric
        self._layers: list[CSKLayerResidual] = []

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(item.layer_name for item in self._layers)

    def add_layer(
        self,
        layer_name: str,
        *,
        reuse_key: torch.Tensor,
        reuse_value: torch.Tensor,
        recompute_key: torch.Tensor,
        recompute_value: torch.Tensor,
    ) -> None:
        self._layers.append(
            compute_layer_residual(
                layer_name,
                reuse_key=reuse_key,
                reuse_value=reuse_value,
                recompute_key=recompute_key,
                recompute_value=recompute_value,
            )
        )

    def decide(self) -> CSKProbeDecision:
        if not self._layers:
            raise RuntimeError(f"CSK probe for {self.req_id} has no captured layers")

        k_values = [item.k_mean for item in self._layers]
        v_values = [item.v_mean for item in self._layers]
        kv_values = [item.kv_mean for item in self._layers]
        k_mean = sum(k_values) / len(k_values)
        v_mean = sum(v_values) / len(v_values)
        kv_mean = sum(kv_values) / len(kv_values)
        k_max = max(k_values)
        v_max = max(v_values)
        kv_max = max(kv_values)

        metric = self.gate_metric.lower()
        if metric == "mean":
            gate_value = kv_mean
        elif metric == "k_mean":
            gate_value = k_mean
        elif metric == "v_mean":
            gate_value = v_mean
        elif metric == "k_max":
            gate_value = k_max
        elif metric == "v_max":
            gate_value = v_max
        else:
            metric = "max"
            gate_value = max(k_max, v_max)

        metrics = CSKProbeMetrics(
            num_layers=len(self._layers),
            k_mean=k_mean,
            v_mean=v_mean,
            kv_mean=kv_mean,
            k_max_layer=k_max,
            v_max_layer=v_max,
            kv_max_layer=kv_max,
            gate_value=gate_value,
            gate_metric=metric,
            layers=tuple(self._layers),
        )
        return CSKProbeDecision(
            req_id=self.req_id,
            cache_id=self.cache_id,
            passed=gate_value <= self.tau,
            tau=self.tau,
            metrics=metrics,
        )
