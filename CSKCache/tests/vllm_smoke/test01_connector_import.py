from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from cskcache.integration.vllm.v1_connector import CSKCacheConnectorV1
from cskcache.v1.compute.gate import CSKProbeAccumulator


def main() -> None:
    acc = CSKProbeAccumulator(req_id="r0", cache_id="skill0", tau=0.1, gate_metric="max")
    x = torch.ones(4, 2, 3)
    acc.add_layer(
        "layer.0",
        reuse_key=x,
        reuse_value=x,
        recompute_key=x.clone(),
        recompute_value=x.clone(),
    )
    decision = acc.decide()
    assert decision.passed
    assert decision.metrics.gate_value < 1e-5
    print("connector import ok:", CSKCacheConnectorV1.__name__)
    print("probe gate smoke ok:", decision.metrics)


if __name__ == "__main__":
    main()
