"""Token selection helpers for SkillBlend selective recompute."""

from __future__ import annotations

from enum import Enum
import math

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on active env
    torch = None  # type: ignore[assignment]


class SelectionStrategy(str, Enum):
    """Ablation strategies for deciding which skill tokens to recompute."""

    HIGH_KV_DEVIATION = "high_kv_deviation"
    FIRST_LAST = "first_last"
    RANDOM = "random"


def token_budget(length: int, ratio: float, *, min_tokens: int = 1) -> int:
    """Convert a recompute ratio to a token count."""

    if length <= 0:
        raise ValueError("length must be positive")
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("ratio must be in [0, 1]")
    if ratio == 0.0:
        return 0
    return min(length, max(min_tokens, int(math.ceil(length * ratio))))


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "skillblend.selection requires torch. Activate the model environment "
            "before running HKVD token-selection experiments."
        )
    return torch


def kv_deviation_scores(
    cached_k: torch.Tensor,
    candidate_k: torch.Tensor,
    *,
    token_dim: int = 0,
) -> torch.Tensor:
    """Return per-token squared L2 deviation between cached and candidate K.

    `cached_k` is the old/reused K, while `candidate_k` is the K computed on a
    check layer under the current context. The output shape is `[num_tokens]`.
    """

    torch_mod = _require_torch()
    if cached_k.shape != candidate_k.shape:
        raise ValueError(
            f"cached_k and candidate_k shape mismatch: {cached_k.shape} vs {candidate_k.shape}"
        )
    if token_dim < 0:
        token_dim += cached_k.ndim
    if token_dim < 0 or token_dim >= cached_k.ndim:
        raise ValueError(f"invalid token_dim={token_dim} for ndim={cached_k.ndim}")

    diff = (candidate_k.float() - cached_k.float()) ** 2
    if token_dim != 0:
        diff = diff.movedim(token_dim, 0)
    return diff.flatten(start_dim=1).sum(dim=1)


def select_high_kv_deviation_tokens(
    cached_k: torch.Tensor,
    candidate_k: torch.Tensor,
    *,
    ratio: float,
    token_dim: int = 0,
    min_tokens: int = 1,
) -> torch.Tensor:
    """Select sorted token indices with the largest K deviation."""

    torch_mod = _require_torch()
    scores = kv_deviation_scores(cached_k, candidate_k, token_dim=token_dim)
    k = token_budget(int(scores.numel()), ratio, min_tokens=min_tokens)
    if k == 0:
        return torch_mod.empty(0, dtype=torch_mod.long, device=scores.device)
    indices = torch.topk(scores, k=k).indices
    return torch.sort(indices).values


def select_first_last_tokens(
    length: int,
    *,
    ratio: float,
    min_tokens: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """A simple ablation baseline: split the budget between head and tail."""

    torch_mod = _require_torch()
    k = token_budget(length, ratio, min_tokens=min_tokens)
    if k == 0:
        return torch_mod.empty(0, dtype=torch_mod.long, device=device)
    head = (k + 1) // 2
    tail = k // 2
    indices = list(range(head))
    if tail:
        indices.extend(range(length - tail, length))
    return torch_mod.tensor(sorted(set(indices)), dtype=torch_mod.long, device=device)


def select_random_tokens(
    length: int,
    *,
    ratio: float,
    min_tokens: int = 1,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random ablation baseline with deterministic seed."""

    torch_mod = _require_torch()
    k = token_budget(length, ratio, min_tokens=min_tokens)
    if k == 0:
        return torch_mod.empty(0, dtype=torch_mod.long, device=device)
    gen_device = (
        device
        if isinstance(device, torch_mod.device) and device.type != "cuda"
        else "cpu"
    )
    generator = torch_mod.Generator(device=gen_device)
    generator.manual_seed(seed)
    perm = torch_mod.randperm(length, generator=generator)[:k]
    return torch_mod.sort(perm.to(device=device)).values
