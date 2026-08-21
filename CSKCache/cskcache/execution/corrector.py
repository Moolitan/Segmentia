"""Context-aware token-axis correction owned by CSKCache."""

from __future__ import annotations

import math

import torch


class ContextAwareKVCorrector:
    """Apply the Section 3.2 shared Key residual to one staged layer.

    The staged tensor contains the offline Key values beginning at the
    calibration interval.  The current request has already recomputed that
    interval in vLLM's paged KV cache.  Their token-mean difference estimates
    one independent offset for every flattened KV-head component.  Only the
    not-yet-computed suffix is modified; Value is deliberately not accepted by
    this API and therefore cannot be changed accidentally.
    """

    def correct_key_(
        self,
        staged_key: torch.Tensor,
        recomputed_calibration_key: torch.Tensor,
        *,
        calibration_tokens: int,
        suffix_offset: int,
        alpha: float,
    ) -> torch.Tensor:
        """Correct ``staged_key[suffix_offset:]`` in place and return offset."""

        if not isinstance(staged_key, torch.Tensor) or not isinstance(
            recomputed_calibration_key, torch.Tensor
        ):
            raise TypeError("CSKCache Key correction requires torch tensors")
        if staged_key.ndim < 2 or staged_key.shape[0] == 0:
            raise ValueError("staged Key must have a non-empty token dimension")
        if not torch.is_floating_point(staged_key) or not torch.is_floating_point(
            recomputed_calibration_key
        ):
            raise ValueError("CSKCache Key correction requires floating tensors")
        if staged_key.device != recomputed_calibration_key.device:
            raise ValueError("cached and recomputed Key must be on the same device")
        if not isinstance(calibration_tokens, int) or isinstance(
            calibration_tokens, bool
        ):
            raise TypeError("calibration_tokens must be an integer")
        if not 0 < calibration_tokens <= staged_key.shape[0]:
            raise ValueError("calibration interval is outside the staged Key")
        if not isinstance(suffix_offset, int) or isinstance(suffix_offset, bool):
            raise TypeError("suffix_offset must be an integer")
        if not calibration_tokens <= suffix_offset < staged_key.shape[0]:
            raise ValueError("corrected suffix must follow the calibration interval")
        expected_shape = staged_key[:calibration_tokens].shape
        if recomputed_calibration_key.shape != expected_shape:
            raise ValueError(
                "cached and recomputed calibration Key shapes differ: "
                f"cached={tuple(expected_shape)}, "
                f"recomputed={tuple(recomputed_calibration_key.shape)}"
            )
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("correction alpha must be numeric")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("correction alpha must be finite and in [0, 1]")

        cached_calibration = staged_key[:calibration_tokens].detach().clone()
        offset = alpha * (
            recomputed_calibration_key.to(torch.float32)
            - cached_calibration.to(torch.float32)
        ).mean(dim=0)
        staged_key[suffix_offset:].add_(offset.to(dtype=staged_key.dtype))
        return offset
