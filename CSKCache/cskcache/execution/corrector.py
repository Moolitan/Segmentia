"""Context-aware token-axis correction owned by CSKCache."""

from __future__ import annotations

import math

import torch


class ContextAwareKVCorrector:
    """Apply the Section 3.2 shared context residual to one staged layer.

    The staged tensor contains the offline Key or Value beginning at the
    calibration interval.  The current request has already recomputed that
    interval in vLLM's paged KV cache.  Their token-mean difference estimates
    one independent offset for every flattened KV-head component.  Only the
    not-yet-computed suffix is modified.

    Key and Value are corrected through separate entry points because they
    reach this class from different staging tensors and because the Value arm
    can be disabled for the Key-only ablation.
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

        return self._correct_(
            staged_key,
            recomputed_calibration_key,
            calibration_tokens=calibration_tokens,
            suffix_offset=suffix_offset,
            alpha=alpha,
            component="Key",
        )

    def correct_value_(
        self,
        staged_value: torch.Tensor,
        recomputed_calibration_value: torch.Tensor,
        *,
        calibration_tokens: int,
        suffix_offset: int,
        alpha: float,
    ) -> torch.Tensor:
        """Correct ``staged_value[suffix_offset:]`` in place and return offset.

        Value carries no positional encoding, so the staged tensor needs no
        position adjustment before the residual is estimated.
        """

        return self._correct_(
            staged_value,
            recomputed_calibration_value,
            calibration_tokens=calibration_tokens,
            suffix_offset=suffix_offset,
            alpha=alpha,
            component="Value",
        )

    def _correct_(
        self,
        staged: torch.Tensor,
        recomputed_calibration: torch.Tensor,
        *,
        calibration_tokens: int,
        suffix_offset: int,
        alpha: float,
        component: str,
    ) -> torch.Tensor:
        if not isinstance(staged, torch.Tensor) or not isinstance(
            recomputed_calibration, torch.Tensor
        ):
            raise TypeError(
                f"CSKCache {component} correction requires torch tensors"
            )
        if staged.ndim < 2 or staged.shape[0] == 0:
            raise ValueError(
                f"staged {component} must have a non-empty token dimension"
            )
        if not torch.is_floating_point(staged) or not torch.is_floating_point(
            recomputed_calibration
        ):
            raise ValueError(
                f"CSKCache {component} correction requires floating tensors"
            )
        if staged.device != recomputed_calibration.device:
            raise ValueError(
                f"cached and recomputed {component} must be on the same device"
            )
        if not isinstance(calibration_tokens, int) or isinstance(
            calibration_tokens, bool
        ):
            raise TypeError("calibration_tokens must be an integer")
        if not 0 < calibration_tokens <= staged.shape[0]:
            raise ValueError(
                f"calibration interval is outside the staged {component}"
            )
        if not isinstance(suffix_offset, int) or isinstance(suffix_offset, bool):
            raise TypeError("suffix_offset must be an integer")
        if not calibration_tokens <= suffix_offset < staged.shape[0]:
            raise ValueError("corrected suffix must follow the calibration interval")
        expected_shape = staged[:calibration_tokens].shape
        if recomputed_calibration.shape != expected_shape:
            raise ValueError(
                f"cached and recomputed calibration {component} shapes differ: "
                f"cached={tuple(expected_shape)}, "
                f"recomputed={tuple(recomputed_calibration.shape)}"
            )
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("correction alpha must be numeric")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("correction alpha must be finite and in [0, 1]")

        cached_calibration = staged[:calibration_tokens].detach().clone()
        offset = alpha * (
            recomputed_calibration.to(torch.float32)
            - cached_calibration.to(torch.float32)
        ).mean(dim=0)
        staged[suffix_offset:].add_(offset.to(dtype=staged.dtype))
        return offset
