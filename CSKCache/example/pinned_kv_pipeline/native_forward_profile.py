"""Worker extension that times one native vLLM model forward with CUDA events."""

from __future__ import annotations

from typing import Any


class NativeForwardProfileWorkerExtension:
    def start_native_forward_profile(self) -> None:
        import torch

        self._native_forward_records = []
        self._native_forward_pending = []
        model = self.model_runner.get_model()

        def before(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            positions = kwargs.get("positions")
            token_count = (
                int(input_ids.numel())
                if input_ids is not None
                else int(inputs_embeds.shape[0])
            )
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._native_forward_pending.append(
                (start, token_count, int(positions.numel()))
            )

        def after(
            _module: Any,
            _args: Any,
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start, input_tokens, position_tokens = (
                self._native_forward_pending.pop()
            )
            self._native_forward_records.append(
                (start, end, input_tokens, position_tokens)
            )

        self._native_forward_handles = (
            model.register_forward_pre_hook(before, with_kwargs=True),
            model.register_forward_hook(after, with_kwargs=True),
        )

    def finish_native_forward_profile(self) -> dict[str, Any]:
        import torch

        torch.cuda.synchronize()
        for handle in self._native_forward_handles:
            handle.remove()
        records = [
            {
                "gpu_ms": float(start.elapsed_time(end)),
                "input_tokens": input_tokens,
                "position_tokens": position_tokens,
            }
            for start, end, input_tokens, position_tokens in (
                self._native_forward_records
            )
        ]
        self._native_forward_handles = ()
        self._native_forward_records = []
        self._native_forward_pending = []
        return {"rank": int(self.rank), "forwards": records}
