from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _coerce_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


@dataclass(frozen=True)
class CSKCacheConfig:
    """Configuration for the CSKCache engine.

    The probe/gate fields are the parameters of the signature probe-gated reuse
    mechanism; the storage fields control the CPU+disk tiering. This object is
    deliberately vLLM-free so the engine can be built from a plain dict, from
    environment variables, or from a vLLM config, without importing vLLM.

    Field names match the original ``CSKCacheVllmConfig`` so existing callers
    that read ``config.probe_tokens`` etc. keep working unchanged.
    """

    kv_dir: Path | None = None
    probe_enabled: bool = False
    probe_tokens: int = 4
    anchor_tokens: int = 32
    probe_tau: float = 0.15
    gate_metric: str = "max"
    # Storage tiering (new). None cpu_max_bytes = keep everything in CPU memory,
    # reproducing the original in-memory registry. Set both to enable disk spill.
    cpu_max_bytes: int | None = None
    disk_dir: Path | None = None
    capture_only: bool = False

    def __post_init__(self) -> None:
        if self.probe_tokens <= 0:
            raise ValueError("cskcache.probe_tokens must be positive")
        if self.anchor_tokens < self.probe_tokens:
            raise ValueError("cskcache.anchor_tokens must be >= cskcache.probe_tokens")

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "CSKCacheConfig":
        """Build from a plain mapping.

        Accepts both bare keys (``probe_tokens``) and dotted keys
        (``cskcache.probe_tokens``); the dotted form is what vLLM's connector
        extra-config uses, so one parser serves both entry points.
        """

        values = dict(values or {})

        def pick(name: str) -> Any:
            return values.get(name, values.get(f"cskcache.{name}"))

        kv_dir = pick("kv_dir")
        disk_dir = pick("disk_dir")
        cpu_max_bytes = pick("cpu_max_bytes")
        return cls(
            kv_dir=Path(str(kv_dir)) if kv_dir else None,
            probe_enabled=_coerce_bool(pick("probe_enabled"), False),
            probe_tokens=_coerce_int(pick("probe_tokens"), 4),
            anchor_tokens=_coerce_int(pick("anchor_tokens"), 32),
            probe_tau=_coerce_float(pick("probe_tau"), 0.15),
            gate_metric=str(pick("gate_metric") or "max"),
            cpu_max_bytes=None if cpu_max_bytes is None else int(cpu_max_bytes),
            disk_dir=Path(str(disk_dir)) if disk_dir else None,
            capture_only=_coerce_bool(pick("capture_only"), False),
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CSKCacheConfig":
        """Build from ``CSKCACHE_*`` environment variables."""

        env = os.environ if environ is None else environ

        def pick(name: str) -> Any:
            return env.get(f"CSKCACHE_{name.upper()}")

        return cls.from_dict(
            {
                "kv_dir": pick("kv_dir"),
                "probe_enabled": pick("probe_enabled"),
                "probe_tokens": pick("probe_tokens"),
                "anchor_tokens": pick("anchor_tokens"),
                "probe_tau": pick("probe_tau"),
                "gate_metric": pick("gate_metric"),
                "cpu_max_bytes": pick("cpu_max_bytes"),
                "disk_dir": pick("disk_dir"),
                "capture_only": pick("capture_only"),
            }
        )

    @classmethod
    def from_vllm(cls, vllm_config: Any) -> "CSKCacheConfig":
        """Build from a vLLM config via duck typing (no vLLM import).

        Reads ``kv_transfer_config.kv_connector_extra_config`` for dotted
        ``cskcache.*`` keys, then fills any gaps from the environment, matching
        the precedence the original ``load_vllm_config`` used.
        """

        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        extra: Mapping[str, Any] = (
            getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
        )
        # Environment is the fallback layer: merge env-derived values under the
        # explicit extra-config values.
        merged = {
            "kv_dir": extra.get("cskcache.kv_dir") or os.environ.get("CSKCACHE_KV_DIR"),
            "probe_enabled": extra.get("cskcache.probe_enabled"),
            "probe_tokens": extra.get("cskcache.probe_tokens"),
            "anchor_tokens": extra.get("cskcache.anchor_tokens"),
            "probe_tau": extra.get("cskcache.probe_tau"),
            "gate_metric": (
                extra.get("cskcache.gate_metric") or os.environ.get("CSKCACHE_GATE_METRIC")
            ),
            "cpu_max_bytes": extra.get("cskcache.cpu_max_bytes"),
            "disk_dir": extra.get("cskcache.disk_dir") or os.environ.get("CSKCACHE_DISK_DIR"),
            "capture_only": extra.get("cskcache.capture_only"),
        }
        env_fallback = cls.from_env()
        # For probe/gate scalars, prefer explicit extra-config, else env value.
        return cls.from_dict(
            {
                "kv_dir": merged["kv_dir"],
                "probe_enabled": (
                    merged["probe_enabled"]
                    if merged["probe_enabled"] is not None
                    else env_fallback.probe_enabled
                ),
                "probe_tokens": (
                    merged["probe_tokens"]
                    if merged["probe_tokens"] is not None
                    else env_fallback.probe_tokens
                ),
                "anchor_tokens": (
                    merged["anchor_tokens"]
                    if merged["anchor_tokens"] is not None
                    else env_fallback.anchor_tokens
                ),
                "probe_tau": (
                    merged["probe_tau"]
                    if merged["probe_tau"] is not None
                    else env_fallback.probe_tau
                ),
                "gate_metric": merged["gate_metric"] or env_fallback.gate_metric,
                "cpu_max_bytes": (
                    merged["cpu_max_bytes"]
                    if merged["cpu_max_bytes"] is not None
                    else env_fallback.cpu_max_bytes
                ),
                "disk_dir": merged["disk_dir"],
                "capture_only": (
                    merged["capture_only"]
                    if merged["capture_only"] is not None
                    else env_fallback.capture_only
                ),
            }
        )
