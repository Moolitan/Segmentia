"""Persistent deployment performance used by the paper profitability gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping


SCHEMA_VERSION = 1


def deployment_fingerprint(environment: Mapping[str, object]) -> str:
    """Hash only stable deployment properties, never request-local values."""

    encoded = json.dumps(
        dict(environment), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_system_profile_path(
    metadata_path: str | os.PathLike[str],
    fingerprint: str,
    explicit_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve an override or the metadata area above the Skill pool."""

    if not fingerprint:
        raise ValueError("deployment fingerprint must be non-empty")
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not str(path):
            raise ValueError("system profile path must not be empty")
        return path.resolve()
    catalog = Path(metadata_path).expanduser().resolve()
    skill_pool = catalog if catalog.is_dir() else catalog.parent
    return (
        skill_pool.parent
        / "cskcache_metadata"
        / "system_profiles"
        / f"{fingerprint}.json"
    )


@dataclass
class RunningAverage:
    value: float | None = None
    samples: int = 0

    def add(self, value: float) -> None:
        if value <= 0:
            raise ValueError("performance samples must be positive")
        total = 0.0 if self.value is None else self.value * self.samples
        self.samples += 1
        self.value = (total + value) / self.samples

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RunningAverage":
        value = payload.get("value")
        samples = payload.get("samples", 0)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError("profile average value must be numeric")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
            raise ValueError("profile sample count must be a non-negative integer")
        result = cls(None if value is None else float(value), samples)
        if (result.value is None) != (result.samples == 0):
            raise ValueError("profile average value and sample count disagree")
        if result.value is not None and result.value <= 0:
            raise ValueError("profile average must be positive")
        return result


@dataclass
class SystemPerformanceProfile:
    """Small measured cost table; this is not a fitted performance model."""

    deployment_fingerprint: str
    environment: dict[str, object]
    prefill_ms_per_token: RunningAverage = field(default_factory=RunningAverage)
    ssd_bytes_per_ms: RunningAverage = field(default_factory=RunningAverage)
    h2d_bytes_per_ms: RunningAverage = field(default_factory=RunningAverage)
    composition_ms_per_token: dict[str, RunningAverage] = field(default_factory=dict)
    epsilon_ms: float = 0.0
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def composition_key(strategy: str, ratio: float) -> str:
        return f"{strategy}:{ratio:.8f}"

    def is_ready(self, strategy: str, ratio: float) -> bool:
        composition = self.composition_ms_per_token.get(
            self.composition_key(strategy, ratio)
        )
        return bool(
            self.prefill_ms_per_token.samples
            and self.ssd_bytes_per_ms.samples
            and composition is not None
            and composition.samples
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SystemPerformanceProfile":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported system performance profile schema")
        fingerprint = payload.get("deployment_fingerprint")
        environment = payload.get("environment")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("system profile requires a deployment fingerprint")
        if not isinstance(environment, Mapping):
            raise ValueError("system profile requires an environment mapping")
        compositions = payload.get("composition_ms_per_token", {})
        if not isinstance(compositions, Mapping):
            raise ValueError("composition profile must be a mapping")
        epsilon = payload.get("epsilon_ms", 0.0)
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("epsilon_ms must be numeric")
        if float(epsilon) < 0:
            raise ValueError("epsilon_ms must be non-negative")
        return cls(
            deployment_fingerprint=fingerprint,
            environment=dict(environment),
            prefill_ms_per_token=RunningAverage.from_dict(
                _mapping(payload, "prefill_ms_per_token")
            ),
            ssd_bytes_per_ms=RunningAverage.from_dict(
                _mapping(payload, "ssd_bytes_per_ms")
            ),
            h2d_bytes_per_ms=RunningAverage.from_dict(
                _mapping(payload, "h2d_bytes_per_ms")
            ),
            composition_ms_per_token={
                str(key): RunningAverage.from_dict(_require_mapping(value))
                for key, value in compositions.items()
            },
            epsilon_ms=float(epsilon),
        )


@dataclass(frozen=True)
class ProfitabilityEstimate:
    t_pf_ms: float
    t_ready_ms: float
    t_comp_ms: float
    t_comp_visible_ms: float
    gain_ms: float

    @property
    def profitable(self) -> bool:
        return self.gain_ms > 0.0


class SystemPerformanceTracker:
    """Load, update and atomically persist one deployment's measurements."""

    def __init__(
        self,
        *,
        enabled: bool,
        environment: Mapping[str, object],
        metadata_path: str | os.PathLike[str],
        explicit_path: str | os.PathLike[str] | None = None,
        writer: bool = True,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("profitability enabled must be a boolean")
        self.enabled = enabled
        self.environment = dict(environment)
        self.fingerprint = deployment_fingerprint(self.environment)
        self.path = resolve_system_profile_path(
            metadata_path, self.fingerprint, explicit_path
        )
        self.writer = writer
        self._lock = threading.RLock()
        self._announced_keys: set[str] = set()
        self.profile = self._load_or_empty()

    def _load_or_empty(self) -> SystemPerformanceProfile:
        if not self.path.exists():
            return SystemPerformanceProfile(self.fingerprint, self.environment)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            profile = SystemPerformanceProfile.from_dict(payload)
            if profile.deployment_fingerprint != self.fingerprint:
                raise ValueError("deployment fingerprint mismatch")
            return profile
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return SystemPerformanceProfile(self.fingerprint, self.environment)

    def first_use_message(self, strategy: str, ratio: float) -> str | None:
        """Return the required first-use line once for each missing cost arm."""

        if (
            not self.enabled
            or not self.writer
            or self.profile.is_ready(strategy, ratio)
        ):
            return None
        key = self.profile.composition_key(strategy, ratio)
        with self._lock:
            if key in self._announced_keys:
                return None
            self._announced_keys.add(key)
        return (
            "CSKCache profitability profile not found. This is the first-use "
            "calibration request: profitability gating is skipped for the "
            f"current request, and measured system performance will be persisted to {self.path}."
        )

    def estimate(
        self,
        *,
        skill_tokens: int,
        storage_bytes: int,
        prefetch_elapsed_ms: float,
        strategy: str,
        ratio: float,
    ) -> ProfitabilityEstimate | None:
        with self._lock:
            if not self.enabled or not self.profile.is_ready(strategy, ratio):
                return None
            prefill = self.profile.prefill_ms_per_token.value
            ssd = self.profile.ssd_bytes_per_ms.value
            composition = self.profile.composition_ms_per_token[
                self.profile.composition_key(strategy, ratio)
            ].value
            epsilon = self.profile.epsilon_ms
        assert prefill is not None and ssd is not None and composition is not None
        t_pf = skill_tokens * prefill
        t_ready = max(0.0, storage_bytes / ssd - prefetch_elapsed_ms)
        t_comp = skill_tokens * composition
        visible = t_ready + t_comp + epsilon
        return ProfitabilityEstimate(t_pf, t_ready, t_comp, visible, t_pf - visible)

    def record_prefill(self, tokens: int, duration_ms: float) -> None:
        if self.enabled and tokens > 0 and duration_ms > 0:
            with self._lock:
                self.profile.prefill_ms_per_token.add(duration_ms / tokens)
                self._persist_locked()

    def record_ssd(self, byte_count: int, duration_ms: float) -> None:
        if self.enabled and byte_count > 0 and duration_ms > 0:
            with self._lock:
                self.profile.ssd_bytes_per_ms.add(byte_count / duration_ms)
                self._persist_locked()

    def record_h2d(self, byte_count: int, duration_ms: float) -> None:
        if self.enabled and byte_count > 0 and duration_ms > 0:
            with self._lock:
                self.profile.h2d_bytes_per_ms.add(byte_count / duration_ms)
                self._persist_locked()

    def record_composition(
        self, *, strategy: str, ratio: float, tokens: int, duration_ms: float
    ) -> None:
        if not self.enabled or tokens <= 0 or duration_ms <= 0:
            return
        key = self.profile.composition_key(strategy, ratio)
        with self._lock:
            self.profile.composition_ms_per_token.setdefault(
                key, RunningAverage()
            ).add(duration_ms / tokens)
            self._persist_locked()

    def _persist_locked(self) -> None:
        if not self.enabled or not self.writer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        encoded = json.dumps(
            self.profile.to_dict(), sort_keys=True, indent=2
        ).encode("utf-8")
        descriptor = os.open(
            temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.path)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _require_mapping(payload.get(key, {}))


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("system profile entry must be a mapping")
    return value
