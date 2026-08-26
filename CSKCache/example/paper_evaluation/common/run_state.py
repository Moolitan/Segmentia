"""Automatic timestamped run directories and crash-safe case state."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import append_csv, append_jsonl


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def input_fingerprint(paths: Iterable[Path], values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted((path.resolve() for path in paths), key=str):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(values, ensure_ascii=False, sort_keys=True, default=str).encode()
    )
    return digest.hexdigest()


@dataclass
class RunContext:
    section: str
    run_id: str
    run_dir: Path
    fingerprint: str
    state: dict[str, Any]

    @classmethod
    def open(
        cls,
        *,
        output_root: Path,
        section: str,
        config_paths: Iterable[Path],
        config_values: Mapping[str, Any],
    ) -> "RunContext":
        section_root = output_root / section
        section_root.mkdir(parents=True, exist_ok=True)
        fingerprint = input_fingerprint(config_paths, config_values)
        active_path = section_root / "active_run.json"
        active = None
        if active_path.is_file():
            active = json.loads(active_path.read_text(encoding="utf-8"))
        if (
            isinstance(active, dict)
            and active.get("status") != "completed"
            and active.get("input_fingerprint") == fingerprint
            and Path(str(active.get("run_dir", ""))).is_dir()
        ):
            run_dir = Path(active["run_dir"])
            run_id = str(active["run_id"])
        else:
            run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            run_dir = section_root / run_id
            suffix = 1
            while run_dir.exists():
                run_dir = section_root / f"{run_id}-{suffix:02d}"
                suffix += 1
            run_id = run_dir.name
            run_dir.mkdir(parents=True)
            _atomic_json(
                active_path,
                {
                    "status": "running",
                    "section": section,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "input_fingerprint": fingerprint,
                    "started_utc": utc_now(),
                },
            )
        state_path = run_dir / "run_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("input_fingerprint") != fingerprint:
                raise RuntimeError("run state fingerprint differs from current config")
        else:
            state = {
                "status": "running",
                "section": section,
                "run_id": run_id,
                "input_fingerprint": fingerprint,
                "started_utc": utc_now(),
                "cases": {},
            }
            _atomic_json(state_path, state)
        _atomic_json(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "section": section,
                "run_id": run_id,
                "hostname": socket.gethostname(),
                "input_fingerprint": fingerprint,
                "config": dict(config_values),
            },
        )
        return cls(section, run_id, run_dir, fingerprint, state)

    @property
    def samples_csv(self) -> Path:
        return self.run_dir / "samples.csv"

    @property
    def samples_jsonl(self) -> Path:
        return self.run_dir / "samples.jsonl"

    def completed(self, case_id: str) -> bool:
        return self.state["cases"].get(case_id, {}).get("status") == "completed"

    def mark(self, case_id: str, status: str, **details: Any) -> None:
        if status not in {"running", "completed", "failed"}:
            raise ValueError(f"invalid case status: {status}")
        self.state["cases"][case_id] = {
            "status": status,
            "updated_utc": utc_now(),
            **details,
        }
        _atomic_json(self.run_dir / "run_state.json", self.state)

    def record(self, row: Mapping[str, Any]) -> None:
        base = {
            "run_id": self.run_id,
            "section": self.section,
            "hostname": socket.gethostname(),
            "input_fingerprint": self.fingerprint,
        }
        payload = {**base, **dict(row)}
        append_jsonl(self.samples_jsonl, payload)
        append_csv(self.samples_csv, payload)

    def finish(self) -> None:
        self.state["status"] = "completed"
        self.state["completed_utc"] = utc_now()
        _atomic_json(self.run_dir / "run_state.json", self.state)
        _atomic_json(
            self.run_dir.parent / "active_run.json",
            {
                "status": "completed",
                "section": self.section,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "input_fingerprint": self.fingerprint,
                "completed_utc": utc_now(),
            },
        )
