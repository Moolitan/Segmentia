"""Independent reader for the Terminal smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {"producer": "openhands-tools==1.43.1", "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    raw = args.artifact.read_bytes()
    payload = json.loads(raw)
    if payload != EXPECTED:
        raise RuntimeError(f"Unexpected Terminal artifact: {payload}")
    print(
        json.dumps(
            {
                "verifier_readable": True,
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "payload": payload,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
