from __future__ import annotations

import json

from cskcache.integrations.lmcache.rank_control import (
    CSKCacheRankControlClient,
)


class _FakeTransport:
    world_size = 2

    def __init__(self) -> None:
        self.frames = None
        self.closed = False

    def send_and_recv_all(self, frames):
        self.frames = frames
        return [
            json.dumps({"rank": 0, "tokens": 16}).encode(),
            json.dumps({"rank": 1, "tokens": 24}).encode(),
        ]

    def close(self) -> None:
        self.closed = True


def test_cskcache_rank_control_preserves_rank_local_results() -> None:
    transport = _FakeTransport()
    client = CSKCacheRankControlClient(transport)

    assert client.execute_all("cskcache.requirement", {"ticket": "call-1"}) == [
        {"rank": 0, "tokens": 16},
        {"rank": 1, "tokens": 24},
    ]
    assert transport.frames == [
        "external_control",
        "cskcache.requirement",
        '{"ticket":"call-1"}',
    ]

    client.close()
    assert transport.closed


def test_cskcache_rank_control_fails_closed_on_partial_response() -> None:
    transport = _FakeTransport()
    transport.send_and_recv_all = lambda _frames: [b"null"]
    client = CSKCacheRankControlClient(transport)

    assert client.execute_all("cskcache.requirement", {}) == []
