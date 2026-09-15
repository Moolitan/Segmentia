"""Rank-wise readiness collection over LMCache's opaque RPC transport."""

from __future__ import annotations

from collections.abc import Mapping
import json
from threading import Lock
from typing import Any

from lmcache.logging import init_logger
from lmcache.v1.lookup_client.external_control_client import EXTERNAL_CONTROL
from lmcache.v1.rpc.zmq_transport import (
    SocketParams,
    ZmqReqRepClientTransport,
)
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache


logger = init_logger(__name__)


class CSKCacheRankControlClient:
    """Return rank-local readiness without requiring byte-identical replies."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self._lock = Lock()

    @classmethod
    def from_lmcache(
        cls, config: Any, metadata: Any
    ) -> "CSKCacheRankControlClient":
        if metadata is None or metadata.engine_id is None:
            raise ValueError("LMCache metadata is unavailable for rank control")
        lookup_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        ranks = list(lookup_ids) if lookup_ids else list(range(metadata.world_size))
        if ranks != list(range(metadata.world_size)):
            raise ValueError("CSKCache requires one lookup server per worker rank")
        kv_extra = metadata.kv_connector_extra_config or {}
        rpc_port = kv_extra.get("lmcache_rpc_port", 0)
        sockets = [
            SocketParams(
                socket_path=get_zmq_rpc_path_lmcache(
                    metadata.engine_id, "lookup", rpc_port, rank
                ),
                rank=rank,
            )
            for rank in ranks
        ]
        return cls(
            ZmqReqRepClientTransport(
                socket_params=sockets,
                timeout_ms=config.lookup_timeout_ms,
            )
        )

    def execute_all(
        self, command: str, payload: Mapping[str, Any]
    ) -> list[Any]:
        payload_json = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":")
        )
        try:
            with self._lock:
                responses = self._transport.send_and_recv_all(
                    [EXTERNAL_CONTROL, command, payload_json]
                )
            if len(responses) != self._transport.world_size:
                return []
            return [json.loads(response.decode("utf-8")) for response in responses]
        except Exception:
            logger.exception("CSKCache rank control failed: %s", command)
            return []

    def close(self) -> None:
        with self._lock:
            self._transport.close()
