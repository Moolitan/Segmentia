"""CSKCache KV movement layer.

Encapsulates every operation that touches the serving engine's paged KV cache:
slicing an offline entry, RoPE-correcting keys for a new position, scattering
reuse K/V into paged slots, and gathering recomputed K/V back out. Isolating
these behind ``KVConnectorInterface`` keeps the engine free of paged-memory
details and gives a single place to add other backends later.
"""

from cskcache.v1.kv_transfer.gpu_connector import (
    KVConnectorInterface,
    VLLMPagedGPUConnector,
)

__all__ = ["KVConnectorInterface", "VLLMPagedGPUConnector"]
