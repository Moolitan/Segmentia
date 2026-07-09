"""Eviction policies for the CSKCache storage tiers."""

from cskcache.v1.storage.cache_policy.lru import LRUCachePolicy, get_cache_policy

__all__ = ["LRUCachePolicy", "get_cache_policy"]
