"""Storage-load plans derived from one persistent KV layout."""

from __future__ import annotations

from dataclasses import dataclass

from ...layouts import KVLayout, KVLayoutPlan


@dataclass(frozen=True)
class StorageLoadGroup:
    """Regions submitted together before advancing to the next group."""

    group_id: int
    region_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if not self.region_ids:
            raise ValueError("a storage load group must not be empty")


@dataclass(frozen=True)
class StorageLoadPlan:
    """Complete SSD-to-host submission grouping for one Skill."""

    layout: KVLayout
    layout_plan: KVLayoutPlan
    groups: tuple[StorageLoadGroup, ...]

    def __post_init__(self) -> None:
        if self.layout is not self.layout_plan.layout:
            raise ValueError("storage load layout differs from its layout plan")
        if tuple(group.group_id for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ValueError("storage load group IDs must be dense and ordered")
        flattened = tuple(
            region_id for group in self.groups for region_id in group.region_ids
        )
        if sorted(flattened) != list(range(len(self.layout_plan.regions))):
            raise ValueError("storage load groups must cover every region once")

    @property
    def ordered_region_ids(self) -> tuple[int, ...]:
        return tuple(
            region_id for group in self.groups for region_id in group.region_ids
        )
