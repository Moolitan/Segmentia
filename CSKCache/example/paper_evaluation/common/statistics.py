"""Small deterministic aggregation helpers used by all subsections."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def median_or_none(values: Iterable[Any]) -> float | None:
    parsed = [float(value) for value in values if value not in (None, "")]
    return statistics.median(parsed) if parsed else None


def group_rows(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(groups)
