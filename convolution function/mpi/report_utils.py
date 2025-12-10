"""Utilities for aggregating benchmark results."""

from __future__ import annotations

from statistics import mean
from typing import Iterable, List, Sequence


def summarize(records: Iterable[dict], key: str) -> float:
    values = [float(r[key]) for r in records if key in r]
    if not values:
        return 0.0
    return mean(values)
