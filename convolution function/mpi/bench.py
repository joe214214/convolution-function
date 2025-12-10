"""Benchmark harness for the MPI convolution implementation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class BenchmarkRecord:
    params: dict
    t_total: float
    t_comp: float
    t_comm: float
    speedup: float
    efficiency: float


def append_csv(path: str, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    csv_path = Path(path)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
