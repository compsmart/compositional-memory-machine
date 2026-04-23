from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hrr.binding import cosine, normalize


@dataclass
class MemoryRecord:
    key: str
    vector: np.ndarray
    payload: dict[str, Any] = field(default_factory=dict)
    writes: int = 1


class AMM:
    """Append-oriented associative memory with per-key vector accumulation."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    def write(self, key: str, vector: np.ndarray, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        if key in self.records:
            record = self.records[key]
            record.vector = normalize(record.vector * record.writes + vector)
            record.payload = payload
            record.writes += 1
            return
        self.records[key] = MemoryRecord(key=key, vector=normalize(vector), payload=payload.copy())

    def get(self, key: str) -> MemoryRecord | None:
        return self.records.get(key)

    def delete(self, key: str) -> None:
        self.records.pop(key, None)

    def reset_by_prefix(self, prefix: str) -> None:
        for key in list(self.records):
            if key.startswith(prefix):
                self.delete(key)

    def nearest(self, vector: np.ndarray, top_k: int = 1) -> list[tuple[MemoryRecord, float]]:
        scored = [(record, cosine(vector, record.vector)) for record in self.records.values()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def query(self, vector: np.ndarray) -> tuple[MemoryRecord | None, float]:
        nearest = self.nearest(vector, top_k=1)
        if not nearest:
            return None, 0.0
        return nearest[0]

    def __len__(self) -> int:
        return len(self.records)
