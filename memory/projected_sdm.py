from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from hrr.binding import cosine, normalize


@dataclass
class ProjectedRecord:
    key: str
    vector: np.ndarray
    payload: dict[str, Any] = field(default_factory=dict)


class ProjectedSDM:
    """Small SDM-style memory with projected binary addresses and cleanup.

    This is intentionally separate from the repo's full-vector AMM. It models
    the failure mode in D-2831/D-2832: high-dimensional HRR keys are compressed
    into a lower-dimensional address space before writes hit shared locations.
    """

    def __init__(
        self,
        *,
        vector_dim: int,
        addr_dim: int = 64,
        n_locations: int = 512,
        k: int = 8,
        write_mode: Literal["sum", "overwrite"] = "sum",
        seed: int = 0,
    ) -> None:
        if k <= 0 or k > n_locations:
            raise ValueError("k must be in [1, n_locations]")
        self.vector_dim = vector_dim
        self.addr_dim = addr_dim
        self.n_locations = n_locations
        self.k = k
        self.write_mode = write_mode
        self.rng = np.random.default_rng(seed)
        self.projection = self.rng.normal(0.0, 1.0, (vector_dim, addr_dim))
        self.locations = self.rng.choice(np.array([-1.0, 1.0]), size=(n_locations, addr_dim))
        self.values = np.zeros((n_locations, vector_dim), dtype=float)
        self.counts = np.zeros(n_locations, dtype=float)
        self.records: dict[str, ProjectedRecord] = {}
        self._keys: list[str] = []
        self._record_locations: dict[str, set[int]] = {}
        self._matrix: np.ndarray | None = None
        self._dirty = False

    def write(self, key: str, vector: np.ndarray, payload: dict[str, Any] | None = None) -> None:
        vector = normalize(vector)
        payload = payload or {}
        indices = self._location_indices(vector)
        if self.write_mode == "overwrite":
            self.values[indices] = vector
            self.counts[indices] = 1.0
        else:
            self.values[indices] += vector
            self.counts[indices] += 1.0
        is_new = key not in self.records
        self.records[key] = ProjectedRecord(key=key, vector=vector, payload=payload.copy())
        if is_new:
            self._keys.append(key)
        self._record_locations[key] = set(int(index) for index in indices)
        self._dirty = True

    def read_vector(self, vector: np.ndarray, *, read_k: int | None = None) -> np.ndarray:
        indices = self._location_indices(vector, k=read_k)
        active = self.counts[indices] > 0
        if not np.any(active):
            return np.zeros(self.vector_dim, dtype=float)
        return normalize(np.sum(self.values[indices[active]], axis=0))

    def query(
        self,
        vector: np.ndarray,
        *,
        cleanup: Literal["global", "address"] = "global",
        read_k: int | None = None,
    ) -> tuple[ProjectedRecord | None, float]:
        scored = self.nearest(vector, cleanup=cleanup, read_k=read_k, top_k=1)
        if not scored:
            return None, 0.0
        return scored[0]

    def nearest(
        self,
        vector: np.ndarray,
        *,
        cleanup: Literal["global", "address"] = "global",
        read_k: int | None = None,
        top_k: int = 1,
    ) -> list[tuple[ProjectedRecord, float]]:
        readout = self.read_vector(vector, read_k=read_k)
        if not self.records or not np.any(readout):
            return []
        candidates = self._candidate_keys(vector, read_k=read_k) if cleanup == "address" else self._keys
        if not candidates:
            return []
        scored = [(self.records[key], cosine(readout, self.records[key].vector)) for key in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def query_many(
        self,
        vectors: np.ndarray,
        *,
        cleanup: Literal["global", "address"] = "global",
        read_k: int | None = None,
    ) -> list[tuple[ProjectedRecord | None, float]]:
        if not self.records:
            return [(None, 0.0) for _vector in vectors]
        if cleanup == "address":
            return [self.query(vector, cleanup="address", read_k=read_k) for vector in vectors]
        matrix = self._record_matrix()
        rows = []
        for vector in vectors:
            rows.append(self.read_vector(vector, read_k=read_k))
        readouts = np.vstack(rows)
        norms = np.linalg.norm(readouts, axis=1)
        valid = norms > 1e-12
        scores = readouts @ matrix.T
        winners = np.argmax(scores, axis=1)
        results: list[tuple[ProjectedRecord | None, float]] = []
        for idx, winner in enumerate(winners):
            if not valid[idx]:
                results.append((None, 0.0))
                continue
            key = self._keys[int(winner)]
            results.append((self.records[key], float(scores[idx, winner])))
        return results

    def _address(self, vector: np.ndarray) -> np.ndarray:
        return np.where(vector @ self.projection >= 0.0, 1.0, -1.0)

    def _location_indices(self, vector: np.ndarray, *, k: int | None = None) -> np.ndarray:
        k = self.k if k is None else k
        if k <= 0 or k > self.n_locations:
            raise ValueError("k must be in [1, n_locations]")
        address = self._address(normalize(vector))
        # For +/-1 signatures, dot product ranking is equivalent to Hamming distance ranking.
        scores = self.locations @ address
        if k == self.n_locations:
            return np.arange(self.n_locations)
        return np.argpartition(scores, -k)[-k:]

    def _record_matrix(self) -> np.ndarray:
        if self._matrix is None or self._dirty:
            self._matrix = np.vstack([self.records[key].vector for key in self._keys])
            self._dirty = False
        return self._matrix

    def _candidate_keys(self, vector: np.ndarray, *, read_k: int | None = None) -> list[str]:
        query_locations = set(int(index) for index in self._location_indices(vector, k=read_k))
        return [
            key
            for key in self._keys
            if self._record_locations.get(key, set()) & query_locations
        ]

    def __len__(self) -> int:
        return len(self.records)
