from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrr.binding import normalize


@dataclass(frozen=True)
class ProjectedQueryResult:
    key: str | None
    score: float
    candidate_count: int
    expected_in_candidates: bool | None = None


class ProjectedAddressIndex:
    """Random-hyperplane address router followed by cosine scoring."""

    def __init__(
        self,
        dim: int,
        addr_dim: int,
        *,
        seed: int = 0,
        radius_fraction: float = 0.35,
    ) -> None:
        if addr_dim <= 0:
            raise ValueError("addr_dim must be positive")
        if not 0.0 <= radius_fraction <= 1.0:
            raise ValueError("radius_fraction must be between 0 and 1")
        self.dim = dim
        self.addr_dim = addr_dim
        self.radius_fraction = radius_fraction
        self.radius = max(1, int(addr_dim * radius_fraction))
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(0.0, 1.0, (dim, addr_dim))
        self.keys: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.matrix: np.ndarray | None = None
        self.signatures: np.ndarray | None = None

    def build(self, rows: list[tuple[str, np.ndarray, dict[str, Any] | None]]) -> None:
        self.keys = [key for key, _vector, _payload in rows]
        self.payloads = [payload or {} for _key, _vector, payload in rows]
        self.matrix = np.vstack([normalize(vector) for _key, vector, _payload in rows])
        self.signatures = self.matrix @ self.projection >= 0.0

    def query(self, vector: np.ndarray, *, expected_key: str | None = None) -> ProjectedQueryResult:
        if self.matrix is None or self.signatures is None:
            raise ValueError("index has not been built")

        probe = normalize(vector)
        probe_signature = probe @ self.projection >= 0.0
        hamming = np.count_nonzero(self.signatures != probe_signature, axis=1)
        candidates = np.flatnonzero(hamming <= self.radius)
        expected_in_candidates = None
        if expected_key is not None:
            expected_in_candidates = any(self.keys[idx] == expected_key for idx in candidates)
        if len(candidates) == 0:
            return ProjectedQueryResult(
                key=None,
                score=0.0,
                candidate_count=0,
                expected_in_candidates=expected_in_candidates,
            )

        scores = self.matrix[candidates] @ probe
        winner_offset = int(np.argmax(scores))
        winner_idx = int(candidates[winner_offset])
        return ProjectedQueryResult(
            key=self.keys[winner_idx],
            score=float(scores[winner_offset]),
            candidate_count=int(len(candidates)),
            expected_in_candidates=expected_in_candidates,
        )

