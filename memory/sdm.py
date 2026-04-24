from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrr.binding import normalize

from .amm import AMM


@dataclass(frozen=True)
class SDMRoute:
    location: int
    entropy: float
    active_locations: int
    weights: np.ndarray


@dataclass(frozen=True)
class SDMQueryResult:
    key: str | None
    score: float
    location: int | None
    routed_location: int
    candidate_locations: tuple[int, ...]
    entropy: float
    active_locations: int


class EntropyGatedSDM:
    """Sparse routed AMM bank with random hard locations and entropy telemetry."""

    def __init__(
        self,
        input_dim: int,
        *,
        addr_dim: int = 64,
        n_locs: int = 64,
        seed: int = 0,
        gate_beta: float = -2.0,
        route_top_k: int = 3,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if addr_dim <= 0:
            raise ValueError("addr_dim must be positive")
        if n_locs <= 0:
            raise ValueError("n_locs must be positive")
        if route_top_k <= 0:
            raise ValueError("route_top_k must be positive")
        self.input_dim = input_dim
        self.addr_dim = addr_dim
        self.n_locs = n_locs
        self.gate_beta = gate_beta
        self.route_top_k = min(route_top_k, n_locs)
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(0.0, 1.0, (input_dim, addr_dim))
        raw_locations = rng.normal(0.0, 1.0, (n_locs, addr_dim))
        self.locations = np.vstack([normalize(row) for row in raw_locations])
        self.memories = [AMM() for _ in range(n_locs)]
        self.location_writes = np.zeros(n_locs, dtype=int)

    def _address(self, vector: np.ndarray) -> np.ndarray:
        return normalize(normalize(vector) @ self.projection)

    def route(self, vector: np.ndarray) -> SDMRoute:
        address = self._address(vector)
        logits = abs(self.gate_beta) * (self.locations @ address)
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights /= np.sum(weights)
        entropy = self._normalized_entropy(weights)
        active_locations = int(np.count_nonzero(weights >= (1.0 / self.n_locs)))
        return SDMRoute(
            location=int(np.argmax(weights)),
            entropy=entropy,
            active_locations=max(active_locations, 1),
            weights=weights,
        )

    def write(self, key: str, vector: np.ndarray, payload: dict[str, Any] | None = None) -> SDMRoute:
        route = self.route(vector)
        stored_payload = (payload or {}).copy()
        stored_payload.update(
            {
                "sdm_location": route.location,
                "sdm_entropy": route.entropy,
                "sdm_active_locations": route.active_locations,
            }
        )
        self.memories[route.location].write(key, vector, stored_payload)
        self.location_writes[route.location] += 1
        return route

    def query(self, vector: np.ndarray) -> SDMQueryResult:
        route = self.route(vector)
        candidate_locations = np.argsort(route.weights)[::-1][: self.route_top_k]
        best_record = None
        best_score = 0.0
        best_location = route.location
        for location in candidate_locations:
            record, score = self.memories[int(location)].query(vector)
            if record is None or score < best_score:
                continue
            best_record = record
            best_score = score
            best_location = int(location)
        return SDMQueryResult(
            key=None if best_record is None else best_record.key,
            score=best_score,
            location=best_location if best_record is not None else route.location,
            routed_location=route.location,
            candidate_locations=tuple(int(location) for location in candidate_locations),
            entropy=route.entropy,
            active_locations=route.active_locations,
        )

    def approx_memory_mb(self) -> float:
        bytes_projection = self.projection.size * 8
        bytes_locations = self.locations.size * 8
        vector_count = sum(len(memory.records) for memory in self.memories)
        bytes_records = vector_count * self.input_dim * 8
        return float((bytes_projection + bytes_locations + bytes_records) / (1024 * 1024))

    @staticmethod
    def _normalized_entropy(weights: np.ndarray) -> float:
        safe = np.clip(weights, 1e-12, 1.0)
        entropy = -np.sum(safe * np.log(safe))
        max_entropy = np.log(len(weights))
        if max_entropy <= 0.0:
            return 0.0
        return float(entropy / max_entropy)
