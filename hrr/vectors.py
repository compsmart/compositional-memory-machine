from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .binding import make_unitary, normalize


@dataclass
class VectorStore:
    dim: int
    seed: int = 0
    _vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def get(self, token: str) -> np.ndarray:
        if token not in self._vectors:
            self._vectors[token] = normalize(self.rng.normal(0.0, 1.0, self.dim))
        return self._vectors[token]

    def get_unitary(self, token: str) -> np.ndarray:
        if token not in self._vectors:
            self._vectors[token] = make_unitary(self.rng.normal(0.0, 1.0, self.dim))
        return self._vectors[token]

    def snapshot(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._vectors.items()}
