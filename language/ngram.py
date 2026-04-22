from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrr.binding import bind, normalize
from hrr.vectors import VectorStore
from memory import AMM


@dataclass
class Prediction:
    token: str | None
    confidence: float
    context_key: str | None


class NGramLanguageMemory:
    """Tiny HRR next-token predictor using bigram contexts as AMM keys."""

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.position_1 = self.store.get_unitary("__POS_1__")
        self.position_2 = self.store.get_unitary("__POS_2__")
        self.memory = AMM()

    def context_vector(self, left: str, right: str) -> np.ndarray:
        return normalize(
            bind(self.position_1, self.store.get(f"tok:{left}"))
            + bind(self.position_2, self.store.get(f"tok:{right}"))
        )

    def context_key(self, left: str, right: str) -> str:
        return f"{left}\t{right}"

    def learn_sequence(self, tokens: list[str], *, cycles: int = 1) -> None:
        for _cycle in range(cycles):
            for idx in range(len(tokens) - 2):
                left, right, next_token = tokens[idx], tokens[idx + 1], tokens[idx + 2]
                self.learn(left, right, next_token)

    def learn(self, left: str, right: str, next_token: str) -> None:
        self.memory.write(
            self.context_key(left, right),
            self.context_vector(left, right),
            {"left": left, "right": right, "next_token": next_token},
        )

    def predict(self, left: str, right: str, *, min_confidence: float = 0.6) -> Prediction:
        record, score = self.memory.query(self.context_vector(left, right))
        if record is None or score < min_confidence:
            return Prediction(token=None, confidence=score, context_key=None)
        return Prediction(
            token=str(record.payload["next_token"]),
            confidence=score,
            context_key=record.key,
        )
