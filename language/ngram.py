from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hrr.binding import bind, normalize
from hrr.vectors import VectorStore
from memory import AMM


@dataclass
class Prediction:
    token: str | None
    confidence: float
    context_key: str | None
    alternatives: list["RankedContinuation"] = field(default_factory=list)


@dataclass(frozen=True)
class RankedContinuation:
    token: str
    score: float
    probability: float


class NGramLanguageMemory:
    """Tiny HRR next-token predictor using bigram contexts as AMM keys."""

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.position_1 = self.store.get_unitary("__POS_1__")
        self.position_2 = self.store.get_unitary("__POS_2__")
        self.memory = AMM()
        self._context_weights: dict[str, dict[str, float]] = {}
        self._known_tokens: set[str] = set()

    def context_vector(self, left: str, right: str) -> np.ndarray:
        return normalize(
            bind(self.position_1, self.store.get(f"tok:{left}"))
            + bind(self.position_2, self.store.get(f"tok:{right}"))
        )

    def context_key(self, left: str, right: str) -> str:
        return f"{left}\t{right}"

    def learn_sequence(self, tokens: list[str], *, cycles: int = 1, weight: float = 1.0) -> None:
        for _cycle in range(cycles):
            for idx in range(len(tokens) - 2):
                left, right, next_token = tokens[idx], tokens[idx + 1], tokens[idx + 2]
                self.learn(left, right, next_token, weight=weight)

    def learn_distribution(self, left: str, right: str, weighted_tokens: dict[str, float]) -> None:
        for token, weight in weighted_tokens.items():
            self.learn(left, right, token, weight=weight)

    def learn(self, left: str, right: str, next_token: str, *, weight: float = 1.0) -> None:
        context_key = self.context_key(left, right)
        weights = self._context_weights.setdefault(context_key, {})
        weights[next_token] = weights.get(next_token, 0.0) + weight
        self._known_tokens.add(next_token)

        continuation_vector = self._continuation_vector(weights)
        self.memory.write(
            context_key,
            self.context_vector(left, right),
            {
                "left": left,
                "right": right,
                "next_token": max(weights.items(), key=lambda item: item[1])[0],
                "distribution": weights.copy(),
                "continuation_vector": continuation_vector,
            },
        )

    def predict(self, left: str, right: str, *, min_confidence: float = 0.6, top_k: int = 5) -> Prediction:
        record, score = self.memory.query(self.context_vector(left, right))
        if record is None or score < min_confidence:
            return Prediction(token=None, confidence=score, context_key=None, alternatives=[])
        alternatives = self.rank_continuations(record.payload, top_k=top_k)
        if not alternatives:
            return Prediction(token=None, confidence=score, context_key=record.key, alternatives=[])
        top = alternatives[0]
        return Prediction(
            token=top.token,
            confidence=score * top.probability,
            context_key=record.key,
            alternatives=alternatives,
        )

    def rank_continuations(self, payload: dict[str, object], *, top_k: int = 5) -> list[RankedContinuation]:
        distribution = payload.get("distribution")
        continuation_vector = payload.get("continuation_vector")
        if not isinstance(distribution, dict) or continuation_vector is None:
            return []
        total_weight = sum(float(weight) for weight in distribution.values()) or 1.0
        ranked = []
        for token, weight in distribution.items():
            token_vector = self.store.get(f"tok:{token}")
            ranked.append(
                RankedContinuation(
                    token=str(token),
                    score=float(np.dot(continuation_vector, token_vector) / (np.linalg.norm(continuation_vector) * np.linalg.norm(token_vector))),
                    probability=float(weight) / total_weight,
                )
            )
        ranked.sort(key=lambda item: (item.probability, item.score), reverse=True)
        return ranked[:top_k]

    def _continuation_vector(self, weights: dict[str, float]) -> np.ndarray:
        vectors = [float(weight) * self.store.get(f"tok:{token}") for token, weight in weights.items()]
        return normalize(np.sum(vectors, axis=0))
