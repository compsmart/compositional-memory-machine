from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrr.binding import bind, normalize
from hrr.vectors import VectorStore
from memory import AMM, ProjectedSDM


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


class ProjectedNGramLanguageMemory(NGramLanguageMemory):
    """N-gram predictor backed by projected SDM instead of full-vector AMM."""

    def __init__(
        self,
        dim: int = 2048,
        seed: int = 0,
        *,
        addr_dim: int = 512,
        n_locations: int = 2048,
        write_k: int = 8,
        read_k: int = 128,
    ) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.position_1 = self.store.get_unitary("__POS_1__")
        self.position_2 = self.store.get_unitary("__POS_2__")
        self.read_k = read_k
        self.memory = ProjectedSDM(
            vector_dim=dim,
            addr_dim=addr_dim,
            n_locations=n_locations,
            k=write_k,
            write_mode="sum",
            seed=seed + addr_dim,
        )

    def predict(self, left: str, right: str, *, min_confidence: float = 0.35) -> Prediction:
        record, score = self.memory.query(
            self.context_vector(left, right),
            cleanup="address",
            read_k=self.read_k,
        )
        if record is None or score < min_confidence:
            return Prediction(token=None, confidence=score, context_key=None)
        return Prediction(
            token=str(record.payload["next_token"]),
            confidence=score,
            context_key=record.key,
        )


class ProjectedTrigramLanguageMemory:
    """Projected SDM context memory for (left, right, filler) -> next token."""

    def __init__(
        self,
        dim: int = 2048,
        seed: int = 0,
        *,
        addr_dim: int = 512,
        n_locations: int = 2048,
        write_k: int = 8,
        read_k: int = 128,
    ) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.position_1 = self.store.get_unitary("__TRI_POS_1__")
        self.position_2 = self.store.get_unitary("__TRI_POS_2__")
        self.position_3 = self.store.get_unitary("__TRI_POS_3__")
        self.read_k = read_k
        self.memory = ProjectedSDM(
            vector_dim=dim,
            addr_dim=addr_dim,
            n_locations=n_locations,
            k=write_k,
            write_mode="sum",
            seed=seed + addr_dim,
        )

    def context_vector(self, left: str, right: str, filler: str) -> np.ndarray:
        return normalize(
            bind(self.position_1, self.store.get(f"tok:{left}"))
            + bind(self.position_2, self.store.get(f"tok:{right}"))
            + bind(self.position_3, self.store.get(f"tok:{filler}"))
        )

    def context_key(self, left: str, right: str, filler: str) -> str:
        return f"{left}\t{right}\t{filler}"

    def learn(self, left: str, right: str, filler: str, next_token: str) -> None:
        self.memory.write(
            self.context_key(left, right, filler),
            self.context_vector(left, right, filler),
            {"left": left, "right": right, "filler": filler, "next_token": next_token},
        )

    def predict(
        self,
        left: str,
        right: str,
        filler: str,
        *,
        min_confidence: float = 0.0,
        min_margin: float = 0.0,
    ) -> Prediction:
        scored = self.memory.nearest(
            self.context_vector(left, right, filler),
            cleanup="address",
            read_k=self.read_k,
            top_k=2,
        )
        if not scored:
            return Prediction(token=None, confidence=0.0, context_key=None)
        record, score = scored[0]
        margin = score - scored[1][1] if len(scored) > 1 else score
        if score < min_confidence or margin < min_margin:
            return Prediction(token=None, confidence=score, context_key=None)
        return Prediction(
            token=str(record.payload["next_token"]),
            confidence=score,
            context_key=record.key,
        )

    def score(self, left: str, right: str, filler: str) -> dict[str, float | str | None]:
        scored = self.memory.nearest(
            self.context_vector(left, right, filler),
            cleanup="address",
            read_k=self.read_k,
            top_k=2,
        )
        if not scored:
            return {"token": None, "score": 0.0, "margin": 0.0}
        record, score = scored[0]
        margin = score - scored[1][1] if len(scored) > 1 else score
        return {
            "token": str(record.payload["next_token"]),
            "score": score,
            "margin": margin,
        }
