from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hrr.binding import bind, cosine, normalize, unbind
from hrr.vectors import VectorStore


ADJECTIVES = tuple(
    [
        "amber",
        "ancient",
        "brisk",
        "calm",
        "crisp",
        "distant",
        "eager",
        "faint",
        "gentle",
        "golden",
        "hidden",
        "icy",
        "jagged",
        "lively",
        "mellow",
        "narrow",
        "quiet",
        "rapid",
        "silver",
        "warm",
    ]
)
NOUNS = tuple(
    [
        "bridge",
        "cedar",
        "cloud",
        "comet",
        "field",
        "forest",
        "garden",
        "harbor",
        "meadow",
        "mirror",
        "mountain",
        "orchard",
        "planet",
        "river",
        "signal",
        "station",
        "stone",
        "temple",
        "thunder",
        "valley",
    ]
)


def make_value_vector(store: VectorStore, adjective: str, noun: str) -> np.ndarray:
    role_adj = store.get_unitary("__ROLE_ADJ__")
    role_noun = store.get_unitary("__ROLE_NOUN__")
    return normalize(bind(role_adj, store.get(f"adj:{adjective}")) + bind(role_noun, store.get(f"noun:{noun}")))


@dataclass(frozen=True)
class DecodedValue:
    adjective: str
    noun: str
    text: str
    strategy: str


@dataclass
class CompositionalValueDecoder:
    store: VectorStore
    adjectives: tuple[str, ...] = ADJECTIVES
    nouns: tuple[str, ...] = NOUNS
    adj_head: np.ndarray | None = None
    noun_head: np.ndarray | None = None

    def fit_linear_head(
        self,
        examples: Iterable[tuple[np.ndarray, str, str]],
        *,
        ridge_alpha: float = 0.1,
    ) -> None:
        rows = list(examples)
        if not rows:
            raise ValueError("examples must not be empty")

        train_x = np.vstack([normalize(np.asarray(vector, dtype=float)) for vector, _adj, _noun in rows])
        train_adj = np.zeros((len(rows), len(self.adjectives)), dtype=float)
        train_noun = np.zeros((len(rows), len(self.nouns)), dtype=float)
        for idx, (_vector, adjective, noun) in enumerate(rows):
            train_adj[idx, self.adjectives.index(adjective)] = 1.0
            train_noun[idx, self.nouns.index(noun)] = 1.0

        gram = train_x.T @ train_x
        identity = np.eye(gram.shape[0], dtype=train_x.dtype)
        self.adj_head = np.linalg.solve(gram + ridge_alpha * identity, train_x.T @ train_adj)
        self.noun_head = np.linalg.solve(gram + ridge_alpha * identity, train_x.T @ train_noun)

    def decode_hrr(self, value_vector: np.ndarray) -> DecodedValue:
        role_adj = self.store.get_unitary("__ROLE_ADJ__")
        role_noun = self.store.get_unitary("__ROLE_NOUN__")
        adj_vectors = [self.store.get(f"adj:{token}") for token in self.adjectives]
        noun_vectors = [self.store.get(f"noun:{token}") for token in self.nouns]

        decoded_adj = unbind(value_vector, role_adj)
        decoded_noun = unbind(value_vector, role_noun)
        pred_adj = self._decode_nearest(decoded_adj, adj_vectors)
        pred_noun = self._decode_nearest(decoded_noun, noun_vectors)
        adjective = self.adjectives[pred_adj]
        noun = self.nouns[pred_noun]
        return DecodedValue(adjective=adjective, noun=noun, text=f"{adjective} {noun}", strategy="hrr_native")

    def decode_linear(self, value_vector: np.ndarray) -> DecodedValue:
        if self.adj_head is None or self.noun_head is None:
            raise ValueError("linear heads have not been fitted")

        probe = normalize(np.asarray(value_vector, dtype=float))
        adj_logits = probe @ self.adj_head
        noun_logits = probe @ self.noun_head
        adjective = self.adjectives[int(np.argmax(adj_logits))]
        noun = self.nouns[int(np.argmax(noun_logits))]
        return DecodedValue(adjective=adjective, noun=noun, text=f"{adjective} {noun}", strategy="linear")

    def decode(self, value_vector: np.ndarray, *, strategy: str = "hrr_native") -> DecodedValue:
        if strategy == "hrr_native":
            return self.decode_hrr(value_vector)
        if strategy == "linear":
            return self.decode_linear(value_vector)
        raise ValueError(f"unknown compositional decode strategy: {strategy}")

    @staticmethod
    def _decode_nearest(vector: np.ndarray, candidates: list[np.ndarray]) -> int:
        scores = [cosine(vector, candidate) for candidate in candidates]
        return int(np.argmax(scores))
