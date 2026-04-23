from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hrr.binding import bind, cosine, normalize
from hrr.vectors import VectorStore


def bind_all(vectors: Iterable[np.ndarray]) -> np.ndarray:
    items = [normalize(np.asarray(vector, dtype=float)) for vector in vectors]
    if not items:
        raise ValueError("bind_all requires at least one vector")
    current = items[0]
    for vector in items[1:]:
        current = bind(current, vector)
    return normalize(current)


def bundle(vectors: Iterable[np.ndarray]) -> np.ndarray:
    items = [np.asarray(vector, dtype=float) for vector in vectors]
    if not items:
        raise ValueError("bundle requires at least one vector")
    return normalize(np.sum(items, axis=0))


def bound_token(store: VectorStore, role: np.ndarray, namespace: str, token: str) -> np.ndarray:
    return bind(role, store.get(f"{namespace}:{token}"))


def nearest_token(vector: np.ndarray, candidates: dict[str, np.ndarray]) -> tuple[str, float]:
    best_token = ""
    best_score = -1.0
    for token, candidate in candidates.items():
        score = cosine(vector, candidate)
        if score > best_score:
            best_token = token
            best_score = score
    return best_token, best_score


def make_similar_vector(
    rng: np.random.Generator,
    dim: int,
    similarity: float,
    *,
    base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left = normalize(base if base is not None else rng.normal(0.0, 1.0, dim))
    noise = rng.normal(0.0, 1.0, dim)
    noise = noise - float(np.dot(noise, left)) * left
    noise = normalize(noise)
    right = normalize(similarity * left + np.sqrt(max(1.0 - similarity**2, 0.0)) * noise)
    return left, right
