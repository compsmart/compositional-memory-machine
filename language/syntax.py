from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrr.binding import bind, cosine, normalize
from hrr.vectors import VectorStore


@dataclass(frozen=True)
class SyntaxTriple:
    subject: str
    verb: str
    object: str


class SyntaxComposer:
    """Encodes syntactic patterns with shared semantic role-filler structure."""

    PATTERNS = ("active", "passive", "relative", "prepositional", "coordinated")

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.pattern_weight = 1.063
        self.variant_weight = 0.714
        self.store = VectorStore(dim=dim, seed=seed)
        self.role_subject = self.store.get_unitary("__SYN_SUBJECT__")
        self.role_verb = self.store.get_unitary("__SYN_VERB__")
        self.role_object = self.store.get_unitary("__SYN_OBJECT__")
        self.role_pattern = self.store.get_unitary("__SYN_PATTERN__")
        self.role_variant = self.store.get_unitary("__SYN_VARIANT__")

    def encode(self, triple: SyntaxTriple, pattern: str, *, variant: str = "a") -> np.ndarray:
        if pattern not in self.PATTERNS:
            raise ValueError(f"Unknown syntax pattern: {pattern}")
        vector = (
            bind(self.role_subject, self.store.get(f"entity:{triple.subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{triple.verb}"))
            + bind(self.role_object, self.store.get(f"entity:{triple.object}"))
            + self.pattern_weight * bind(self.role_pattern, self.store.get(f"pattern:{pattern}"))
            + self.variant_weight * bind(self.role_variant, self.store.get(f"variant:{pattern}:{variant}"))
        )
        return normalize(vector)

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return cosine(left, right)


def build_syntax_triples(domains: int = 5, triples_per_domain: int = 30) -> list[tuple[str, SyntaxTriple]]:
    rows: list[tuple[str, SyntaxTriple]] = []
    for domain_idx in range(domains):
        domain = f"domain_{domain_idx}"
        for idx in range(triples_per_domain):
            rows.append(
                (
                    domain,
                    SyntaxTriple(
                        subject=f"{domain}_subject_{idx}",
                        verb=f"{domain}_verb_{idx % 6}",
                        object=f"{domain}_object_{idx}",
                    ),
                )
            )
    return rows
