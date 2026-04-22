from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binding import bind, normalize
from .vectors import VectorStore


@dataclass(frozen=True)
class SVOFact:
    subject: str
    verb: str
    object: str


class SVOEncoder:
    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.role_subject = self.store.get_unitary("__ROLE_SUBJECT__")
        self.role_verb = self.store.get_unitary("__ROLE_VERB__")
        self.role_object = self.store.get_unitary("__ROLE_OBJECT__")

    def encode(self, subject: str, verb: str, object_: str) -> np.ndarray:
        sentence = (
            bind(self.role_subject, self.store.get(f"subj:{subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{verb}"))
            + bind(self.role_object, self.store.get(f"obj:{object_}"))
        )
        return normalize(sentence)

    def encode_fact(self, fact: SVOFact) -> np.ndarray:
        return self.encode(fact.subject, fact.verb, fact.object)
