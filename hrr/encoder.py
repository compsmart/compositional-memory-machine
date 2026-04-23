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


@dataclass(frozen=True)
class TemporalFact:
    subject: str
    verb: str
    object: str
    time_token: str
    state_token: str = "current"


@dataclass(frozen=True)
class PragmaticFact:
    subject: str
    verb: str
    object: str
    sentiment: str
    certainty: str
    negation: str
    modality: str


@dataclass(frozen=True)
class HierarchicalClause:
    subject: str
    verb: str
    object: str
    embedded: "HierarchicalClause | None" = None


class SVOEncoder:
    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.role_subject = self.store.get_unitary("__ROLE_SUBJECT__")
        self.role_verb = self.store.get_unitary("__ROLE_VERB__")
        self.role_object = self.store.get_unitary("__ROLE_OBJECT__")
        self.role_time = self.store.get_unitary("__ROLE_TIME__")
        self.role_state = self.store.get_unitary("__ROLE_STATE__")
        self.role_sentiment = self.store.get_unitary("__ROLE_SENTIMENT__")
        self.role_certainty = self.store.get_unitary("__ROLE_CERTAINTY__")
        self.role_negation = self.store.get_unitary("__ROLE_NEGATION__")
        self.role_modality = self.store.get_unitary("__ROLE_MODALITY__")
        self.role_rel_clause = self.store.get_unitary("__ROLE_REL_CLAUSE__")

    def encode(self, subject: str, verb: str, object_: str) -> np.ndarray:
        sentence = (
            bind(self.role_subject, self.store.get(f"subj:{subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{verb}"))
            + bind(self.role_object, self.store.get(f"obj:{object_}"))
        )
        return normalize(sentence)

    def encode_fact(self, fact: SVOFact) -> np.ndarray:
        return self.encode(fact.subject, fact.verb, fact.object)

    def encode_temporal(
        self,
        subject: str,
        verb: str,
        object_: str,
        *,
        time_token: str | None = None,
        state_token: str | None = None,
    ) -> np.ndarray:
        sentence = (
            bind(self.role_subject, self.store.get(f"subj:{subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{verb}"))
            + bind(self.role_object, self.store.get(f"obj:{object_}"))
        )
        if time_token is not None:
            sentence = sentence + bind(self.role_time, self.store.get(f"time:{time_token}"))
        if state_token is not None:
            sentence = sentence + bind(self.role_state, self.store.get(f"state:{state_token}"))
        return normalize(sentence)

    def encode_temporal_fact(self, fact: TemporalFact) -> np.ndarray:
        return self.encode_temporal(
            fact.subject,
            fact.verb,
            fact.object,
            time_token=fact.time_token,
            state_token=fact.state_token,
        )

    def encode_pragmatic(
        self,
        subject: str,
        verb: str,
        object_: str,
        *,
        sentiment: str,
        certainty: str,
        negation: str,
        modality: str,
    ) -> np.ndarray:
        sentence = (
            bind(self.role_subject, self.store.get(f"subj:{subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{verb}"))
            + bind(self.role_object, self.store.get(f"obj:{object_}"))
            + bind(self.role_sentiment, self.store.get(f"sentiment:{sentiment}"))
            + bind(self.role_certainty, self.store.get(f"certainty:{certainty}"))
            + bind(self.role_negation, self.store.get(f"negation:{negation}"))
            + bind(self.role_modality, self.store.get(f"modality:{modality}"))
        )
        return normalize(sentence)

    def encode_pragmatic_fact(self, fact: PragmaticFact) -> np.ndarray:
        return self.encode_pragmatic(
            fact.subject,
            fact.verb,
            fact.object,
            sentiment=fact.sentiment,
            certainty=fact.certainty,
            negation=fact.negation,
            modality=fact.modality,
        )

    def encode_hierarchical_clause(self, clause: HierarchicalClause) -> np.ndarray:
        sentence = (
            bind(self.role_subject, self.store.get(f"subj:{clause.subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{clause.verb}"))
            + bind(self.role_object, self.store.get(f"obj:{clause.object}"))
        )
        if clause.embedded is not None:
            sentence = sentence + bind(self.role_rel_clause, self.encode_hierarchical_clause(clause.embedded))
        return normalize(sentence)
