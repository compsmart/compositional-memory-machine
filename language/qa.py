from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrr.binding import bind, normalize, unbind
from hrr.vectors import VectorStore
from memory import AMM


@dataclass(frozen=True)
class QAFact:
    subject: str
    verb: str
    object: str
    domain: str = "general"


@dataclass(frozen=True)
class QAResult:
    answer: str | None
    verb: str | None
    confidence: float
    verb_confidence: float
    object_confidence: float
    fact_key: str | None


class ClosedLoopQAMemory:
    """Retrieves HRR facts and decodes answer roles by unbinding."""

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.role_subject = self.store.get_unitary("__QA_SUBJECT__")
        self.role_verb = self.store.get_unitary("__QA_VERB__")
        self.role_object = self.store.get_unitary("__QA_OBJECT__")
        self.facts = AMM()
        self.verbs = AMM()
        self.objects = AMM()

    def fact_key(self, fact: QAFact) -> str:
        return f"{fact.domain}:{fact.subject}:{fact.verb}"

    def encode_fact(self, fact: QAFact) -> np.ndarray:
        return normalize(
            bind(self.role_subject, self.store.get(f"subject:{fact.subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{fact.verb}"))
            + bind(self.role_object, self.store.get(f"object:{fact.object}"))
        )

    def encode_query(self, subject: str, verb: str) -> np.ndarray:
        return normalize(
            bind(self.role_subject, self.store.get(f"subject:{subject}"))
            + bind(self.role_verb, self.store.get(f"verb:{verb}"))
        )

    def learn(self, fact: QAFact) -> None:
        self.verbs.write(f"verb:{fact.verb}", self.store.get(f"verb:{fact.verb}"), {"verb": fact.verb})
        self.objects.write(
            f"object:{fact.object}",
            self.store.get(f"object:{fact.object}"),
            {"object": fact.object},
        )
        self.facts.write(
            self.fact_key(fact),
            self.encode_fact(fact),
            {
                "domain": fact.domain,
                "subject": fact.subject,
                "verb": fact.verb,
                "object": fact.object,
            },
        )

    def ask(self, subject: str, verb: str, *, min_confidence: float = 0.55) -> QAResult:
        record, fact_score = self.facts.query(self.encode_query(subject, verb))
        if record is None or fact_score < min_confidence:
            return QAResult(
                answer=None,
                verb=None,
                confidence=fact_score,
                verb_confidence=0.0,
                object_confidence=0.0,
                fact_key=None,
            )

        verb_record, verb_score = self.verbs.query(normalize(unbind(record.vector, self.role_verb)))
        object_record, object_score = self.objects.query(normalize(unbind(record.vector, self.role_object)))
        return QAResult(
            answer=str(object_record.payload["object"]) if object_record else None,
            verb=str(verb_record.payload["verb"]) if verb_record else None,
            confidence=fact_score,
            verb_confidence=verb_score,
            object_confidence=object_score,
            fact_key=record.key,
        )


def build_qa_facts(domains: int = 5, facts_per_domain: int = 50) -> list[QAFact]:
    domain_specs = [
        ("medical", "diagnoses", "patient"),
        ("education", "teaches", "student"),
        ("aviation", "flies", "route"),
        ("kitchen", "prepares", "meal"),
        ("legal", "reviews", "evidence"),
    ]
    selected = domain_specs[:domains]
    facts: list[QAFact] = []
    for domain, verb, object_base in selected:
        for idx in range(facts_per_domain):
            facts.append(
                QAFact(
                    domain=domain,
                    subject=f"{domain}_entity_{idx}",
                    verb=verb,
                    object=f"{object_base}_{idx}",
                )
            )
    return facts
