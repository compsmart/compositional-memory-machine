from __future__ import annotations

from dataclasses import dataclass

from factgraph import FactGraph
from hrr.encoder import SVOEncoder
from memory.amm import AMM


@dataclass(frozen=True)
class ConversationFact:
    session: int
    turn: int
    subject: str
    relation: str
    object: str

    @property
    def key(self) -> str:
        return f"s{self.session}:t{self.turn}:{self.subject}:{self.relation}:{self.object}"


class EpisodicMemory:
    """Persistent conversation memory with current graph state and AMM evidence."""

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.encoder = SVOEncoder(dim=dim, seed=seed)
        self.memory = AMM()
        self.graph = FactGraph()
        self.history: list[ConversationFact] = []

    def state_fact(self, fact: ConversationFact) -> None:
        vector = self.encoder.encode(fact.subject, fact.relation, fact.object)
        self.memory.write(
            fact.key,
            vector,
            {
                "session": fact.session,
                "turn": fact.turn,
                "subject": fact.subject,
                "verb": fact.relation,
                "object": fact.object,
            },
        )
        self.graph.write(fact.subject, fact.relation, fact.object)
        self.history.append(fact)

    def revise_fact(self, fact: ConversationFact) -> None:
        vector = self.encoder.encode(fact.subject, fact.relation, fact.object)
        self.memory.write(
            fact.key,
            vector,
            {
                "session": fact.session,
                "turn": fact.turn,
                "subject": fact.subject,
                "verb": fact.relation,
                "object": fact.object,
                "revision": True,
            },
        )
        self.graph.revise(fact.subject, fact.relation, fact.object)
        self.history.append(fact)

    def recall_current(self, subject: str, relation: str) -> str | None:
        return self.graph.read(subject, relation)

    def recall_evidence(self, fact: ConversationFact, *, min_confidence: float = 0.9) -> bool:
        record, confidence = self.memory.query(self.encoder.encode(fact.subject, fact.relation, fact.object))
        return record is not None and record.key == fact.key and confidence >= min_confidence

