from __future__ import annotations

from dataclasses import dataclass

from factgraph import FactGraph
from hrr.encoder import SVOEncoder, SVOFact, TemporalFact
from memory.amm import AMM
from memory.chunked_kg import ChunkedKGMemory


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
        self.chunk_memory = ChunkedKGMemory(chunk_size=8)
        self.history: list[ConversationFact] = []

    def state_fact(self, fact: ConversationFact) -> None:
        payload = self._payload_for_fact(fact)
        vector = self.encoder.encode_temporal_fact(
            TemporalFact(
                fact.subject,
                fact.relation,
                fact.object,
                time_token=f"s{fact.session}:t{fact.turn}",
                state_token="observed",
            )
        )
        self.chunk_memory.write_fact(
            fact.key,
            f"session{fact.session}",
            SVOFact(fact.subject, fact.relation, fact.object),
            vector,
            payload,
        )
        self.memory.write(fact.key, vector, payload)
        self.graph.write(fact.subject, fact.relation, fact.object)
        self.history.append(fact)

    def revise_fact(self, fact: ConversationFact) -> None:
        payload = self._payload_for_fact(fact)
        payload["revision"] = True
        vector = self.encoder.encode_temporal_fact(
            TemporalFact(
                fact.subject,
                fact.relation,
                fact.object,
                time_token=f"s{fact.session}:t{fact.turn}",
                state_token="revised",
            )
        )
        self.chunk_memory.write_fact(
            fact.key,
            f"session{fact.session}",
            SVOFact(fact.subject, fact.relation, fact.object),
            vector,
            payload,
        )
        self.memory.write(fact.key, vector, payload)
        self.graph.revise(fact.subject, fact.relation, fact.object)
        self.history.append(fact)

    def recall_current(self, subject: str, relation: str) -> str | None:
        return self.graph.read(subject, relation)

    def recall_history(self, subject: str, relation: str) -> list[str]:
        return [event.target for event in self.graph.history(subject, relation)]

    def recall_at_turn(self, subject: str, relation: str, *, revision: int) -> str | None:
        return self.graph.read_at_revision(subject, relation, revision)

    def recall_evidence(self, fact: ConversationFact, *, min_confidence: float = 0.9) -> bool:
        candidates = [
            self.encoder.encode_temporal_fact(
                TemporalFact(
                    fact.subject,
                    fact.relation,
                    fact.object,
                    time_token=f"s{fact.session}:t{fact.turn}",
                    state_token=state,
                )
            )
            for state in ("observed", "revised")
        ]
        best_record = None
        best_confidence = 0.0
        for vector in candidates:
            record, confidence = self.memory.query(vector)
            if confidence > best_confidence:
                best_record = record
                best_confidence = confidence
        return best_record is not None and best_record.key == fact.key and best_confidence >= min_confidence

    @staticmethod
    def _payload_for_fact(fact: ConversationFact) -> dict[str, object]:
        return {
            "session": fact.session,
            "turn": fact.turn,
            "subject": fact.subject,
            "verb": fact.relation,
            "object": fact.object,
        }

