from __future__ import annotations

from dataclasses import dataclass, replace

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
    confidence: float = 1.0
    kind: str = "explicit"
    source: str = "episodic"
    source_id: str = ""
    source_chunk_id: str = ""
    excerpt: str = ""
    char_start: int | None = None
    char_end: int | None = None
    sentence_index: int | None = None

    @property
    def key(self) -> str:
        return f"s{self.session}:t{self.turn}:{self.subject}:{self.relation}:{self.object}"


@dataclass(frozen=True)
class ConversationTurn:
    session: int
    turn: int
    speaker: str
    utterance: str
    intent: str = ""
    facts: tuple[ConversationFact, ...] = ()


class EpisodicMemory:
    """Persistent conversation memory with current graph state and AMM evidence."""

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        from ingestion import RelationRegistry

        self.encoder = SVOEncoder(dim=dim, seed=seed)
        self.memory = AMM()
        self.graph = FactGraph()
        self.chunk_memory = ChunkedKGMemory(chunk_size=8)
        self.relation_registry = RelationRegistry()
        self.history: list[ConversationFact] = []

    def state_fact(self, fact: ConversationFact) -> None:
        canonical_fact = self._canonicalize_fact(fact)
        payload = self._payload_for_fact(fact, canonical_fact=canonical_fact, state_token="observed")
        vector = self.encoder.encode_temporal_fact(
            TemporalFact(
                canonical_fact.subject,
                canonical_fact.relation,
                canonical_fact.object,
                time_token=self._time_token(canonical_fact),
                state_token="observed",
            )
        )
        key = self._fact_key(canonical_fact)
        chunk_record = self.chunk_memory.write_fact(
            key,
            self._domain_for_fact(canonical_fact),
            SVOFact(canonical_fact.subject, canonical_fact.relation, canonical_fact.object),
            vector,
            payload,
        )
        payload["chunk_id"] = chunk_record.chunk_id
        self.memory.write(key, vector, payload)
        self.graph.write(
            canonical_fact.subject,
            canonical_fact.relation,
            canonical_fact.object,
            provenance=payload["provenance"],
        )
        self.history.append(canonical_fact)

    def revise_fact(self, fact: ConversationFact) -> None:
        canonical_fact = self._canonicalize_fact(fact)
        payload = self._payload_for_fact(fact, canonical_fact=canonical_fact, state_token="revised")
        payload["revision"] = True
        payload["provenance"]["revision"] = True
        vector = self.encoder.encode_temporal_fact(
            TemporalFact(
                canonical_fact.subject,
                canonical_fact.relation,
                canonical_fact.object,
                time_token=self._time_token(canonical_fact),
                state_token="revised",
            )
        )
        key = self._fact_key(canonical_fact)
        chunk_record = self.chunk_memory.write_fact(
            key,
            self._domain_for_fact(canonical_fact),
            SVOFact(canonical_fact.subject, canonical_fact.relation, canonical_fact.object),
            vector,
            payload,
        )
        payload["chunk_id"] = chunk_record.chunk_id
        self.memory.write(key, vector, payload)
        self.graph.revise(
            canonical_fact.subject,
            canonical_fact.relation,
            canonical_fact.object,
            provenance=payload["provenance"],
        )
        self.history.append(canonical_fact)

    def ingest_turn(self, turn: ConversationTurn) -> list[ConversationFact]:
        emitted: list[ConversationFact] = []
        if turn.speaker:
            emitted.append(
                ConversationFact(
                    session=turn.session,
                    turn=turn.turn,
                    subject=self._turn_subject(turn.session, turn.turn),
                    relation="speaker",
                    object=turn.speaker,
                    source="episodic_turn",
                    source_id=f"s{turn.session}:t{turn.turn}",
                    excerpt=turn.utterance,
                )
            )
        if turn.intent:
            emitted.append(
                ConversationFact(
                    session=turn.session,
                    turn=turn.turn,
                    subject=self._turn_subject(turn.session, turn.turn),
                    relation="intent",
                    object=turn.intent,
                    source="episodic_turn",
                    source_id=f"s{turn.session}:t{turn.turn}",
                    excerpt=turn.utterance,
                )
            )
        emitted.extend(turn.facts)
        for fact in emitted:
            self.state_fact(fact)
        return emitted

    def ingest_episode(self, turns: list[ConversationTurn]) -> list[ConversationFact]:
        emitted: list[ConversationFact] = []
        for turn in turns:
            emitted.extend(self.ingest_turn(turn))
        return emitted

    def recall_current(self, subject: str, relation: str) -> str | None:
        return self.graph.read(self._clean_slot(subject), self._canonical_relation(relation))

    def recall_history(self, subject: str, relation: str) -> list[str]:
        return [event.target for event in self.graph.history(self._clean_slot(subject), self._canonical_relation(relation))]

    def recall_at_turn(self, subject: str, relation: str, *, revision: int) -> str | None:
        return self.graph.read_at_revision(self._clean_slot(subject), self._canonical_relation(relation), revision)

    def current_truth(self, subject: str, relation: str) -> dict[str, object]:
        return self.graph.evidence_summary(self._clean_slot(subject), self._canonical_relation(relation))

    def claim_history(self, subject: str, relation: str) -> list[dict[str, object]]:
        return [
            {
                "target": event.target,
                "revision": event.revision,
                "status": event.status,
                "provenance": event.provenance,
            }
            for event in self.graph.history(self._clean_slot(subject), self._canonical_relation(relation))
        ]

    def recall_evidence(self, fact: ConversationFact, *, min_confidence: float = 0.9) -> bool:
        canonical_fact = self._canonicalize_fact(fact)
        candidates = [
            self.encoder.encode_temporal_fact(
                TemporalFact(
                    canonical_fact.subject,
                    canonical_fact.relation,
                    canonical_fact.object,
                    time_token=self._time_token(canonical_fact),
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
        return (
            best_record is not None
            and best_record.key == self._fact_key(canonical_fact)
            and best_confidence >= min_confidence
        )

    def _canonicalize_fact(self, fact: ConversationFact) -> ConversationFact:
        return replace(
            fact,
            subject=self._clean_slot(fact.subject),
            relation=self._canonical_relation(fact.relation),
            object=self._clean_slot(fact.object),
        )

    def _canonical_relation(self, relation: str) -> str:
        return self.relation_registry.normalize(relation).canonical

    @staticmethod
    def _clean_slot(value: str) -> str:
        return " ".join(value.strip().split())

    @staticmethod
    def _time_token(fact: ConversationFact) -> str:
        return f"s{fact.session}:t{fact.turn}"

    @staticmethod
    def _domain_for_fact(fact: ConversationFact) -> str:
        return f"session{fact.session}"

    @staticmethod
    def _turn_subject(session: int, turn: int) -> str:
        return f"s{session}:t{turn}"

    @staticmethod
    def _fact_key(fact: ConversationFact) -> str:
        return f"s{fact.session}:t{fact.turn}:{fact.subject}:{fact.relation}:{fact.object}"

    def _payload_for_fact(
        self,
        fact: ConversationFact,
        *,
        canonical_fact: ConversationFact,
        state_token: str,
    ) -> dict[str, object]:
        normalized = self.relation_registry.normalize(fact.relation)
        source_name = fact.source or "episodic"
        provenance: dict[str, object] = {
            "source": source_name,
            "kind": fact.kind,
            "confidence": fact.confidence,
            "raw_relation": normalized.raw,
            "normalized_relation": normalized.canonical,
            "matched_alias": normalized.matched_alias,
            "state_token": state_token,
        }
        if fact.source_id:
            provenance["source_id"] = fact.source_id
        if fact.source_chunk_id:
            provenance["source_chunk_id"] = fact.source_chunk_id
        if fact.excerpt:
            provenance["excerpt"] = fact.excerpt
        if fact.char_start is not None:
            provenance["char_start"] = fact.char_start
        if fact.char_end is not None:
            provenance["char_end"] = fact.char_end
        if fact.sentence_index is not None:
            provenance["sentence_index"] = fact.sentence_index
        return {
            "session": canonical_fact.session,
            "turn": canonical_fact.turn,
            "subject": canonical_fact.subject,
            "verb": canonical_fact.relation,
            "object": canonical_fact.object,
            "domain": self._domain_for_fact(canonical_fact),
            "source": source_name,
            "kind": fact.kind,
            "confidence": fact.confidence,
            "raw_relation": normalized.raw,
            "normalized_relation": normalized.canonical,
            "matched_alias": normalized.matched_alias,
            "provenance": provenance,
        }

