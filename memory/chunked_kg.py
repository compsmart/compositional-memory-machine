from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hrr.binding import cosine, normalize
from hrr.encoder import SVOFact

from .amm import AMM, MemoryRecord


@dataclass
class ChunkedFactRecord:
    key: str
    domain: str
    chunk_id: str
    fact: SVOFact
    vector: np.ndarray
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGChunk:
    chunk_id: str
    domain: str
    fact_keys: list[str] = field(default_factory=list)
    entities: set[str] = field(default_factory=set)
    relations: set[str] = field(default_factory=set)
    bridge_entities: set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.fact_keys)

    def summary(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "domain": self.domain,
            "size": self.size,
            "entities": sorted(self.entities),
            "relations": sorted(self.relations),
            "bridge_entities": sorted(self.bridge_entities),
        }


class ChunkedKGMemory:
    """Chunk-oriented fact store that layers local retrieval over flat AMM writes."""

    def __init__(self, *, chunk_size: int = 25) -> None:
        self.chunk_size = chunk_size
        self.chunks: dict[str, KGChunk] = {}
        self.facts: dict[str, ChunkedFactRecord] = {}
        self._tuple_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self._entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self._domain_chunk_ids: dict[str, list[str]] = defaultdict(list)
        self._chunk_memories: dict[str, AMM] = {}

    def write_fact(
        self,
        key: str,
        domain: str,
        fact: SVOFact,
        vector: np.ndarray,
        payload: dict[str, Any] | None = None,
        *,
        chunk_id: str | None = None,
    ) -> ChunkedFactRecord:
        payload = payload or {}
        existing = self.facts.get(key)
        if existing is not None:
            chunk_id = existing.chunk_id
        if chunk_id is None:
            chunk_id = self._assign_chunk(domain, fact)
        chunk = self._ensure_chunk(chunk_id, domain)

        merged_payload = payload.copy()
        merged_payload["chunk_id"] = chunk_id
        merged_payload["domain"] = domain

        if key not in self.facts:
            chunk.fact_keys.append(key)
            tuple_key = (fact.subject, fact.verb, fact.object)
            self._tuple_index[tuple_key].append(key)

        chunk.entities.update({fact.subject, fact.object})
        chunk.relations.add(fact.verb)

        record = ChunkedFactRecord(
            key=key,
            domain=domain,
            chunk_id=chunk_id,
            fact=fact,
            vector=normalize(vector),
            payload=merged_payload,
        )
        self.facts[key] = record

        chunk_memory = self._chunk_memories.setdefault(chunk_id, AMM())
        chunk_memory.write(key, record.vector, merged_payload)

        for entity in (fact.subject, fact.object):
            self._entity_to_chunks[entity].add(chunk_id)
            self._refresh_bridge_entity(entity)
        return record

    def get_fact(self, key: str) -> ChunkedFactRecord | None:
        return self.facts.get(key)

    def lookup(
        self,
        subject: str,
        relation: str,
        object_: str,
        *,
        domain: str | None = None,
        chunk_id: str | None = None,
    ) -> ChunkedFactRecord | None:
        candidates = self._tuple_index.get((subject, relation, object_), [])
        for key in candidates:
            record = self.facts[key]
            if domain is not None and record.domain != domain:
                continue
            if chunk_id is not None and record.chunk_id != chunk_id:
                continue
            return record
        return None

    def chunks_for_entity(self, entity: str, *, domain: str | None = None) -> list[KGChunk]:
        chunk_ids = sorted(self._entity_to_chunks.get(entity, ()))
        items = [self.chunks[chunk_id] for chunk_id in chunk_ids]
        if domain is None:
            return items
        return [chunk for chunk in items if chunk.domain == domain]

    def facts_for_chunk(self, chunk_id: str) -> list[ChunkedFactRecord]:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return []
        return [self.facts[key] for key in chunk.fact_keys]

    def nearest(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 1,
        chunk_id: str | None = None,
        candidate_chunks: list[str] | None = None,
    ) -> list[tuple[ChunkedFactRecord, float]]:
        scored: list[tuple[ChunkedFactRecord, float]] = []
        if chunk_id is not None:
            keys = self.chunks.get(chunk_id, KGChunk(chunk_id=chunk_id, domain="")).fact_keys
        elif candidate_chunks is not None:
            keys = [key for candidate in candidate_chunks for key in self.chunks.get(candidate, KGChunk(candidate, "")).fact_keys]
        else:
            keys = list(self.facts.keys())

        seen: set[str] = set()
        for key in keys:
            if key in seen or key not in self.facts:
                continue
            seen.add(key)
            record = self.facts[key]
            scored.append((record, cosine(vector, record.vector)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def score_fact(self, fact: SVOFact, vector: np.ndarray, *, preferred_chunk: str | None = None) -> float:
        record = self.lookup(fact.subject, fact.verb, fact.object, chunk_id=preferred_chunk)
        if record is None:
            record = self.lookup(fact.subject, fact.verb, fact.object)
        if record is None:
            return 0.0
        return cosine(vector, record.vector)

    def chunk_summaries(self) -> list[dict[str, Any]]:
        return [self.chunks[chunk_id].summary() for chunk_id in sorted(self.chunks)]

    def _assign_chunk(self, domain: str, fact: SVOFact) -> str:
        best_chunk_id: str | None = None
        best_score = -1
        candidate_entities = {fact.subject, fact.object}
        for chunk_id in self._domain_chunk_ids.get(domain, []):
            chunk = self.chunks[chunk_id]
            if chunk.size >= self.chunk_size:
                continue
            overlap = len(candidate_entities & chunk.entities)
            relation_bonus = int(fact.verb in chunk.relations)
            score = overlap * 10 + relation_bonus * 2 - chunk.size
            if score > best_score:
                best_score = score
                best_chunk_id = chunk_id

        if best_chunk_id is not None and best_score > 0:
            return best_chunk_id

        for chunk_id in reversed(self._domain_chunk_ids.get(domain, [])):
            if self.chunks[chunk_id].size < self.chunk_size:
                return chunk_id
        return self._create_chunk(domain)

    def _create_chunk(self, domain: str) -> str:
        chunk_id = f"{domain}:chunk{len(self._domain_chunk_ids[domain])}"
        self._ensure_chunk(chunk_id, domain)
        return chunk_id

    def _ensure_chunk(self, chunk_id: str, domain: str) -> KGChunk:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            chunk = KGChunk(chunk_id=chunk_id, domain=domain)
            self.chunks[chunk_id] = chunk
            self._domain_chunk_ids[domain].append(chunk_id)
        return chunk

    def _refresh_bridge_entity(self, entity: str) -> None:
        chunk_ids = self._entity_to_chunks.get(entity, set())
        is_bridge = len(chunk_ids) > 1
        for chunk_id in chunk_ids:
            chunk = self.chunks[chunk_id]
            if is_bridge:
                chunk.bridge_entities.add(entity)
            else:
                chunk.bridge_entities.discard(entity)
