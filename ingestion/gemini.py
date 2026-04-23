from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from factgraph import FactGraph
from hrr.datasets import fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory.amm import AMM
from memory.chunked_kg import ChunkedKGMemory
from .relations import NormalizedRelation, RelationRegistry


class ExtractedFact(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    kind: Literal["explicit", "missed", "implied", "derived"] = "explicit"
    source: str = ""
    source_id: str = ""
    source_chunk_id: str = ""
    excerpt: str = ""
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    sentence_index: int | None = Field(default=None, ge=0)


class ExtractionResponse(BaseModel):
    estimated_fact_count: int = 0
    facts: list[ExtractedFact] = Field(default_factory=list)


@dataclass(frozen=True)
class IngestionResult:
    facts: list[ExtractedFact]
    pass1_count: int
    pass2_count: int
    estimated_fact_count: int
    written_facts: int
    relation_stats: dict[str, object]

    @property
    def enrichment(self) -> float:
        if self.pass1_count == 0:
            return 0.0
        return len(self.facts) / self.pass1_count


class GeminiExtractor:
    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash-lite",
        google_api_key: str | None = None,
        google_api_key_env: str = "GOOGLE_API_KEY",
    ) -> None:
        self.model = model
        self.google_api_key = google_api_key
        self.google_api_key_env = google_api_key_env

    def extract(self, text: str, *, source: str = "") -> tuple[ExtractionResponse, ExtractionResponse]:
        pass1 = self._pass1_extract(text)
        pass2 = self._pass2_enrich(text, pass1)
        if source:
            for fact in [*pass1.facts, *pass2.facts]:
                fact.source = fact.source or source
        return pass1, pass2

    def _api_key(self) -> str | None:
        return self.google_api_key or os.getenv(self.google_api_key_env) or os.getenv("GEMINI_API_KEY")

    def _client(self):
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                f"{self.google_api_key_env} or GEMINI_API_KEY is required for Gemini extraction"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini extraction") from exc
        return genai.Client(api_key=api_key)

    def _generate_json(self, prompt: str) -> ExtractionResponse:
        client = self._client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ExtractionResponse,
                "temperature": 0.0,
            },
        )
        payload = getattr(response, "parsed", None)
        if isinstance(payload, ExtractionResponse):
            return payload
        text = getattr(response, "text", "")
        try:
            return ExtractionResponse.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Gemini response did not match extraction schema: {text[:500]}") from exc

    def _pass1_extract(self, text: str) -> ExtractionResponse:
        prompt = (
            "You are a strict factual triple extraction system. Extract explicit atomic facts from "
            "the text as subject-relation-object triples. Count the extractable factual claims. "
            "Use short canonical relation labels in snake_case. Do not invent facts. Return JSON "
            "matching the provided schema.\n\n"
            f"TEXT:\n{text[:60000]}"
        )
        return self._generate_json(prompt)

    def _pass2_enrich(self, text: str, pass1: ExtractionResponse) -> ExtractionResponse:
        known = [fact.model_dump() for fact in pass1.facts]
        prompt = (
            "Re-read the text and find triples missed by pass 1, plus only strongly implied or "
            "directly derived facts. Do not repeat existing triples. Use kind='missed', 'implied', "
            "or 'derived'. Use conservative confidence for non-explicit facts. Return JSON matching "
            "the provided schema.\n\n"
            f"PASS_1_ESTIMATED_COUNT: {pass1.estimated_fact_count}\n"
            f"PASS_1_FACTS: {json.dumps(known, ensure_ascii=True)}\n\n"
            f"TEXT:\n{text[:60000]}"
        )
        return self._generate_json(prompt)


class TextIngestionPipeline:
    def __init__(
        self,
        encoder: SVOEncoder,
        memory: AMM,
        factgraph: FactGraph,
        *,
        chunk_memory: ChunkedKGMemory | None = None,
        extractor: GeminiExtractor | None = None,
        relation_registry: RelationRegistry | None = None,
        enable_typed_relation_fallback: bool | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.memory = memory
        self.factgraph = factgraph
        self.chunk_memory = chunk_memory
        self.extractor = extractor or GeminiExtractor()
        self.relation_registry = relation_registry or RelationRegistry(
            enable_typed_fallback=enable_typed_relation_fallback
        )
        self.min_confidence = min_confidence

    def ingest_text(self, text: str, *, source: str = "text", domain: str = "real_text") -> IngestionResult:
        pass1, pass2 = self.extractor.extract(text, source=source)
        return self.ingest_facts(
            [*pass1.facts, *pass2.facts],
            source=source,
            domain=domain,
            pass1_count=len(pass1.facts),
            pass2_count=len(pass2.facts),
            estimated_fact_count=pass1.estimated_fact_count,
        )

    def ingest_facts(
        self,
        facts: list[ExtractedFact],
        *,
        source: str = "structured",
        domain: str = "structured",
        pass1_count: int = 0,
        pass2_count: int = 0,
        estimated_fact_count: int = 0,
    ) -> IngestionResult:
        learned_aliases = self.relation_registry.learn_from_facts(facts, slot_cleaner=self._clean_slot)
        resolved_facts: dict[tuple[str, str, str], tuple[ExtractedFact, NormalizedRelation]] = {}
        for fact in facts:
            normalized = self.relation_registry.normalize_fact(
                fact,
                domain=domain,
                slot_cleaner=self._clean_slot,
            )
            self.relation_registry.observe_resolved_fact(
                fact,
                canonical_relation=normalized.canonical,
                domain=domain,
                slot_cleaner=self._clean_slot,
            )
            key = (
                self._clean_slot(fact.subject).lower(),
                normalized.canonical,
                self._clean_slot(fact.object).lower(),
            )
            existing = resolved_facts.get(key)
            if existing is None or fact.confidence > existing[0].confidence:
                resolved_facts[key] = (fact, normalized)
        written = 0
        raw_relations: set[str] = set()
        normalized_relations: set[str] = set()
        unresolved_relations: set[str] = set()
        alias_hits = 0
        typed_fallback_hits = 0
        deduplicated_facts: list[ExtractedFact] = []
        for fact, normalized in resolved_facts.values():
            deduplicated_facts.append(fact)
            raw_relations.add(normalized.raw)
            normalized_relations.add(normalized.canonical)
            if not normalized.registry_hit:
                unresolved_relations.add(normalized.slug)
            alias_hits += int(normalized.matched_alias)
            typed_fallback_hits += int(normalized.resolution_source == "typed_fallback")
            if self._write_fact(fact, normalized=normalized, source=source, domain=domain):
                written += 1
        return IngestionResult(
            facts=deduplicated_facts,
            pass1_count=pass1_count,
            pass2_count=pass2_count,
            estimated_fact_count=estimated_fact_count,
            written_facts=written,
            relation_stats={
                "raw_relation_labels": len(raw_relations),
                "normalized_relation_labels": len(normalized_relations),
                "alias_hits": alias_hits,
                "unresolved_relation_labels": len(unresolved_relations),
                "unresolved_relation_examples": sorted(unresolved_relations),
                "learned_alias_labels": len(learned_aliases),
                "learned_alias_examples": learned_aliases,
                "alias_candidates": self.relation_registry.candidate_aliases(),
                "alias_proposals": self.relation_registry.proposal_log(),
                "typed_fallback_hits": typed_fallback_hits,
                "chunk_count": len(self.chunk_memory.chunks) if self.chunk_memory is not None else 0,
            },
        )

    def write_structured_fact(
        self,
        fact: ExtractedFact,
        *,
        source: str = "structured",
        domain: str = "structured",
    ) -> bool:
        self.relation_registry.observe_fact(
            fact.subject,
            fact.relation,
            fact.object,
            slot_cleaner=self._clean_slot,
        )
        normalized = self.relation_registry.normalize_fact(
            fact,
            domain=domain,
            slot_cleaner=self._clean_slot,
        )
        self.relation_registry.observe_resolved_fact(
            fact,
            canonical_relation=normalized.canonical,
            domain=domain,
            slot_cleaner=self._clean_slot,
        )
        return self._write_fact(fact, normalized=normalized, source=source, domain=domain)

    def _deduplicate(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        seen: dict[tuple[str, str, str], ExtractedFact] = {}
        for fact in facts:
            key = (
                self._clean_slot(fact.subject).lower(),
                self.relation_registry.normalize(fact.relation).canonical,
                self._clean_slot(fact.object).lower(),
            )
            existing = seen.get(key)
            if existing is None or fact.confidence > existing.confidence:
                seen[key] = fact
        return list(seen.values())

    def _write_fact(
        self,
        fact: ExtractedFact,
        *,
        normalized: NormalizedRelation,
        source: str,
        domain: str,
    ) -> bool:
        if fact.confidence < self.min_confidence:
            return False

        svo = SVOFact(
            subject=self._clean_slot(fact.subject),
            verb=normalized.canonical,
            object=self._clean_slot(fact.object),
        )
        key = fact_key(domain, svo)
        vector = self.encoder.encode_fact(svo)
        source_name = fact.source or source
        payload = self._build_payload(fact, source_name=source_name, domain=domain, svo=svo, normalized=normalized)
        if self.chunk_memory is not None:
            chunk_record = self.chunk_memory.write_fact(key, domain, svo, vector, payload)
            payload["chunk_id"] = chunk_record.chunk_id
        self.memory.write(key, vector, payload)
        self.factgraph.write(svo.subject, svo.verb, svo.object, provenance=payload["provenance"])
        return True

    def _build_payload(
        self,
        fact: ExtractedFact,
        *,
        source_name: str,
        domain: str,
        svo: SVOFact,
        normalized: NormalizedRelation,
    ) -> dict[str, object]:
        provenance: dict[str, object] = {
            "source": source_name,
            "kind": fact.kind,
            "confidence": fact.confidence,
            "raw_relation": normalized.raw,
            "normalized_relation": normalized.canonical,
            "matched_alias": normalized.matched_alias,
            "resolution_source": normalized.resolution_source,
            "relation_evidence_count": normalized.evidence_count,
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

        payload: dict[str, object] = {
            "domain": domain,
            "subject": svo.subject,
            "verb": svo.verb,
            "object": svo.object,
            "confidence": fact.confidence,
            "kind": fact.kind,
            "source": source_name,
            "raw_relation": normalized.raw,
            "normalized_relation": normalized.canonical,
            "matched_alias": normalized.matched_alias,
            "resolution_source": normalized.resolution_source,
            "relation_evidence_count": normalized.evidence_count,
            "provenance": provenance,
        }
        return payload

    @staticmethod
    def _clean_slot(value: str) -> str:
        return " ".join(value.strip().split())
