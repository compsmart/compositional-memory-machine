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
from .relations import RelationRegistry


class ExtractedFact(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    kind: Literal["explicit", "missed", "implied", "derived"] = "explicit"
    source: str = ""


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
    relation_stats: dict[str, int]

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
        min_confidence: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.memory = memory
        self.factgraph = factgraph
        self.chunk_memory = chunk_memory
        self.extractor = extractor or GeminiExtractor()
        self.relation_registry = relation_registry or RelationRegistry()
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
        facts = self._deduplicate(facts)
        written = 0
        raw_relations: set[str] = set()
        normalized_relations: set[str] = set()
        alias_hits = 0
        for fact in facts:
            if fact.confidence < self.min_confidence:
                continue
            normalized = self.relation_registry.normalize(fact.relation)
            raw_relations.add(normalized.raw)
            normalized_relations.add(normalized.canonical)
            alias_hits += int(normalized.matched_alias)
            svo = SVOFact(
                subject=self._clean_slot(fact.subject),
                verb=normalized.canonical,
                object=self._clean_slot(fact.object),
            )
            key = fact_key(domain, svo)
            vector = self.encoder.encode_fact(svo)
            source_name = fact.source or source
            payload = {
                "domain": domain,
                "subject": svo.subject,
                "verb": svo.verb,
                "object": svo.object,
                "confidence": fact.confidence,
                "kind": fact.kind,
                "source": source_name,
                "raw_relation": normalized.raw,
                "normalized_relation": normalized.canonical,
                "provenance": {
                    "source": source_name,
                    "kind": fact.kind,
                    "confidence": fact.confidence,
                    "raw_relation": normalized.raw,
                },
            }
            if self.chunk_memory is not None:
                chunk_record = self.chunk_memory.write_fact(key, domain, svo, vector, payload)
                payload["chunk_id"] = chunk_record.chunk_id
            self.memory.write(key, vector, payload)
            self.factgraph.write(svo.subject, svo.verb, svo.object)
            written += 1
        return IngestionResult(
            facts=facts,
            pass1_count=pass1_count,
            pass2_count=pass2_count,
            estimated_fact_count=estimated_fact_count,
            written_facts=written,
            relation_stats={
                "raw_relation_labels": len(raw_relations),
                "normalized_relation_labels": len(normalized_relations),
                "alias_hits": alias_hits,
                "unresolved_relation_labels": len(raw_relations) - alias_hits,
                "chunk_count": len(self.chunk_memory.chunks) if self.chunk_memory is not None else 0,
            },
        )

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

    @staticmethod
    def _clean_slot(value: str) -> str:
        return " ".join(value.strip().split())
