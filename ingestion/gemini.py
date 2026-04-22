from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from factgraph import FactGraph
from hrr.datasets import fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory.amm import AMM


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
        extractor: GeminiExtractor | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.memory = memory
        self.factgraph = factgraph
        self.extractor = extractor or GeminiExtractor()
        self.min_confidence = min_confidence

    def ingest_text(self, text: str, *, source: str = "text", domain: str = "real_text") -> IngestionResult:
        pass1, pass2 = self.extractor.extract(text, source=source)
        facts = self._deduplicate([*pass1.facts, *pass2.facts])
        written = 0
        for fact in facts:
            if fact.confidence < self.min_confidence:
                continue
            relation = self._canonical_relation(fact.relation)
            svo = SVOFact(
                subject=self._clean_slot(fact.subject),
                verb=relation,
                object=self._clean_slot(fact.object),
            )
            key = fact_key(domain, svo)
            vector = self.encoder.encode_fact(svo)
            payload = {
                "domain": domain,
                "subject": svo.subject,
                "verb": svo.verb,
                "object": svo.object,
                "confidence": fact.confidence,
                "kind": fact.kind,
                "source": fact.source or source,
            }
            self.memory.write(key, vector, payload)
            self.factgraph.write(svo.subject, svo.verb, svo.object)
            written += 1
        return IngestionResult(
            facts=facts,
            pass1_count=len(pass1.facts),
            pass2_count=len(pass2.facts),
            estimated_fact_count=pass1.estimated_fact_count,
            written_facts=written,
        )

    def _deduplicate(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        seen: dict[tuple[str, str, str], ExtractedFact] = {}
        for fact in facts:
            key = (
                self._clean_slot(fact.subject).lower(),
                self._canonical_relation(fact.relation),
                self._clean_slot(fact.object).lower(),
            )
            existing = seen.get(key)
            if existing is None or fact.confidence > existing.confidence:
                seen[key] = fact
        return list(seen.values())

    @staticmethod
    def _canonical_relation(value: str) -> str:
        relation = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return relation or "related_to"

    @staticmethod
    def _clean_slot(value: str) -> str:
        return " ".join(value.strip().split())
