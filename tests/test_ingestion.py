from __future__ import annotations

from factgraph import FactGraph
from hrr import SVOEncoder
from ingestion import ExtractedFact, ExtractionResponse, GeminiExtractor, TextIngestionPipeline
from memory import AMM
from query import QueryEngine


class FakeExtractor(GeminiExtractor):
    def extract(self, text: str, *, source: str = "") -> tuple[ExtractionResponse, ExtractionResponse]:
        pass1 = ExtractionResponse(
            estimated_fact_count=2,
            facts=[
                ExtractedFact(
                    subject="Ada Lovelace",
                    relation="collaborated with",
                    object="Charles Babbage",
                    confidence=0.95,
                    kind="explicit",
                    source=source,
                    source_id="fixture-pass1",
                    excerpt="Ada Lovelace collaborated with Charles Babbage.",
                    char_start=0,
                    char_end=46,
                    sentence_index=0,
                )
            ],
        )
        pass2 = ExtractionResponse(
            facts=[
                ExtractedFact(
                    subject="Ada Lovelace",
                    relation="worked on with",
                    object="Charles Babbage",
                    confidence=0.6,
                    kind="missed",
                    source=source,
                ),
                ExtractedFact(
                    subject="Ada Lovelace",
                    relation="described",
                    object="an algorithm for Bernoulli numbers",
                    confidence=0.8,
                    kind="missed",
                    source=source,
                )
            ]
        )
        return pass1, pass2


def test_text_ingestion_writes_extracted_facts_to_memory_and_graph() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph, extractor=FakeExtractor())

    result = pipeline.ingest_text("Ada text", source="fixture", domain="history")
    query = QueryEngine(encoder=encoder, memory=memory)
    probe = query.ask_svo("Ada Lovelace", "worked_with", "Charles Babbage")

    assert result.written_facts == 2
    assert result.relation_stats["alias_hits"] == 1
    assert result.relation_stats["unresolved_relation_labels"] == 0
    assert result.relation_stats["unresolved_relation_examples"] == []
    assert probe["found"] is True
    assert probe["domain"] == "history"
    assert probe["verb"] == "worked_with"
    assert graph.read("Ada Lovelace", "worked_with") == "Charles Babbage"
    record = memory.get("history:Ada Lovelace:worked_with:Charles Babbage")
    assert record is not None
    assert record.payload["raw_relation"] == "collaborated with"
    assert (
        record.payload["provenance"]["source"] == "fixture"
    )
    assert record.payload["provenance"]["source_id"] == "fixture-pass1"
    assert record.payload["provenance"]["excerpt"] == "Ada Lovelace collaborated with Charles Babbage."
    assert record.payload["provenance"]["char_start"] == 0
    assert record.payload["provenance"]["char_end"] == 46
    assert record.payload["provenance"]["sentence_index"] == 0
    assert record.payload["provenance"]["matched_alias"] is True
    assert record.payload["normalized_relation"] == "worked_with"


def test_write_structured_fact_tracks_unresolved_relations_separately() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph)

    written = pipeline.write_structured_fact(
        ExtractedFact(
            subject="Ada Lovelace",
            relation="invented",
            object="analytical poetry",
            confidence=0.9,
            source="fixture",
            source_id="structured-1",
            excerpt="Ada Lovelace invented analytical poetry.",
        ),
        source="fixture",
        domain="history",
    )

    assert written is True
    record = memory.get("history:Ada Lovelace:invented:analytical poetry")
    assert record is not None
    assert record.payload["normalized_relation"] == "invented"
    assert record.payload["matched_alias"] is False
    assert record.payload["provenance"]["source_id"] == "structured-1"
    assert graph.read("Ada Lovelace", "invented") == "analytical poetry"

    result = pipeline.ingest_facts(
        [
            ExtractedFact(
                subject="Ada Lovelace",
                relation="invented",
                object="analytical poetry",
                confidence=0.9,
                source="fixture",
            )
        ],
        source="fixture",
        domain="history",
    )

    assert result.relation_stats["unresolved_relation_labels"] == 1
    assert result.relation_stats["unresolved_relation_examples"] == ["invented"]
