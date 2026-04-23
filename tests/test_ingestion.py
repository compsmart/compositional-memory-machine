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
    query = QueryEngine(encoder=encoder, memory=memory, relation_registry=pipeline.relation_registry)
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


def test_relation_registry_learns_unknown_alias_from_repeated_pair_overlap() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph)
    query = QueryEngine(encoder=encoder, memory=memory, relation_registry=pipeline.relation_registry)

    result = pipeline.ingest_facts(
        [
            ExtractedFact(
                subject="Ada Lovelace",
                relation="collaborated with",
                object="Charles Babbage",
                confidence=0.95,
                source="fixture",
            ),
            ExtractedFact(
                subject="Ada Lovelace",
                relation="teamed up with",
                object="Charles Babbage",
                confidence=0.9,
                source="fixture",
            ),
            ExtractedFact(
                subject="Grace Hopper",
                relation="collaborated with",
                object="Howard Aiken",
                confidence=0.95,
                source="fixture",
            ),
            ExtractedFact(
                subject="Grace Hopper",
                relation="teamed up with",
                object="Howard Aiken",
                confidence=0.9,
                source="fixture",
            ),
        ],
        source="fixture",
        domain="history",
    )

    assert result.written_facts == 2
    assert result.relation_stats["learned_alias_labels"] == 1
    assert result.relation_stats["learned_alias_examples"] == ["teamed_up_with"]
    assert result.relation_stats["unresolved_relation_labels"] == 0
    assert result.relation_stats["unresolved_relation_examples"] == []
    assert pipeline.relation_registry.normalize("teamed up with").canonical == "worked_with"

    probe = query.ask_svo("Grace Hopper", "teamed up with", "Howard Aiken")
    assert probe["found"] is True
    assert probe["verb"] == "worked_with"

    record = memory.get("history:Grace Hopper:worked_with:Howard Aiken")
    assert record is not None
    assert record.payload["raw_relation"] == "collaborated with"
    assert record.payload["resolution_source"] == "seed"
    assert record.payload["provenance"]["resolution_source"] == "seed"


def test_relation_registry_keeps_single_overlap_as_candidate_before_promotion() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph)

    result = pipeline.ingest_facts(
        [
            ExtractedFact(
                subject="Ada Lovelace",
                relation="collaborated with",
                object="Charles Babbage",
                confidence=0.95,
                source="fixture",
            ),
            ExtractedFact(
                subject="Ada Lovelace",
                relation="teamed up with",
                object="Charles Babbage",
                confidence=0.9,
                source="fixture",
            ),
        ],
        source="fixture",
        domain="history",
    )

    assert result.written_facts == 2
    assert result.relation_stats["learned_alias_labels"] == 0
    assert result.relation_stats["unresolved_relation_labels"] == 1
    assert result.relation_stats["unresolved_relation_examples"] == ["teamed_up_with"]
    assert result.relation_stats["alias_candidates"] == [
        {"alias": "teamed_up_with", "canonical": "worked_with", "support_pairs": 1}
    ]
    assert pipeline.relation_registry.normalize("teamed up with").canonical == "teamed_up_with"


def test_typed_relation_fallback_stays_off_by_default_for_disjoint_unknown_relation() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph)

    result = pipeline.ingest_facts(
        [
            ExtractedFact(
                subject="Ada Lovelace",
                relation="collaborated with",
                object="Charles Babbage",
                confidence=0.95,
                source="fixture",
                excerpt="Ada Lovelace and Charles Babbage worked on a research lab prototype project.",
            ),
            ExtractedFact(
                subject="Grace Hopper",
                relation="described",
                object="Howard Aiken",
                confidence=0.95,
                source="fixture",
                excerpt="Grace Hopper described a formal architecture pattern for Howard Aiken.",
            ),
            ExtractedFact(
                subject="Barbara Liskov",
                relation="teamed up with",
                object="John McCarthy",
                confidence=0.92,
                source="fixture",
                excerpt="Barbara Liskov and John McCarthy joined the same research lab prototype project.",
            ),
        ],
        source="fixture",
        domain="research",
    )

    assert result.relation_stats["typed_fallback_hits"] == 0
    assert result.relation_stats["unresolved_relation_examples"] == ["teamed_up_with"]
    assert pipeline.relation_registry.normalize("teamed up with").canonical == "teamed_up_with"
    assert graph.read("Barbara Liskov", "worked_with") is None
    assert graph.read("Barbara Liskov", "teamed_up_with") == "John McCarthy"


def test_typed_relation_fallback_maps_disjoint_unknown_relation_when_enabled() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(
        encoder,
        memory,
        graph,
        enable_typed_relation_fallback=True,
    )
    query = QueryEngine(encoder=encoder, memory=memory, relation_registry=pipeline.relation_registry)

    result = pipeline.ingest_facts(
        [
            ExtractedFact(
                subject="Ada Lovelace",
                relation="collaborated with",
                object="Charles Babbage",
                confidence=0.95,
                source="fixture",
                excerpt="Ada Lovelace and Charles Babbage worked on a research lab prototype project.",
            ),
            ExtractedFact(
                subject="Grace Hopper",
                relation="worked on with",
                object="Howard Aiken",
                confidence=0.95,
                source="fixture",
                excerpt="Grace Hopper and Howard Aiken shared a research team prototype effort in the lab.",
            ),
            ExtractedFact(
                subject="Linus Torvalds",
                relation="described",
                object="a distributed protocol",
                confidence=0.95,
                source="fixture",
                excerpt="Linus Torvalds described a distributed protocol in a technical design note.",
            ),
            ExtractedFact(
                subject="Leslie Lamport",
                relation="published notes about",
                object="a proof outline",
                confidence=0.95,
                source="fixture",
                excerpt="Leslie Lamport published notes about a proof outline for the systems workshop.",
            ),
            ExtractedFact(
                subject="Barbara Liskov",
                relation="teamed up with",
                object="John McCarthy",
                confidence=0.92,
                source="fixture",
                excerpt="Barbara Liskov and John McCarthy joined the same research lab prototype project.",
            ),
        ],
        source="fixture",
        domain="research",
    )

    assert result.relation_stats["typed_fallback_hits"] == 1
    assert result.relation_stats["unresolved_relation_examples"] == []
    normalized = pipeline.relation_registry.normalize("teamed up with")
    assert normalized.canonical == "worked_with"
    assert normalized.resolution_source == "typed_fallback"

    record = memory.get("research:Barbara Liskov:worked_with:John McCarthy")
    assert record is not None
    assert record.payload["raw_relation"] == "teamed up with"
    assert record.payload["resolution_source"] == "typed_fallback"
    assert graph.read("Barbara Liskov", "worked_with") == "John McCarthy"

    probe = query.ask_svo("Barbara Liskov", "teamed up with", "John McCarthy")
    assert probe["found"] is True
    assert probe["verb"] == "worked_with"
