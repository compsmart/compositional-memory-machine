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
                    relation="worked with",
                    object="Charles Babbage",
                    confidence=0.95,
                    kind="explicit",
                    source=source,
                )
            ],
        )
        pass2 = ExtractionResponse(
            facts=[
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
    assert probe["found"] is True
    assert probe["domain"] == "history"
    assert graph.read("Ada Lovelace", "worked_with") == "Charles Babbage"
