from __future__ import annotations

from factgraph import FactGraph
from generation import FrozenGeneratorAdapter
from hrr import SVOEncoder
from ingestion import TextIngestionPipeline
from memory import AMM
from query import QueryEngine


SAMPLE_TEXT = """
Ada Lovelace worked with Charles Babbage on the Analytical Engine.
Lovelace published notes about the machine in 1843.
Her notes described an algorithm for computing Bernoulli numbers.
The Analytical Engine was a proposed mechanical general-purpose computer.
"""


def main() -> None:
    encoder = SVOEncoder(dim=2048, seed=7)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph)
    result = pipeline.ingest_text(SAMPLE_TEXT, source="ada_fixture", domain="history")

    print(
        {
            "pass1_count": result.pass1_count,
            "pass2_count": result.pass2_count,
            "deduplicated": len(result.facts),
            "written_facts": result.written_facts,
            "enrichment": result.enrichment,
        }
    )
    for fact in result.facts:
        print(f"{fact.kind}: {fact.subject} --{fact.relation}--> {fact.object} ({fact.confidence})")

    query = QueryEngine(encoder=encoder, memory=memory)
    generator = FrozenGeneratorAdapter()
    probe = query.ask_svo("Ada Lovelace", "worked_with", "Charles Babbage")
    print("Probe:", probe)
    print("Answer:", generator.answer("Who did Ada Lovelace work with?", probe))


if __name__ == "__main__":
    main()
