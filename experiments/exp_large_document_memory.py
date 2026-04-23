from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph
from hrr import SVOEncoder
from ingestion import ExtractedFact, TextIngestionPipeline
from memory import AMM, ChunkedKGMemory
from query import QueryEngine


def _fact(
    subject: str,
    relation: str,
    object_: str,
    *,
    source_id: str,
    chunk_id: str,
    excerpt: str,
    confidence: float = 0.95,
) -> ExtractedFact:
    return ExtractedFact(
        subject=subject,
        relation=relation,
        object=object_,
        confidence=confidence,
        source="large_doc_fixture",
        source_id=source_id,
        source_chunk_id=chunk_id,
        excerpt=excerpt,
    )


def _corpus() -> list[ExtractedFact]:
    return [
        _fact("Ada Lovelace", "collaborated with", "Charles Babbage", source_id="doc-1", chunk_id="d1:c0", excerpt="Ada Lovelace collaborated with Charles Babbage on the analytical engine notes."),
        _fact("Charles Babbage", "described", "the analytical engine", source_id="doc-1", chunk_id="d1:c1", excerpt="Charles Babbage described the analytical engine in the briefing."),
        _fact("the analytical engine", "used by", "the forecasting lab", source_id="doc-2", chunk_id="d2:c0", excerpt="The analytical engine was used by the forecasting lab."),
        _fact("the forecasting lab", "located in", "London", source_id="doc-2", chunk_id="d2:c1", excerpt="The forecasting lab remained located in London."),
        _fact("Robot Atlas", "location", "hangar", source_id="doc-3", chunk_id="d3:c0", excerpt="Robot Atlas was initially stationed in the hangar."),
        _fact("Robot Atlas", "location", "field", source_id="doc-4", chunk_id="d4:c0", excerpt="A later report corrected Robot Atlas's location to the field."),
        _fact("Meridian Team", "published notes about", "safety checks", source_id="doc-4", chunk_id="d4:c1", excerpt="The Meridian Team published notes about safety checks."),
        _fact("Safety Memo", "described", "an override path", source_id="doc-5", chunk_id="d5:c0", excerpt="The safety memo described an override path for field deployment."),
        _fact("Override Path", "tested in", "wind tunnel", source_id="doc-5", chunk_id="d5:c1", excerpt="The override path was tested in the wind tunnel."),
        _fact("Field Unit", "reported by", "Sensor Grid", source_id="doc-6", chunk_id="d6:c0", excerpt="The field unit was reported by the sensor grid."),
        _fact("Sensor Grid", "managed by", "Ops Desk", source_id="doc-6", chunk_id="d6:c1", excerpt="The sensor grid was managed by the ops desk."),
        _fact("Ops Desk", "located in", "North Wing", source_id="doc-6", chunk_id="d6:c2", excerpt="The ops desk was located in the north wing."),
    ]


def run(*, dim: int = 2048, seeds: tuple[int, ...] = (42, 123)) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    facts = _corpus()
    for seed in seeds:
        encoder = SVOEncoder(dim=dim, seed=seed)
        memory = AMM()
        graph = FactGraph()
        chunk_memory = ChunkedKGMemory(dim=dim, role_count=4)
        pipeline = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
        query = QueryEngine(
            encoder=encoder,
            memory=memory,
            graph=graph,
            chunk_memory=chunk_memory,
            relation_registry=pipeline.relation_registry,
        )

        result = pipeline.ingest_facts(facts, source="large_doc_fixture", domain="large_doc")
        graph.add_evidence(
            "Robot Atlas",
            "location",
            "harbor",
            provenance={
                "source": "large_doc_fixture",
                "source_id": "doc-7",
                "source_chunk_id": "d7:c0",
                "excerpt": "A conflicting witness note placed Robot Atlas near the harbor.",
            },
        )
        graph.add_evidence(
            "Drone Echo",
            "status",
            "offline",
            provenance={
                "source": "large_doc_fixture",
                "source_id": "doc-8",
                "source_chunk_id": "d8:c0",
                "excerpt": "A single incomplete note claimed Drone Echo was offline.",
            },
        )

        recall = query.ask_svo("Ada Lovelace", "worked_with", "Charles Babbage")
        chain = query.ask_chain("Field Unit", ["reported_by", "managed_by", "located_in"])
        truth = query.ask_current_truth("Robot Atlas", "location")
        history = query.ask_history("Robot Atlas", "location")
        refusal = query.ask_current_truth("Drone Echo", "status")

        rows.append(
            {
                "seed": float(seed),
                "written_facts": float(result.written_facts),
                "chunk_count": float(result.relation_stats["chunk_count"]),
                "recall_em": float(recall["found"] is True and recall["verb"] == "worked_with"),
                "chain_em": float(chain["found"] is True and chain["target"] == "North Wing"),
                "current_truth_em": float(truth["target"] == "field"),
                "history_em": float(
                    [event["target"] for event in history["events"] if event["status"] != "evidence"] == ["hangar", "field"]
                ),
                "competing_evidence_em": float(truth["competing_targets"] == ["harbor"]),
                "refusal_em": float(refusal["found"] is False and refusal["unresolved"] is True),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds)):
        print(row)


if __name__ == "__main__":
    main()
