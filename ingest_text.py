from __future__ import annotations

import argparse
from pathlib import Path

from factgraph import FactGraph
from hrr import SVOEncoder
from ingestion import TextIngestionPipeline
from memory import AMM
from query import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real text with Gemini and write facts to HRR memory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--text")
    parser.add_argument("--domain", default="real_text")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-subject")
    parser.add_argument("--probe-relation")
    parser.add_argument("--probe-object")
    args = parser.parse_args()

    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    source_name = "inline_text" if args.text is not None else str(args.file)

    encoder = SVOEncoder(dim=args.dim, seed=args.seed)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(encoder, memory, graph)
    result = pipeline.ingest_text(text, source=source_name, domain=args.domain)

    print(
        {
            "pass1_count": result.pass1_count,
            "pass2_count": result.pass2_count,
            "deduplicated": len(result.facts),
            "written_facts": result.written_facts,
            "enrichment": result.enrichment,
            "relation_stats": result.relation_stats,
        }
    )
    for fact in result.facts:
        print(f"{fact.kind}: {fact.subject} --{fact.relation}--> {fact.object} ({fact.confidence})")

    if args.probe_subject and args.probe_relation and args.probe_object:
        query = QueryEngine(encoder=encoder, memory=memory, relation_registry=pipeline.relation_registry)
        print("Probe:", query.ask_svo(args.probe_subject, args.probe_relation, args.probe_object))


if __name__ == "__main__":
    main()
