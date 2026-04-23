from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph
from hrr.encoder import SVOEncoder, SVOFact
from ingestion import ExtractedFact, TextIngestionPipeline
from memory import AMM, ChunkedKGMemory
from query import QueryEngine


def _write_fact(
    pipeline: TextIngestionPipeline,
    domain: str,
    fact: SVOFact,
) -> None:
    written = pipeline.write_structured_fact(
        ExtractedFact(
            subject=fact.subject,
            relation=fact.verb,
            object=fact.object,
            confidence=1.0,
            kind="explicit",
            source="exp_chunked_multihop",
            source_id=f"exp_chunked_multihop:{domain}",
        ),
        source="exp_chunked_multihop",
        domain=domain,
    )
    if not written:
        raise RuntimeError(f"failed to write fact for domain={domain}: {fact}")


def run(
    *,
    dim: int = 2048,
    seeds: tuple[int, ...] = (42, 123),
    chunk_sizes: tuple[int, ...] = (2, 3),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    facts = [
        ("alpha", SVOFact("alice", "knows", "bob")),
        ("alpha", SVOFact("bob", "works_with", "carol")),
        ("beta", SVOFact("carol", "guides", "delta")),
        ("beta", SVOFact("delta", "maintains", "engine")),
    ]

    for chunk_size in chunk_sizes:
        for seed in seeds:
            encoder = SVOEncoder(dim=dim, seed=seed)
            memory = AMM()
            graph = FactGraph()
            chunk_memory = ChunkedKGMemory(chunk_size=chunk_size, dim=dim, role_count=4)
            pipeline = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
            query = QueryEngine(
                encoder=encoder,
                memory=memory,
                graph=graph,
                chunk_memory=chunk_memory,
                relation_registry=pipeline.relation_registry,
            )

            for domain, fact in facts:
                _write_fact(pipeline, domain, fact)

            hop2 = query.ask_chain("alice", ["knows", "works_with"])
            hop3 = query.ask_chain("alice", ["knows", "works_with", "guides"])
            hop4 = query.ask_chain("alice", ["knows", "works_with", "guides", "maintains"])

            rows.append(
                {
                    "seed": float(seed),
                    "chunk_size": float(chunk_size),
                    "chunk_count": float(len(chunk_memory.chunks)),
                    "bridge_chunks": float(sum(1 for chunk in chunk_memory.chunks.values() if chunk.bridge_entities)),
                    "hop2_em": float(hop2["found"] and hop2.get("target") == "carol"),
                    "hop3_em": float(hop3["found"] and hop3.get("target") == "delta"),
                    "cross_domain_em": float(hop4["found"] and hop4.get("target") == "engine"),
                    "hop4_confidence": float(hop4["confidence"]),
                    "shared_entity_pool": 1.0,
                    "explicit_chain_construction": 1.0,
                    "capacity_budget": float(chunk_memory.capacity_budget),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[2, 3])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds), chunk_sizes=tuple(args.chunk_sizes)):
        print(row)


if __name__ == "__main__":
    main()
