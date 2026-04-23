from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph
from hrr.datasets import fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory import AMM, ChunkedKGMemory
from query import QueryEngine


def _write_fact(
    encoder: SVOEncoder,
    memory: AMM,
    chunk_memory: ChunkedKGMemory,
    graph: FactGraph,
    domain: str,
    fact: SVOFact,
) -> None:
    key = fact_key(domain, fact)
    vector = encoder.encode_fact(fact)
    payload = {
        "domain": domain,
        "subject": fact.subject,
        "verb": fact.verb,
        "object": fact.object,
        "source": "exp_chunked_multihop",
        "confidence": 1.0,
    }
    chunk_record = chunk_memory.write_fact(key, domain, fact, vector, payload)
    payload["chunk_id"] = chunk_record.chunk_id
    memory.write(key, vector, payload)
    graph.write(fact.subject, fact.verb, fact.object)


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
            chunk_memory = ChunkedKGMemory(chunk_size=chunk_size)
            query = QueryEngine(
                encoder=encoder,
                memory=memory,
                graph=graph,
                chunk_memory=chunk_memory,
            )

            for domain, fact in facts:
                _write_fact(encoder, memory, chunk_memory, graph, domain, fact)

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
