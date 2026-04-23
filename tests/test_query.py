from __future__ import annotations

from factgraph import FactGraph
from hrr.datasets import fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory import AMM, ChunkedKGMemory
from query import QueryEngine


def _write(encoder: SVOEncoder, memory: AMM, chunk_memory: ChunkedKGMemory, graph: FactGraph, domain: str, fact: SVOFact) -> None:
    key = fact_key(domain, fact)
    vector = encoder.encode_fact(fact)
    payload = {"domain": domain, "subject": fact.subject, "verb": fact.verb, "object": fact.object}
    chunk_record = chunk_memory.write_fact(key, domain, fact, vector, payload)
    payload["chunk_id"] = chunk_record.chunk_id
    memory.write(key, vector, payload)
    graph.write(fact.subject, fact.verb, fact.object)


def test_query_engine_chain_returns_structured_path() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(chunk_size=2)
    query = QueryEngine(encoder=encoder, memory=memory, graph=graph, chunk_memory=chunk_memory)

    _write(encoder, memory, chunk_memory, graph, "alpha", SVOFact("alice", "knows", "bob"))
    _write(encoder, memory, chunk_memory, graph, "alpha", SVOFact("bob", "works_with", "carol"))
    _write(encoder, memory, chunk_memory, graph, "beta", SVOFact("carol", "guides", "delta"))

    result = query.ask_chain("alice", ["knows", "works_with", "guides"])

    assert result["found"] is True
    assert result["target"] == "delta"
    assert result["path"] == ["alice", "bob", "carol", "delta"]
    assert len(result["steps"]) == 3
    assert all(step["chunk_id"] is not None for step in result["steps"])


def test_query_engine_chain_refuses_missing_path() -> None:
    encoder = SVOEncoder(dim=512, seed=0)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(chunk_size=2)
    query = QueryEngine(encoder=encoder, memory=memory, graph=graph, chunk_memory=chunk_memory)

    _write(encoder, memory, chunk_memory, graph, "alpha", SVOFact("alice", "knows", "bob"))

    result = query.ask_chain("alice", ["knows", "works_with"])

    assert result["found"] is False
    assert result["failed_hop"] == 2
