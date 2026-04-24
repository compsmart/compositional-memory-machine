from __future__ import annotations

from factgraph import FactGraph
from hrr.encoder import SVOEncoder, SVOFact
from ingestion import ExtractedFact, TextIngestionPipeline
from memory import AMM, ChunkedKGMemory, capacity_budget
from query import QueryEngine


def _write(pipeline: TextIngestionPipeline, domain: str, fact: SVOFact) -> None:
    written = pipeline.write_structured_fact(
        ExtractedFact(
            subject=fact.subject,
            relation=fact.verb,
            object=fact.object,
            confidence=1.0,
            kind="explicit",
            source="test_query",
            source_id=f"test_query:{domain}",
        ),
        source="test_query",
        domain=domain,
    )
    assert written is True


def test_query_engine_chain_returns_structured_path() -> None:
    encoder = SVOEncoder(dim=1024, seed=0)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(chunk_size=2)
    pipeline = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
    query = QueryEngine(
        encoder=encoder,
        memory=memory,
        graph=graph,
        chunk_memory=chunk_memory,
        relation_registry=pipeline.relation_registry,
    )

    _write(pipeline, "alpha", SVOFact("alice", "knows", "bob"))
    _write(pipeline, "alpha", SVOFact("bob", "worked on with", "carol"))
    _write(pipeline, "beta", SVOFact("carol", "guides", "delta"))

    result = query.ask_chain("alice", ["knows", "worked_with", "guides"])
    alias_result = query.ask_chain("alice", ["knows", "worked on with", "guides"])

    assert result["found"] is True
    assert alias_result["found"] is True
    assert result["target"] == "delta"
    assert alias_result["target"] == "delta"
    assert result["path"] == ["alice", "bob", "carol", "delta"]
    assert len(result["steps"]) == 3
    assert all(step["chunk_id"] is not None for step in result["steps"])
    assert graph.read("bob", "worked_with") == "carol"
    record = memory.get("alpha:bob:worked_with:carol")
    assert record is not None
    assert record.payload["raw_relation"] == "worked on with"
    assert record.payload["normalized_relation"] == "worked_with"
    assert record.payload["provenance"]["source"] == "test_query"
    assert record.payload["provenance"]["source_id"] == "test_query:alpha"


def test_query_engine_chain_refuses_missing_path() -> None:
    encoder = SVOEncoder(dim=512, seed=0)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(chunk_size=2)
    pipeline = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
    query = QueryEngine(
        encoder=encoder,
        memory=memory,
        graph=graph,
        chunk_memory=chunk_memory,
        relation_registry=pipeline.relation_registry,
    )

    _write(pipeline, "alpha", SVOFact("alice", "knows", "bob"))

    result = query.ask_chain("alice", ["knows", "works_with"])

    assert result["found"] is False
    assert result["failed_hop"] == 2


def test_query_engine_chain_refuses_when_dimension_budget_is_too_low() -> None:
    encoder = SVOEncoder(dim=256, seed=0)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(dim=256, role_count=4)
    pipeline = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
    query = QueryEngine(
        encoder=encoder,
        memory=memory,
        graph=graph,
        chunk_memory=chunk_memory,
        relation_registry=pipeline.relation_registry,
    )

    _write(pipeline, "alpha", SVOFact("alice", "knows", "bob"))
    _write(pipeline, "alpha", SVOFact("bob", "works_with", "carol"))

    result = query.ask_chain("alice", ["knows", "works_with"])

    assert result["found"] is False
    assert result["budget_exceeded"] is True


def test_query_engine_branching_chain_surfaces_current_and_competing_paths() -> None:
    encoder = SVOEncoder(dim=2048, seed=0)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(dim=2048, role_count=4)
    pipeline = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
    query = QueryEngine(
        encoder=encoder,
        memory=memory,
        graph=graph,
        chunk_memory=chunk_memory,
        relation_registry=pipeline.relation_registry,
    )

    _write(pipeline, "alpha", SVOFact("alice", "knows", "bob"))
    _write(pipeline, "alpha", SVOFact("bob", "works_with", "carol"))
    _write(pipeline, "alpha", SVOFact("bob", "works_with", "dana"))
    _write(pipeline, "alpha", SVOFact("carol", "guides", "delta"))
    _write(pipeline, "alpha", SVOFact("dana", "guides", "echo"))

    result = query.ask_branching_chain("alice", ["knows", "works_with", "guides"])

    assert result["found"] is True
    assert len(result["branches"]) >= 2
    best_branch = result["branches"][0]
    assert best_branch["path"] == ["alice", "bob", "dana", "echo"]
    branch_targets = {tuple(branch["path"]) for branch in result["branches"]}
    assert ("alice", "bob", "carol", "delta") in branch_targets
    assert ("alice", "bob", "dana", "echo") in branch_targets


def test_capacity_budget_uses_large_dimension_revision_for_role4() -> None:
    assert capacity_budget(4096, role_count=4) == 49
    assert capacity_budget(8192, role_count=4) == 50
    assert capacity_budget(16384, role_count=4) == 100
