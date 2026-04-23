from __future__ import annotations

from hrr.datasets import fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory import ChunkedKGMemory


def test_chunked_kg_assigns_provenance_and_bridge_entities() -> None:
    encoder = SVOEncoder(dim=512, seed=0)
    store = ChunkedKGMemory(chunk_size=2)

    facts = [
        ("alpha", SVOFact("alice", "knows", "bob")),
        ("alpha", SVOFact("bob", "works_with", "carol")),
        ("alpha", SVOFact("carol", "guides", "delta")),
    ]

    for domain, fact in facts:
        record = store.write_fact(
            fact_key(domain, fact),
            domain,
            fact,
            encoder.encode_fact(fact),
            {"domain": domain, "subject": fact.subject, "verb": fact.verb, "object": fact.object},
        )
        assert record.payload["chunk_id"] == record.chunk_id

    assert len(store.chunks) == 2
    assert store.lookup("alice", "knows", "bob") is not None
    assert any("carol" in chunk.bridge_entities for chunk in store.chunks.values())
