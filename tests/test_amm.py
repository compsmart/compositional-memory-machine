from __future__ import annotations

from experiments.common import build_memory, evaluate_known
from query import QueryEngine


def test_amm_retrieves_known_svo_facts() -> None:
    encoder, memory = build_memory(dim=2048, seed=0, cycles=3)

    metrics = evaluate_known(encoder, memory)

    assert metrics["top1"] == 1.0


def test_query_engine_returns_structured_payload() -> None:
    encoder, memory = build_memory(dim=2048, seed=0, cycles=1)
    query = QueryEngine(encoder=encoder, memory=memory)

    result = query.ask_svo("doctor", "treats", "patient")

    assert result["found"] is True
    assert result["subject"] == "doctor"
    assert result["verb"] == "treats"
    assert result["object"] == "patient"
    assert result["domain"] == "medical"
