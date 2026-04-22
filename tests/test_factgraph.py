from __future__ import annotations

from experiments.exp_revision_chain3 import run
from factgraph import FactGraph


def test_factgraph_revises_single_key() -> None:
    graph = FactGraph()
    graph.write("doctor", "works_at", "clinic")

    graph.revise("doctor", "works_at", "hospital")

    assert graph.read("doctor", "works_at") == "hospital"


def test_chain3_revision_experiment() -> None:
    assert run()["chain3_revision_em"] == 1.0
