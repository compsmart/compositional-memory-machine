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


def test_factgraph_preserves_history_for_revisions() -> None:
    graph = FactGraph()
    graph.write("doctor", "works_at", "clinic")
    graph.revise("doctor", "works_at", "hospital")

    history = graph.history("doctor", "works_at")

    assert [event.target for event in history] == ["clinic", "hospital"]
    assert graph.read_at_revision("doctor", "works_at", history[0].revision) == "clinic"
