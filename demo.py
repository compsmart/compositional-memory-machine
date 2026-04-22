from __future__ import annotations

from experiments.common import build_memory
from factgraph import FactGraph
from generation import FrozenGeneratorAdapter
from query import QueryEngine


def main() -> None:
    encoder, memory = build_memory(dim=2048, seed=0, cycles=10)
    query = QueryEngine(encoder=encoder, memory=memory)
    generator = FrozenGeneratorAdapter()

    known = query.ask_svo("doctor", "treats", "patient")
    novel = query.ask_svo("doctor", "monitors", "patient")

    graph = FactGraph()
    graph.write("doctor", "works_at", "clinic")
    before = graph.read("doctor", "works_at")
    graph.revise("doctor", "works_at", "hospital")
    after = graph.read("doctor", "works_at")

    print("Known retrieval:", known)
    print("Known answer:", generator.answer("Who treats the patient?", known))
    print("Novel composition route:", novel)
    print("FactGraph revision:", {"before": before, "after": after})


if __name__ == "__main__":
    main()
