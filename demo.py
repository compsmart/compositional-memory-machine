from __future__ import annotations

from experiments.common import build_memory
from factgraph import FactGraph
from generation import CompositionalValueDecoder, FrozenGeneratorAdapter, make_value_vector
from hrr.vectors import VectorStore
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

    value_store = VectorStore(dim=256, seed=0)
    value_examples = []
    for adjective, noun in (("amber", "bridge"), ("calm", "forest"), ("silver", "signal"), ("warm", "valley")):
        value_examples.append((make_value_vector(value_store, adjective, noun), adjective, noun))
    value_decoder = CompositionalValueDecoder(store=value_store)
    value_decoder.fit_linear_head(value_examples)
    value_adapter = FrozenGeneratorAdapter(value_decoder=value_decoder, value_strategy="linear")
    compositional = value_adapter.answer(
        "What property does entity_demo have?",
        {
            "confidence": 0.99,
            "entity": "entity_demo",
            "value_vector": make_value_vector(value_store, "silver", "signal"),
        },
    )

    print("Known retrieval:", known)
    print("Known answer:", generator.answer("Who treats the patient?", known))
    print("Novel composition route:", novel)
    print("Compositional value answer:", compositional)
    print("FactGraph revision:", {"before": before, "after": after})


if __name__ == "__main__":
    main()
