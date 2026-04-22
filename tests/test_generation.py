from __future__ import annotations

from generation import ADJECTIVES, NOUNS, CompositionalValueDecoder, FrozenGeneratorAdapter, make_value_vector
from hrr.vectors import VectorStore


def _value_rows(store: VectorStore, count: int = 60) -> list[tuple[str, str, str, object]]:
    rows: list[tuple[str, str, str, object]] = []
    for idx in range(count):
        entity = f"entity_{idx:03d}"
        adjective = ADJECTIVES[idx % len(ADJECTIVES)]
        noun = NOUNS[(idx * 7) % len(NOUNS)]
        rows.append((entity, adjective, noun, make_value_vector(store, adjective, noun)))
    return rows


def test_compositional_value_decoder_decodes_held_out_value() -> None:
    store = VectorStore(dim=128, seed=0)
    rows = _value_rows(store)
    decoder = CompositionalValueDecoder(store=store)
    decoder.fit_linear_head([(vector, adjective, noun) for _entity, adjective, noun, vector in rows[:48]])

    entity, adjective, noun, value_vector = rows[55]

    hrr_decoded = decoder.decode(value_vector, strategy="hrr_native")
    linear_decoded = decoder.decode(value_vector, strategy="linear")

    assert entity == "entity_055"
    assert hrr_decoded.text == f"{adjective} {noun}"
    assert linear_decoded.text == f"{adjective} {noun}"


def test_frozen_generator_adapter_answers_from_value_vector() -> None:
    store = VectorStore(dim=128, seed=0)
    rows = _value_rows(store)
    decoder = CompositionalValueDecoder(store=store)
    decoder.fit_linear_head([(vector, adjective, noun) for _entity, adjective, noun, vector in rows[:48]])

    entity, adjective, noun, value_vector = rows[55]
    adapter = FrozenGeneratorAdapter(value_decoder=decoder, value_strategy="linear")

    answer = adapter.answer(
        f"What property does {entity} have?",
        {
            "confidence": 0.99,
            "entity": entity,
            "value_vector": value_vector,
        },
    )

    assert answer == f"{entity} has property {adjective} {noun}."
