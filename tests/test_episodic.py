from __future__ import annotations

from memory import ConversationFact, EpisodicMemory


def test_episodic_memory_normalizes_relation_aliases_and_keeps_provenance() -> None:
    memory = EpisodicMemory(dim=1024, seed=0)

    fact = ConversationFact(
        0,
        0,
        "Ada Lovelace",
        "collaborated with",
        "Charles Babbage",
        source="fixture",
        source_id="turn-0",
        excerpt="Ada Lovelace collaborated with Charles Babbage.",
        char_start=0,
        char_end=46,
        sentence_index=0,
    )
    memory.state_fact(fact)

    assert memory.recall_current("Ada Lovelace", "collaborated with") == "Charles Babbage"
    assert memory.recall_current("Ada Lovelace", "worked_with") == "Charles Babbage"

    record = next(iter(memory.memory.records.values()))
    assert record.payload["raw_relation"] == "collaborated with"
    assert record.payload["normalized_relation"] == "worked_with"
    assert record.payload["matched_alias"] is True
    assert record.payload["provenance"]["source"] == "fixture"
    assert record.payload["provenance"]["source_id"] == "turn-0"
    assert record.payload["provenance"]["excerpt"] == "Ada Lovelace collaborated with Charles Babbage."
    assert record.payload["provenance"]["char_start"] == 0
    assert record.payload["provenance"]["char_end"] == 46
    assert record.payload["provenance"]["sentence_index"] == 0
    assert memory.history[0].relation == "worked_with"


def test_episodic_memory_revision_collapses_relation_aliases() -> None:
    memory = EpisodicMemory(dim=1024, seed=0)

    memory.state_fact(
        ConversationFact(
            0,
            0,
            "Ada Lovelace",
            "collaborated with",
            "Charles Babbage",
            source="fixture",
        )
    )
    revised = ConversationFact(
        0,
        1,
        "Ada Lovelace",
        "worked on with",
        "Mary Somerville",
        source="fixture",
        source_id="turn-1",
    )
    memory.revise_fact(revised)

    assert memory.recall_current("Ada Lovelace", "worked_with") == "Mary Somerville"
    assert memory.recall_history("Ada Lovelace", "collaborated with") == [
        "Charles Babbage",
        "Mary Somerville",
    ]
    assert memory.recall_evidence(revised) is True

    revised_record = memory.memory.get("s0:t1:Ada Lovelace:worked_with:Mary Somerville")
    assert revised_record is not None
    assert revised_record.payload["provenance"]["revision"] is True
    assert revised_record.payload["provenance"]["source_id"] == "turn-1"
