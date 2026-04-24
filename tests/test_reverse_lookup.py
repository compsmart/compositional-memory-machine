from __future__ import annotations

from reverse_lookup import ReverseAttributeIndex, parse_reverse_attribute_query, scan_reverse_attribute_candidates


def _medical_facts() -> list[dict[str, object]]:
    return [
        {
            "subject": "Sulfametrole",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > ATC code: - J01EE03 (WHO) (with trimethoprim)",
        },
        {
            "subject": "Sulfametrole",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > Routes of administration: Oral",
        },
        {
            "subject": "Sulfamethazine",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > Routes of administration: Oral",
        },
        {
            "subject": "Amikacin",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > Routes of administration: Intravenous",
        },
        {
            "subject": "Amikacin",
            "relation": "infobox_has",
            "object": "Drugbox > Identifiers > CAS Number: 39831-55-5",
        },
    ]


def test_reverse_lookup_index_finds_exact_identifier_match() -> None:
    query = parse_reverse_attribute_query("Which chemical compound has ATC code J01EE03?")
    assert query is not None

    hits = ReverseAttributeIndex.from_facts(_medical_facts()).lookup(query)

    assert hits[0].subject == "Sulfametrole"
    assert hits[0].matched_identifiers == ("J01EE03",)


def test_reverse_lookup_scan_and_index_agree_on_ambiguous_query() -> None:
    query = parse_reverse_attribute_query("Which drug has route of administration oral?")
    assert query is not None

    scan_hits = scan_reverse_attribute_candidates(_medical_facts(), query)
    index_hits = ReverseAttributeIndex.from_facts(_medical_facts()).lookup(query)

    assert {hit.subject for hit in scan_hits[:2]} == {"Sulfametrole", "Sulfamethazine"}
    assert {hit.subject for hit in index_hits[:2]} == {"Sulfametrole", "Sulfamethazine"}


def test_reverse_lookup_index_returns_no_hits_for_near_miss_identifier() -> None:
    query = parse_reverse_attribute_query("Which chemical compound has ATC code J01EE30?")
    assert query is not None

    hits = ReverseAttributeIndex.from_facts(_medical_facts()).lookup(query)

    assert hits == []
