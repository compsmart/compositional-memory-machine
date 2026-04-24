from __future__ import annotations

from experiments.exp_hf_corpus_ingest import _ingest_records_by_domain
from ingestion import (
    HF_JOTSCHI_KB,
    HF_JOTSCHI_KG,
    HF_STRUCTURED_WIKIPEDIA,
    HF_WIKIDATA_ALL,
    SqliteFactLedger,
    dataset_row_to_fact_records,
    preload_writer_from_jsonl,
    write_fact_jsonl,
)
from web import HHRWebState


def test_jotschi_kb_row_maps_fact_texts_to_structured_records() -> None:
    records = dataset_row_to_fact_records(
        HF_JOTSCHI_KB,
        {
            "title": "Artificial intelligence",
            "id": "1164",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "facts": [
                {"text": "Convolutional neural networks were introduced by Kunihiko Fukushima in 1980."},
                {"text": "AI algorithms experience exponential slowdown for large problems."},
            ],
        },
    )

    assert len(records) == 2
    assert records[0].domain == "hf_wikipedia_kb"
    assert records[0].fact.subject == "Artificial intelligence"
    assert records[0].fact.relation == "described_by"
    assert "Kunihiko Fukushima" in records[0].fact.object
    assert records[0].fact.source_id == "1164:0"


def test_jotschi_kg_row_maps_relationships_to_atomic_facts() -> None:
    records = dataset_row_to_fact_records(
        HF_JOTSCHI_KG,
        {
            "entry_url": "https://en.wikipedia.org/wiki/Alps",
            "fact_nr": 73,
            "source_fact": "The total Alpine population is 14 million across eight countries.",
            "relationships": [
                {
                    "entity_a": {"name": "Alpine population", "type": "Value", "attribute": ""},
                    "rel": "HAS_QUANTITY",
                    "entity_b": {"name": "14 million", "type": "Value", "attribute": ""},
                },
                {
                    "entity_a": {"name": "Alpine", "type": "Location", "attribute": ""},
                    "rel": "LOCATED_IN",
                    "entity_b": {"name": "eight countries", "type": "Location", "attribute": ""},
                },
            ],
        },
    )

    assert len(records) == 2
    assert records[0].fact.relation == "has_quantity"
    assert records[0].fact.object == "14 million"
    assert records[1].fact.relation == "located_in"
    assert records[1].fact.source_chunk_id == "https://en.wikipedia.org/wiki/Alps"


def test_structured_wikipedia_row_maps_description_and_infobox() -> None:
    records = dataset_row_to_fact_records(
        HF_STRUCTURED_WIKIPEDIA,
        {
            "title": "Ada Lovelace",
            "id": "ada-1",
            "description": "English mathematician and writer.",
            "infobox": {
                "occupation": "Mathematician",
                "known_for": ["Analytical Engine notes", "Bernoulli number algorithm"],
            },
        },
    )

    assert len(records) == 3
    assert records[0].fact.relation == "described_by"
    assert records[1].fact.relation == "infobox_has"
    assert "occupation: Mathematician" in records[1].fact.object
    assert "known_for" in records[2].fact.source_id


def test_structured_wikipedia_medical_row_uses_domain_override_and_nested_infobox() -> None:
    records = dataset_row_to_fact_records(
        HF_STRUCTURED_WIKIPEDIA,
        {
            "name": "Example syndrome",
            "identifier": "syndrome-1",
            "description": "Rare genetic disease affecting connective tissue.",
            "main_entity": {"identifier": "Q12136"},
            "infoboxes": [
                {
                    "type": "infobox",
                    "name": "Medical condition",
                    "has_parts": [
                        {"type": "field", "name": "Symptoms", "value": "Fever"},
                        {"type": "field", "name": "Specialty", "value": "Medical genetics"},
                    ],
                }
            ],
        },
        medical_only=True,
        structured_wikipedia_qid_map={"Q12136": "medical.disease"},
    )

    assert len(records) == 3
    assert {record.domain for record in records} == {"medical.disease"}
    assert any(record.fact.object == "Rare genetic disease affecting connective tissue." for record in records)
    assert any("Medical condition > Symptoms: Fever" in record.fact.object for record in records)
    assert any("Medical condition > Specialty: Medical genetics" in record.fact.object for record in records)


def test_structured_wikipedia_medical_only_skips_non_medical_rows() -> None:
    records = dataset_row_to_fact_records(
        HF_STRUCTURED_WIKIPEDIA,
        {
            "name": "Ada Lovelace",
            "identifier": "ada-1",
            "description": "English mathematician and writer.",
        },
        medical_only=True,
    )

    assert records == []


def test_wikidata_row_maps_claims_with_capped_output() -> None:
    records = dataset_row_to_fact_records(
        HF_WIKIDATA_ALL,
        {
            "id": "Q7259",
            "labels": {"en": {"language": "en", "value": "Ada Lovelace"}},
            "claims": {
                "P31": [
                    {
                        "id": "Q7259$1",
                        "mainsnak": {
                            "property": "P31",
                            "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q5"}},
                        },
                    }
                ],
                "P569": [
                    {
                        "id": "Q7259$2",
                        "mainsnak": {
                            "property": "P569",
                            "datavalue": {"type": "time", "value": {"time": "+1815-12-10T00:00:00Z"}},
                        },
                    }
                ],
            },
        },
        max_claims_per_entity=1,
    )

    assert len(records) == 1
    assert records[0].domain == "hf_wikidata"
    assert records[0].fact.subject == "Ada Lovelace"
    assert records[0].fact.relation == "p31"
    assert records[0].fact.object == "Q5"


def test_jsonl_preload_round_trip_writes_into_web_state(tmp_path) -> None:
    path = tmp_path / "facts.jsonl"
    records = dataset_row_to_fact_records(
        HF_JOTSCHI_KG,
        {
            "entry_url": "https://en.wikipedia.org/wiki/Alps",
            "fact_nr": 73,
            "source_fact": "The total Alpine population is 14 million across eight countries.",
            "relationships": [
                {
                    "entity_a": {"name": "Alpine population", "type": "Value", "attribute": ""},
                    "rel": "HAS_QUANTITY",
                    "entity_b": {"name": "14 million", "type": "Value", "attribute": ""},
                }
            ],
        },
    )
    write_fact_jsonl(path, records)

    state = HHRWebState()
    before = state.status()["stored_facts"]
    loaded = preload_writer_from_jsonl(state.pipeline, path)
    after = state.status()["stored_facts"]

    assert loaded == 1
    assert after == before + 1
    assert state.graph.read("Alpine population", "has_quantity") == "14 million"


def test_sqlite_fact_ledger_deduplicates_and_tracks_progress(tmp_path) -> None:
    ledger = SqliteFactLedger(tmp_path / "hf_ingest.sqlite")
    try:
        records = dataset_row_to_fact_records(
            HF_JOTSCHI_KG,
            {
                "entry_url": "https://en.wikipedia.org/wiki/Alps",
                "fact_nr": 73,
                "source_fact": "The total Alpine population is 14 million across eight countries.",
                "relationships": [
                    {
                        "entity_a": {"name": "Alpine population", "type": "Value", "attribute": ""},
                        "rel": "HAS_QUANTITY",
                        "entity_b": {"name": "14 million", "type": "Value", "attribute": ""},
                    }
                ],
            },
        )

        first = ledger.insert_records(records, row_offset=0)
        second = ledger.insert_records(records, row_offset=1)
        ledger.update_progress(
            dataset=HF_JOTSCHI_KG,
            split="train",
            next_offset=5,
            written_facts=1,
            updated_at="2026-04-23T00:00:00Z",
        )
        progress = ledger.load_progress(dataset=HF_JOTSCHI_KG)

        assert len(first) == 1
        assert second == []
        assert progress == {
            "dataset": HF_JOTSCHI_KG,
            "split": "train",
            "next_offset": 5,
            "written_facts": 1,
            "updated_at": "2026-04-23T00:00:00Z",
        }
    finally:
        ledger.close()


def test_ingest_groups_records_by_domain() -> None:
    disease_records = dataset_row_to_fact_records(
        HF_STRUCTURED_WIKIPEDIA,
        {
            "name": "Example syndrome",
            "identifier": "syndrome-1",
            "description": "Rare genetic disease affecting connective tissue.",
        },
        medical_only=True,
        structured_wikipedia_qid_map={"Q12136": "medical.disease"},
    )
    drug_records = dataset_row_to_fact_records(
        HF_STRUCTURED_WIKIPEDIA,
        {
            "name": "Examplecillin",
            "identifier": "drug-1",
            "description": "Antibiotic medication used to treat infections.",
        },
        medical_only=True,
    )

    pipeline, batch_records, relation_stats = _ingest_records_by_domain(
        [*disease_records, *drug_records],
        dim=1024,
        seed=0,
        source="hf:wikimedia/structured-wikipedia",
        default_domain="hf_structured_wikipedia",
    )

    assert {record.domain for record in batch_records} == {"medical.disease", "medical.drug"}
    assert relation_stats["per_domain"].keys() == {"medical.disease", "medical.drug"}
    assert len(pipeline.chunk_memory.chunks) >= 2
