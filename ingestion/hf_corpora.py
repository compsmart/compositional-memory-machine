from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol

from .gemini import ExtractedFact


HF_JOTSCHI_KB = "Jotschi/wikipedia_knowledge_base_en"
HF_JOTSCHI_KG = "Jotschi/wikipedia_knowledge_graph_en"
HF_WIKIDATA_ALL = "Wikimedians/wikidata-all"
HF_STRUCTURED_WIKIPEDIA = "wikimedia/structured-wikipedia"
HF_FINEWIKI = "HuggingFaceFW/finewiki"

HF_SUPPORTED_DATASETS = (
    HF_JOTSCHI_KB,
    HF_JOTSCHI_KG,
    HF_WIKIDATA_ALL,
    HF_STRUCTURED_WIKIPEDIA,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDICAL_QID_MAP_PATH = WORKSPACE_ROOT / "data" / "medical_wikidata_qid_map.json"
DEFAULT_MEDICAL_DOMAIN = "medical.core"
MEDICAL_DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "medical.pathogen",
        (
            "bacterium",
            "bacteria",
            "virus",
            "viral",
            "fungus",
            "fungi",
            "parasite",
            "pathogen",
            "microorganism",
            "microbe",
        ),
    ),
    (
        "medical.disease",
        (
            "disease",
            "syndrome",
            "disorder",
            "infection",
            "cancer",
            "tumor",
            "tumour",
            "illness",
            "pathology",
            "medical condition",
        ),
    ),
    (
        "medical.drug",
        (
            "drug",
            "medication",
            "medicine",
            "antibiotic",
            "antiviral",
            "vaccine",
            "pharmaceutical",
            "analgesic",
            "steroid",
            "anticoagulant",
        ),
    ),
    (
        "medical.anatomy",
        (
            "anatomical",
            "anatomy",
            "artery",
            "vein",
            "nerve",
            "muscle",
            "bone",
            "organ",
            "tissue",
            "gland",
            "ligament",
            "tendon",
        ),
    ),
    (
        "medical.procedure",
        (
            "procedure",
            "surgery",
            "surgical",
            "therapy",
            "treatment",
            "diagnosis",
            "diagnostic",
            "screening",
            "biopsy",
            "transplant",
            "rehabilitation",
        ),
    ),
    (
        "medical.specialty",
        (
            "medical specialty",
            "medical speciality",
            "branch of medicine",
            "specialty",
            "speciality",
            "physician",
            "doctor",
            "surgeon",
            "clinician",
        ),
    ),
    (
        "medical.symptom",
        (
            "symptom",
            "clinical sign",
            "manifestation",
            "pain",
            "fever",
            "nausea",
            "fatigue",
        ),
    ),
)
MEDICAL_DOMAIN_PRIORITY = {
    domain: len(MEDICAL_DOMAIN_KEYWORDS) - index
    for index, (domain, _keywords) in enumerate(MEDICAL_DOMAIN_KEYWORDS)
}
MEDICAL_INFOBOX_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("drugbox", "medical.drug"),
    ("automatic taxobox", "medical.pathogen"),
    ("medical condition", "medical.disease"),
    ("disease", "medical.disease"),
    ("symptoms", "medical.symptom"),
    ("specialty", "medical.specialty"),
)
MEDICAL_GATE_KEYWORDS: tuple[str, ...] = (
    "medical",
    "medicine",
    "health",
    "disease",
    "drug",
    "anatomy",
    "symptom",
    "therapy",
    "treatment",
    "diagnosis",
    "surgery",
    "infection",
    "pathogen",
)


@dataclass(frozen=True)
class StructuredFactRecord:
    fact: ExtractedFact
    domain: str


class StructuredFactWriter(Protocol):
    def write_structured_fact(
        self,
        fact: ExtractedFact,
        *,
        source: str = "structured",
        domain: str = "structured",
    ) -> bool: ...


def dataset_default_domain(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == HF_JOTSCHI_KB.lower():
        return "hf_wikipedia_kb"
    if normalized == HF_JOTSCHI_KG.lower():
        return "hf_wikipedia_kg"
    if normalized == HF_WIKIDATA_ALL.lower():
        return "hf_wikidata"
    if normalized == HF_STRUCTURED_WIKIPEDIA.lower():
        return "hf_structured_wikipedia"
    return "hf_corpus"


def dataset_source_name(dataset_name: str) -> str:
    return f"hf:{dataset_name}"


@lru_cache(maxsize=8)
def _load_qid_domain_map_cached(path: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"medical qid map must be a JSON object: {path}")
    qid_map: dict[str, str] = {}
    for raw_qid, raw_domain in payload.items():
        qid = str(raw_qid).strip()
        domain = str(raw_domain).strip()
        if qid and domain:
            qid_map[qid] = domain
    return qid_map


def load_medical_wikidata_qid_map(path: str | Path | None = None) -> dict[str, str]:
    map_path = Path(path) if path is not None else DEFAULT_MEDICAL_QID_MAP_PATH
    if not map_path.exists():
        return {}
    return dict(_load_qid_domain_map_cached(str(map_path.resolve())))


def classify_structured_wikipedia_medical_domain(
    row: dict[str, Any],
    *,
    qid_domain_map: Mapping[str, str] | None = None,
) -> str | None:
    qid_lookup = qid_domain_map if qid_domain_map is not None else load_medical_wikidata_qid_map()
    for entity_id in _structured_wikipedia_entity_ids(row):
        domain = qid_lookup.get(entity_id)
        if domain:
            return domain

    domain_scores: dict[str, int] = {}
    medical_signal = 0
    for infobox_name in _structured_wikipedia_infobox_names(row):
        lowered_name = infobox_name.lower()
        for keyword, domain in MEDICAL_INFOBOX_KEYWORDS:
            if keyword in lowered_name:
                domain_scores[domain] = domain_scores.get(domain, 0) + 3
                medical_signal += 2

    for text in _structured_wikipedia_medical_texts(row):
        lowered = text.lower()
        for keyword in MEDICAL_GATE_KEYWORDS:
            if _contains_keyword(lowered, keyword):
                medical_signal += 1
        for domain, keywords in MEDICAL_DOMAIN_KEYWORDS:
            matches = sum(1 for keyword in keywords if _contains_keyword(lowered, keyword))
            if matches:
                domain_scores[domain] = domain_scores.get(domain, 0) + matches

    if domain_scores:
        return max(domain_scores.items(), key=lambda item: (item[1], MEDICAL_DOMAIN_PRIORITY.get(item[0], 0)))[0]
    if medical_signal:
        return DEFAULT_MEDICAL_DOMAIN
    return None


def dataset_row_to_fact_records(
    dataset_name: str,
    row: dict[str, Any],
    *,
    max_claims_per_entity: int | None = None,
    medical_only: bool = False,
    structured_wikipedia_qid_map: Mapping[str, str] | None = None,
) -> list[StructuredFactRecord]:
    normalized = dataset_name.strip().lower()
    if normalized == HF_JOTSCHI_KB.lower():
        return jotschi_kb_row_to_records(row)
    if normalized == HF_JOTSCHI_KG.lower():
        return jotschi_kg_row_to_records(row)
    if normalized == HF_WIKIDATA_ALL.lower():
        return wikidata_row_to_records(row, max_claims_per_entity=max_claims_per_entity)
    if normalized == HF_STRUCTURED_WIKIPEDIA.lower():
        medical_domain = classify_structured_wikipedia_medical_domain(
            row,
            qid_domain_map=structured_wikipedia_qid_map,
        )
        if medical_only and medical_domain is None:
            return []
        return structured_wikipedia_row_to_records(
            row,
            domain=medical_domain or dataset_default_domain(HF_STRUCTURED_WIKIPEDIA),
        )
    raise ValueError(f"unsupported dataset: {dataset_name}")


def jotschi_kb_row_to_records(row: dict[str, Any]) -> list[StructuredFactRecord]:
    title = _first_text(row, "title", "name", "page_title")
    source_id = _first_text(row, "id", "page_id", "row_id")
    url = _first_text(row, "url", "entry_url")
    facts = row.get("facts")
    if not title or not isinstance(facts, list):
        return []

    records: list[StructuredFactRecord] = []
    for index, item in enumerate(facts):
        text = _extract_fact_text(item)
        if not text:
            continue
        records.append(
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject=_trim_slot(title),
                    relation="described_by",
                    object=_trim_slot(text, limit=320),
                    confidence=0.7,
                    kind="explicit",
                    source=dataset_source_name(HF_JOTSCHI_KB),
                    source_id=f"{source_id or title}:{index}",
                    excerpt=_trim_excerpt(text),
                    source_chunk_id=url,
                    sentence_index=index,
                ),
                domain=dataset_default_domain(HF_JOTSCHI_KB),
            )
        )
    return records


def jotschi_kg_row_to_records(row: dict[str, Any]) -> list[StructuredFactRecord]:
    relationships = row.get("relationships")
    if not isinstance(relationships, list):
        return []
    entry_url = _first_text(row, "entry_url", "url")
    fact_nr = _first_text(row, "fact_nr", "id")
    source_fact = _first_text(row, "source_fact", "fact", "text")
    records: list[StructuredFactRecord] = []
    for index, item in enumerate(relationships):
        if not isinstance(item, dict):
            continue
        entity_a = _entity_name(item.get("entity_a"))
        relation = _trim_relation(_first_text(item, "rel", "relation"))
        entity_b = _entity_name(item.get("entity_b"))
        if not entity_a or not relation or not entity_b:
            continue
        records.append(
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject=_trim_slot(entity_a),
                    relation=relation,
                    object=_trim_slot(entity_b),
                    confidence=0.8,
                    kind="explicit",
                    source=dataset_source_name(HF_JOTSCHI_KG),
                    source_id=f"{fact_nr or entry_url or 'kg'}:{index}",
                    source_chunk_id=entry_url,
                    excerpt=_trim_excerpt(source_fact),
                    sentence_index=index,
                ),
                domain=dataset_default_domain(HF_JOTSCHI_KG),
            )
        )
    return records


def structured_wikipedia_row_to_records(
    row: dict[str, Any],
    *,
    domain: str | None = None,
) -> list[StructuredFactRecord]:
    title = _first_text(row, "title", "name", "page_title")
    if not title:
        return []
    source_id = _first_text(row, "id", "identifier", "page_id", "url", "entry_url") or title
    record_domain = domain or dataset_default_domain(HF_STRUCTURED_WIKIPEDIA)
    records: list[StructuredFactRecord] = []

    description = _first_text(row, "description", "lead", "abstract")
    if description:
        records.append(
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject=_trim_slot(title),
                    relation="described_by",
                    object=_trim_slot(description, limit=320),
                    confidence=0.9,
                    kind="explicit",
                    source=dataset_source_name(HF_STRUCTURED_WIKIPEDIA),
                    source_id=f"{source_id}:description",
                    excerpt=_trim_excerpt(description),
                ),
                domain=record_domain,
            )
        )

    for key, value in _iter_structured_wikipedia_infobox_entries(row):
        flattened = _flatten_value(value)
        if not flattened:
            continue
        records.append(
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject=_trim_slot(title),
                    relation="infobox_has",
                    object=_trim_slot(f"{_trim_slot(str(key), limit=80)}: {flattened}", limit=320),
                    confidence=0.85,
                    kind="explicit",
                    source=dataset_source_name(HF_STRUCTURED_WIKIPEDIA),
                    source_id=f"{source_id}:infobox:{_slug(str(key)) or 'entry'}",
                    excerpt=_trim_excerpt(flattened),
                ),
                domain=record_domain,
            )
        )
    return records


def _iter_structured_wikipedia_infobox_entries(row: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for key in ("infobox", "infoboxes"):
        infobox = row.get(key)
        if isinstance(infobox, dict):
            for entry_key, entry_value in infobox.items():
                yield str(entry_key), entry_value
        elif isinstance(infobox, list):
            for index, item in enumerate(infobox):
                if isinstance(item, dict):
                    yielded = False
                    for label, value in _iter_structured_wikipedia_infobox_item(item):
                        yield label, value
                        yielded = True
                    if not yielded:
                        label = _first_text(item, "label", "name", "key", "type") or f"entry_{index}"
                        value = item.get("value")
                        if value is None:
                            value = item
                        yield label, value


def _iter_structured_wikipedia_infobox_item(
    item: dict[str, Any],
    *,
    prefix: str = "",
) -> Iterator[tuple[str, Any]]:
    node_type = _first_text(item, "type")
    name = _first_text(item, "label", "name", "key")
    values = item.get("values")
    value = item.get("value")
    label = _join_infobox_labels(prefix, name)

    if name and values:
        yield label, values
    elif name and value is not None:
        yield label, value

    child_prefix = prefix
    if node_type in {"infobox", "section"} and name:
        child_prefix = label
    for child in item.get("has_parts", []):
        if isinstance(child, dict):
            yield from _iter_structured_wikipedia_infobox_item(child, prefix=child_prefix)


def wikidata_row_to_records(
    row: dict[str, Any],
    *,
    max_claims_per_entity: int | None = None,
) -> list[StructuredFactRecord]:
    entity = _unwrap_wikidata_entity(row)
    if not isinstance(entity, dict):
        return []
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        return []

    entity_id = _first_text(entity, "id", "entity_id")
    subject = _wikidata_entity_label(entity) or entity_id
    if not subject:
        return []

    records: list[StructuredFactRecord] = []
    emitted = 0
    for property_id, statements in claims.items():
        if not isinstance(statements, list):
            continue
        for index, statement in enumerate(statements):
            if max_claims_per_entity is not None and emitted >= max_claims_per_entity:
                return records
            record = _wikidata_statement_to_record(
                subject=subject,
                entity_id=entity_id,
                property_id=str(property_id),
                statement=statement,
                index=index,
            )
            if record is None:
                continue
            records.append(record)
            emitted += 1
    return records


def write_fact_jsonl(path: str | Path, records: Iterable[StructuredFactRecord]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(structured_fact_record_to_json(record), ensure_ascii=True) + "\n")
            written += 1
    return written


def iter_fact_jsonl(path: str | Path, *, limit: int = 0) -> Iterator[StructuredFactRecord]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit > 0 and index >= limit:
                break
            raw = line.strip()
            if not raw:
                continue
            yield structured_fact_record_from_json(json.loads(raw))


def preload_writer_from_jsonl(
    writer: StructuredFactWriter,
    path: str | Path,
    *,
    limit: int = 0,
) -> int:
    loaded = 0
    for record in iter_fact_jsonl(path, limit=limit):
        writer.write_structured_fact(record.fact, source=record.fact.source or "hf-jsonl", domain=record.domain)
        loaded += 1
    return loaded


def structured_fact_record_to_json(record: StructuredFactRecord) -> dict[str, Any]:
    payload = record.fact.model_dump()
    payload["domain"] = record.domain
    return payload


def structured_fact_record_from_json(payload: dict[str, Any]) -> StructuredFactRecord:
    domain = str(payload.get("domain") or "hf_corpus").strip() or "hf_corpus"
    fact_payload = dict(payload)
    fact_payload.pop("domain", None)
    return StructuredFactRecord(
        fact=ExtractedFact.model_validate(fact_payload),
        domain=domain,
    )


class SqliteFactLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_hash TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_chunk_id TEXT NOT NULL,
                excerpt TEXT NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                sentence_index INTEGER,
                row_offset INTEGER NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                dataset TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                next_offset INTEGER NOT NULL,
                written_facts INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def insert_records(self, records: Iterable[StructuredFactRecord], *, row_offset: int) -> list[StructuredFactRecord]:
        accepted: list[StructuredFactRecord] = []
        for record in records:
            payload = structured_fact_record_to_json(record)
            fact_hash = fact_record_hash(record)
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO facts (
                    fact_hash, domain, subject, relation, object, confidence, kind, source,
                    source_id, source_chunk_id, excerpt, char_start, char_end, sentence_index, row_offset
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_hash,
                    record.domain,
                    payload["subject"],
                    payload["relation"],
                    payload["object"],
                    float(payload["confidence"]),
                    payload["kind"],
                    payload["source"],
                    payload["source_id"],
                    payload["source_chunk_id"],
                    payload["excerpt"],
                    payload["char_start"],
                    payload["char_end"],
                    payload["sentence_index"],
                    row_offset,
                ),
            )
            if cursor.rowcount:
                accepted.append(record)
        self._connection.commit()
        return accepted

    def update_progress(
        self,
        *,
        dataset: str,
        split: str,
        next_offset: int,
        written_facts: int,
        updated_at: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO progress (dataset, split, next_offset, written_facts, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset) DO UPDATE SET
                split = excluded.split,
                next_offset = excluded.next_offset,
                written_facts = excluded.written_facts,
                updated_at = excluded.updated_at
            """,
            (dataset, split, next_offset, written_facts, updated_at),
        )
        self._connection.commit()

    def load_progress(self, *, dataset: str) -> dict[str, Any] | None:
        cursor = self._connection.execute(
            "SELECT dataset, split, next_offset, written_facts, updated_at FROM progress WHERE dataset = ?",
            (dataset,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "dataset": row[0],
            "split": row[1],
            "next_offset": int(row[2]),
            "written_facts": int(row[3]),
            "updated_at": str(row[4]),
        }


def fact_record_hash(record: StructuredFactRecord) -> str:
    payload = structured_fact_record_to_json(record)
    stable = json.dumps(
        {
            "domain": payload["domain"],
            "subject": payload["subject"],
            "relation": payload["relation"],
            "object": payload["object"],
            "source": payload["source"],
            "source_id": payload["source_id"],
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _structured_wikipedia_entity_ids(row: dict[str, Any]) -> Iterator[str]:
    main_entity = row.get("main_entity")
    if isinstance(main_entity, dict):
        entity_id = _first_text(main_entity, "identifier", "id")
        if entity_id:
            yield entity_id
    additional_entities = row.get("additional_entities")
    if isinstance(additional_entities, list):
        for item in additional_entities:
            if isinstance(item, dict):
                entity_id = _first_text(item, "identifier", "id")
                if entity_id:
                    yield entity_id


def _structured_wikipedia_infobox_names(row: dict[str, Any]) -> Iterator[str]:
    infoboxes = row.get("infoboxes")
    if not isinstance(infoboxes, list):
        return
    for item in infoboxes:
        if not isinstance(item, dict):
            continue
        name = _first_text(item, "name", "label", "type")
        if name:
            yield name


def _structured_wikipedia_medical_texts(row: dict[str, Any]) -> Iterator[str]:
    for key in ("name", "title", "description", "abstract"):
        text = _first_text(row, key)
        if text:
            yield text
    for label, value in _iter_structured_wikipedia_infobox_entries(row):
        if label:
            yield label
        flattened = _flatten_value(value)
        if flattened:
            yield flattened


def _join_infobox_labels(prefix: str, name: str) -> str:
    if prefix and name:
        return f"{prefix} > {name}"
    return prefix or name


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


def _first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = " ".join(value.split())
            if cleaned:
                return cleaned
        elif isinstance(value, (int, float)):
            return str(value)
    return ""


def _extract_fact_text(item: Any) -> str:
    if isinstance(item, str):
        return " ".join(item.split())
    if isinstance(item, dict):
        return _first_text(item, "text", "fact", "source_fact", "value")
    return ""


def _entity_name(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    name = _first_text(payload, "name", "label", "id")
    attribute = _first_text(payload, "attribute")
    if attribute and attribute.lower() not in name.lower():
        return f"{name} ({attribute})" if name else attribute
    return name


def _trim_relation(value: str) -> str:
    return _slug(value).strip("_")


def _slug(value: str) -> str:
    lowered = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", lowered).strip("_")


def _trim_slot(value: str, *, limit: int = 240) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _trim_excerpt(value: str, *, limit: int = 400) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _flatten_value(value: Any) -> str:
    if isinstance(value, str):
        return _trim_excerpt(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_flatten_value(item) for item in value]
        parts = [part for part in parts if part]
        return _trim_excerpt("; ".join(parts))
    if isinstance(value, dict):
        for key in ("text", "value", "label", "name", "plain_text"):
            text = _first_text(value, key)
            if text:
                return _trim_excerpt(text)
        pieces = [f"{key}={_flatten_value(item)}" for key, item in value.items() if _flatten_value(item)]
        return _trim_excerpt("; ".join(pieces))
    return ""


def _unwrap_wikidata_entity(row: dict[str, Any]) -> dict[str, Any] | None:
    if "claims" in row:
        return row
    for key in ("item", "entity", "data", "record"):
        value = row.get(key)
        if isinstance(value, dict) and "claims" in value:
            return value
    return None


def _wikidata_entity_label(entity: dict[str, Any]) -> str:
    labels = entity.get("labels")
    if isinstance(labels, dict):
        english = labels.get("en")
        if isinstance(english, dict):
            value = _first_text(english, "value")
            if value:
                return value
        if isinstance(english, str) and english.strip():
            return english.strip()
        for label in labels.values():
            if isinstance(label, dict):
                value = _first_text(label, "value")
                if value:
                    return value
            elif isinstance(label, str) and label.strip():
                return label.strip()
    return _first_text(entity, "title", "name", "id")


def _wikidata_statement_to_record(
    *,
    subject: str,
    entity_id: str,
    property_id: str,
    statement: Any,
    index: int,
) -> StructuredFactRecord | None:
    if not isinstance(statement, dict):
        return None
    mainsnak = statement.get("mainsnak")
    if not isinstance(mainsnak, dict):
        return None
    datavalue = mainsnak.get("datavalue")
    if not isinstance(datavalue, dict):
        return None
    object_value = _wikidata_value_to_text(datavalue.get("value"))
    if not object_value:
        return None
    relation = _trim_relation(_first_text(mainsnak, "property") or property_id)
    if not relation:
        return None
    statement_id = _first_text(statement, "id") or f"{entity_id}:{property_id}:{index}"
    return StructuredFactRecord(
        fact=ExtractedFact(
            subject=_trim_slot(subject),
            relation=relation,
            object=_trim_slot(object_value),
            confidence=0.95,
            kind="explicit",
            source=dataset_source_name(HF_WIKIDATA_ALL),
            source_id=statement_id,
            excerpt=_trim_excerpt(f"{subject} {relation} {object_value}"),
            sentence_index=index,
        ),
        domain=dataset_default_domain(HF_WIKIDATA_ALL),
    )


def _wikidata_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        entity_id = _first_text(value, "id")
        if entity_id:
            return entity_id
        amount = _first_text(value, "amount")
        unit = _first_text(value, "unit")
        if amount:
            return amount if not unit else f"{amount} {unit}"
        time = _first_text(value, "time")
        if time:
            return time
        latitude = _first_text(value, "latitude")
        longitude = _first_text(value, "longitude")
        if latitude and longitude:
            return f"{latitude},{longitude}"
        text = _first_text(value, "text", "value")
        if text:
            return text
    return ""
