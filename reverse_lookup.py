from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

GENERIC_QUERY_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "chemical",
    "code",
    "codes",
    "compound",
    "compounds",
    "does",
    "drug",
    "drugs",
    "for",
    "has",
    "have",
    "htc",
    "identifier",
    "identifiers",
    "is",
    "number",
    "of",
    "the",
    "what",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class ReverseAttributeQuery:
    raw_message: str
    normalized_message: str
    tokens: frozenset[str]
    identifiers: frozenset[str]


@dataclass(frozen=True)
class ReverseFactRecord:
    subject: str
    relation: str
    object_text: str
    field_path: str
    field_value: str
    normalized_object: str
    field_tokens: frozenset[str]
    value_tokens: frozenset[str]
    search_tokens: frozenset[str]
    identifiers: frozenset[str]


@dataclass(frozen=True)
class ReverseLookupHit:
    subject: str
    relation: str
    object_text: str
    score: float
    matched_tokens: tuple[str, ...]
    matched_identifiers: tuple[str, ...]


def normalize_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    compact = ascii_only.lower().replace(">", " ")
    compact = re.sub(r"[^a-z0-9:\-]+", " ", compact)
    return re.sub(r"\s+", " ", compact).strip()


def extract_identifier_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    identifiers: set[str] = set()
    for match in re.finditer(r"\b\d{2,7}-\d{2}-\d\b", ascii_only):
        identifiers.add(match.group(0).upper())
    for match in re.finditer(r"\b(?=[A-Za-z0-9:-]{5,}\b)(?=\S*[A-Za-z])(?=\S*\d)[A-Za-z0-9:-]+\b", ascii_only):
        identifiers.add(match.group(0).upper())
    return identifiers


def parse_reverse_fact_record(fact: dict[str, object]) -> ReverseFactRecord | None:
    relation = str(fact.get("relation") or "").strip()
    if relation != "infobox_has":
        return None
    subject = str(fact.get("subject") or "").strip()
    object_text = str(fact.get("object") or "").strip()
    if not subject or not object_text:
        return None
    if ":" in object_text:
        field_path, field_value = object_text.split(":", 1)
    else:
        field_path, field_value = object_text, object_text
    normalized_object = normalize_lookup_text(object_text)
    field_tokens = frozenset(_lookup_tokens(field_path))
    value_tokens = frozenset(_lookup_tokens(field_value))
    search_tokens = frozenset(token for token in [*field_tokens, *value_tokens] if token not in GENERIC_QUERY_TOKENS)
    return ReverseFactRecord(
        subject=subject,
        relation=relation,
        object_text=object_text,
        field_path=field_path.strip(),
        field_value=field_value.strip(),
        normalized_object=normalized_object,
        field_tokens=field_tokens,
        value_tokens=value_tokens,
        search_tokens=search_tokens,
        identifiers=frozenset(extract_identifier_tokens(object_text)),
    )


def parse_reverse_attribute_query(message: str) -> ReverseAttributeQuery | None:
    normalized = normalize_lookup_text(message)
    if not re.match(r"^(which|what|who)\b", normalized):
        return None
    if " has " not in f" {normalized} ":
        return None
    tokens = frozenset(token for token in _lookup_tokens(message) if token not in GENERIC_QUERY_TOKENS)
    identifiers = frozenset(extract_identifier_tokens(message))
    if not tokens and not identifiers:
        return None
    return ReverseAttributeQuery(
        raw_message=message,
        normalized_message=normalized,
        tokens=tokens,
        identifiers=identifiers,
    )


def scan_reverse_attribute_candidates(
    facts: Iterable[dict[str, object]],
    query: ReverseAttributeQuery,
) -> list[ReverseLookupHit]:
    grouped: dict[str, ReverseLookupHit] = {}
    for fact in facts:
        record = parse_reverse_fact_record(fact)
        if record is None:
            continue
        score, matched_tokens, matched_identifiers = _score_record(record, query)
        if score <= 0:
            continue
        hit = ReverseLookupHit(
            subject=record.subject,
            relation=record.relation,
            object_text=record.object_text,
            score=score,
            matched_tokens=tuple(sorted(matched_tokens)),
            matched_identifiers=tuple(sorted(matched_identifiers)),
        )
        existing = grouped.get(record.subject)
        if existing is None or hit.score > existing.score:
            grouped[record.subject] = hit
    return sorted(grouped.values(), key=lambda item: (-item.score, item.subject.lower()))


class ReverseAttributeIndex:
    def __init__(self, records: list[ReverseFactRecord]) -> None:
        self.records = records
        self._identifier_index: dict[str, set[int]] = defaultdict(set)
        self._token_index: dict[str, set[int]] = defaultdict(set)
        for idx, record in enumerate(records):
            for identifier in record.identifiers:
                self._identifier_index[identifier].add(idx)
            for token in record.search_tokens:
                self._token_index[token].add(idx)

    @classmethod
    def from_facts(cls, facts: Iterable[dict[str, object]]) -> "ReverseAttributeIndex":
        records = [record for fact in facts if (record := parse_reverse_fact_record(fact)) is not None]
        return cls(records)

    def lookup(self, query: ReverseAttributeQuery) -> list[ReverseLookupHit]:
        candidate_indexes: set[int] = set()
        for identifier in query.identifiers:
            candidate_indexes.update(self._identifier_index.get(identifier, set()))
        if not candidate_indexes:
            for token in query.tokens:
                candidate_indexes.update(self._token_index.get(token, set()))
        grouped: dict[str, ReverseLookupHit] = {}
        for idx in candidate_indexes:
            record = self.records[idx]
            score, matched_tokens, matched_identifiers = _score_record(record, query)
            if score <= 0:
                continue
            hit = ReverseLookupHit(
                subject=record.subject,
                relation=record.relation,
                object_text=record.object_text,
                score=score,
                matched_tokens=tuple(sorted(matched_tokens)),
                matched_identifiers=tuple(sorted(matched_identifiers)),
            )
            existing = grouped.get(record.subject)
            if existing is None or hit.score > existing.score:
                grouped[record.subject] = hit
        return sorted(grouped.values(), key=lambda item: (-item.score, item.subject.lower()))


def _score_record(
    record: ReverseFactRecord,
    query: ReverseAttributeQuery,
) -> tuple[float, set[str], set[str]]:
    matched_identifiers = set(record.identifiers & query.identifiers)
    matched_tokens = set(record.search_tokens & query.tokens)
    score = 0.0
    if matched_identifiers:
        score += 25.0 * len(matched_identifiers)
    if matched_tokens:
        score += 3.0 * len(matched_tokens)
        score += 1.0 * len(record.field_tokens & query.tokens)
        score += 1.5 * len(record.value_tokens & query.tokens)
    if query.identifiers and not matched_identifiers:
        score -= 10.0
    if not query.identifiers and len(matched_tokens) < 2:
        score = 0.0
    return score, matched_tokens, matched_identifiers


def _lookup_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_lookup_text(value))
