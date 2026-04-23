from __future__ import annotations

from collections import defaultdict
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .relation_concepts import RelationConceptMemory


DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "worked_with": (
        "collaborated_with",
        "collaborated with",
        "worked on with",
        "worked alongside",
    ),
    "published_notes_about": (
        "published notes about",
        "wrote notes about",
        "authored notes about",
    ),
    "described": (
        "describes",
        "described in",
    ),
    "proposed_as": (
        "was proposed as",
        "proposed mechanical",
    ),
}


@dataclass(frozen=True)
class NormalizedRelation:
    raw: str
    slug: str
    canonical: str
    matched_alias: bool
    registry_hit: bool
    resolution_source: str
    evidence_count: int = 0


@dataclass(frozen=True)
class RelationProposal:
    alias: str
    canonical: str
    status: str
    support_pairs: int
    typed_support: int
    mean_score: float
    mean_margin: float
    source: str


class RelationRegistry:
    """Canonical relation labels plus seed and learned aliases for extracted triples."""

    def __init__(
        self,
        aliases: dict[str, tuple[str, ...]] | None = None,
        *,
        min_support_pairs: int = 2,
        enable_typed_fallback: bool | None = None,
        typed_fallback_score_threshold: float = 0.45,
        typed_fallback_margin_threshold: float = 0.08,
        typed_fallback_min_support: int = 1,
        concept_memory: RelationConceptMemory | None = None,
    ) -> None:
        self.aliases = aliases or DEFAULT_ALIASES
        self.min_support_pairs = max(1, min_support_pairs)
        self.enable_typed_fallback = self._env_flag("HHR_ENABLE_TYPED_RELATION_FALLBACK") if enable_typed_fallback is None else enable_typed_fallback
        self.typed_fallback_score_threshold = typed_fallback_score_threshold
        self.typed_fallback_margin_threshold = typed_fallback_margin_threshold
        self.typed_fallback_min_support = max(1, typed_fallback_min_support)
        self.concept_memory = concept_memory or (RelationConceptMemory() if self.enable_typed_fallback else None)
        self._seed_lookup: dict[str, str] = {}
        self._learned_lookup: dict[str, str] = {}
        self._learned_sources: dict[str, str] = {}
        self._canonical_labels: set[str] = set()
        self._learned_targets: set[str] = set()
        self._pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._support: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        self._typed_support: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        self._proposal_status: dict[tuple[str, str], str] = {}
        self._proposal_sources: dict[tuple[str, str], str] = {}
        for canonical, values in self.aliases.items():
            normalized = self._slug(canonical)
            self._seed_lookup[normalized] = normalized
            self._canonical_labels.add(normalized)
            for value in values:
                self._seed_lookup[self._slug(value)] = normalized

    def normalize(self, value: str) -> NormalizedRelation:
        slug = self._slug(value)
        canonical, resolution_source = self._resolve(slug)
        registry_hit = slug in self._seed_lookup or slug in self._learned_lookup or slug in self._canonical_labels
        evidence_count = 0
        if slug != canonical and resolution_source in {"pair_overlap", "typed_fallback"}:
            evidence_count = len(self._support.get((slug, canonical), set()))
        return NormalizedRelation(
            raw=value,
            slug=slug,
            canonical=canonical,
            matched_alias=canonical != slug,
            registry_hit=registry_hit,
            resolution_source=resolution_source,
            evidence_count=evidence_count,
        )

    def normalize_fact(
        self,
        fact: object,
        *,
        domain: str = "",
        slot_cleaner: Callable[[str], str] | None = None,
    ) -> NormalizedRelation:
        normalized = self.normalize(str(getattr(fact, "relation", "")))
        if normalized.registry_hit:
            return normalized
        if not self.enable_typed_fallback or self.concept_memory is None:
            return normalized
        match = self.concept_memory.classify_fact(fact, domain=domain, slot_cleaner=slot_cleaner)
        if match is None:
            return normalized
        if match.score < self.typed_fallback_score_threshold or match.margin < self.typed_fallback_margin_threshold:
            self._record_proposal(
                normalized.slug,
                match.canonical,
                status="rejected_low_confidence",
                source="typed_fallback",
            )
            return normalized
        if match.canonical == normalized.slug:
            return normalized
        self._typed_support[(normalized.slug, match.canonical)].append((match.score, match.margin))
        support = len(self._typed_support[(normalized.slug, match.canonical)])
        if support < self.typed_fallback_min_support:
            self._record_proposal(
                normalized.slug,
                match.canonical,
                status="pending",
                source="typed_fallback",
            )
            return normalized
        self.register_alias(normalized.slug, match.canonical, source="typed_fallback")
        return self.normalize(str(getattr(fact, "relation", "")))

    def observe_fact(
        self,
        subject: str,
        relation: str,
        object_: str,
        *,
        slot_cleaner: Callable[[str], str] | None = None,
    ) -> list[str]:
        cleaner = slot_cleaner or self._clean_slot
        pair = (cleaner(subject).lower(), cleaner(object_).lower())
        if not pair[0] or not pair[1]:
            return []

        normalized = self.normalize(relation)
        labels = self._pair_relations[pair]
        labels.add(normalized.canonical)

        known_targets = {label for label in labels if self._is_canonical_target(label)}
        unresolved_labels = {label for label in labels if not self._is_registered_label(label)}
        for unresolved in unresolved_labels:
            for canonical in known_targets:
                if canonical != unresolved:
                    self._support[(unresolved, canonical)].add(pair)

        return self._promote_candidates()

    def learn_from_facts(
        self,
        facts: Iterable[object],
        *,
        slot_cleaner: Callable[[str], str] | None = None,
    ) -> list[str]:
        learned: set[str] = set()
        for fact in facts:
            learned.update(
                self.observe_fact(
                    str(getattr(fact, "subject", "")),
                    str(getattr(fact, "relation", "")),
                    str(getattr(fact, "object", "")),
                    slot_cleaner=slot_cleaner,
                )
            )
        return sorted(learned)

    def learned_aliases(self) -> dict[str, str]:
        return dict(sorted((alias, self.normalize(canonical).canonical) for alias, canonical in self._learned_lookup.items()))

    def register_alias(self, alias: str, canonical: str, *, source: str) -> None:
        alias_slug = self._slug(alias)
        canonical_slug = self.normalize(canonical).canonical
        if alias_slug == canonical_slug:
            self._canonical_labels.add(canonical_slug)
            return
        self._learned_lookup[alias_slug] = canonical_slug
        self._learned_sources[alias_slug] = source
        self._canonical_labels.add(canonical_slug)
        self._learned_targets.add(canonical_slug)
        self._record_proposal(alias_slug, canonical_slug, status="accepted", source=source)
        self._relabel_pairs(alias_slug, canonical_slug)

    def observe_resolved_fact(
        self,
        fact: object,
        *,
        canonical_relation: str,
        domain: str = "",
        slot_cleaner: Callable[[str], str] | None = None,
    ) -> None:
        if self.concept_memory is not None:
            self.concept_memory.observe_fact(
                canonical_relation,
                fact,
                domain=domain,
                slot_cleaner=slot_cleaner,
            )

    def candidate_aliases(self, *, limit: int = 10) -> list[dict[str, object]]:
        rows = [self._proposal_row(alias, canonical) for alias, canonical in self._proposal_keys(include_accepted=False)]
        rows.sort(
            key=lambda row: (
                -int(row["support_pairs"]),
                -int(row["typed_support"]),
                -float(row["mean_score"]),
                str(row["alias"]),
                str(row["canonical"]),
            )
        )
        return rows[:limit]

    def proposal_log(self, *, limit: int = 20) -> list[dict[str, object]]:
        rows = [self._proposal_row(alias, canonical) for alias, canonical in self._proposal_keys(include_accepted=True)]
        rows.sort(
            key=lambda row: (
                0 if row["status"] == "accepted" else 1,
                -int(row["support_pairs"]),
                -int(row["typed_support"]),
                -float(row["mean_score"]),
                str(row["alias"]),
            )
        )
        return rows[:limit]

    @staticmethod
    def _slug(value: str) -> str:
        relation = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return relation or "related_to"

    @staticmethod
    def _clean_slot(value: str) -> str:
        return " ".join(value.strip().split())

    def _resolve(self, slug: str) -> tuple[str, str]:
        current = slug
        resolution_source = "self"
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            if current in self._learned_lookup:
                alias = current
                current = self._learned_lookup[alias]
                resolution_source = self._learned_sources.get(alias, "pair_overlap")
                continue
            if current in self._seed_lookup:
                resolved = self._seed_lookup[current]
                if resolved != current:
                    resolution_source = "seed"
                    current = resolved
                    continue
                if resolution_source == "self":
                    resolution_source = "seed"
                break
            break
        return current, resolution_source

    def _promote_candidates(self) -> list[str]:
        promotions: list[str] = []
        grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (alias, canonical), pairs in self._support.items():
            if alias == canonical or alias in self._learned_lookup:
                continue
            grouped[alias].append((self.normalize(canonical).canonical, len(pairs)))

        for alias, candidates in grouped.items():
            candidates.sort(key=lambda item: (-item[1], item[0]))
            best_canonical, best_support = candidates[0]
            next_support = candidates[1][1] if len(candidates) > 1 else -1
            if best_support < self.min_support_pairs or best_support == next_support:
                self._record_proposal(alias, best_canonical, status="pending", source="pair_overlap")
                continue
            self.register_alias(alias, best_canonical, source="pair_overlap")
            promotions.append(alias)
        return sorted(promotions)

    def _relabel_pairs(self, alias: str, canonical: str) -> None:
        for labels in self._pair_relations.values():
            if alias in labels:
                labels.discard(alias)
                labels.add(canonical)

    def _is_registered_label(self, label: str) -> bool:
        return label in self._seed_lookup or label in self._learned_lookup or label in self._canonical_labels

    def _is_canonical_target(self, label: str) -> bool:
        return label in self._canonical_labels or label in self._learned_targets

    def _record_proposal(self, alias: str, canonical: str, *, status: str, source: str) -> None:
        key = (self._slug(alias), self.normalize(canonical).canonical)
        previous = self._proposal_status.get(key)
        if previous == "accepted" and status != "accepted":
            return
        self._proposal_status[key] = status
        self._proposal_sources[key] = source

    def _proposal_keys(self, *, include_accepted: bool) -> list[tuple[str, str]]:
        keys = set(self._proposal_status)
        keys.update(self._support)
        keys.update(self._typed_support)
        rows = []
        for alias, canonical in keys:
            canonical_slug = self.normalize(canonical).canonical
            if alias == canonical_slug:
                continue
            if not include_accepted and alias in self._learned_lookup:
                continue
            rows.append((alias, canonical_slug))
        return rows

    def _proposal_row(self, alias: str, canonical: str) -> dict[str, object]:
        typed_rows = self._typed_support.get((alias, canonical), [])
        mean_score = sum(score for score, _ in typed_rows) / len(typed_rows) if typed_rows else 0.0
        mean_margin = sum(margin for _, margin in typed_rows) / len(typed_rows) if typed_rows else 0.0
        status = self._proposal_status.get((alias, canonical), "pending")
        return {
            "alias": alias,
            "canonical": self.normalize(canonical).canonical,
            "status": status,
            "support_pairs": len(self._support.get((alias, canonical), set())),
            "typed_support": len(typed_rows),
            "mean_score": round(mean_score, 3),
            "mean_margin": round(mean_margin, 3),
            "source": self._proposal_sources.get((alias, canonical), "pair_overlap"),
        }

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

