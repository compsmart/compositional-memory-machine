from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from hrr.binding import bind, normalize
from hrr.vectors import VectorStore
from memory import AMM


_ORG_MARKERS = {
    "company",
    "corp",
    "corporation",
    "inc",
    "ltd",
    "lab",
    "labs",
    "institute",
    "group",
    "team",
    "university",
}
_LOCATION_MARKERS = {
    "city",
    "region",
    "campus",
    "office",
    "site",
    "hq",
    "harbor",
    "valley",
    "station",
}
_PERSON_ROLE_MARKERS = {
    "engineer",
    "scientist",
    "designer",
    "advisor",
    "principal",
    "student",
    "intern",
    "manager",
    "director",
    "lead",
    "analyst",
    "researcher",
    "mentor",
    "founder",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class RelationConceptMatch:
    canonical: str
    score: float
    margin: float
    support_count: int = 0


class RelationConceptMemory:
    """Experimental typed relation concept memory for unknown-relation fallback."""

    def __init__(self, *, dim: int = 512, seed: int = 0, cue_limit: int = 6) -> None:
        self.dim = dim
        self.seed = seed
        self.cue_limit = cue_limit
        self.store = VectorStore(dim=dim, seed=seed)
        self.memory = AMM()
        self.role_domain = self.store.get_unitary("__REL_DOMAIN__")
        self.role_subject_kind = self.store.get_unitary("__REL_SUBJECT_KIND__")
        self.role_object_kind = self.store.get_unitary("__REL_OBJECT_KIND__")
        self.role_subject_hint = self.store.get_unitary("__REL_SUBJECT_HINT__")
        self.role_object_hint = self.store.get_unitary("__REL_OBJECT_HINT__")
        self.role_cue = self.store.get_unitary("__REL_CUE__")
        self.role_relation_surface = self.store.get_unitary("__REL_SURFACE__")
        self.role_relation_bigram = self.store.get_unitary("__REL_BIGRAM__")

    def observe_fact(
        self,
        canonical: str,
        fact: object,
        *,
        domain: str = "",
        slot_cleaner: Callable[[str], str] | None = None,
    ) -> None:
        canonical_slug = self._slug(canonical)
        vector = self._encode_fact(fact, domain=domain, slot_cleaner=slot_cleaner)
        existing = self.memory.get(f"relation:{canonical_slug}")
        support_count = 1 if existing is None else existing.writes + 1
        self.memory.write(
            f"relation:{canonical_slug}",
            vector,
            {"canonical": canonical_slug, "support_count": support_count},
        )

    def classify_fact(
        self,
        fact: object,
        *,
        domain: str = "",
        slot_cleaner: Callable[[str], str] | None = None,
    ) -> RelationConceptMatch | None:
        vector = self._encode_fact(fact, domain=domain, slot_cleaner=slot_cleaner)
        nearest = self.memory.nearest(vector, top_k=2)
        if not nearest:
            return None
        best_record, best_score = nearest[0]
        second_score = nearest[1][1] if len(nearest) > 1 else 0.0
        return RelationConceptMatch(
            canonical=str(best_record.payload["canonical"]),
            score=float(best_score),
            margin=float(best_score - second_score),
            support_count=int(best_record.payload.get("support_count", best_record.writes)),
        )

    def _encode_fact(
        self,
        fact: object,
        *,
        domain: str,
        slot_cleaner: Callable[[str], str] | None,
    ) -> np.ndarray:
        cleaner = slot_cleaner or self._clean_slot
        subject = cleaner(str(getattr(fact, "subject", "")))
        object_ = cleaner(str(getattr(fact, "object", "")))
        relation = str(getattr(fact, "relation", ""))
        excerpt = str(getattr(fact, "excerpt", ""))

        parts: list[np.ndarray] = [
            bind(self.role_domain, self.store.get(f"domain:{self._slug(domain or 'default')}")),
            bind(self.role_subject_kind, self.store.get(f"subject_kind:{self._entity_kind(subject)}")),
            bind(self.role_object_kind, self.store.get(f"object_kind:{self._entity_kind(object_)}")),
            bind(self.role_subject_hint, self.store.get(f"subject_hint:{self._entity_hint(subject)}")),
            bind(self.role_object_hint, self.store.get(f"object_hint:{self._entity_hint(object_)}")),
        ]
        for token in self._relation_surface_tokens(relation):
            parts.append(bind(self.role_relation_surface, self.store.get(f"relation_surface:{token}")))
        for cue in self._cue_tokens(excerpt, relation=relation, subject=subject, object_=object_):
            parts.append(bind(self.role_cue, self.store.get(f"cue:{cue}")))
        for bigram in self._bigrams(excerpt):
            parts.append(bind(self.role_relation_bigram, self.store.get(f"cue_bigram:{bigram}")))
        return normalize(np.sum(parts, axis=0))

    def _cue_tokens(self, excerpt: str, *, relation: str, subject: str, object_: str) -> list[str]:
        blocked = {
            *self._tokens(subject),
            *self._tokens(object_),
            *self._tokens(relation),
        }
        cues: list[str] = []
        for token in self._tokens(excerpt):
            if token in _STOPWORDS or token in blocked:
                continue
            cues.append(token)
            if len(cues) >= self.cue_limit:
                break
        if not cues:
            cues.append("no_excerpt_cue")
        return cues

    def _relation_surface_tokens(self, relation: str) -> list[str]:
        tokens = [token for token in self._tokens(relation) if token not in _STOPWORDS]
        return tokens[: self.cue_limit]

    def _bigrams(self, excerpt: str) -> list[str]:
        tokens = [token for token in self._tokens(excerpt) if token not in _STOPWORDS]
        bigrams: list[str] = []
        for index in range(max(0, len(tokens) - 1)):
            bigrams.append(f"{tokens[index]}_{tokens[index + 1]}")
            if len(bigrams) >= self.cue_limit:
                break
        return bigrams

    def _entity_kind(self, value: str) -> str:
        tokens = self._tokens(value)
        if not tokens:
            return "empty"
        if any(token in _ORG_MARKERS for token in tokens):
            return "organization"
        if any(token in _LOCATION_MARKERS for token in tokens):
            return "location"
        if any(token in _PERSON_ROLE_MARKERS for token in tokens):
            return "person_role"
        if any(char.isdigit() for char in value):
            return "contains_digit"
        if len(value.split()) >= 2 and all(part[:1].isupper() for part in value.split() if part):
            return "proper_name"
        if len(tokens) == 1:
            return "single_token"
        return "generic_entity"

    def _entity_hint(self, value: str) -> str:
        tokens = self._tokens(value)
        if not tokens:
            return "empty"
        role_tokens = [token for token in tokens if token in _PERSON_ROLE_MARKERS | _ORG_MARKERS | _LOCATION_MARKERS]
        if role_tokens:
            return role_tokens[0]
        return tokens[-1]

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value.lower())

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return slug or "related_to"

    @staticmethod
    def _clean_slot(value: str) -> str:
        return " ".join(value.strip().split())
