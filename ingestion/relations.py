from __future__ import annotations

import re
from dataclasses import dataclass


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


class RelationRegistry:
    """Canonical relation labels plus aliases for extracted triples."""

    def __init__(self, aliases: dict[str, tuple[str, ...]] | None = None) -> None:
        self.aliases = aliases or DEFAULT_ALIASES
        self._lookup: dict[str, str] = {}
        for canonical, values in self.aliases.items():
            normalized = self._slug(canonical)
            self._lookup[normalized] = normalized
            for value in values:
                self._lookup[self._slug(value)] = normalized

    def normalize(self, value: str) -> NormalizedRelation:
        slug = self._slug(value)
        canonical = self._lookup.get(slug, slug)
        return NormalizedRelation(
            raw=value,
            slug=slug,
            canonical=canonical,
            matched_alias=canonical != slug,
            registry_hit=slug in self._lookup,
        )

    @staticmethod
    def _slug(value: str) -> str:
        relation = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return relation or "related_to"

