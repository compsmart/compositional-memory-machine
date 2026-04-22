from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str


class FactGraph:
    """Small directed fact graph with local per-key reset revisions."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], str] = {}

    def write(self, source: str, relation: str, target: str) -> None:
        self._edges[(source, relation)] = target

    def per_key_reset(self, source: str, relation: str) -> None:
        self._edges.pop((source, relation), None)

    def revise(self, source: str, relation: str, target: str) -> None:
        self.per_key_reset(source, relation)
        self.write(source, relation, target)

    def read(self, source: str, relation: str) -> str | None:
        return self._edges.get((source, relation))

    def follow_chain(self, source: str, relations: list[str]) -> list[str] | None:
        path = [source]
        current = source
        for relation in relations:
            target = self.read(current, relation)
            if target is None:
                return None
            path.append(target)
            current = target
        return path

    def edges(self) -> list[Edge]:
        return [Edge(source, relation, target) for (source, relation), target in self._edges.items()]
