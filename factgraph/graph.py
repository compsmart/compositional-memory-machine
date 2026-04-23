from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    revision: int = 0


@dataclass(frozen=True)
class EdgeEvent:
    source: str
    relation: str
    target: str
    revision: int


class FactGraph:
    """Small directed fact graph with local per-key reset revisions."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], str] = {}
        self._history: dict[tuple[str, str], list[EdgeEvent]] = {}
        self._revision = 0

    def write(self, source: str, relation: str, target: str) -> None:
        self._edges[(source, relation)] = target
        self._revision += 1
        self._history.setdefault((source, relation), []).append(
            EdgeEvent(source=source, relation=relation, target=target, revision=self._revision)
        )

    def per_key_reset(self, source: str, relation: str) -> None:
        self._edges.pop((source, relation), None)

    def revise(self, source: str, relation: str, target: str) -> None:
        self.per_key_reset(source, relation)
        self.write(source, relation, target)

    def read(self, source: str, relation: str) -> str | None:
        return self._edges.get((source, relation))

    def history(self, source: str, relation: str) -> list[EdgeEvent]:
        return list(self._history.get((source, relation), ()))

    def read_at_revision(self, source: str, relation: str, revision: int) -> str | None:
        history = self._history.get((source, relation), ())
        current: str | None = None
        for event in history:
            if event.revision > revision:
                break
            current = event.target
        return current

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

    def trace_chain(self, source: str, relations: list[str]) -> list[EdgeEvent] | None:
        current = source
        events: list[EdgeEvent] = []
        for relation in relations:
            target = self.read(current, relation)
            if target is None:
                return None
            history = self.history(current, relation)
            if history:
                events.append(history[-1])
            else:
                events.append(EdgeEvent(source=current, relation=relation, target=target, revision=0))
            current = target
        return events

    def edges(self) -> list[Edge]:
        output: list[Edge] = []
        for (source, relation), target in self._edges.items():
            revision = self._history.get((source, relation), [EdgeEvent(source, relation, target, 0)])[-1].revision
            output.append(Edge(source, relation, target, revision=revision))
        return output
