from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    revision: int = 0


@dataclass
class EdgeEvent:
    source: str
    relation: str
    target: str
    revision: int
    status: str = "current"
    provenance: dict[str, object] = field(default_factory=dict)


class FactGraph:
    """Small directed fact graph with local per-key reset revisions."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], str] = {}
        self._history: dict[tuple[str, str], list[EdgeEvent]] = {}
        self._revision = 0

    def write(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        provenance: dict[str, object] | None = None,
    ) -> None:
        self._edges[(source, relation)] = target
        self._revision += 1
        history = self._history.setdefault((source, relation), [])
        if history:
            previous = history[-1]
            if previous.status == "current":
                previous.status = "superseded"
        history.append(
            EdgeEvent(
                source=source,
                relation=relation,
                target=target,
                revision=self._revision,
                status="current",
                provenance=(provenance or {}).copy(),
            )
        )

    def add_evidence(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        provenance: dict[str, object] | None = None,
        make_current: bool = False,
    ) -> None:
        if make_current:
            self.write(source, relation, target, provenance=provenance)
            return
        self._revision += 1
        self._history.setdefault((source, relation), []).append(
            EdgeEvent(
                source=source,
                relation=relation,
                target=target,
                revision=self._revision,
                status="evidence",
                provenance=(provenance or {}).copy(),
            )
        )

    def per_key_reset(self, source: str, relation: str) -> None:
        self._edges.pop((source, relation), None)

    def revise(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        provenance: dict[str, object] | None = None,
    ) -> None:
        self.per_key_reset(source, relation)
        self.write(source, relation, target, provenance=provenance)

    def read(self, source: str, relation: str) -> str | None:
        return self._edges.get((source, relation))

    def history(self, source: str, relation: str) -> list[EdgeEvent]:
        return list(self._history.get((source, relation), ()))

    def current_claim(self, source: str, relation: str) -> EdgeEvent | None:
        history = self._history.get((source, relation), ())
        for event in reversed(history):
            if event.status == "current":
                return event
        return None

    def evidence(self, source: str, relation: str) -> list[EdgeEvent]:
        return self.history(source, relation)

    def evidence_summary(self, source: str, relation: str) -> dict[str, object]:
        history = self.history(source, relation)
        current = self.current_claim(source, relation)
        competing = sorted(
            {
                event.target
                for event in history
                if event.status == "evidence" and (current is None or event.target != current.target)
            }
        )
        return {
            "source": source,
            "relation": relation,
            "current_target": None if current is None else current.target,
            "current_revision": None if current is None else current.revision,
            "current_provenance": {} if current is None else current.provenance.copy(),
            "claim_count": len(history),
            "historical_targets": [event.target for event in history if event.status == "superseded"],
            "competing_targets": competing,
        }

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
                current_claim = self.current_claim(current, relation)
                events.append(current_claim or history[-1])
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
