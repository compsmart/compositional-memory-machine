from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from factgraph import FactGraph
from hrr.binding import cosine
from hrr.encoder import SVOEncoder, SVOFact
from memory.amm import AMM
from memory.chunked_kg import ChunkedKGMemory

if TYPE_CHECKING:
    from ingestion.relations import RelationRegistry


def _canonical_key(domain: str | None, subject: str, verb: str, object_: str) -> str | None:
    if domain is None:
        return None
    return f"{domain}:{subject}:{verb}:{object_}"


@dataclass
class QueryEngine:
    encoder: SVOEncoder
    memory: AMM
    graph: FactGraph | None = None
    chunk_memory: ChunkedKGMemory | None = None
    relation_registry: "RelationRegistry | None" = None
    min_confidence: float = 0.35
    hop_decay: float = 0.9

    def _dimension_hop_base(self) -> float:
        """Data-fitted per-hop base from the sequential-unbinding frontier sweep."""
        if self.encoder.dim >= 2048:
            return 1.0
        if self.encoder.dim >= 1024:
            return 0.90
        if self.encoder.dim >= 256:
            return 0.45
        return 0.25

    def _dimension_hop_budget(self, hops: int) -> float:
        return self._dimension_hop_base() ** max(hops, 1)

    def _canonical_relation(self, relation: str) -> str:
        if self.relation_registry is None:
            return relation
        return self.relation_registry.normalize(relation).canonical

    def ask_svo(self, subject: str, verb: str, object_: str) -> dict[str, object]:
        canonical_verb = self._canonical_relation(verb)
        vector = self.encoder.encode(subject, canonical_verb, object_)
        record, confidence = self.memory.query(vector)
        if record is None or confidence < self.min_confidence:
            return {
                "found": False,
                "confidence": confidence,
                "subject": subject,
                "verb": canonical_verb,
                "object": object_,
                "source": "amm",
            }

        payload = record.payload
        domain = payload.get("domain")
        return {
            "found": True,
            "key": record.key,
            "confidence": confidence,
            "subject": payload.get("subject", subject),
            "verb": payload.get("verb", canonical_verb),
            "object": payload.get("object", object_),
            "domain": domain,
            "chunk_id": payload.get("chunk_id"),
            "source": "amm",
            "novel_composition": record.key != _canonical_key(domain, subject, canonical_verb, object_),
            "provenance": payload.get("provenance", {}),
        }

    def ask_current_truth(self, subject: str, relation: str) -> dict[str, object]:
        if self.graph is None:
            raise ValueError("graph is required for truth queries")
        canonical_relation = self._canonical_relation(relation)
        summary = self.graph.evidence_summary(subject, canonical_relation)
        current_target = summary["current_target"]
        unresolved = current_target is None
        return {
            "found": not unresolved,
            "subject": subject,
            "relation": canonical_relation,
            "target": current_target,
            "unresolved": unresolved,
            "claim_count": summary["claim_count"],
            "historical_targets": summary["historical_targets"],
            "competing_targets": summary["competing_targets"],
            "provenance": summary["current_provenance"],
            "source": "factgraph",
        }

    def ask_history(self, subject: str, relation: str) -> dict[str, object]:
        if self.graph is None:
            raise ValueError("graph is required for truth queries")
        canonical_relation = self._canonical_relation(relation)
        events = self.graph.history(subject, canonical_relation)
        return {
            "subject": subject,
            "relation": canonical_relation,
            "events": [
                {
                    "target": event.target,
                    "revision": event.revision,
                    "status": event.status,
                    "provenance": event.provenance,
                }
                for event in events
            ],
            "source": "factgraph",
        }

    def ask_chain(self, subject: str, relations: list[str]) -> dict[str, object]:
        if self.graph is None:
            raise ValueError("graph is required for chain queries")
        canonical_relations = [self._canonical_relation(relation) for relation in relations]
        current = subject
        path = [subject]
        steps: list[dict[str, object]] = []
        path_confidence = self._dimension_hop_budget(len(canonical_relations))
        budget_trace: list[dict[str, object]] = []

        for hop, relation in enumerate(canonical_relations, start=1):
            target = self.graph.read(current, relation)
            if target is None:
                return {
                    "found": False,
                    "subject": subject,
                    "relations": canonical_relations,
                    "path": path,
                    "steps": steps,
                    "confidence": 0.0 if not steps else path_confidence,
                    "failed_hop": hop,
                    "budget_trace": budget_trace,
                    "source": "graph+chunk",
                }

            fact = SVOFact(current, relation, target)
            vector = self.encoder.encode_fact(fact)
            evidence = self._step_evidence(fact, vector)
            step_confidence = float(evidence["confidence"])
            budget_trace.append(
                {
                    "hop": hop,
                    "chunk_id": evidence.get("chunk_id"),
                    "estimated_hop1": evidence.get("estimated_hop1"),
                    "chunk_size": evidence.get("chunk_size"),
                    "capacity_budget": evidence.get("capacity_budget"),
                    "perfect_chain_budget": evidence.get("perfect_chain_budget"),
                }
            )
            path_confidence *= max(min(step_confidence, float(evidence.get("estimated_hop1", 1.0))) * self.hop_decay, 1e-6)
            steps.append(
                {
                    "hop": hop,
                    "subject": current,
                    "relation": relation,
                    "target": target,
                    **evidence,
                }
            )
            current = target
            path.append(target)

        if path_confidence < self.min_confidence:
            return {
                "found": False,
                "subject": subject,
                "relations": canonical_relations,
                "path": path,
                "steps": steps,
                "confidence": path_confidence,
                "target": current,
                "budget_trace": budget_trace,
                "budget_exceeded": True,
                "dimension_budget": self._dimension_hop_budget(len(canonical_relations)),
                "source": "graph+chunk",
            }

        return {
            "found": True,
            "subject": subject,
            "relations": canonical_relations,
            "path": path,
            "steps": steps,
            "confidence": path_confidence,
            "target": current,
            "budget_trace": budget_trace,
            "source": "graph+chunk",
            "dimension_budget": self._dimension_hop_budget(len(canonical_relations)),
        }

    def ask_branching_chain(
        self,
        subject: str,
        relations: list[str],
        *,
        branch_limit: int = 6,
    ) -> dict[str, object]:
        if self.graph is None:
            raise ValueError("graph is required for chain queries")
        canonical_relations = [self._canonical_relation(relation) for relation in relations]
        branches: list[dict[str, object]] = [
            {
                "path": [subject],
                "steps": [],
                "confidence": 1.0,
            }
        ]
        for hop, relation in enumerate(canonical_relations, start=1):
            next_branches: list[dict[str, object]] = []
            for branch in branches:
                current = str(branch["path"][-1])
                candidates = self._branch_candidates(current, relation)
                for event in candidates:
                    fact = SVOFact(current, relation, event.target)
                    vector = self.encoder.encode_fact(fact)
                    evidence = self._step_evidence(fact, vector)
                    step_confidence = float(evidence["confidence"])
                    status_penalty = 1.0 if event.status == "current" else 0.8
                    next_branches.append(
                        {
                            "path": [*branch["path"], event.target],
                            "steps": [
                                *branch["steps"],
                                {
                                    "hop": hop,
                                    "subject": current,
                                    "relation": relation,
                                    "target": event.target,
                                    "status": event.status,
                                    "revision": event.revision,
                                    "provenance": event.provenance,
                                    **evidence,
                                },
                            ],
                            "confidence": float(branch["confidence"]) * max(step_confidence * status_penalty, 1e-6),
                        }
                    )
            if not next_branches:
                return {
                    "found": False,
                    "subject": subject,
                    "relations": canonical_relations,
                    "branches": [],
                    "failed_hop": hop,
                    "source": "factgraph+chunk",
                    "dimension_budget": self._dimension_hop_budget(len(canonical_relations)),
                }
            next_branches.sort(key=lambda row: float(row["confidence"]), reverse=True)
            branches = next_branches[:branch_limit]
        return {
            "found": True,
            "subject": subject,
            "relations": canonical_relations,
            "branches": branches,
            "source": "factgraph+chunk",
            "dimension_budget": self._dimension_hop_budget(len(canonical_relations)),
        }

    def ask_relational(
        self,
        subject: str,
        relation_sequence: list[str],
        constraints: dict[str, object] | None = None,
    ) -> dict[str, object]:
        constraints = constraints or {}
        response = self.ask_chain(subject, relation_sequence)
        if response["found"] and "target" in constraints and response.get("target") != constraints["target"]:
            response["found"] = False
            response["constraint_mismatch"] = True
        response["constraints"] = constraints
        return response

    def _branch_candidates(self, subject: str, relation: str) -> list[object]:
        assert self.graph is not None
        candidates: list[object] = []
        current = self.graph.current_claim(subject, relation)
        if current is not None:
            candidates.append(current)
        seen_targets = {current.target} if current is not None else set()
        for event in self.graph.history(subject, relation):
            if event.status == "current" or event.target in seen_targets:
                continue
            seen_targets.add(event.target)
            candidates.append(event)
        return candidates

    def _step_evidence(self, fact: SVOFact, vector) -> dict[str, object]:
        if self.chunk_memory is None:
            return {
                "confidence": 1.0,
                "chunk_id": None,
                "candidate_chunks": [],
                "chunk_size": 0,
                "capacity_budget": None,
                "perfect_chain_budget": None,
                "estimated_hop1": 1.0,
            }

        exact = self.chunk_memory.lookup(fact.subject, fact.verb, fact.object)
        if exact is not None:
            budget = self.chunk_memory.chunk_budget(exact.chunk_id)
            return {
                "confidence": cosine(vector, exact.vector),
                "chunk_id": exact.chunk_id,
                "candidate_chunks": [chunk.chunk_id for chunk in self.chunk_memory.chunks_for_entity(fact.subject)],
                "chunk_size": exact.payload.get("chunk_size", self.chunk_memory.chunks[exact.chunk_id].size),
                **budget,
            }

        candidate_chunks = [chunk.chunk_id for chunk in self.chunk_memory.chunks_for_entity(fact.subject)]
        nearest = self.chunk_memory.nearest(vector, top_k=1, candidate_chunks=candidate_chunks or None)
        if not nearest:
            return {
                "confidence": 0.0,
                "chunk_id": None,
                "candidate_chunks": candidate_chunks,
                "chunk_size": 0,
                "capacity_budget": self.chunk_memory.capacity_budget,
                "perfect_chain_budget": self.chunk_memory.perfect_chain_budget,
                "estimated_hop1": 0.0,
            }
        best_record, score = nearest[0]
        budget = self.chunk_memory.chunk_budget(best_record.chunk_id)
        return {
            "confidence": score,
            "chunk_id": best_record.chunk_id,
            "candidate_chunks": candidate_chunks,
            "chunk_size": self.chunk_memory.chunks[best_record.chunk_id].size,
            **budget,
        }
