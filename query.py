from __future__ import annotations

from dataclasses import dataclass

from factgraph import FactGraph
from hrr.binding import cosine
from hrr.encoder import SVOEncoder, SVOFact
from memory.amm import AMM
from memory.chunked_kg import ChunkedKGMemory


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
    min_confidence: float = 0.35
    hop_decay: float = 0.9

    def ask_svo(self, subject: str, verb: str, object_: str) -> dict[str, object]:
        vector = self.encoder.encode(subject, verb, object_)
        record, confidence = self.memory.query(vector)
        if record is None or confidence < self.min_confidence:
            return {
                "found": False,
                "confidence": confidence,
                "subject": subject,
                "verb": verb,
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
            "verb": payload.get("verb", verb),
            "object": payload.get("object", object_),
            "domain": domain,
            "chunk_id": payload.get("chunk_id"),
            "source": "amm",
            "novel_composition": record.key != _canonical_key(domain, subject, verb, object_),
        }

    def ask_chain(self, subject: str, relations: list[str]) -> dict[str, object]:
        if self.graph is None:
            raise ValueError("graph is required for chain queries")
        current = subject
        path = [subject]
        steps: list[dict[str, object]] = []
        path_confidence = 1.0

        for hop, relation in enumerate(relations, start=1):
            target = self.graph.read(current, relation)
            if target is None:
                return {
                    "found": False,
                    "subject": subject,
                    "relations": relations,
                    "path": path,
                    "steps": steps,
                    "confidence": 0.0 if not steps else path_confidence,
                    "failed_hop": hop,
                    "source": "graph+chunk",
                }

            fact = SVOFact(current, relation, target)
            vector = self.encoder.encode_fact(fact)
            evidence = self._step_evidence(fact, vector)
            step_confidence = float(evidence["confidence"])
            path_confidence *= max(step_confidence * self.hop_decay, 1e-6)
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

        return {
            "found": True,
            "subject": subject,
            "relations": relations,
            "path": path,
            "steps": steps,
            "confidence": path_confidence,
            "target": current,
            "source": "graph+chunk",
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

    def _step_evidence(self, fact: SVOFact, vector) -> dict[str, object]:
        if self.chunk_memory is None:
            return {"confidence": 1.0, "chunk_id": None, "candidate_chunks": []}

        exact = self.chunk_memory.lookup(fact.subject, fact.verb, fact.object)
        if exact is not None:
            return {
                "confidence": cosine(vector, exact.vector),
                "chunk_id": exact.chunk_id,
                "candidate_chunks": [chunk.chunk_id for chunk in self.chunk_memory.chunks_for_entity(fact.subject)],
            }

        candidate_chunks = [chunk.chunk_id for chunk in self.chunk_memory.chunks_for_entity(fact.subject)]
        nearest = self.chunk_memory.nearest(vector, top_k=1, candidate_chunks=candidate_chunks or None)
        if not nearest:
            return {"confidence": 0.0, "chunk_id": None, "candidate_chunks": candidate_chunks}
        best_record, score = nearest[0]
        return {
            "confidence": score,
            "chunk_id": best_record.chunk_id,
            "candidate_chunks": candidate_chunks,
        }
