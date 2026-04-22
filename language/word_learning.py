from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrr.binding import bind, cosine, normalize, unbind
from hrr.vectors import VectorStore
from memory import AMM


@dataclass(frozen=True)
class ContextExample:
    subject: str
    action: str
    object: str
    property_hint: str | None = None


class WordLearningMemory:
    """Learns an unknown action word from HRR role-unbound context examples."""

    def __init__(self, dim: int = 2048, seed: int = 0) -> None:
        self.dim = dim
        self.store = VectorStore(dim=dim, seed=seed)
        self.role_subject = self.store.get_unitary("__WL_SUBJECT__")
        self.role_action = self.store.get_unitary("__WL_ACTION__")
        self.role_object = self.store.get_unitary("__WL_OBJECT__")
        self.role_property = self.store.get_unitary("__WL_PROPERTY__")
        self.lexicon = AMM()
        self.clusters = AMM()
        self.action_properties: dict[str, str] = {}

    def add_known_action(self, action: str, cluster: str, property_hint: str | None = None) -> None:
        property_hint = property_hint or cluster
        self.action_properties[action] = property_hint
        action_vector = self._action_property_vector(action, property_hint)
        self.clusters.write(
            f"cluster:{cluster}:{action}",
            action_vector,
            {"action": action, "cluster": cluster, "property": property_hint},
        )

    def context_vector(self, example: ContextExample) -> np.ndarray:
        vector = (
            bind(self.role_subject, self.store.get(f"entity:{example.subject}"))
            + bind(self.role_action, self.store.get(f"action:{example.action}"))
            + bind(self.role_object, self.store.get(f"entity:{example.object}"))
        )
        if example.property_hint:
            vector = vector + bind(
                self.role_action,
                bind(self.role_property, self.store.get(f"property:{example.property_hint}")),
            )
        return normalize(vector)

    def learn_word(self, word: str, examples: list[ContextExample]) -> dict[str, object]:
        if not examples:
            raise ValueError("At least one context example is required")
        extracted = [normalize(unbind(self.context_vector(example), self.role_action)) for example in examples]
        centroid = normalize(np.mean(extracted, axis=0))
        self.lexicon.write(f"word:{word}", centroid, {"word": word, "examples": len(examples)})
        cluster_record, cluster_score = self.clusters.query(centroid)
        return {
            "word": word,
            "examples": len(examples),
            "cluster": cluster_record.payload["cluster"] if cluster_record else None,
            "nearest_action": cluster_record.payload["action"] if cluster_record else None,
            "confidence": cluster_score,
        }

    def retrieve_word(self, word: str) -> dict[str, object]:
        record = self.lexicon.get(f"word:{word}")
        if record is None:
            return {"found": False, "word": word}
        cluster_record, cluster_score = self.clusters.query(record.vector)
        return {
            "found": True,
            "word": word,
            "cluster": cluster_record.payload["cluster"] if cluster_record else None,
            "nearest_action": cluster_record.payload["action"] if cluster_record else None,
            "confidence": cluster_score,
            "writes": record.writes,
        }

    def plausibility(self, word: str, candidate_action: str) -> float:
        record = self.lexicon.get(f"word:{word}")
        if record is None:
            return 0.0
        property_hint = self.action_properties.get(candidate_action)
        if property_hint is None:
            candidate = self.store.get(f"action:{candidate_action}")
        else:
            candidate = self._action_property_vector(candidate_action, property_hint)
        return cosine(record.vector, candidate)

    def _action_property_vector(self, action: str, property_hint: str) -> np.ndarray:
        return normalize(
            self.store.get(f"action:{action}")
            + bind(self.role_property, self.store.get(f"property:{property_hint}"))
        )
