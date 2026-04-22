from __future__ import annotations

from dataclasses import dataclass

from hrr.encoder import SVOEncoder
from memory.amm import AMM


@dataclass
class QueryEngine:
    encoder: SVOEncoder
    memory: AMM
    min_confidence: float = 0.35

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
        return {
            "found": True,
            "key": record.key,
            "confidence": confidence,
            "subject": payload.get("subject", subject),
            "verb": payload.get("verb", verb),
            "object": payload.get("object", object_),
            "domain": payload.get("domain"),
            "source": "amm",
            "novel_composition": record.key
            != f"{payload.get('domain')}:{subject}:{verb}:{object_}",
        }
