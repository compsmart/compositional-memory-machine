from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .compositional import CompositionalValueDecoder


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class FrozenGeneratorAdapter:
    generator: TextGenerator | None = None
    min_confidence: float = 0.35
    value_decoder: CompositionalValueDecoder | None = None
    value_strategy: str = "hrr_native"

    def answer(self, question: str, result: dict[str, object]) -> str:
        confidence = float(result.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return "I do not have a reliable memory for that question."

        if self.value_decoder is not None and "value_vector" in result:
            return self._answer_compositional(question, result)

        subject = result.get("subject")
        verb = result.get("verb")
        object_ = result.get("object")
        if self.generator is None:
            return f"{subject} {verb} {object_}."

        prompt = (
            "Answer the question using only this structured memory.\n"
            f"Question: {question}\n"
            f"Subject: {subject}\n"
            f"Verb: {verb}\n"
            f"Object: {object_}\n"
            "Answer:"
        )
        return self.generator.generate(prompt)

    def _answer_compositional(self, question: str, result: dict[str, object]) -> str:
        decoded = self.value_decoder.decode(
            np.asarray(result["value_vector"], dtype=float),
            strategy=str(result.get("decode_strategy", self.value_strategy)),
        )
        entity = result.get("entity") or result.get("subject") or "The entity"
        if self.generator is None:
            return f"{entity} has property {decoded.text}."

        prompt = (
            "Answer the question using only this structured compositional memory.\n"
            f"Question: {question}\n"
            f"Entity: {entity}\n"
            f"Decoded adjective: {decoded.adjective}\n"
            f"Decoded noun: {decoded.noun}\n"
            f"Decoded text: {decoded.text}\n"
            "Answer:"
        )
        return self.generator.generate(prompt)
