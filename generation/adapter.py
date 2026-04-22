from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class FrozenGeneratorAdapter:
    generator: TextGenerator | None = None
    min_confidence: float = 0.35

    def answer(self, question: str, result: dict[str, object]) -> str:
        confidence = float(result.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return "I do not have a reliable memory for that question."

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
