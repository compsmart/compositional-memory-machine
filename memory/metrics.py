from __future__ import annotations

from collections.abc import Iterable

from .amm import AMM


def top1_accuracy(memory: AMM, examples: Iterable[tuple[str, object]]) -> float:
    examples = list(examples)
    if not examples:
        return 0.0
    correct = 0
    for expected_key, vector in examples:
        record, _score = memory.query(vector)
        correct += int(record is not None and record.key == expected_key)
    return correct / len(examples)


def exact_match_rate(predicted: Iterable[str], expected: Iterable[str]) -> float:
    pairs = list(zip(predicted, expected, strict=False))
    if not pairs:
        return 0.0
    return sum(int(left == right) for left, right in pairs) / len(pairs)


def forgetting_rate(before: dict[str, str], after: dict[str, str]) -> float:
    if not before:
        return 0.0
    forgotten = sum(1 for key, value in before.items() if after.get(key) != value)
    return forgotten / len(before)
