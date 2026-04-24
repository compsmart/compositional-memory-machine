from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reverse_lookup import ReverseAttributeIndex, parse_reverse_attribute_query, scan_reverse_attribute_candidates


@dataclass(frozen=True)
class ReverseLookupCase:
    question: str
    expected_subjects: tuple[str, ...]


@dataclass(frozen=True)
class StrategyResult:
    name: str
    accuracy: float
    exact_match_accuracy: float
    abstain_accuracy: float
    mean_latency_ms: float
    notes: str


def _fixture_facts(noise_facts: int = 0) -> list[dict[str, object]]:
    facts = [
        {
            "subject": "Sulfametrole",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > ATC code: - J01EE03 (WHO) (with trimethoprim)",
        },
        {
            "subject": "Sulfametrole",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > Routes of administration: Oral",
        },
        {
            "subject": "Sulfamethazine",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > Routes of administration: Oral",
        },
        {
            "subject": "Amikacin",
            "relation": "infobox_has",
            "object": "Drugbox > Clinical data > Routes of administration: Intravenous",
        },
        {
            "subject": "Amikacin",
            "relation": "infobox_has",
            "object": "Drugbox > Identifiers > CAS Number: 39831-55-5",
        },
        {
            "subject": "Amikacin",
            "relation": "infobox_has",
            "object": "Drugbox > Identifiers > PubChem CID: 37768",
        },
    ]
    for idx in range(noise_facts):
        facts.append(
            {
                "subject": f"NoiseCompound{idx}",
                "relation": "infobox_has",
                "object": f"Drugbox > Identifiers > Noise Code: ZX{idx:05d}",
            }
        )
    return facts


def _cases() -> list[ReverseLookupCase]:
    return [
        ReverseLookupCase(
            question="Which chemical compound has HTC code of J01EE03?",
            expected_subjects=("Sulfametrole",),
        ),
        ReverseLookupCase(
            question="Which drug has CAS Number 39831-55-5?",
            expected_subjects=("Amikacin",),
        ),
        ReverseLookupCase(
            question="Which drug has route of administration intravenous?",
            expected_subjects=("Amikacin",),
        ),
        ReverseLookupCase(
            question="Which drug has route of administration oral?",
            expected_subjects=("Sulfamethazine", "Sulfametrole"),
        ),
        ReverseLookupCase(
            question="Which chemical compound has ATC code J01EE30?",
            expected_subjects=(),
        ),
    ]


def run(*, noise_facts: int = 0) -> dict[str, object]:
    facts = _fixture_facts(noise_facts=noise_facts)
    cases = _cases()
    index = ReverseAttributeIndex.from_facts(facts)
    strategies = {
        "controller_scan": lambda question: scan_reverse_attribute_candidates(facts, parse_reverse_attribute_query(question)),
        "reverse_index": lambda question: index.lookup(parse_reverse_attribute_query(question)),
    }
    results = [_evaluate_strategy(name, strategy, cases) for name, strategy in strategies.items()]
    return {
        "noise_facts": noise_facts,
        "case_count": len(cases),
        "results": [asdict(item) for item in results],
        "recommended": max(results, key=lambda item: (item.accuracy, item.abstain_accuracy, -item.mean_latency_ms)).name,
    }


def _evaluate_strategy(
    name: str,
    resolver: Callable[[str], list],
    cases: list[ReverseLookupCase],
) -> StrategyResult:
    exact_scores: list[float] = []
    abstain_scores: list[float] = []
    latencies: list[float] = []
    for case in cases:
        start = perf_counter()
        hits = resolver(case.question)
        latencies.append((perf_counter() - start) * 1000.0)
        subjects = tuple(hit.subject for hit in hits[: len(case.expected_subjects) or 1])
        if case.expected_subjects:
            exact_scores.append(1.0 if subjects == case.expected_subjects else 0.0)
        else:
            abstain_scores.append(1.0 if not hits else 0.0)
    exact_match_accuracy = mean(exact_scores) if exact_scores else 1.0
    abstain_accuracy = mean(abstain_scores) if abstain_scores else 1.0
    accuracy = mean([*exact_scores, *abstain_scores]) if [*exact_scores, *abstain_scores] else 1.0
    notes = (
        "Indexed lookup avoids scanning all facts and stays deterministic on code-based questions."
        if name == "reverse_index"
        else "Baseline scan is simple but scales linearly with loaded facts."
    )
    return StrategyResult(
        name=name,
        accuracy=accuracy,
        exact_match_accuracy=exact_match_accuracy,
        abstain_accuracy=abstain_accuracy,
        mean_latency_ms=mean(latencies) if latencies else 0.0,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-facts", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(run(noise_facts=args.noise_facts), indent=2))


if __name__ == "__main__":
    main()
