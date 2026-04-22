from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language.qa import ClosedLoopQAMemory, QAFact, build_qa_facts


def _evaluate(memory: ClosedLoopQAMemory, facts: list[QAFact]) -> dict[str, float]:
    answer_correct = 0
    verb_correct = 0
    object_correct = 0
    confidence_sum = 0.0
    object_confidence_sum = 0.0
    for fact in facts:
        result = memory.ask(fact.subject, fact.verb)
        answer_correct += int(result.answer == fact.object)
        verb_correct += int(result.verb == fact.verb)
        object_correct += int(result.answer == fact.object)
        confidence_sum += result.confidence
        object_confidence_sum += result.object_confidence
    total = len(facts)
    return {
        "answer_em": answer_correct / total if total else 0.0,
        "verb_accuracy": verb_correct / total if total else 0.0,
        "object_accuracy": object_correct / total if total else 0.0,
        "mean_fact_confidence": confidence_sum / total if total else 0.0,
        "mean_object_confidence": object_confidence_sum / total if total else 0.0,
    }


def run(
    dim: int = 2048,
    seeds: tuple[int, ...] = (0, 1, 2),
    domains: int = 5,
    facts_per_domain: int = 50,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    facts = build_qa_facts(domains=domains, facts_per_domain=facts_per_domain)
    for seed in seeds:
        memory = ClosedLoopQAMemory(dim=dim, seed=seed)
        learned: list[QAFact] = []
        retention_scores: list[float] = []

        for domain_idx in range(domains):
            cycle_facts = facts[domain_idx * facts_per_domain : (domain_idx + 1) * facts_per_domain]
            for fact in cycle_facts:
                memory.learn(fact)
            learned.extend(cycle_facts)
            retention_scores.append(_evaluate(memory, learned)["answer_em"])

        final = _evaluate(memory, facts)
        rows.append(
            {
                "dim": float(dim),
                "seed": float(seed),
                "domains": float(domains),
                "facts": float(len(facts)),
                "cycles": float(domains),
                "answer_em": final["answer_em"],
                "verb_accuracy": final["verb_accuracy"],
                "object_accuracy": final["object_accuracy"],
                "forgetting": 1.0 - min(retention_scores) if retention_scores else 0.0,
                "mean_fact_confidence": final["mean_fact_confidence"],
                "mean_object_confidence": final["mean_object_confidence"],
            }
        )
    return rows


if __name__ == "__main__":
    for row in run():
        print(row)
