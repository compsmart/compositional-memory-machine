from __future__ import annotations

from hrr.datasets import all_facts, fact_key
from hrr.encoder import SVOEncoder, SVOFact
from memory.amm import AMM


def payload(domain: str, fact: SVOFact) -> dict[str, str]:
    return {
        "domain": domain,
        "subject": fact.subject,
        "verb": fact.verb,
        "object": fact.object,
    }


def build_memory(dim: int = 2048, seed: int = 0, cycles: int = 1) -> tuple[SVOEncoder, AMM]:
    encoder = SVOEncoder(dim=dim, seed=seed)
    memory = AMM()
    for _cycle in range(cycles):
        for domain, fact in all_facts():
            memory.write(fact_key(domain, fact), encoder.encode_fact(fact), payload(domain, fact))
    return encoder, memory


def evaluate_known(encoder: SVOEncoder, memory: AMM) -> dict[str, float]:
    correct = 0
    total = 0
    min_margin = 1.0
    for domain, fact in all_facts():
        key = fact_key(domain, fact)
        nearest = memory.nearest(encoder.encode_fact(fact), top_k=2)
        total += 1
        correct += int(nearest and nearest[0][0].key == key)
        if len(nearest) == 2:
            min_margin = min(min_margin, nearest[0][1] - nearest[1][1])
    return {
        "top1": correct / total if total else 0.0,
        "min_margin": min_margin,
        "count": float(total),
    }
