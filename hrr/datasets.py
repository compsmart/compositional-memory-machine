from __future__ import annotations

import random

from hrr.encoder import SVOFact


DOMAINS: dict[str, list[SVOFact]] = {
    "medical": [
        SVOFact("doctor", "treats", "patient"),
        SVOFact("nurse", "monitors", "patient"),
        SVOFact("surgeon", "repairs", "injury"),
        SVOFact("pharmacist", "dispenses", "medicine"),
    ],
    "education": [
        SVOFact("teacher", "explains", "lesson"),
        SVOFact("student", "reads", "book"),
        SVOFact("tutor", "coaches", "learner"),
        SVOFact("principal", "leads", "school"),
    ],
    "aviation": [
        SVOFact("pilot", "flies", "plane"),
        SVOFact("mechanic", "inspects", "engine"),
        SVOFact("controller", "guides", "flight"),
        SVOFact("attendant", "serves", "passenger"),
    ],
    "kitchen": [
        SVOFact("chef", "prepares", "meal"),
        SVOFact("baker", "bakes", "bread"),
        SVOFact("server", "carries", "plate"),
        SVOFact("barista", "brews", "coffee"),
    ],
    "legal": [
        SVOFact("lawyer", "argues", "case"),
        SVOFact("judge", "reviews", "evidence"),
        SVOFact("clerk", "files", "record"),
        SVOFact("witness", "answers", "question"),
    ],
}


def all_facts() -> list[tuple[str, SVOFact]]:
    return [(domain, fact) for domain, facts in DOMAINS.items() for fact in facts]


def fact_key(domain: str, fact: SVOFact) -> str:
    return f"{domain}:{fact.subject}:{fact.verb}:{fact.object}"


def synthetic_facts(
    domains: int = 8,
    subjects: int = 160,
    verbs: int = 80,
    objects: int = 160,
    facts_per_domain: int = 1000,
    seed: int = 0,
) -> list[tuple[str, SVOFact]]:
    """Generate many overlapping SVO facts for collision/capacity stress tests."""
    rng = random.Random(seed)
    subject_pool = [f"s{i}" for i in range(subjects)]
    verb_pool = [f"v{i}" for i in range(verbs)]
    object_pool = [f"o{i}" for i in range(objects)]

    rows: list[tuple[str, SVOFact]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for domain_idx in range(domains):
        domain = f"domain{domain_idx}"
        while sum(1 for current_domain, _fact in rows if current_domain == domain) < facts_per_domain:
            # Each domain gets a biased slice plus shared global tokens. This creates
            # realistic overlap and many near neighbors without making domains disjoint.
            offset = domain_idx * 17
            subject = subject_pool[(offset + rng.randrange(subjects // 2)) % subjects]
            verb = verb_pool[(offset + rng.randrange(verbs // 2)) % verbs]
            object_ = object_pool[(offset + rng.randrange(objects // 2)) % objects]
            key = (domain, subject, verb, object_)
            if key in seen:
                continue
            seen.add(key)
            rows.append((domain, SVOFact(subject, verb, object_)))
    return rows
