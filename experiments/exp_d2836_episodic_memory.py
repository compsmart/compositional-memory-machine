from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.episodic import ConversationFact, EpisodicMemory


RELATIONS = ("likes", "works_at", "owns")


def _fact(session: int, turn: int, idx: int) -> ConversationFact:
    relation = RELATIONS[idx % len(RELATIONS)]
    return ConversationFact(
        session=session,
        turn=turn,
        subject=f"user{session}_{turn}_{idx}",
        relation=relation,
        object=f"value{session}_{turn}_{idx}",
    )


def _revision(original: ConversationFact, session: int, turn: int) -> ConversationFact:
    return ConversationFact(
        session=session,
        turn=turn,
        subject=original.subject,
        relation=original.relation,
        object=f"revised{session}_{turn}",
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def run(
    *,
    dim: int = 2048,
    seeds: tuple[int, ...] = (42, 123, 7),
    sessions: int = 3,
    turns: int = 10,
    facts_per_turn: int = 3,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for seed in seeds:
        memory = EpisodicMemory(dim=dim, seed=seed)
        first_fact_by_session: list[ConversationFact] = []
        immediate_hits = immediate_total = 0
        distant_hits = distant_total = 0
        cross_session_hits = cross_session_total = 0
        revision_hits = revision_total = 0

        for session in range(sessions):
            first_fact_this_session: ConversationFact | None = None
            for turn in range(turns):
                turn_facts = [_fact(session, turn, idx) for idx in range(facts_per_turn)]
                for fact in turn_facts:
                    memory.state_fact(fact)
                    immediate_total += 1
                    if memory.recall_current(fact.subject, fact.relation) == fact.object:
                        immediate_hits += 1
                    if first_fact_this_session is None:
                        first_fact_this_session = fact

                if turn >= 3 and first_fact_this_session is not None:
                    distant_total += 1
                    if (
                        memory.recall_current(first_fact_this_session.subject, first_fact_this_session.relation)
                        == first_fact_this_session.object
                    ):
                        distant_hits += 1

                if session > 0 and first_fact_by_session:
                    prior = first_fact_by_session[session - 1]
                    cross_session_total += 1
                    if memory.recall_current(prior.subject, prior.relation) == prior.object:
                        cross_session_hits += 1

                if turn == turns // 2 and first_fact_this_session is not None:
                    revised = _revision(first_fact_this_session, session, turn)
                    memory.revise_fact(revised)
                    revision_total += 1
                    if memory.recall_current(revised.subject, revised.relation) == revised.object:
                        revision_hits += 1
                    first_fact_this_session = revised

            if first_fact_this_session is not None:
                first_fact_by_session.append(first_fact_this_session)

        retention_hits = sum(1 for fact in memory.history if memory.recall_evidence(fact))
        retention_total = len(memory.history)

        rows.append(
            {
                "seed": float(seed),
                "sessions": float(sessions),
                "turns": float(turns),
                "facts_per_turn": float(facts_per_turn),
                "immediate_em": _rate(immediate_hits, immediate_total),
                "distant_em": _rate(distant_hits, distant_total),
                "cross_session_em": _rate(cross_session_hits, cross_session_total),
                "revision_em": _rate(revision_hits, revision_total),
                "retention_em": _rate(retention_hits, retention_total),
                "stored_facts": float(retention_total),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--facts-per-turn", type=int, default=3)
    args = parser.parse_args()

    for row in run(
        dim=args.dim,
        seeds=tuple(args.seeds),
        sessions=args.sessions,
        turns=args.turns,
        facts_per_turn=args.facts_per_turn,
    ):
        print(row)


if __name__ == "__main__":
    main()
