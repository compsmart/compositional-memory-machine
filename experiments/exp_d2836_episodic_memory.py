from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.episodic import ConversationFact, ConversationTurn, EpisodicMemory


PSEUDOWORDS = (
    "dax",
    "blick",
    "wug",
    "fep",
    "zup",
    "krell",
    "mip",
    "troma",
    "vash",
    "lurn",
)
MEANINGS = (
    "ingest",
    "move",
    "observe",
    "build",
    "repair",
)
CORRECTED_MEANINGS = {
    "ingest": "consume",
    "move": "travel",
    "observe": "inspect",
    "build": "assemble",
    "repair": "maintain",
}


def _turn_subject(session: int, turn: int) -> str:
    return f"session{session}:turn{turn}"


def _word_subject(session: int, word: str) -> str:
    return f"session{session}:word:{word}"


def _word_pair(session: int, turn: int) -> tuple[str, str]:
    index = session * 100 + turn
    word = PSEUDOWORDS[index % len(PSEUDOWORDS)]
    meaning = MEANINGS[index % len(MEANINGS)]
    return word, meaning


def _speaker_fact(session: int, turn: int, speaker: str, intent: str) -> ConversationFact:
    return ConversationFact(
        session=session,
        turn=turn,
        subject=_turn_subject(session, turn),
        relation="speaker",
        object=speaker,
        source="exp_d2836_episodic_memory",
        source_id=f"s{session}:t{turn}",
        excerpt=f"{speaker} turn with intent {intent}",
    )


def _intent_fact(session: int, turn: int, intent: str) -> ConversationFact:
    return ConversationFact(
        session=session,
        turn=turn,
        subject=_turn_subject(session, turn),
        relation="intent",
        object=intent,
        source="exp_d2836_episodic_memory",
        source_id=f"s{session}:t{turn}",
        excerpt=f"Intent {intent}",
    )


def _introduced_word_fact(session: int, turn: int, word: str) -> ConversationFact:
    return ConversationFact(
        session=session,
        turn=turn,
        subject=_turn_subject(session, turn),
        relation="introduced_word",
        object=word,
        source="exp_d2836_episodic_memory",
        source_id=f"s{session}:t{turn}",
        excerpt=f"Introduced word {word}",
    )


def _meaning_fact(session: int, turn: int, word: str, meaning: str, *, intent: str) -> ConversationFact:
    return ConversationFact(
        session=session,
        turn=turn,
        subject=_word_subject(session, word),
        relation="means",
        object=meaning,
        source="exp_d2836_episodic_memory",
        source_id=f"s{session}:t{turn}",
        excerpt=f"{intent}: {word} means {meaning}",
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _controller_episode_metrics(memory: EpisodicMemory) -> dict[str, float]:
    turns = [
        ConversationTurn(
            session=9,
            turn=0,
            speaker="user",
            utterance="My project is Atlas.",
            intent="declare_project",
            facts=(
                ConversationFact(
                    session=9,
                    turn=0,
                    subject="user",
                    relation="project",
                    object="Atlas",
                    source="controller_episode",
                    source_id="s9:t0",
                    excerpt="My project is Atlas.",
                ),
            ),
        ),
        ConversationTurn(
            session=9,
            turn=1,
            speaker="assistant",
            utterance="I track Atlas for you.",
            intent="confirm_project",
            facts=(
                ConversationFact(
                    session=9,
                    turn=1,
                    subject="assistant",
                    relation="tracks_project",
                    object="Atlas",
                    source="controller_episode",
                    source_id="s9:t1",
                    excerpt="I track Atlas for you.",
                ),
            ),
        ),
        ConversationTurn(
            session=9,
            turn=2,
            speaker="user",
            utterance="It launches tomorrow.",
            intent="project_update",
            facts=(
                ConversationFact(
                    session=9,
                    turn=2,
                    subject="Atlas",
                    relation="launches_on",
                    object="tomorrow",
                    source="controller_episode",
                    source_id="s9:t2",
                    excerpt="It launches tomorrow.",
                ),
            ),
        ),
        ConversationTurn(
            session=9,
            turn=3,
            speaker="assistant",
            utterance="I am the memory workbench.",
            intent="self_reference",
            facts=(
                ConversationFact(
                    session=9,
                    turn=3,
                    subject="assistant",
                    relation="identity",
                    object="memory_workbench",
                    source="controller_episode",
                    source_id="s9:t3",
                    excerpt="I am the memory workbench.",
                ),
            ),
        ),
        ConversationTurn(
            session=9,
            turn=4,
            speaker="user",
            utterance="Teach me dax while Atlas stays green.",
            intent="mixed_word_and_fact",
            facts=(
                ConversationFact(
                    session=9,
                    turn=4,
                    subject="s9:t4",
                    relation="introduced_word",
                    object="dax",
                    source="controller_episode",
                    source_id="s9:t4",
                    excerpt="Teach me dax while Atlas stays green.",
                ),
                ConversationFact(
                    session=9,
                    turn=4,
                    subject="Atlas",
                    relation="status",
                    object="green",
                    source="controller_episode",
                    source_id="s9:t4",
                    excerpt="Teach me dax while Atlas stays green.",
                ),
            ),
        ),
    ]
    emitted = memory.ingest_episode(turns)
    return {
        "controller_episode_em": float(len(emitted) >= 10),
        "pronoun_carryover_em": float(memory.recall_current("Atlas", "launches_on") == "tomorrow"),
        "self_reference_em": float(memory.recall_current("assistant", "identity") == "memory_workbench"),
        "mixed_turn_em": float(
            memory.recall_current("s9:t4", "introduced_word") == "dax"
            and memory.recall_current("Atlas", "status") == "green"
        ),
    }


def run(
    *,
    dim: int = 2048,
    seeds: tuple[int, ...] = (42, 123, 7),
    sessions: int = 3,
    turns: int = 10,
    facts_per_turn: int = 3,
) -> list[dict[str, float]]:
    if facts_per_turn < 3:
        raise ValueError("facts_per_turn must be >= 3 for the D-2836-style dialogue benchmark")

    rows: list[dict[str, float]] = []

    for seed in seeds:
        memory = EpisodicMemory(dim=dim, seed=seed)
        first_fact_by_session: list[ConversationFact] = []
        immediate_hits = immediate_total = 0
        distant_hits = distant_total = 0
        cross_session_hits = cross_session_total = 0
        revision_hits = revision_total = 0
        metadata_hits = metadata_total = 0
        answer_hits = answer_total = 0
        correction_hits = correction_total = 0

        for session in range(sessions):
            first_fact_this_session: ConversationFact | None = None
            first_word_fact_this_session: ConversationFact | None = None
            taught_words: list[tuple[str, str, str]] = []
            for turn in range(turns):
                if turn == turns // 2 and first_word_fact_this_session is not None:
                    current_meaning = first_word_fact_this_session.object
                    corrected_meaning = CORRECTED_MEANINGS[current_meaning]
                    turn_facts = [
                        _speaker_fact(session, turn, "user", "correct_word"),
                        _intent_fact(session, turn, "correct_word"),
                    ]
                    revised = _meaning_fact(
                        session,
                        turn,
                        first_word_fact_this_session.subject.split(":word:", 1)[1],
                        corrected_meaning,
                        intent="correct_word",
                    )
                    old_value = first_word_fact_this_session.object
                    for fact in turn_facts:
                        memory.state_fact(fact)
                        immediate_total += 1
                        if memory.recall_current(fact.subject, fact.relation) == fact.object:
                            immediate_hits += 1
                    memory.revise_fact(revised)
                    immediate_total += 1
                    revision_total += 1
                    correction_total += 1
                    if memory.recall_current(revised.subject, revised.relation) == revised.object:
                        immediate_hits += 1
                        revision_hits += 1
                    if memory.recall_history(revised.subject, revised.relation) == [old_value, revised.object]:
                        correction_hits += 1
                    first_word_fact_this_session = revised
                    taught_words[0] = (taught_words[0][0], revised.object, revised.object)
                else:
                    if turn % 2 == 0:
                        word, meaning = _word_pair(session, turn)
                        turn_facts = [
                            _speaker_fact(session, turn, "user", "teach_word"),
                            _intent_fact(session, turn, "teach_word"),
                            _introduced_word_fact(session, turn, word),
                        ]
                        taught_words.append((word, meaning, CORRECTED_MEANINGS[meaning]))
                    else:
                        word, meaning, _corrected = taught_words[-1]
                        turn_facts = [
                            _speaker_fact(session, turn, "assistant", "assistant_answer"),
                            _intent_fact(session, turn, "assistant_answer"),
                            _meaning_fact(session, turn, word, meaning, intent="assistant_answer"),
                        ]

                for fact in turn_facts:
                    memory.state_fact(fact)
                    immediate_total += 1
                    if memory.recall_current(fact.subject, fact.relation) == fact.object:
                        immediate_hits += 1
                    if fact.relation in {"speaker", "intent"}:
                        metadata_total += 1
                        if memory.recall_current(fact.subject, fact.relation) == fact.object:
                            metadata_hits += 1
                    if fact.relation == "means":
                        answer_total += 1
                        if memory.recall_current(fact.subject, fact.relation) == fact.object:
                            answer_hits += 1
                        if first_word_fact_this_session is None:
                            first_word_fact_this_session = fact
                    if first_fact_this_session is None and fact.relation == "introduced_word":
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
                "speaker_intent_em": _rate(metadata_hits, metadata_total),
                "assistant_answer_em": _rate(answer_hits, answer_total),
                "correction_em": _rate(correction_hits, correction_total),
                "stored_facts": float(retention_total),
                **_controller_episode_metrics(memory),
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
