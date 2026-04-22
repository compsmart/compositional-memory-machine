from __future__ import annotations

from experiments.exp_d2829_next_token import run as run_next_token
from experiments.exp_d2830_word_learning import run as run_word_learning
from experiments.exp_d2833_emergent_syntax import run as run_emergent_syntax
from experiments.exp_d2834_closed_loop_qa import run as run_closed_loop_qa
from experiments.exp_projected_sdm_ngram import run as run_projected_ngram
from experiments.exp_projected_sdm_trigram import run as run_projected_trigram
from language import (
    ClosedLoopQAMemory,
    ContextExample,
    NGramLanguageMemory,
    ProjectedNGramLanguageMemory,
    ProjectedTrigramLanguageMemory,
    QAFact,
    SyntaxComposer,
    SyntaxTriple,
    WordLearningMemory,
)


def test_ngram_memory_predicts_seen_context() -> None:
    memory = NGramLanguageMemory(dim=1024, seed=0)
    memory.learn_sequence(["the", "doctor", "treats", "the", "patient"], cycles=2)

    prediction = memory.predict("the", "doctor")

    assert prediction.token == "treats"
    assert prediction.confidence > 0.99


def test_word_learning_routes_unknown_action_to_known_cluster() -> None:
    memory = WordLearningMemory(dim=1024, seed=0)
    memory.add_known_action("eat", "ingest", "ingest")
    memory.add_known_action("run", "move", "move")

    result = memory.learn_word(
        "dax",
        [
            ContextExample("child", "dax", "apple", "ingest"),
            ContextExample("student", "dax", "meal", "ingest"),
            ContextExample("chef", "dax", "soup", "ingest"),
        ],
    )

    assert result["cluster"] == "ingest"
    assert memory.plausibility("dax", "eat") > memory.plausibility("dax", "run")


def test_d2829_experiment_smoke() -> None:
    row = run_next_token(dim=1024, seeds=(0,), cycles=2)[0]

    assert row["seen_em"] == 1.0
    assert row["familiar_em"] == 1.0
    assert row["novel_hit_rate"] == 0.0


def test_d2830_experiment_smoke() -> None:
    row = run_word_learning(dim=1024, seeds=(0,))[0]

    assert row["dax_cluster_correct"] == 1.0
    assert row["blick_cluster_correct"] == 1.0
    assert row["retention"] == 1.0


def test_projected_ngram_memory_predicts_seen_context() -> None:
    memory = ProjectedNGramLanguageMemory(dim=512, seed=0, addr_dim=128, n_locations=256, write_k=4, read_k=16)
    memory.learn_sequence(["the", "doctor", "treats", "the", "patient"], cycles=2)

    assert memory.predict("the", "doctor").token == "treats"


def test_projected_sdm_ngram_experiment_smoke() -> None:
    row = run_projected_ngram(dim=512, addr_dim=128, seeds=(0,), cycles=2, n_locations=512, write_k=4, read_k=32)[0]

    assert row["seen_em"] >= 0.8
    assert 0.0 <= row["familiar_em"] <= 1.0
    assert 0.0 <= row["novel_hit_rate"] <= 1.0
    assert 0.0 <= row["calibrated_novel_hit_rate"] <= 1.0


def test_projected_trigram_memory_predicts_seen_context() -> None:
    memory = ProjectedTrigramLanguageMemory(dim=512, seed=0, addr_dim=128, n_locations=512, write_k=4, read_k=32)
    memory.learn("doctor", "treats", "near", "patient")

    assert memory.predict("doctor", "treats", "near").token == "patient"


def test_projected_sdm_trigram_experiment_smoke() -> None:
    row = run_projected_trigram(dim=512, addr_dim=128, seeds=(0,), cycles=2, n_locations=512, write_k=4, read_k=32)[0]

    assert row["seen_em"] >= 0.8
    assert 0.0 <= row["score_calibrated_familiar_em"] <= 1.0
    assert 0.0 <= row["score_calibrated_novel_hit_rate"] <= 1.0
    assert 0.0 <= row["calibrated_familiar_em"] <= 1.0
    assert 0.0 <= row["calibrated_novel_hit_rate"] <= 1.0


def test_syntax_composer_routes_same_meaning_patterns_above_random() -> None:
    composer = SyntaxComposer(dim=512, seed=0)
    triple = SyntaxTriple("doctor", "treats", "patient")
    other = SyntaxTriple("pilot", "flies", "route")

    active = composer.encode(triple, "active")
    passive = composer.encode(triple, "passive")
    random = composer.encode(other, "passive")

    assert composer.similarity(active, passive) > composer.similarity(active, random)


def test_d2833_experiment_smoke() -> None:
    row = run_emergent_syntax(dim=512, seeds=(0,), domains=2, triples_per_domain=6)[0]

    assert row["mean_cross_pattern_cosine"] > row["mean_random_cosine"]
    assert row["cross_over_random_margin"] > 0.0


def test_closed_loop_qa_unbinds_answer_roles() -> None:
    memory = ClosedLoopQAMemory(dim=512, seed=0)
    fact = QAFact("Ada", "worked_with", "Babbage", "history")
    memory.learn(fact)

    result = memory.ask("Ada", "worked_with")

    assert result.answer == "Babbage"
    assert result.verb == "worked_with"


def test_d2834_experiment_smoke() -> None:
    row = run_closed_loop_qa(dim=512, seeds=(0,), domains=2, facts_per_domain=6)[0]

    assert row["answer_em"] == 1.0
    assert row["verb_accuracy"] == 1.0
    assert row["object_accuracy"] == 1.0
