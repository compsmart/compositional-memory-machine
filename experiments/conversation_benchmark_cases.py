from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable

from experiments.exp_codebase_memory import run as run_codebase_memory
from experiments.exp_d2836_episodic_memory import run as run_episodic_memory
from experiments.exp_large_document_memory import run as run_large_document_memory
from experiments.exp_structural_generalization import run as run_structural_generalization
from experiments.exp_temporal_state_tracking import run as run_temporal_state_tracking
from experiments.exp_truth_provenance_conflicts import run as run_truth_provenance_conflicts
from ingestion import ExtractedFact, ExtractionResponse, GeminiExtractor
from web import HHRWebState, to_jsonable


GRACEFUL_FAILURE_ROUTES = {
    "fallback",
    "fact_prompt_miss",
    "ingest_prompt",
    "ingest_unavailable",
    "multi_hop_miss",
    "no_reliable_match",
    "pattern_miss",
    "word_learning_prompt",
    "word_recall_miss",
}
GRACEFUL_FAILURE_PHRASES = (
    "i do not have",
    "i do not know",
    "i could not",
    "i have not learned",
    "i need",
    "memory-backed fact questions",
    "try one of:",
)

SMOKE_CASE_IDS = (
    "memory_fact_recall",
    "multihop_bridge_query",
    "word_meaning_learning",
    "alias_normalization_ingest",
    "temporal_state_tracking_substrate",
    "trick_unknown_fact_refusal",
    "coding_python_function",
)
ROADMAP_SERIOUS_CASE_IDS = (
    "memory_fact_recall",
    "context_pronoun_carryover",
    "inverse_relation_lookup",
    "multihop_bridge_query",
    "pattern_completion_doctor",
    "pattern_completion_artist_distribution",
    "word_meaning_learning",
    "word_retention_after_interference",
    "alias_normalization_ingest",
    "temporal_state_tracking_substrate",
    "episodic_dialogue_memory_substrate",
    "episodic_dialogue_metadata_substrate",
    "truth_provenance_substrate",
    "large_document_memory_substrate",
    "codebase_dependency_memory_substrate",
    "structural_generalization_substrate",
    "trick_unknown_fact_refusal",
    "explanation_from_memory",
    "logic_transitive_order",
    "puzzle_number_sequence",
    "multilingual_spanish_recall",
    "coding_python_function",
    "sentiment_labeling",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    chat_dim: int = 2048
    chat_seed: int = 0
    episodic_dim: int = 2048
    episodic_seeds: tuple[int, ...] = (42, 123, 7)
    episodic_sessions: int = 3
    episodic_turns: int = 10
    episodic_facts_per_turn: int = 3
    temporal_dim: int = 2048
    temporal_seeds: tuple[int, ...] = (42, 123)


@dataclass(frozen=True)
class CaseVerdict:
    score: float
    notes: str
    observed: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseRun:
    prompts: tuple[str, ...]
    history: list[dict[str, Any]]
    replies: list[dict[str, Any]]
    observed: dict[str, Any] = field(default_factory=dict)


ChatValidator = Callable[[CaseRun], CaseVerdict]
MetricExecutor = Callable[[BenchmarkConfig], CaseVerdict]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    track: str
    surface: str
    description: str
    expected_behavior: str
    scorer_type: str
    prompts: tuple[str, ...] = ()
    weight: float = 1.0
    pass_threshold: float = 1.0
    validator: ChatValidator | None = None
    metric_executor: MetricExecutor | None = None


class ConversationBenchmarkExtractor(GeminiExtractor):
    def extract(self, text: str, *, source: str = "") -> tuple[ExtractionResponse, ExtractionResponse]:
        if "Alice" in text:
            pass1 = ExtractionResponse(
                estimated_fact_count=3,
                facts=[
                    ExtractedFact(
                        subject="Alice",
                        relation="knows",
                        object="Bob",
                        confidence=0.95,
                        kind="explicit",
                        source=source,
                    ),
                    ExtractedFact(
                        subject="Bob",
                        relation="works with",
                        object="Carol",
                        confidence=0.95,
                        kind="explicit",
                        source=source,
                    ),
                ],
            )
            pass2 = ExtractionResponse(
                facts=[
                    ExtractedFact(
                        subject="Carol",
                        relation="guides",
                        object="Delta",
                        confidence=0.8,
                        kind="missed",
                        source=source,
                    )
                ]
            )
            return pass1, pass2

        pass1 = ExtractionResponse(
            estimated_fact_count=2,
            facts=[
                ExtractedFact(
                    subject="Ada Lovelace",
                    relation="collaborated with",
                    object="Charles Babbage",
                    confidence=0.95,
                    kind="explicit",
                    source=source,
                )
            ],
        )
        pass2 = ExtractionResponse(
            facts=[
                ExtractedFact(
                    subject="Ada Lovelace",
                    relation="described",
                    object="an algorithm for Bernoulli numbers",
                    confidence=0.8,
                    kind="missed",
                    source=source,
                )
            ]
        )
        return pass1, pass2

    def _api_key(self) -> str | None:
        return "fixture-key"


def build_chat_state(config: BenchmarkConfig) -> HHRWebState:
    return HHRWebState(dim=config.chat_dim, seed=config.chat_seed, extractor=ConversationBenchmarkExtractor())


def run_chat_case(case: BenchmarkCase, config: BenchmarkConfig) -> CaseVerdict:
    state = build_chat_state(config)
    replies: list[dict[str, Any]] = []
    for prompt in case.prompts:
        payload = state.chat({"message": prompt})
        replies.append(to_jsonable(payload["reply"]))
    run = CaseRun(
        prompts=case.prompts,
        history=to_jsonable(state.chat_history),
        replies=replies,
        observed={
            "final_reply": replies[-1] if replies else {},
            "history_length": len(state.chat_history),
        },
    )
    if case.validator is None:
        raise ValueError(f"chat case {case.case_id} is missing a validator")
    return case.validator(run)


def run_metric_case(case: BenchmarkCase, config: BenchmarkConfig) -> CaseVerdict:
    if case.metric_executor is None:
        raise ValueError(f"metric case {case.case_id} is missing an executor")
    return case.metric_executor(config)


def _reply_text(reply: dict[str, Any]) -> str:
    return str(reply.get("text", ""))


def _reply_route(reply: dict[str, Any]) -> str:
    return str(reply.get("route", ""))


def _contains_all(text: str, fragments: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(fragment.lower() in lowered for fragment in fragments)


def _is_graceful_failure(reply: dict[str, Any]) -> bool:
    route = _reply_route(reply)
    text = _reply_text(reply).lower()
    return route in GRACEFUL_FAILURE_ROUTES or any(phrase in text for phrase in GRACEFUL_FAILURE_PHRASES)


def _fact_recall_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    success = (
        _reply_route(reply) == "fact_query"
        and _contains_all(_reply_text(reply), ("Ada Lovelace", "Charles Babbage"))
        and str(reply.get("graph_target", "")) == "Charles Babbage"
    )
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Fact recall via chat after raw-text ingestion.",
        observed=run.observed,
    )


def _pronoun_carryover_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    success = (
        _reply_route(reply) == "fact_query"
        and _contains_all(_reply_text(reply), ("Charles Babbage",))
        and str(reply.get("graph_target", "")) == "Charles Babbage"
    )
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Pronoun carryover resolved to the last fact subject.",
        observed=run.observed,
    )


def _inverse_lookup_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    chain_path = tuple(str(node) for node in reply.get("chain_path", []))
    success = (
        _reply_route(reply) == "fact_query"
        and str(reply.get("graph_target", "")) == "Ada Lovelace"
        and chain_path == ("Charles Babbage", "Ada Lovelace")
    )
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Reverse relation lookup through the graph-backed chat path.",
        observed={**run.observed, "chain_path": list(chain_path)},
    )


def _multihop_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    chain_path = tuple(str(node) for node in reply.get("chain_path", []))
    success = (
        _reply_route(reply) == "multi_hop_query"
        and chain_path == ("Alice", "Bob", "Carol")
        and _contains_all(_reply_text(reply), ("Alice knows Bob", "Bob works with Carol", "So the answer is Bob"))
    )
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Two-hop bridge question answered from linked chat-ingested facts.",
        observed={**run.observed, "chain_path": list(chain_path)},
    )


def _pattern_validator(expected_token: str, *, alternatives: tuple[str, ...] = ()) -> ChatValidator:
    def validate(run: CaseRun) -> CaseVerdict:
        reply = run.replies[-1]
        text = _reply_text(reply)
        success = _reply_route(reply) == "pattern_prediction" and expected_token in text
        if success and alternatives:
            success = all(option in text for option in alternatives)
        return CaseVerdict(
            score=1.0 if success else 0.0,
            notes=f"Pattern continuation should surface `{expected_token}`.",
            observed=run.observed,
        )

    return validate


def _word_meaning_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    success = (
        _reply_route(reply) == "word_recall"
        and _contains_all(_reply_text(reply), ("dax", "ingest"))
    )
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Learned word should map to the ingest cluster when recalled.",
        observed=run.observed,
    )


def _word_retention_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    text = _reply_text(reply)
    success = _reply_route(reply) == "word_recall" and "dax" in text and "ingest" in text
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Original learned meaning should survive interference from a second new word.",
        observed=run.observed,
    )


def _trick_refusal_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    text = _reply_text(reply)
    graceful = _is_graceful_failure(reply)
    success = graceful and "telephone" not in text.lower()
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Unsupported factual claim should not be answered as if known.",
        observed=run.observed,
    )


def _explanation_validator(run: CaseRun) -> CaseVerdict:
    reply = run.replies[-1]
    text = _reply_text(reply)
    if "because" in text.lower() or "evidence" in text.lower() or "memory" in text.lower():
        score = 1.0 if "Charles Babbage" in text else 0.5
    elif "Charles Babbage" in text:
        score = 0.5
    elif _is_graceful_failure(reply):
        score = 0.25
    else:
        score = 0.0
    return CaseVerdict(
        score=score,
        notes="Explanation cases reward evidence-aware answers over bare fact restatement.",
        observed=run.observed,
    )


def _exact_answer_validator(expected_fragments: tuple[str, ...]) -> ChatValidator:
    def validate(run: CaseRun) -> CaseVerdict:
        text = _reply_text(run.replies[-1]).lower()
        success = all(fragment.lower() in text for fragment in expected_fragments)
        return CaseVerdict(
            score=1.0 if success else 0.0,
            notes=f"Exact-answer challenge expecting: {', '.join(expected_fragments)}.",
            observed=run.observed,
        )

    return validate


def _correct_or_graceful_validator(expected_fragments: tuple[str, ...]) -> ChatValidator:
    def validate(run: CaseRun) -> CaseVerdict:
        reply = run.replies[-1]
        text = _reply_text(reply)
        if _contains_all(text, expected_fragments):
            score = 1.0
            notes = "Correct answer returned."
        elif _is_graceful_failure(reply):
            score = 0.5
            notes = "Graceful limitation was surfaced instead of bluffing."
        else:
            score = 0.0
            notes = "Neither correct nor safely uncertain."
        return CaseVerdict(score=score, notes=notes, observed=run.observed)

    return validate


def _alias_normalization_executor(config: BenchmarkConfig) -> CaseVerdict:
    state = build_chat_state(config)
    payload = state.ingest_text({"text": "Ada text", "domain": "history", "source": "benchmark"})
    facts = list(payload["facts"]["facts"])
    ada_fact = next(
        fact
        for fact in facts
        if fact["subject"] == "Ada Lovelace" and fact["object"] == "Charles Babbage"
    )
    provenance = dict(ada_fact.get("provenance", {}))
    success = (
        ada_fact["relation"] == "worked_with"
        and provenance.get("raw_relation") == "collaborated with"
        and bool(provenance.get("matched_alias"))
    )
    return CaseVerdict(
        score=1.0 if success else 0.0,
        notes="Ingestion should normalize alias-equivalent relations to canonical registry forms.",
        observed={"fact": ada_fact},
    )


def _episodic_memory_executor(config: BenchmarkConfig) -> CaseVerdict:
    rows = run_episodic_memory(
        dim=config.episodic_dim,
        seeds=config.episodic_seeds,
        sessions=config.episodic_sessions,
        turns=config.episodic_turns,
        facts_per_turn=config.episodic_facts_per_turn,
    )
    metrics = (
        "immediate_em",
        "distant_em",
        "cross_session_em",
        "revision_em",
        "retention_em",
    )
    observed = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    return CaseVerdict(
        score=mean(observed.values()),
        notes="D-2836-style episodic dialogue memory metrics averaged across seeds.",
        observed=observed,
    )


def _episodic_metadata_executor(config: BenchmarkConfig) -> CaseVerdict:
    rows = run_episodic_memory(
        dim=config.episodic_dim,
        seeds=config.episodic_seeds,
        sessions=config.episodic_sessions,
        turns=config.episodic_turns,
        facts_per_turn=config.episodic_facts_per_turn,
    )
    metrics = (
        "speaker_intent_em",
        "assistant_answer_em",
        "correction_em",
    )
    observed = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    return CaseVerdict(
        score=mean(observed.values()),
        notes="Dialogue-turn metadata and correction fidelity averaged across seeds.",
        observed=observed,
    )


def _temporal_state_executor(config: BenchmarkConfig) -> CaseVerdict:
    rows = run_temporal_state_tracking(dim=config.temporal_dim, seeds=config.temporal_seeds)
    metrics = ("latest_state_em", "history_em", "historical_em", "retention_em")
    observed = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    return CaseVerdict(
        score=mean(observed.values()),
        notes="Temporal state tracking metrics averaged across seeds.",
        observed=observed,
    )


def _truth_provenance_executor(config: BenchmarkConfig) -> CaseVerdict:
    rows = run_truth_provenance_conflicts(dim=config.temporal_dim, seeds=config.temporal_seeds)
    metrics = (
        "current_truth_em",
        "history_em",
        "competing_evidence_em",
        "provenance_em",
        "unresolved_refusal_em",
    )
    observed = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    return CaseVerdict(
        score=mean(observed.values()),
        notes="Current truth, conflict evidence, and provenance are preserved across revisions.",
        observed=observed,
    )


def _large_document_executor(config: BenchmarkConfig) -> CaseVerdict:
    rows = run_large_document_memory(dim=config.chat_dim, seeds=config.temporal_seeds)
    metrics = (
        "recall_em",
        "chain_em",
        "current_truth_em",
        "history_em",
        "competing_evidence_em",
        "refusal_em",
    )
    observed = {metric: mean(float(row[metric]) for row in rows) for metric in metrics}
    observed["chunk_count"] = mean(float(row["chunk_count"]) for row in rows)
    return CaseVerdict(
        score=mean(observed[metric] for metric in metrics),
        notes="Large-document extracted corpus benchmark with contradiction and refusal checks.",
        observed=observed,
    )


def _codebase_executor(config: BenchmarkConfig) -> CaseVerdict:
    row = run_codebase_memory(dim=config.chat_dim, seed=config.chat_seed)
    metrics = ("imports_em", "calls_em", "defined_in_em")
    observed = {metric: float(row[metric]) for metric in metrics}
    observed["fact_count"] = float(row["fact_count"])
    return CaseVerdict(
        score=mean(observed[metric] for metric in metrics),
        notes="Python codebase structure can be ingested and queried as dependency-style facts.",
        observed=observed,
    )


def _structural_generalization_executor(_config: BenchmarkConfig) -> CaseVerdict:
    observed = run_structural_generalization()
    return CaseVerdict(
        score=float(observed["breadth_score"]),
        notes="Structural suite combines prefix thresholds, hierarchical clauses, and surface pattern completion.",
        observed=observed,
    )


ALL_CASES = (
    BenchmarkCase(
        case_id="memory_fact_recall",
        category="memory",
        track="implemented",
        surface="web_chat",
        description="Ingest text and recall a stored fact.",
        expected_behavior="Return the learned Ada/Charles collaboration fact from chat memory.",
        scorer_type="structured_match",
        prompts=("Remember Ada text", "Who did Ada Lovelace work with?"),
        validator=_fact_recall_validator,
    ),
    BenchmarkCase(
        case_id="context_pronoun_carryover",
        category="general_context",
        track="implemented",
        surface="web_chat",
        description="Resolve a pronoun using the prior chat subject.",
        expected_behavior="Carry the subject from the prior factual turn and answer the pronoun question correctly.",
        scorer_type="structured_match",
        prompts=("Remember Ada text", "Who did Ada Lovelace work with?", "Who did she work with?"),
        validator=_pronoun_carryover_validator,
    ),
    BenchmarkCase(
        case_id="inverse_relation_lookup",
        category="multi_hop",
        track="implemented",
        surface="web_chat",
        description="Answer an inverse relation question.",
        expected_behavior="Traverse the reverse edge and identify Ada Lovelace from Charles Babbage.",
        scorer_type="structured_match",
        prompts=("Remember Ada text", "Who worked with Charles Babbage?"),
        validator=_inverse_lookup_validator,
    ),
    BenchmarkCase(
        case_id="multihop_bridge_query",
        category="multi_hop",
        track="implemented",
        surface="web_chat",
        description="Answer a bridge question over linked facts.",
        expected_behavior="Use linked facts to answer who Alice knows that works with Carol.",
        scorer_type="structured_match",
        prompts=(
            "Remember Alice knows Bob. Bob works with Carol. Carol guides Delta.",
            "Who does Alice know who works with Carol?",
        ),
        validator=_multihop_validator,
    ),
    BenchmarkCase(
        case_id="pattern_completion_doctor",
        category="language_patterning",
        track="implemented",
        surface="web_chat",
        description="Complete a deterministic learned pattern.",
        expected_behavior="Predict `treats` for the `the doctor ...` context.",
        scorer_type="exact_match",
        prompts=("Complete this learned pattern: 'the doctor ...'",),
        validator=_pattern_validator("treats"),
    ),
    BenchmarkCase(
        case_id="pattern_completion_artist_distribution",
        category="language_patterning",
        track="implemented",
        surface="web_chat",
        description="Surface the ranked alternatives for a probabilistic pattern.",
        expected_behavior="Predict `paints` and show `sketches` and `draws` as alternatives.",
        scorer_type="structured_match",
        prompts=("Complete this learned pattern: 'the artist ...'",),
        validator=_pattern_validator("paints", alternatives=("sketches", "draws")),
    ),
    BenchmarkCase(
        case_id="word_meaning_learning",
        category="canonical_meanings",
        track="implemented",
        surface="web_chat",
        description="Learn a new word from examples and recall its cluster.",
        expected_behavior="Learn `dax` as an ingest-like action and recall that meaning.",
        scorer_type="structured_match",
        prompts=(
            "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed.",
            "What does dax mean?",
        ),
        validator=_word_meaning_validator,
    ),
    BenchmarkCase(
        case_id="word_retention_after_interference",
        category="memory",
        track="implemented",
        surface="web_chat",
        description="Retain a learned word after another word is taught.",
        expected_behavior="Still recall `dax` after teaching `blick`.",
        scorer_type="structured_match",
        prompts=(
            "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed.",
            "Learn another word: blick. A runner blicks a track; a traveler blicks a road; a hiker blicks a trail.",
            "Do you still remember dax?",
        ),
        validator=_word_retention_validator,
    ),
    BenchmarkCase(
        case_id="alias_normalization_ingest",
        category="canonical_meanings",
        track="implemented",
        surface="structured_ingest",
        description="Normalize an aliased relation during ingestion.",
        expected_behavior="Map `collaborated with` onto canonical `worked_with` with provenance.",
        scorer_type="structured_match",
        metric_executor=_alias_normalization_executor,
    ),
    BenchmarkCase(
        case_id="temporal_state_tracking_substrate",
        category="temporal",
        track="implemented",
        surface="episodic_substrate",
        description="Track latest and historical state across a revision.",
        expected_behavior="Preserve latest state, full history, and historical lookup fidelity.",
        scorer_type="programmatic_validator",
        metric_executor=_temporal_state_executor,
    ),
    BenchmarkCase(
        case_id="episodic_dialogue_memory_substrate",
        category="memory",
        track="implemented",
        surface="episodic_substrate",
        description="Measure D-2836-style dialogue memory retention.",
        expected_behavior="Maintain immediate, distant, cross-session, revision, and retention EM.",
        scorer_type="programmatic_validator",
        metric_executor=_episodic_memory_executor,
    ),
    BenchmarkCase(
        case_id="episodic_dialogue_metadata_substrate",
        category="general_context",
        track="implemented",
        surface="episodic_substrate",
        description="Measure dialogue-turn metadata and correction fidelity.",
        expected_behavior="Retain speaker, intent, assistant-answer, and correction facts across turns.",
        scorer_type="programmatic_validator",
        metric_executor=_episodic_metadata_executor,
    ),
    BenchmarkCase(
        case_id="truth_provenance_substrate",
        category="temporal",
        track="implemented",
        surface="episodic_substrate",
        description="Track current truth, conflicting evidence, and provenance together.",
        expected_behavior="Return the latest truth while preserving superseded claims and competing evidence.",
        scorer_type="programmatic_validator",
        metric_executor=_truth_provenance_executor,
    ),
    BenchmarkCase(
        case_id="large_document_memory_substrate",
        category="memory",
        track="implemented",
        surface="structured_ingest",
        description="Use a larger extracted corpus with contradiction and refusal checks.",
        expected_behavior="Preserve recall, multihop access, revision history, and safe refusal on incomplete claims.",
        scorer_type="programmatic_validator",
        metric_executor=_large_document_executor,
    ),
    BenchmarkCase(
        case_id="codebase_dependency_memory_substrate",
        category="coding",
        track="implemented",
        surface="structured_ingest",
        description="Parse Python code into dependency-style graph facts.",
        expected_behavior="Answer imports, calls, and symbol ownership queries from codebase ingestion.",
        scorer_type="programmatic_validator",
        metric_executor=_codebase_executor,
    ),
    BenchmarkCase(
        case_id="structural_generalization_substrate",
        category="language_patterning",
        track="implemented",
        surface="episodic_substrate",
        description="Measure structural generalization beyond a single prefix benchmark.",
        expected_behavior="Score well on prefix, hierarchical, and chat-surface pattern tasks together.",
        scorer_type="programmatic_validator",
        metric_executor=_structural_generalization_executor,
    ),
    BenchmarkCase(
        case_id="trick_unknown_fact_refusal",
        category="trick_questions",
        track="implemented",
        surface="web_chat",
        description="Refuse to hallucinate an unsupported claim.",
        expected_behavior="Avoid inventing an answer when asked what Ada Lovelace invented.",
        scorer_type="graceful_failure_validator",
        prompts=("Remember Ada text", "What did Ada Lovelace invent?"),
        validator=_trick_refusal_validator,
    ),
    BenchmarkCase(
        case_id="explanation_from_memory",
        category="explanation_understanding",
        track="frontier",
        surface="web_chat",
        description="Explain an answer rather than merely restating it.",
        expected_behavior="Use evidence-aware language when asked why Ada worked with Charles Babbage.",
        scorer_type="programmatic_validator",
        prompts=("Remember Ada text", "Why do you think Ada Lovelace worked with Charles Babbage?"),
        validator=_explanation_validator,
    ),
    BenchmarkCase(
        case_id="logic_transitive_order",
        category="logic",
        track="frontier",
        surface="web_chat",
        description="Solve a simple transitive-order reasoning problem.",
        expected_behavior="Answer that Alice is tallest in a three-person order chain.",
        scorer_type="exact_match",
        prompts=("If Alice is taller than Bob and Bob is taller than Carol, who is tallest?",),
        validator=_exact_answer_validator(("alice", "tallest")),
    ),
    BenchmarkCase(
        case_id="puzzle_number_sequence",
        category="puzzles",
        track="frontier",
        surface="web_chat",
        description="Continue a simple doubling sequence.",
        expected_behavior="Return `32` for the sequence 2, 4, 8, 16, ?.",
        scorer_type="exact_match",
        prompts=("Which number comes next in the sequence 2, 4, 8, 16, ?",),
        validator=_exact_answer_validator(("32",)),
    ),
    BenchmarkCase(
        case_id="multilingual_spanish_recall",
        category="multilingual",
        track="frontier",
        surface="web_chat",
        description="Answer a simple memory question in Spanish.",
        expected_behavior="Return Charles Babbage, or at least fail safely instead of bluffing.",
        scorer_type="correct_or_graceful",
        prompts=("Remember Ada text", "Con quien trabajo Ada Lovelace?"),
        validator=_correct_or_graceful_validator(("Charles Babbage",)),
    ),
    BenchmarkCase(
        case_id="coding_python_function",
        category="coding",
        track="frontier",
        surface="web_chat",
        description="Write a tiny Python helper function.",
        expected_behavior="Return a correct `add(a, b)` implementation.",
        scorer_type="exact_match",
        prompts=("Write a Python function add(a, b) that returns the sum.",),
        validator=_exact_answer_validator(("def add", "return", "a", "b")),
    ),
    BenchmarkCase(
        case_id="sentiment_labeling",
        category="sentiment",
        track="frontier",
        surface="web_chat",
        description="Label simple sentiment from a short utterance.",
        expected_behavior="Classify the example movie review as positive.",
        scorer_type="exact_match",
        prompts=("The movie was amazing and I loved every minute. Was the sentiment positive, negative, or neutral?",),
        validator=_exact_answer_validator(("positive",)),
    ),
)

CASE_INDEX = {case.case_id: case for case in ALL_CASES}
