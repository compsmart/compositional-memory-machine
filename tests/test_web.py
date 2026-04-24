from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

from hrr.encoder import SVOFact
from ingestion import ExtractedFact, ExtractionResponse, GeminiExtractor, write_fact_jsonl
from ingestion.hf_corpora import StructuredFactRecord
from web import HHRWebState, build_web_state, make_web_server


class FakeExtractor(GeminiExtractor):
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


@contextmanager
def running_server(state: HHRWebState):
    server = make_web_server(state=state, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _get(url: str) -> dict[str, object]:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def _post(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _write_fact(state: HHRWebState, domain: str, fact: SVOFact) -> None:
    state.pipeline.write_structured_fact(
        ExtractedFact(
            subject=fact.subject,
            relation=fact.verb,
            object=fact.object,
            confidence=1.0,
            kind="explicit",
            source="fixture",
            source_id=f"fixture:{domain}",
        ),
        source="fixture",
        domain=domain,
    )


def test_web_status_and_facts_routes() -> None:
    with running_server(HHRWebState()) as base_url:
        status = _get(f"{base_url}/api/status")
        facts = _get(f"{base_url}/api/facts")
        fast_facts = _get(f"{base_url}/api/facts?fast=1&include_graph=0&include_chunks=0&limit=5")
        chat = _get(f"{base_url}/api/chat/history")
        snapshot = _get(f"{base_url}/api/snapshot")
        home = _get_text(f"{base_url}/")
        script = _get_text(f"{base_url}/static/app.js")

    assert status["stored_facts"] == 20
    assert status["graph_facts"] == 20
    assert status["memory_records"] == 20
    assert status["chunk_count"] == 5
    assert status["chunk_budget"] == 25
    assert status["perfect_chain_budget"] == 12
    assert facts["total"] == 20
    assert len(facts["chunks"]) == 5
    assert fast_facts["total"] == 20
    assert len(fast_facts["facts"]) == 5
    assert fast_facts["graph"] == {"nodes": [], "edges": []}
    assert fast_facts["chunks"] == []
    assert chat["history"][0]["route"] == "ready"
    assert len(facts["graph"]["nodes"]) >= 10
    assert len(facts["graph"]["edges"]) == 20
    assert snapshot["status"]["stored_facts"] == status["stored_facts"]
    assert snapshot["facts"]["total"] == facts["total"]
    assert "HRR + AMM Language Memory" in home
    assert 'id="chatForm"' in home
    assert "MemoryGraph3D" in script


def test_web_query_route_returns_structured_answer() -> None:
    with running_server(HHRWebState()) as base_url:
        payload = _post(
            f"{base_url}/api/query/svo",
            {"subject": "doctor", "relation": "treats", "object": "patient"},
        )

    result = payload["result"]
    assert result["found"] is True
    assert result["route"] == "exact_fact"
    assert result["answer"] == "doctor treats patient."
    assert result["graph_target"] == "patient"
    assert result["evidence"][0]["subject"] == "doctor"


def test_web_chain_query_route_returns_multi_hop_answer() -> None:
    state = HHRWebState()
    for domain, fact in [
        ("bridge", SVOFact("alice", "knows", "bob")),
        ("bridge", SVOFact("bob", "works_with", "carol")),
        ("bridge", SVOFact("carol", "guides", "delta")),
    ]:
        _write_fact(state, domain, fact)

    with running_server(state) as base_url:
        payload = _post(
            f"{base_url}/api/query/chain",
            {"subject": "alice", "relations": ["knows", "works_with", "guides"]},
        )

    result = payload["result"]
    assert result["found"] is True
    assert result["target"] == "delta"
    assert result["path"] == ["alice", "bob", "carol", "delta"]


def test_web_truth_history_and_branching_query_routes() -> None:
    state = HHRWebState()
    for domain, fact in [
        ("branch", SVOFact("alice", "knows", "bob")),
        ("branch", SVOFact("bob", "works_with", "carol")),
        ("branch", SVOFact("bob", "works_with", "dana")),
        ("branch", SVOFact("carol", "guides", "delta")),
        ("branch", SVOFact("dana", "guides", "echo")),
    ]:
        _write_fact(state, domain, fact)

    with running_server(state) as base_url:
        current_truth = _post(
            f"{base_url}/api/query/current-truth",
            {"subject": "bob", "relation": "works_with"},
        )
        history = _post(
            f"{base_url}/api/query/history",
            {"subject": "bob", "relation": "works_with"},
        )
        branching = _post(
            f"{base_url}/api/query/branching-chain",
            {"subject": "alice", "relations": ["knows", "works_with", "guides"]},
        )

    truth_result = current_truth["result"]
    assert truth_result["found"] is True
    assert truth_result["target"] == "dana"

    history_result = history["result"]
    assert [event["target"] for event in history_result["events"]] == ["carol", "dana"]

    branching_result = branching["result"]
    assert branching_result["found"] is True
    assert branching_result["branches"][0]["path"] == ["alice", "bob", "dana", "echo"]
    branch_paths = {tuple(branch["path"]) for branch in branching_result["branches"]}
    assert ("alice", "bob", "carol", "delta") in branch_paths
    assert ("alice", "bob", "dana", "echo") in branch_paths


def test_web_ingest_and_compositional_routes() -> None:
    state = HHRWebState(extractor=FakeExtractor())
    with running_server(state) as base_url:
        ingest = _post(
            f"{base_url}/api/ingest/text",
            {"text": "Ada text", "domain": "history", "source": "fixture"},
        )
        compositional = _get(f"{base_url}/api/demo/compositional")
        facts = _get(f"{base_url}/api/facts")

    assert ingest["ingestion"]["written_facts"] == 2
    assert compositional["linear"]["text"] == "silver signal"
    assert compositional["hrr_native"]["text"] == "silver signal"
    assert "silver signal" in compositional["answer"]
    assert any(fact["subject"] == "Ada Lovelace" and fact["relation"] == "worked_with" for fact in facts["facts"])
    ada_fact = next(fact for fact in facts["facts"] if fact["subject"] == "Ada Lovelace" and fact["relation"] == "worked_with")
    assert ada_fact["provenance"]["raw_relation"] == "collaborated with"
    assert ada_fact["provenance"]["matched_alias"] is True


def test_web_chat_route_supports_multi_turn_memory() -> None:
    state = HHRWebState(extractor=FakeExtractor())
    with running_server(state) as base_url:
        _post(
            f"{base_url}/api/ingest/text",
            {"text": "Ada text", "domain": "history", "source": "fixture"},
        )
        fact_reply = _post(f"{base_url}/api/chat", {"message": "Who did Ada Lovelace work with?"})
        pronoun_reply = _post(f"{base_url}/api/chat", {"message": "Who did she work with?"})
        pattern_reply = _post(f"{base_url}/api/chat", {"message": "Complete this learned pattern: 'the doctor ...'"})
        probabilistic_pattern_reply = _post(
            f"{base_url}/api/chat", {"message": "Complete this learned pattern: 'the artist ...'"}
        )
        learn_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed."},
        )
        recall_reply = _post(f"{base_url}/api/chat", {"message": "Do you still remember dax?"})

    assert fact_reply["reply"]["route"] == "fact_query"
    assert "Charles Babbage" in fact_reply["reply"]["text"]
    assert pronoun_reply["reply"]["route"] == "fact_query"
    assert "Charles Babbage" in pronoun_reply["reply"]["text"]
    assert pattern_reply["reply"]["route"] == "pattern_prediction"
    assert "treats" in pattern_reply["reply"]["text"]
    assert probabilistic_pattern_reply["reply"]["route"] == "pattern_prediction"
    assert probabilistic_pattern_reply["reply"]["alternatives"][0]["token"] == "sketches"
    assert "Alternatives:" in probabilistic_pattern_reply["reply"]["text"]
    assert learn_reply["reply"]["route"] == "word_learning"
    assert "ingest action" in learn_reply["reply"]["text"]
    assert recall_reply["reply"]["route"] == "word_recall"
    assert "nearest action is" in recall_reply["reply"]["text"]


def test_build_web_state_can_preload_jsonl(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "facts.jsonl"
    write_fact_jsonl(
        jsonl_path,
        [
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject="Grossglockner",
                    relation="first_climbed_on",
                    object="1800",
                    confidence=0.8,
                    kind="explicit",
                    source="fixture",
                    source_id="fixture:1",
                    excerpt="The Grossglockner was first climbed in 1800.",
                ),
                domain="hf_wikipedia_kg",
            )
        ],
    )

    state = build_web_state(preload_jsonl=jsonl_path, preload_limit=1)

    assert state.graph.read("Grossglockner", "first_climbed_on") == "1800"


def test_web_memory_bank_routes_list_and_switch_archives(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "reports" / "hf_ingest_runs" / "fixture_bank" / "facts.jsonl"
    write_fact_jsonl(
        jsonl_path,
        [
            StructuredFactRecord(
                fact=ExtractedFact(
                    subject="Grossglockner",
                    relation="first_climbed_on",
                    object="1800",
                    confidence=0.8,
                    kind="explicit",
                    source="fixture",
                    source_id="fixture:1",
                    excerpt="The Grossglockner was first climbed in 1800.",
                ),
                domain="hf_wikipedia_kg",
            )
        ],
    )

    state = HHRWebState(bank_root=tmp_path)
    with running_server(state) as base_url:
        banks = _get(f"{base_url}/api/memory-banks")
        switched = _post(
            f"{base_url}/api/memory-bank/select",
            {"bank_id": "reports/hf_ingest_runs/fixture_bank/facts.jsonl"},
        )

    assert banks["current_bank_id"] == "seed"
    assert any(bank["id"] == "reports/hf_ingest_runs/fixture_bank/facts.jsonl" for bank in banks["banks"])
    assert switched["selected_bank_id"] == "reports/hf_ingest_runs/fixture_bank/facts.jsonl"
    assert switched["loaded_archive_facts"] == 1
    assert switched["status"]["stored_facts"] == 21
    assert switched["status"]["current_memory_bank_id"] == "reports/hf_ingest_runs/fixture_bank/facts.jsonl"
    assert state.graph.read("Grossglockner", "first_climbed_on") == "1800"


def test_web_chat_route_end_to_end_multiturn_capabilities() -> None:
    state = HHRWebState(extractor=FakeExtractor())
    with running_server(state) as base_url:
        help_reply = _post(f"{base_url}/api/chat", {"message": "What can you do?"})
        ingest_ada_reply = _post(f"{base_url}/api/chat", {"message": "Remember Ada text"})
        ada_fact_reply = _post(f"{base_url}/api/chat", {"message": "Who did Ada Lovelace work with?"})
        ada_pronoun_reply = _post(f"{base_url}/api/chat", {"message": "Who did she work with?"})
        ada_reverse_reply = _post(f"{base_url}/api/chat", {"message": "Who worked with Charles Babbage?"})
        pattern_reply = _post(f"{base_url}/api/chat", {"message": "Complete this learned pattern: 'the doctor ...'"})
        probabilistic_pattern_reply = _post(
            f"{base_url}/api/chat", {"message": "Complete this learned pattern: 'the artist ...'"}
        )
        learn_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed."},
        )
        recall_reply = _post(f"{base_url}/api/chat", {"message": "What does it mean?"})
        ingest_alice_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Remember Alice knows Bob. Bob works with Carol. Carol guides Delta."},
        )
        reverse_lookup_reply = _post(f"{base_url}/api/chat", {"message": "Who knows Bob?"})
        multihop_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Who does Alice know who works with Carol?"},
        )
        history = _get(f"{base_url}/api/chat/history")

    assert help_reply["reply"]["route"] == "capability_overview"
    assert "memory-grounded controller" in help_reply["reply"]["text"]

    assert ingest_ada_reply["reply"]["route"] == "text_ingest"
    assert "2 distinct facts" in ingest_ada_reply["reply"]["text"]
    assert "chunk(s)" in ingest_ada_reply["reply"]["text"]

    assert ada_fact_reply["reply"]["route"] == "fact_query"
    assert "Ada Lovelace worked with Charles Babbage." in ada_fact_reply["reply"]["text"]
    assert ada_fact_reply["reply"]["graph_target"] == "Charles Babbage"

    assert ada_pronoun_reply["reply"]["route"] == "fact_query"
    assert "Charles Babbage" in ada_pronoun_reply["reply"]["text"]

    assert ada_reverse_reply["reply"]["route"] == "fact_query"
    assert "Ada Lovelace worked with Charles Babbage." in ada_reverse_reply["reply"]["text"]
    assert ada_reverse_reply["reply"]["graph_target"] == "Ada Lovelace"
    assert ada_reverse_reply["reply"]["chain_path"] == ["Charles Babbage", "Ada Lovelace"]

    assert pattern_reply["reply"]["route"] == "pattern_prediction"
    assert "The next token is 'treats'" in pattern_reply["reply"]["text"]

    assert probabilistic_pattern_reply["reply"]["route"] == "pattern_prediction"
    assert probabilistic_pattern_reply["reply"]["alternatives"][0]["token"] == "sketches"
    assert "Alternatives: sketches (0.29), draws (0.14)." in probabilistic_pattern_reply["reply"]["text"]

    assert learn_reply["reply"]["route"] == "word_learning"
    assert "I learned 'dax' as an ingest action." in learn_reply["reply"]["text"]

    assert recall_reply["reply"]["route"] == "word_recall"
    assert "'dax' still routes to ingest" in recall_reply["reply"]["text"]

    assert ingest_alice_reply["reply"]["route"] == "text_ingest"
    assert "3 distinct facts" in ingest_alice_reply["reply"]["text"]

    assert reverse_lookup_reply["reply"]["route"] == "fact_query"
    assert "Alice knows Bob." in reverse_lookup_reply["reply"]["text"]
    assert reverse_lookup_reply["reply"]["graph_target"] == "Alice"
    assert reverse_lookup_reply["reply"]["chain_path"] == ["Bob", "Alice"]

    assert multihop_reply["reply"]["route"] == "multi_hop_query"
    assert "Alice knows Bob, and Bob works with Carol." in multihop_reply["reply"]["text"]
    assert "So the answer is Bob." in multihop_reply["reply"]["text"]
    assert multihop_reply["reply"]["chain_path"] == ["Alice", "Bob", "Carol"]

    routes = [message["route"] for message in history["history"] if message["role"] == "assistant"]
    assert routes == [
        "ready",
        "capability_overview",
        "text_ingest",
        "fact_query",
        "fact_query",
        "fact_query",
        "pattern_prediction",
        "pattern_prediction",
        "word_learning",
        "word_recall",
        "text_ingest",
        "fact_query",
        "multi_hop_query",
    ]
    assert len(history["history"]) == 25


def test_web_chat_route_handles_frontier_controller_helpers() -> None:
    state = HHRWebState(extractor=FakeExtractor())
    with running_server(state) as base_url:
        _post(f"{base_url}/api/chat", {"message": "Remember Ada text"})
        explanation_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Why do you think Ada Lovelace worked with Charles Babbage?"},
        )
        spanish_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Con quien trabajo Ada Lovelace?"},
        )
        coding_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Write a Python function add(a, b) that returns the sum."},
        )
        logic_reply = _post(
            f"{base_url}/api/chat",
            {"message": "If Alice is taller than Bob and Bob is taller than Carol, who is tallest?"},
        )
        sequence_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Which number comes next in the sequence 2, 4, 8, 16, ?"},
        )
        sentiment_reply = _post(
            f"{base_url}/api/chat",
            {"message": "The movie was amazing and I loved every minute. Was the sentiment positive, negative, or neutral?"},
        )

    assert explanation_reply["reply"]["route"] == "explanation_query"
    assert "because" in explanation_reply["reply"]["text"].lower()
    assert "evidence" in explanation_reply["reply"]["text"].lower()
    assert "Charles Babbage" in explanation_reply["reply"]["text"]

    assert spanish_reply["reply"]["route"] == "multilingual_fact_query"
    assert "Charles Babbage" in spanish_reply["reply"]["text"]

    assert coding_reply["reply"]["route"] == "builtin_coding"
    assert "def add(a, b):" in coding_reply["reply"]["text"]
    assert "return a + b" in coding_reply["reply"]["text"]

    assert logic_reply["reply"]["route"] == "builtin_logic"
    assert "Alice is tallest." == logic_reply["reply"]["text"]

    assert sequence_reply["reply"]["route"] == "builtin_sequence"
    assert sequence_reply["reply"]["text"] == "32"

    assert sentiment_reply["reply"]["route"] == "builtin_sentiment"
    assert sentiment_reply["reply"]["text"] == "positive"


def test_web_chat_route_supports_multihop_after_chat_learning() -> None:
    state = HHRWebState(extractor=FakeExtractor())
    with running_server(state) as base_url:
        ingest_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Remember Alice knows Bob. Bob works with Carol. Carol guides Delta."},
        )
        multihop_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Who does Alice know who works with Carol?"},
        )

    assert ingest_reply["reply"]["route"] == "text_ingest"
    assert "3 distinct facts" in ingest_reply["reply"]["text"]
    assert multihop_reply["reply"]["route"] == "multi_hop_query"
    assert "Alice knows Bob" in multihop_reply["reply"]["text"]
    assert "Bob works with Carol" in multihop_reply["reply"]["text"]
    assert multihop_reply["reply"]["chain_path"] == ["Alice", "Bob", "Carol"]


def test_web_chat_route_handles_inverse_and_natural_relational_questions() -> None:
    state = HHRWebState()
    for fact in [
        SVOFact("Ada", "founded", "Meridian Labs"),
        SVOFact("Ada", "hired", "Ben"),
        SVOFact("Ben", "manages", "Cara"),
        SVOFact("Dev", "works_on", "Project Atlas"),
        SVOFact("Project Atlas", "uses", "Graphite Engine"),
        SVOFact("Jon", "maintains", "Graphite Engine"),
        SVOFact("Iris", "supports", "Jon"),
    ]:
        _write_fact(state, "assessment", fact)

    with running_server(state) as base_url:
        inverse_reply = _post(f"{base_url}/api/chat", {"message": "Who founded Meridian Labs?"})
        direct_reply = _post(f"{base_url}/api/chat", {"message": "Who did Ada hire?"})
        bridge_reply = _post(f"{base_url}/api/chat", {"message": "Who does Ada hire who manages Cara?"})
        forward_chain_reply = _post(
            f"{base_url}/api/chat",
            {"message": "What engine does the project Dev works on use?"},
        )
        reverse_chain_reply = _post(
            f"{base_url}/api/chat",
            {"message": "Who supports the person who maintains Graphite Engine?"},
        )

    assert inverse_reply["reply"]["route"] == "fact_query"
    assert "Ada founded Meridian Labs" in inverse_reply["reply"]["text"]
    assert direct_reply["reply"]["route"] == "fact_query"
    assert "Ada hired Ben" in direct_reply["reply"]["text"]
    assert bridge_reply["reply"]["route"] == "multi_hop_query"
    assert bridge_reply["reply"]["chain_path"] == ["Ada", "Ben", "Cara"]
    assert "So the answer is Ben" in bridge_reply["reply"]["text"]
    assert forward_chain_reply["reply"]["route"] == "multi_hop_query"
    assert "Project Atlas uses Graphite Engine" in forward_chain_reply["reply"]["text"]
    assert "So the answer is Graphite Engine" in forward_chain_reply["reply"]["text"]
    assert reverse_chain_reply["reply"]["route"] == "multi_hop_query"
    assert "Jon maintains Graphite Engine" in reverse_chain_reply["reply"]["text"]
    assert "Iris supports Jon" in reverse_chain_reply["reply"]["text"]
    assert "So the answer is Iris" in reverse_chain_reply["reply"]["text"]


def test_web_can_load_headless_scenario_and_export_snapshot() -> None:
    state = HHRWebState()
    with running_server(state) as base_url:
        snapshot = _post(
            f"{base_url}/api/scenario/load",
            {
                "reset": True,
                "facts": [
                    {
                        "subject": "Ada",
                        "relation": "founded",
                        "object": "Meridian Labs",
                        "domain": "scenario",
                        "source": "fixture",
                        "source_id": "scenario-1",
                        "excerpt": "Ada founded Meridian Labs.",
                    }
                ],
                "messages": [{"message": "Who founded Meridian Labs?"}],
            },
        )

    assert snapshot["facts"]["total"] >= 21
    ada_fact = next(
        fact for fact in snapshot["facts"]["facts"] if fact["subject"] == "Ada" and fact["relation"] == "founded"
    )
    assert ada_fact["provenance"]["source_id"] == "scenario-1"
    assert snapshot["chat"]["history"][-1]["route"] == "fact_query"
