from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.request import Request, urlopen

from hrr.encoder import SVOFact
from ingestion import ExtractedFact, ExtractionResponse, GeminiExtractor
from web import HHRWebState, make_web_server


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
        chat = _get(f"{base_url}/api/chat/history")
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
    assert chat["history"][0]["route"] == "ready"
    assert len(facts["graph"]["nodes"]) >= 10
    assert len(facts["graph"]["edges"]) == 20
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
