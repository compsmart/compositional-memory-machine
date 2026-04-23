from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.request import Request, urlopen

from ingestion import ExtractedFact, ExtractionResponse, GeminiExtractor
from web import HHRWebState, make_web_server


class FakeExtractor(GeminiExtractor):
    def extract(self, text: str, *, source: str = "") -> tuple[ExtractionResponse, ExtractionResponse]:
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
    assert facts["total"] == 20
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
    assert learn_reply["reply"]["route"] == "word_learning"
    assert "ingest action" in learn_reply["reply"]["text"]
    assert recall_reply["reply"]["route"] == "word_recall"
    assert "nearest action is" in recall_reply["reply"]["text"]
