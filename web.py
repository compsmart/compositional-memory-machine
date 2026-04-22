from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from factgraph import FactGraph
from generation import CompositionalValueDecoder, FrozenGeneratorAdapter, make_value_vector
from hrr import SVOEncoder, VectorStore
from hrr.datasets import all_facts, fact_key
from ingestion import GeminiExtractor, TextIngestionPipeline
from memory import AMM
from query import QueryEngine


STATIC_DIR = Path(__file__).with_name("web_static")


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class HHRWebState:
    def __init__(
        self,
        *,
        dim: int = 2048,
        seed: int = 0,
        extractor: GeminiExtractor | None = None,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._extractor = extractor
        self.reset_demo()

    def reset_demo(self) -> None:
        self.encoder = SVOEncoder(dim=self.dim, seed=self.seed)
        self.memory = AMM()
        self.graph = FactGraph()
        self.pipeline = TextIngestionPipeline(
            self.encoder,
            self.memory,
            self.graph,
            extractor=self._extractor or GeminiExtractor(),
        )
        self.query = QueryEngine(encoder=self.encoder, memory=self.memory)
        self.generator = FrozenGeneratorAdapter()
        self._seed_fact_memory()
        self._build_compositional_demo()

    def status(self) -> dict[str, Any]:
        facts = self._list_facts()
        return {
            "title": "HHR",
            "stored_facts": len(facts),
            "graph_facts": len(self.graph.edges()),
            "memory_records": len(self.memory.records),
            "google_api_key": bool(self.pipeline.extractor._api_key()),
            "dim": self.dim,
            "demo_entity": self.compositional_demo["entity"],
            "demo_value": self.compositional_demo["expected_value"],
        }

    def facts(self, *, limit: int = 1000) -> dict[str, Any]:
        facts = self._list_facts()
        return {
            "total": len(facts),
            "facts": facts[-limit:],
            "graph": self._graph_payload(),
        }

    def query_svo(self, payload: dict[str, Any]) -> dict[str, Any]:
        subject = str(payload.get("subject", "")).strip()
        relation = str(payload.get("relation", "")).strip()
        object_value = str(payload.get("object", "")).strip()
        if not subject or not relation or not object_value:
            raise ValueError("subject, relation, and object are required")

        probe = self.query.ask_svo(subject, relation, object_value)
        answer = self.generator.answer(f"What does {subject} {relation}?", probe)
        vector = self.encoder.encode(subject, relation, object_value)
        evidence = self._candidate_facts(vector, top_k=5)
        current_target = self.graph.read(subject, relation)
        route = "no_reliable_match"
        if probe["found"]:
            route = "novel_composition" if probe.get("novel_composition") else "exact_fact"
        return {
            "result": {
                "answer": answer,
                "route": route,
                "confidence": float(probe["confidence"]),
                "found": bool(probe["found"]),
                "subject": probe.get("subject", subject),
                "relation": probe.get("verb", relation),
                "object": probe.get("object", object_value),
                "graph_target": current_target,
                "novel_composition": bool(probe.get("novel_composition", False)),
                "evidence": evidence,
            }
        }

    def ingest_text(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("text is required")
        if not self.pipeline.extractor._api_key():
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for text ingestion")

        domain = str(payload.get("domain") or "web").strip() or "web"
        source = str(payload.get("source") or "web-ui").strip() or "web-ui"
        result = self.pipeline.ingest_text(text, source=source, domain=domain)
        return {
            "ingestion": {
                "written_facts": result.written_facts,
                "pass1_count": result.pass1_count,
                "pass2_count": result.pass2_count,
                "estimated_fact_count": result.estimated_fact_count,
                "relation_stats": result.relation_stats,
                "facts": result.facts,
            },
            "status": self.status(),
            "facts": self.facts(),
        }

    def demo_reset(self, _payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.reset_demo()
        return {
            "status": self.status(),
            "facts": self.facts(),
            "compositional": self.demo_compositional(),
        }

    def demo_compositional(self) -> dict[str, Any]:
        hrr_native = self.value_decoder.decode(self.value_vector, strategy="hrr_native")
        linear = self.value_decoder.decode(self.value_vector, strategy="linear")
        answer = self.value_adapter.answer(
            self.compositional_demo["question"],
            {
                "confidence": 0.99,
                "entity": self.compositional_demo["entity"],
                "value_vector": self.value_vector,
                "decode_strategy": "linear",
            },
        )
        return {
            **self.compositional_demo,
            "hrr_native": to_jsonable(hrr_native),
            "linear": to_jsonable(linear),
            "answer": answer,
        }

    def _seed_fact_memory(self) -> None:
        for domain, fact in all_facts():
            key = fact_key(domain, fact)
            self.memory.write(
                key,
                self.encoder.encode_fact(fact),
                {
                    "domain": domain,
                    "subject": fact.subject,
                    "verb": fact.verb,
                    "object": fact.object,
                    "source": "seed",
                    "kind": "explicit",
                    "confidence": 1.0,
                },
            )
            self.graph.write(fact.subject, fact.verb, fact.object)

    def _build_compositional_demo(self) -> None:
        self.value_store = VectorStore(dim=256, seed=self.seed)
        self.value_decoder = CompositionalValueDecoder(store=self.value_store)
        examples: list[tuple[np.ndarray, str, str]] = []
        training_pairs = [
            ("amber", "bridge"),
            ("ancient", "cedar"),
            ("brisk", "cloud"),
            ("calm", "forest"),
            ("distant", "harbor"),
            ("golden", "garden"),
            ("hidden", "mirror"),
            ("rapid", "river"),
            ("warm", "valley"),
            ("quiet", "meadow"),
            ("mellow", "orchard"),
            ("silver", "signal"),
        ]
        for adjective, noun in training_pairs:
            examples.append((make_value_vector(self.value_store, adjective, noun), adjective, noun))
        self.value_decoder.fit_linear_head(examples)
        self.value_entity = "entity_demo"
        self.value_vector = make_value_vector(self.value_store, "silver", "signal")
        self.value_adapter = FrozenGeneratorAdapter(value_decoder=self.value_decoder, value_strategy="linear")
        self.compositional_demo = {
            "entity": self.value_entity,
            "question": f"What property does {self.value_entity} have?",
            "expected_value": "silver signal",
        }

    def _list_facts(self) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for record in self.memory.records.values():
            payload = record.payload
            if not {"subject", "verb", "object"}.issubset(payload):
                continue
            facts.append(
                {
                    "key": record.key,
                    "subject": payload["subject"],
                    "relation": payload["verb"],
                    "object": payload["object"],
                    "confidence": float(payload.get("confidence", 1.0)),
                    "source": str(payload.get("source", "seed")),
                    "kind": str(payload.get("kind", "explicit")),
                    "domain": str(payload.get("domain", "unknown")),
                    "metadata": {
                        "domain": str(payload.get("domain", "unknown")),
                    },
                }
            )
        return facts

    def _graph_payload(self) -> dict[str, Any]:
        edges = [to_jsonable(edge) for edge in self.graph.edges()]
        node_names = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
        return {
            "nodes": [{"id": name, "label": name} for name in node_names],
            "edges": edges,
        }

    def _candidate_facts(self, vector: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for record, score in self.memory.nearest(vector, top_k=top_k):
            payload = record.payload
            if not {"subject", "verb", "object"}.issubset(payload):
                continue
            evidence.append(
                {
                    "key": record.key,
                    "subject": payload["subject"],
                    "relation": payload["verb"],
                    "object": payload["object"],
                    "confidence": float(payload.get("confidence", 1.0)),
                    "score": float(score),
                    "source": str(payload.get("source", "seed")),
                    "domain": str(payload.get("domain", "unknown")),
                }
            )
        return evidence


class HHRWebHandler(BaseHTTPRequestHandler):
    state: HHRWebState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self._send_file(STATIC_DIR / parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/status":
                self._send_json(self.state.status())
            elif parsed.path == "/api/facts":
                self._send_json(self.state.facts())
            elif parsed.path == "/api/demo/compositional":
                self._send_json(self.state.demo_compositional())
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            routes = {
                "/api/query/svo": self.state.query_svo,
                "/api/ingest/text": self.state.ingest_text,
                "/api/demo/reset": self.state.demo_reset,
            }
            handler = routes.get(parsed.path)
            if handler is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send_json(handler(payload))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(to_jsonable(payload), indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": status.value}, status=status)

    def _send_file(self, path: Path) -> None:
        resolved = path.resolve()
        static_root = STATIC_DIR.resolve()
        if static_root not in resolved.parents and resolved != static_root:
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not resolved.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_handler(state: HHRWebState) -> type[HHRWebHandler]:
    class Handler(HHRWebHandler):
        pass

    Handler.state = state
    return Handler


def make_web_server(
    state: HHRWebState | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> HTTPServer:
    return HTTPServer((host, port), build_handler(state or HHRWebState()))


def run_web_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = make_web_server(host=host, port=port)
    print(f"HHR web UI running at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hhr-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    run_web_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
