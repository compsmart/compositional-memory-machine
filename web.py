from __future__ import annotations

import argparse
import json
import mimetypes
import re
from collections import Counter
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
from ingestion import ExtractedFact, GeminiExtractor, TextIngestionPipeline
from language import ContextExample, NGramLanguageMemory, WordLearningMemory
from memory import AMM, ChunkedKGMemory
from query import QueryEngine


STATIC_DIR = Path(__file__).with_name("web_static")
CHAT_INGEST_OBJECTS = {"apple", "meal", "seed", "soup", "sandwich", "water", "tea", "food"}
CHAT_MOVE_OBJECTS = {"track", "road", "trail", "route", "path", "lane", "street"}


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
        self.chunk_memory = ChunkedKGMemory(dim=self.dim, role_count=4)
        self.graph = FactGraph()
        self.pipeline = TextIngestionPipeline(
            self.encoder,
            self.memory,
            self.graph,
            chunk_memory=self.chunk_memory,
            extractor=self._extractor or GeminiExtractor(),
        )
        self.query = QueryEngine(
            encoder=self.encoder,
            memory=self.memory,
            graph=self.graph,
            chunk_memory=self.chunk_memory,
            relation_registry=self.pipeline.relation_registry,
        )
        self.generator = FrozenGeneratorAdapter()
        self._seed_fact_memory()
        self._build_compositional_demo()
        self._build_language_demo()
        self._reset_chat_state()

    def status(self) -> dict[str, Any]:
        facts = self._list_facts()
        return {
            "title": "HHR",
            "stored_facts": len(facts),
            "graph_facts": len(self.graph.edges()),
            "memory_records": len(self.memory.records),
            "chunk_count": len(self.chunk_memory.chunks),
            "chunk_budget": self.chunk_memory.capacity_budget,
            "perfect_chain_budget": self.chunk_memory.perfect_chain_budget,
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
            "chunks": self.chunk_memory.chunk_summaries(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status(),
            "facts": self.facts(),
            "chat": self.chat_history_payload(),
            "compositional": self.demo_compositional(),
            "relation_registry": {
                "learned_aliases": self.pipeline.relation_registry.learned_aliases(),
                "alias_candidates": self.pipeline.relation_registry.candidate_aliases(),
                "alias_proposals": self.pipeline.relation_registry.proposal_log(),
            },
        }

    def chat_history_payload(self) -> dict[str, Any]:
        return {"history": self.chat_history}

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
                "chunk_id": probe.get("chunk_id"),
                "dimension_budget": self.query._dimension_hop_budget(1),
            }
        }

    def query_chain(self, payload: dict[str, Any]) -> dict[str, Any]:
        subject = str(payload.get("subject", "")).strip()
        relations = payload.get("relations")
        if not subject:
            raise ValueError("subject is required")
        if not isinstance(relations, list) or not relations or not all(isinstance(item, str) and item.strip() for item in relations):
            raise ValueError("relations must be a non-empty list of strings")

        result = self.query.ask_chain(subject, [item.strip() for item in relations])
        if result["found"]:
            target = str(result["target"])
            answer = f"{subject} reaches {target} via {' -> '.join(relations)}."
        else:
            answer = f"I could not trace a reliable chain for {subject}."
        return {"result": {"answer": answer, **result}}

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

    def load_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        reset = bool(payload.get("reset", True))
        if reset:
            self.reset_demo()

        for item in payload.get("texts", []):
            if not isinstance(item, dict):
                raise ValueError("scenario texts must be objects")
            self.ingest_text(item)

        for item in payload.get("facts", []):
            if not isinstance(item, dict):
                raise ValueError("scenario facts must be objects")
            fact = ExtractedFact(
                subject=str(item.get("subject", "")),
                relation=str(item.get("relation", "")),
                object=str(item.get("object", "")),
                confidence=float(item.get("confidence", 1.0)),
                kind=str(item.get("kind", "explicit")),
                source=str(item.get("source", "scenario")),
                source_id=str(item.get("source_id", "")),
                source_chunk_id=str(item.get("source_chunk_id", "")),
                excerpt=str(item.get("excerpt", "")),
            )
            domain = str(item.get("domain", "scenario")).strip() or "scenario"
            self.pipeline.write_structured_fact(fact, source=fact.source or "scenario", domain=domain)

        for item in payload.get("messages", []):
            if not isinstance(item, dict):
                raise ValueError("scenario messages must be objects")
            self.chat(item)

        return self.snapshot()

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message", "")).strip()
        if not message:
            raise ValueError("message is required")
        self._append_chat_message("user", message)
        reply = self._reply_to_chat(message)
        assistant = self._append_chat_message("assistant", **reply)
        return {
            "reply": assistant,
            "history": self.chat_history,
            "status": self.status(),
            "facts": self.facts(),
        }

    def demo_reset(self, _payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.reset_demo()
        return {
            "status": self.status(),
            "facts": self.facts(),
            "compositional": self.demo_compositional(),
            "chat": self.chat_history_payload(),
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
            self.pipeline.write_structured_fact(
                ExtractedFact(
                    subject=fact.subject,
                    relation=fact.verb,
                    object=fact.object,
                    confidence=1.0,
                    kind="explicit",
                    source="seed",
                    source_id=f"seed:{domain}",
                ),
                source="seed",
                domain=domain,
            )

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

    def _build_language_demo(self) -> None:
        self.ngram = NGramLanguageMemory(dim=self.dim, seed=self.seed + 1)
        self.ngram.learn_sequence(["the", "doctor", "treats", "the", "patient"], cycles=5)
        self.ngram.learn_sequence(["the", "chef", "prepares", "the", "meal"], cycles=5)
        self.ngram.learn_distribution("the", "artist", {"paints": 4.0, "sketches": 2.0, "draws": 1.0})
        self.word_learning = WordLearningMemory(dim=self.dim, seed=self.seed + 2)
        for action in ["eat", "drink", "consume"]:
            self.word_learning.add_known_action(action, "ingest", "ingest")
        for action in ["run", "walk", "travel"]:
            self.word_learning.add_known_action(action, "move", "move")

    def _reset_chat_state(self) -> None:
        self.chat_history: list[dict[str, Any]] = []
        self.chat_subject: str | None = None
        self.chat_word: str | None = None
        self._append_chat_message(
            "assistant",
            text=(
                "Memory ready. Ask about a stored fact, continue a learned pattern like "
                "\"Complete this learned pattern: 'the doctor ...'\", or teach me a new "
                "word. Use the Ingest pane or say \"Remember: <passage>\" to add new text. "
                "I can also trace simple multi-hop chains after you teach me linked facts."
            ),
            route="ready",
        )

    def _append_chat_message(self, role: str, text: str, **metadata: Any) -> dict[str, Any]:
        message = {"role": role, "text": text}
        for key, value in metadata.items():
            if value is None:
                continue
            message[key] = value
        self.chat_history.append(message)
        return message

    def _reply_to_chat(self, message: str) -> dict[str, Any]:
        for handler in (
            self._reply_to_ingest_prompt,
            self._reply_to_word_learning,
            self._reply_to_multihop_prompt,
            self._reply_to_word_recall,
            self._reply_to_pattern_prompt,
            self._reply_to_fact_prompt,
        ):
            reply = handler(message)
            if reply is not None:
                return reply
        return {
            "text": (
                "I can answer memory-backed fact questions, continue a learned pattern, "
                "or learn a demo word from examples. Try \"Who did Ada Lovelace work "
                "with?\", \"Complete this learned pattern: 'the doctor ...'\", or "
                "\"Learn a new word: dax. A child daxes an apple; a chef daxes soup; "
                "a bird daxes seed.\" You can also teach me linked facts and ask a "
                "multi-hop question like \"Who does Alice know who works with Carol?\""
            ),
            "route": "fallback",
        }

    def _reply_to_ingest_prompt(self, message: str) -> dict[str, Any] | None:
        if not re.match(r"(?is)^\s*(remember|memorize|ingest|read)\b", message):
            return None
        parts = re.split(r":|\n", message, maxsplit=1)
        text = parts[1].strip() if len(parts) >= 2 else ""
        if not text:
            text = re.sub(r"(?is)^\s*(remember|memorize|ingest|read)\b\s*:?\s*", "", message, count=1).strip()
        if not text:
            return {
                "text": "Paste the passage after ':' or use the Ingest pane so I can write it into memory.",
                "route": "ingest_prompt",
            }
        if not self.pipeline.extractor._api_key():
            return {
                "text": "I need GOOGLE_API_KEY or GEMINI_API_KEY before I can ingest raw text in chat.",
                "route": "ingest_unavailable",
            }
        result = self.pipeline.ingest_text(text, source="chat", domain="chat")
        if result.facts:
            self.chat_subject = result.facts[0].subject
        return {
            "text": (
                f"I extracted {len(result.facts)} distinct facts and wrote "
                f"{result.written_facts} of them into HRR memory across "
                f"{result.relation_stats.get('chunk_count', 0)} chunk(s)."
            ),
            "route": "text_ingest",
        }

    def _reply_to_multihop_prompt(self, message: str) -> dict[str, Any] | None:
        if "?" not in message and not re.match(r"(?i)^\s*(who|what|which|where|whom)\b", message):
            return None
        match = self._match_multihop_path(message)
        if match is None:
            return None
        path = match["path"]
        if len(path) == 1 and path[0]["direction"] == "forward":
            return None

        result = self._probe_relational_path(message, match)
        if result is None:
            return None
        self.chat_subject = str(result["graph_target"])
        return result

    def _reply_to_word_learning(self, message: str) -> dict[str, Any] | None:
        match = re.search(r"\blearn(?:\s+\w+){0,3}\s+word:\s*([A-Za-z][\w-]*)", message, re.IGNORECASE)
        if match is None:
            return None
        word = match.group(1).lower()
        examples = self._extract_word_examples(message, word)
        if not examples:
            return {
                "text": (
                    f"I need a few examples for '{word}', for example "
                    "\"A child daxes an apple; a chef daxes soup.\""
                ),
                "route": "word_learning_prompt",
            }
        learned = self.word_learning.learn_word(word, examples)
        self.chat_word = word
        cluster = learned.get("cluster") or "unknown"
        nearest_action = learned.get("nearest_action") or "unknown"
        confidence = float(learned.get("confidence") or 0.0)
        return {
            "text": (
                f"I learned '{word}' as an {cluster} action. Nearest known action: "
                f"{nearest_action} (confidence {confidence:.3f})."
            ),
            "route": "word_learning",
            "confidence": confidence,
        }

    def _reply_to_word_recall(self, message: str) -> dict[str, Any] | None:
        match = re.search(r"\b(?:remember|recall|know|mean)\s+([A-Za-z][\w-]*)\b", message, re.IGNORECASE)
        if match is None:
            match = re.search(r"\bwhat does\s+([A-Za-z][\w-]*)\s+mean\b", message, re.IGNORECASE)
        if match is None:
            return None
        word = match.group(1).lower()
        if word in {"who", "what", "where", "when", "why", "how"}:
            return None
        if word in {"it", "that"} and self.chat_word:
            word = self.chat_word
        retrieved = self.word_learning.retrieve_word(word)
        if not retrieved.get("found"):
            return {
                "text": f"I have not learned '{word}' yet.",
                "route": "word_recall_miss",
            }
        self.chat_word = word
        confidence = float(retrieved.get("confidence") or 0.0)
        return {
            "text": (
                f"Yes. '{word}' still routes to {retrieved['cluster']}; nearest action is "
                f"{retrieved['nearest_action']} (confidence {confidence:.3f})."
            ),
            "route": "word_recall",
            "confidence": confidence,
        }

    def _reply_to_pattern_prompt(self, message: str) -> dict[str, Any] | None:
        context = self._extract_pattern_context(message)
        if context is None:
            return None
        tokens = re.findall(r"[A-Za-z0-9']+", context.lower())
        if len(tokens) < 2:
            return {
                "text": "Give me at least two tokens of context before the ellipsis.",
                "route": "pattern_prompt",
            }
        prediction = self.ngram.predict(tokens[-2], tokens[-1])
        if prediction.token is None:
            return {
                "text": "I do not know a reliable continuation for that pattern yet.",
                "route": "pattern_miss",
                "confidence": float(prediction.confidence),
            }
        alternatives = [candidate for candidate in prediction.alternatives if candidate.token != prediction.token]
        alternative_text = ""
        if alternatives:
            rendered = ", ".join(
                f"{candidate.token} ({candidate.probability:.2f})" for candidate in alternatives[:3]
            )
            alternative_text = f" Alternatives: {rendered}."
        return {
            "text": (
                f"The next token is '{prediction.token}' from context "
                f"'{prediction.context_key}' (confidence {prediction.confidence:.3f})."
                f"{alternative_text}"
            ),
            "route": "pattern_prediction",
            "confidence": float(prediction.confidence),
            "alternatives": alternatives,
        }

    def _reply_to_fact_prompt(self, message: str) -> dict[str, Any] | None:
        subject = self._resolve_subject(message)
        if subject is None:
            return None
        edge = self._match_relation(subject, message)
        if edge is None:
            relations = sorted(
                {edge.relation.replace("_", " ") for edge in self.graph.edges() if edge.source == subject}
            )
            if not relations:
                return None
            return {
                "text": (
                    f"I have memories for {subject}, but I could not match the relation in that question. "
                    f"Try one of: {', '.join(relations[:4])}."
                ),
                "route": "fact_prompt_miss",
            }
        target = edge.target
        probe = self.query.ask_svo(subject, edge.relation, target)
        self.chat_subject = subject
        if not probe["found"]:
            return {
                "text": f"I do not have a reliable memory for that. Best confidence was {probe['confidence']:.3f}.",
                "route": "no_reliable_match",
                "confidence": float(probe["confidence"]),
            }
        evidence = self._candidate_facts(self.encoder.encode(subject, edge.relation, target), top_k=5)
        return {
            "text": f"{self._sentence(subject, edge.relation, target)} Confidence: {probe['confidence']:.3f}.",
            "route": "fact_query",
            "confidence": float(probe["confidence"]),
            "graph_target": target,
            "evidence": evidence,
        }

    def _extract_pattern_context(self, message: str) -> str | None:
        quoted = re.search(r"""["']([^"']+?)\s*\.\.\.["']""", message)
        if quoted:
            return quoted.group(1)
        raw = re.search(r"([A-Za-z][A-Za-z0-9' -]+)\s*\.\.\.", message)
        if raw:
            return raw.group(1)
        return None

    def _extract_word_examples(self, message: str, word: str) -> list[ContextExample]:
        examples: list[ContextExample] = []
        pattern = re.compile(
            rf"(?i)\b(?:a|an|the)\s+(?P<subject>[a-z][\w-]*)\s+{re.escape(word)}(?:es|s)?\s+"
            rf"(?:a|an|the)\s+(?P<object>[a-z][\w-]*)"
        )
        for clause in re.split(r"[;\n]+", message):
            match = pattern.search(clause)
            if match is None:
                continue
            subject = match.group("subject").lower()
            object_ = match.group("object").lower()
            examples.append(
                ContextExample(
                    subject,
                    word,
                    object_,
                    self._infer_property_hint(subject, object_),
                )
            )
        hints = [example.property_hint for example in examples if example.property_hint]
        if hints:
            default_hint = Counter(hints).most_common(1)[0][0]
            examples = [
                ContextExample(example.subject, example.action, example.object, example.property_hint or default_hint)
                for example in examples
            ]
        return examples

    def _infer_property_hint(self, subject: str, object_: str) -> str | None:
        if object_ in CHAT_INGEST_OBJECTS:
            return "ingest"
        if object_ in CHAT_MOVE_OBJECTS:
            return "move"
        if subject in {"chef", "doctor", "child", "bird", "student"}:
            return "ingest"
        if subject in {"runner", "traveler", "hiker", "pilot"}:
            return "move"
        return None

    def _resolve_subject(self, message: str) -> str | None:
        lowered = message.lower()
        if re.search(r"\b(she|her|he|him|they|them)\b", lowered) and self.chat_subject:
            return self.chat_subject
        subjects = list({fact["subject"] for fact in self._list_facts()})
        exact_hits = []
        for subject in subjects:
            position = lowered.find(subject.lower())
            if position != -1:
                exact_hits.append((position, -len(subject), subject))
        if exact_hits:
            exact_hits.sort()
            return exact_hits[0][2]
        subjects = sorted(subjects, key=len, reverse=True)
        for subject in subjects:
            parts = subject.lower().split()
            if len(parts) > 1 and lowered.find(parts[-1]) != -1:
                return subject
        return None

    def _match_relation(self, subject: str, message: str) -> Any | None:
        edges = [edge for edge in self.graph.edges() if edge.source == subject]
        if not edges:
            return None
        message_tokens = self._question_tokens(message)
        best_edge = None
        best_score = 0
        for edge in edges:
            relation_token_count = len(self._question_tokens(edge.relation.replace("_", " ")))
            overlap = self._relation_overlap(edge.relation, message_tokens)
            if overlap == relation_token_count and overlap > best_score:
                best_edge = edge
                best_score = overlap
        if best_edge is not None:
            return best_edge
        for edge in edges:
            overlap = self._relation_overlap(edge.relation, message_tokens)
            if overlap > best_score:
                best_edge = edge
                best_score = overlap
        if best_edge is not None:
            return best_edge
        if len(edges) == 1 and "?" in message:
            return edges[0]
        return None

    def _match_multihop_path(self, message: str, *, max_hops: int = 4) -> dict[str, Any] | None:
        message_tokens = self._question_tokens(message)
        anchors = self._path_anchors(message)
        if not anchors:
            return None
        mentioned_entities = {entity.lower() for entity in self._mentioned_entities(message)}
        best_path = None
        best_score = float("-inf")
        for anchor_index, subject in enumerate(anchors):
            for path in self._graph_paths_from(subject, max_hops=max_hops):
                score = self._score_multihop_path(
                    path,
                    message_tokens=message_tokens,
                    mentioned_entities=mentioned_entities,
                    anchor_index=anchor_index,
                )
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_path = {"subject": subject, "path": path, "score": score}
        return best_path

    def _graph_paths_from(self, subject: str, *, max_hops: int) -> list[list[Any]]:
        adjacency: dict[str, list[Any]] = {}
        reverse_adjacency: dict[str, list[Any]] = {}
        for edge in self.graph.edges():
            adjacency.setdefault(edge.source, []).append(edge)
            reverse_adjacency.setdefault(edge.target, []).append(edge)

        paths: list[list[Any]] = []

        def walk(current: str, current_path: list[Any], seen: set[str]) -> None:
            if current_path:
                paths.append(current_path.copy())
            if len(current_path) >= max_hops:
                return
            for edge in adjacency.get(current, []):
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                current_path.append({"edge": edge, "direction": "forward", "from": current, "to": edge.target})
                walk(edge.target, current_path, seen)
                current_path.pop()
                seen.remove(edge.target)
            for edge in reverse_adjacency.get(current, []):
                if edge.source in seen:
                    continue
                seen.add(edge.source)
                current_path.append({"edge": edge, "direction": "reverse", "from": current, "to": edge.source})
                walk(edge.source, current_path, seen)
                current_path.pop()
                seen.remove(edge.source)

        walk(subject, [], {subject})
        return paths

    def _mentioned_entities(self, message: str) -> list[str]:
        lowered = message.lower()
        entities = sorted(
            {edge.source for edge in self.graph.edges()} | {edge.target for edge in self.graph.edges()},
            key=len,
            reverse=True,
        )
        return [entity for entity in entities if entity.lower() in lowered]

    def _chain_sentence(self, steps: list[dict[str, Any]]) -> str:
        if not steps:
            return "I could not trace a chain."
        clauses = [
            f"{step['subject']} {step['relation'].replace('_', ' ')} {step['target']}"
            for step in steps
        ]
        if len(clauses) == 1:
            return clauses[0] + "."
        if len(clauses) == 2:
            return f"{clauses[0]}, and {clauses[1]}."
        return ", ".join(clauses[:-1]) + f", and {clauses[-1]}."

    def _normalized_tokens(self, value: str) -> list[str]:
        tokens: list[str] = []
        for token in self._question_tokens(value):
            tokens.extend(sorted(self._token_forms(token)))
        return tokens

    def _path_anchors(self, message: str) -> list[str]:
        anchors = self._mentioned_entities_with_positions(message)
        lowered = message.lower()
        if re.search(r"\b(she|her|he|him|they|them)\b", lowered) and self.chat_subject:
            anchors = [self.chat_subject, *anchors]
        ordered: list[str] = []
        seen: set[str] = set()
        for anchor in anchors:
            key = anchor.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(anchor)
        return ordered

    def _mentioned_entities_with_positions(self, message: str) -> list[str]:
        lowered = message.lower()
        entities = sorted(
            {edge.source for edge in self.graph.edges()} | {edge.target for edge in self.graph.edges()},
            key=len,
            reverse=True,
        )
        hits: list[tuple[int, int, str]] = []
        for entity in entities:
            position = lowered.find(entity.lower())
            if position != -1:
                hits.append((position, -len(entity), entity))
        hits.sort()
        ordered: list[str] = []
        seen: set[str] = set()
        for _position, _neg_len, entity in hits:
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(entity)
        return ordered

    @staticmethod
    def _question_tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value.lower())

    def _relation_overlap(self, relation: str, message_tokens: list[str]) -> int:
        relation_tokens = self._question_tokens(relation.replace("_", " "))
        return sum(
            1
            for relation_token in relation_tokens
            if any(self._token_matches(relation_token, message_token) for message_token in message_tokens)
        )

    def _score_multihop_path(
        self,
        path: list[dict[str, Any]],
        *,
        message_tokens: list[str],
        mentioned_entities: set[str],
        anchor_index: int,
    ) -> float | None:
        relation_hits = 0
        relation_score = 0
        entity_score = 0
        reverse_steps = 0
        for step in path:
            edge = step["edge"]
            overlap = self._relation_overlap(edge.relation, message_tokens)
            if overlap > 0:
                relation_hits += 1
                relation_score += overlap
            if step["direction"] == "reverse":
                reverse_steps += 1
            if edge.source.lower() in mentioned_entities or edge.target.lower() in mentioned_entities:
                entity_score += 1
        if len(path) == 1:
            if path[0]["direction"] == "forward" or relation_hits == 0:
                return None
        elif relation_hits < 2:
            return None
        extra_steps = max(0, len(path) - relation_hits)
        return (
            relation_hits * 20
            + relation_score * 5
            + entity_score * 4
            - extra_steps * 15
            - reverse_steps * 2
            - len(path)
            - anchor_index * 5
        )

    def _probe_relational_path(self, message: str, match: dict[str, Any]) -> dict[str, Any] | None:
        subject = str(match["subject"])
        path = list(match["path"])
        if not path:
            return None
        current = subject
        chain_path = [subject]
        actual_steps: list[dict[str, Any]] = []
        confidence = self.query._dimension_hop_budget(len(path))
        for hop, traversal in enumerate(path, start=1):
            edge = traversal["edge"]
            probe = self.query.ask_svo(edge.source, edge.relation, edge.target)
            if not probe["found"]:
                return None
            current = str(traversal["to"])
            chain_path.append(current)
            step_confidence = float(probe["confidence"])
            confidence *= max(step_confidence * self.query.hop_decay, 1e-6)
            actual_steps.append(
                {
                    "hop": hop,
                    "subject": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "direction": traversal["direction"],
                    "confidence": step_confidence,
                    "chunk_id": probe.get("chunk_id"),
                }
            )
        if confidence < self.query.min_confidence:
            return {
                "text": f"I found a possible chain start for {subject}, but I could not complete a reliable multi-hop trace.",
                "route": "multi_hop_miss",
                "confidence": confidence,
            }
        answer_node = self._answer_node_for_path(message, chain_path)
        text = self._relational_answer_text(actual_steps, answer_node, confidence)
        route = "fact_query" if len(actual_steps) == 1 else "multi_hop_query"
        evidence = [
            {
                "subject": step["subject"],
                "relation": step["relation"],
                "object": step["target"],
                "score": float(step["confidence"]),
                "chunk_id": step.get("chunk_id"),
            }
            for step in actual_steps
        ]
        payload = {
            "text": text,
            "route": route,
            "confidence": confidence,
            "graph_target": answer_node,
            "evidence": evidence,
        }
        if len(actual_steps) > 1 or actual_steps[0]["direction"] == "reverse":
            payload["chain_path"] = chain_path
        return payload

    @staticmethod
    def _answer_node_for_path(message: str, chain_path: list[str]) -> str:
        lowered = message.lower()
        if len(chain_path) >= 3 and re.match(r"\s*who\s+does\b", lowered) and re.search(r"\bwho\b", lowered[8:]):
            return chain_path[1]
        return chain_path[-1]

    def _relational_answer_text(self, steps: list[dict[str, Any]], answer_node: str, confidence: float) -> str:
        explanation = self._chain_sentence(steps)
        if len(steps) == 1 and answer_node == steps[0]["subject"]:
            return f"{explanation} Confidence: {confidence:.3f}."
        if len(steps) == 1 and answer_node == steps[0]["target"]:
            return f"{explanation} Confidence: {confidence:.3f}."
        return f"{explanation} So the answer is {answer_node}. Confidence: {confidence:.3f}."

    def _token_matches(self, left: str, right: str) -> bool:
        left_forms = self._token_forms(left)
        right_forms = self._token_forms(right)
        for left_form in left_forms:
            for right_form in right_forms:
                if left_form == right_form:
                    return True
                if min(len(left_form), len(right_form)) >= 3 and (
                    left_form.startswith(right_form) or right_form.startswith(left_form)
                ):
                    return True
        return False

    @staticmethod
    def _token_forms(token: str) -> set[str]:
        forms = {token}
        if token.endswith("ies") and len(token) > 4:
            forms.add(token[:-3] + "y")
        if token.endswith("ing") and len(token) > 5:
            base = token[:-3]
            forms.add(base)
            forms.add(base + "e")
        if token.endswith("ed") and len(token) > 4:
            base = token[:-2]
            forms.add(base)
            if not base.endswith("e"):
                forms.add(base + "e")
        if token.endswith("es") and len(token) > 4:
            base = token[:-2]
            forms.add(base)
            if not base.endswith("e"):
                forms.add(base + "e")
        if token.endswith("s") and len(token) > 3:
            forms.add(token[:-1])
        return {form for form in forms if form}

    @staticmethod
    def _sentence(subject: str, relation: str, object_: str) -> str:
        return f"{subject} {relation.replace('_', ' ')} {object_}."

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
                    "chunk_id": payload.get("chunk_id"),
                    "provenance": to_jsonable(payload.get("provenance", {})),
                    "metadata": {
                        "domain": str(payload.get("domain", "unknown")),
                        "chunk_id": payload.get("chunk_id"),
                        "raw_relation": payload.get("raw_relation"),
                        "normalized_relation": payload.get("normalized_relation"),
                        "matched_alias": bool(payload.get("matched_alias", False)),
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
                    "chunk_id": payload.get("chunk_id"),
                    "provenance": to_jsonable(payload.get("provenance", {})),
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
            elif parsed.path == "/api/chat/history":
                self._send_json(self.state.chat_history_payload())
            elif parsed.path == "/api/snapshot":
                self._send_json(self.state.snapshot())
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
                "/api/chat": self.state.chat,
                "/api/query/svo": self.state.query_svo,
                "/api/query/chain": self.state.query_chain,
                "/api/ingest/text": self.state.ingest_text,
                "/api/scenario/load": self.state.load_scenario,
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
