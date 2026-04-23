from __future__ import annotations

from factgraph import FactGraph
from hrr import SVOEncoder
from hrr.datasets import fact_key
from hrr.encoder import SVOFact
from ingestion import TextIngestionPipeline
from language import ContextExample, NGramLanguageMemory, WordLearningMemory
from memory import AMM, ChunkedKGMemory
from query import QueryEngine


SOURCE_TEXT = """
Ada Lovelace worked with Charles Babbage on the Analytical Engine.
Lovelace published notes about the machine in 1843.
Her notes described an algorithm for computing Bernoulli numbers.
The Analytical Engine was a proposed mechanical general-purpose computer.
"""


def say(role: str, text: str) -> None:
    print(f"{role}: {text}")


def sentence(subject: str, relation: str, object_: str) -> str:
    return f"{subject} {relation.replace('_', ' ')} {object_}."


def find_object(graph: FactGraph, subject: str, relation: str) -> str | None:
    return graph.read(subject, relation)


def main() -> None:
    encoder = SVOEncoder(dim=2048, seed=12)
    memory = AMM()
    chunk_memory = ChunkedKGMemory(chunk_size=3)
    graph = FactGraph()
    ingestion = TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)
    query = QueryEngine(encoder=encoder, memory=memory, graph=graph, chunk_memory=chunk_memory)

    ngram = NGramLanguageMemory(dim=2048, seed=13)
    ngram.learn_sequence(["the", "doctor", "treats", "the", "patient"], cycles=5)
    ngram.learn_sequence(["the", "chef", "prepares", "the", "meal"], cycles=5)

    learner = WordLearningMemory(dim=2048, seed=14)
    for action in ["eat", "drink", "consume"]:
        learner.add_known_action(action, "ingest", "ingest")
    for action in ["run", "walk", "travel"]:
        learner.add_known_action(action, "move", "move")

    say("User", "Read this short passage and remember it.")
    result = ingestion.ingest_text(SOURCE_TEXT, source="conversation_demo", domain="history")
    say(
        "Assistant",
        f"I extracted {len(result.facts)} distinct facts and wrote {result.written_facts} of them into HRR memory.",
    )

    say("User", "Who did Ada Lovelace work with?")
    target = find_object(graph, "Ada Lovelace", "worked_with")
    if target:
        probe = query.ask_svo("Ada Lovelace", "worked_with", target)
        say("Assistant", f"{sentence('Ada Lovelace', 'worked_with', target)} Confidence: {probe['confidence']:.3f}.")
    else:
        say("Assistant", "I do not have that fact in memory.")

    say("User", "What did she publish notes about?")
    notes_target = find_object(graph, "Ada Lovelace", "published_notes_about") or find_object(
        graph, "Lovelace", "published_notes_about"
    )
    if notes_target:
        say("Assistant", sentence("Ada Lovelace", "published_notes_about", notes_target))
    else:
        say("Assistant", "I do not have that fact in memory.")

    bridge_fact = SVOFact("Charles Babbage", "worked_on", "Analytical Engine")
    bridge_key = fact_key("history", bridge_fact)
    bridge_vector = encoder.encode_fact(bridge_fact)
    bridge_payload = {
        "domain": "history",
        "subject": bridge_fact.subject,
        "verb": bridge_fact.verb,
        "object": bridge_fact.object,
        "source": "conversation_demo",
        "kind": "explicit",
        "confidence": 1.0,
    }
    chunk_record = chunk_memory.write_fact(bridge_key, "history", bridge_fact, bridge_vector, bridge_payload)
    bridge_payload["chunk_id"] = chunk_record.chunk_id
    memory.write(bridge_key, bridge_vector, bridge_payload)
    graph.write(bridge_fact.subject, bridge_fact.verb, bridge_fact.object)

    say("User", "Can you follow a two-step chain from Ada Lovelace?")
    chain = query.ask_chain("Ada Lovelace", ["worked_with", "worked_on"])
    if chain["found"]:
        say("Assistant", f"Ada Lovelace reaches {chain['target']} via worked_with -> worked_on.")
    else:
        say("Assistant", "I could not trace a reliable chain.")

    say("User", "Complete this learned pattern: 'the doctor ...'")
    prediction = ngram.predict("the", "doctor")
    if prediction.token:
        say(
            "Assistant",
            f"The next token is '{prediction.token}' from context '{prediction.context_key}' "
            f"(confidence {prediction.confidence:.3f}).",
        )
    else:
        say("Assistant", "I do not know a reliable continuation.")

    say("User", "Now learn a new word: dax. A child daxes an apple; a chef daxes soup; a bird daxes seed.")
    learned = learner.learn_word(
        "dax",
        [
            ContextExample("child", "dax", "apple", "ingest"),
            ContextExample("chef", "dax", "soup", "ingest"),
            ContextExample("bird", "dax", "seed", "ingest"),
            ContextExample("student", "dax", "sandwich", "ingest"),
            ContextExample("doctor", "dax", "meal", "ingest"),
        ],
    )
    say(
        "Assistant",
        f"I learned 'dax' as an {learned['cluster']} action. Nearest known action: "
        f"{learned['nearest_action']} (confidence {learned['confidence']:.3f}).",
    )

    say("User", "Learn another word, then tell me if you still remember dax.")
    learner.learn_word(
        "blick",
        [
            ContextExample("runner", "blick", "track", "move"),
            ContextExample("traveler", "blick", "road", "move"),
            ContextExample("hiker", "blick", "trail", "move"),
            ContextExample("pilot", "blick", "route", "move"),
            ContextExample("child", "blick", "path", "move"),
        ],
    )
    retained = learner.retrieve_word("dax")
    say(
        "Assistant",
        f"Yes. 'dax' still routes to {retained['cluster']}; nearest action is "
        f"{retained['nearest_action']} (confidence {retained['confidence']:.3f}).",
    )

    say("User", "What happens if I ask for something you did not learn?")
    unknown = query.ask_svo("Ada Lovelace", "invented", "the telephone")
    if not unknown["found"]:
        say("Assistant", f"I do not have a reliable memory for that. Best confidence was {unknown['confidence']:.3f}.")
    else:
        say("Assistant", sentence(str(unknown["subject"]), str(unknown["verb"]), str(unknown["object"])))


if __name__ == "__main__":
    main()
