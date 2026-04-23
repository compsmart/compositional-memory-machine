from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import unbind
from hrr.vectors import VectorStore

from experiments.hrr_claim_utils import bind_all, bound_token, bundle, nearest_token


@dataclass(frozen=True)
class PragmaticSentence:
    subject: str
    verb: str
    object: str
    sentiment: str
    certainty: str
    negation: str
    modality: str


def _sentences(n_sentences: int) -> list[PragmaticSentence]:
    subjects = [f"entity_{idx:03d}" for idx in range(80)]
    verbs = [f"verb_{idx:03d}" for idx in range(80)]
    objects = [f"object_{idx:03d}" for idx in range(80)]
    sentiments = ("positive", "neutral", "negative")
    certainties = ("certain", "likely", "uncertain")
    negations = ("affirmed", "negated")
    modalities = ("observed", "reported", "possible")
    return [
        PragmaticSentence(
            subject=subjects[idx % len(subjects)],
            verb=verbs[(idx * 3 + 1) % len(verbs)],
            object=objects[(idx * 5 + 2) % len(objects)],
            sentiment=sentiments[idx % len(sentiments)],
            certainty=certainties[(idx * 2) % len(certainties)],
            negation=negations[idx % len(negations)],
            modality=modalities[(idx * 4) % len(modalities)],
        )
        for idx in range(n_sentences)
    ]


def _components(store: VectorStore) -> dict[str, tuple[str, object]]:
    return {
        "subject": ("subj", store.get_unitary("__ROLE_SUBJECT__")),
        "verb": ("verb", store.get_unitary("__ROLE_VERB__")),
        "object": ("obj", store.get_unitary("__ROLE_OBJECT__")),
        "sentiment": ("sentiment", store.get_unitary("__ROLE_SENTIMENT__")),
        "certainty": ("certainty", store.get_unitary("__ROLE_CERTAINTY__")),
        "negation": ("negation", store.get_unitary("__ROLE_NEGATION__")),
        "modality": ("modality", store.get_unitary("__ROLE_MODALITY__")),
    }


def _sentence_vector(store: VectorStore, roles: dict[str, tuple[str, object]], sentence: PragmaticSentence):
    return bind_all(
        [
            bound_token(store, roles["subject"][1], roles["subject"][0], sentence.subject),
            bound_token(store, roles["verb"][1], roles["verb"][0], sentence.verb),
            bound_token(store, roles["object"][1], roles["object"][0], sentence.object),
            bound_token(store, roles["sentiment"][1], roles["sentiment"][0], sentence.sentiment),
            bound_token(store, roles["certainty"][1], roles["certainty"][0], sentence.certainty),
            bound_token(store, roles["negation"][1], roles["negation"][0], sentence.negation),
            bound_token(store, roles["modality"][1], roles["modality"][0], sentence.modality),
        ]
    )


def run(
    *,
    dim: int = 4096,
    seeds: tuple[int, ...] = (0, 1, 2),
    sentence_counts: tuple[int, ...] = (10, 25, 50),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        store = VectorStore(dim=dim, seed=seed)
        roles = _components(store)
        for n_sentences in sentence_counts:
            sentences = _sentences(n_sentences)
            memory = bundle(_sentence_vector(store, roles, sentence) for sentence in sentences)
            candidate_vectors: dict[str, dict[str, object]] = {
                "subject": {sentence.subject: store.get(f"subj:{sentence.subject}") for sentence in sentences},
                "verb": {sentence.verb: store.get(f"verb:{sentence.verb}") for sentence in sentences},
                "object": {sentence.object: store.get(f"obj:{sentence.object}") for sentence in sentences},
                "sentiment": {token: store.get(f"sentiment:{token}") for token in ("positive", "neutral", "negative")},
                "certainty": {token: store.get(f"certainty:{token}") for token in ("certain", "likely", "uncertain")},
                "negation": {token: store.get(f"negation:{token}") for token in ("affirmed", "negated")},
                "modality": {token: store.get(f"modality:{token}") for token in ("observed", "reported", "possible")},
            }

            hits = {name: 0 for name in candidate_vectors}
            total = len(sentences)

            for sentence in sentences:
                values = sentence.__dict__
                for role_name, (namespace, role_vec) in roles.items():
                    cue = bind_all(
                        [
                            bound_token(store, roles[other_name][1], roles[other_name][0], str(values[other_name]))
                            for other_name in roles
                            if other_name != role_name
                        ]
                    )
                    recovered = unbind(unbind(memory, cue), role_vec)
                    predicted, _score = nearest_token(recovered, candidate_vectors[role_name])
                    hits[role_name] += int(predicted == values[role_name])

            core_roles = ("subject", "verb", "object")
            nuanced_roles = ("sentiment", "certainty", "negation", "modality")
            rows.append(
                {
                    "dim": float(dim),
                    "seed": float(seed),
                    "n_sentences": float(n_sentences),
                    "core_acc": sum(hits[name] for name in core_roles) / (len(core_roles) * total),
                    "nuanced_acc": sum(hits[name] for name in nuanced_roles) / (len(nuanced_roles) * total),
                    "subject_acc": hits["subject"] / total,
                    "verb_acc": hits["verb"] / total,
                    "object_acc": hits["object"] / total,
                    "sentiment_acc": hits["sentiment"] / total,
                    "certainty_acc": hits["certainty"] / total,
                    "negation_acc": hits["negation"] / total,
                    "modality_acc": hits["modality"] / total,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sentences", type=int, nargs="+", default=[10, 25, 50])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds), sentence_counts=tuple(args.sentences)):
        print(row)


if __name__ == "__main__":
    main()
