from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_d2839_sequence_chain import summarize as summarize_sequence_chain
from experiments.exp_d2839_sequence_chain import run as run_sequence_chain
from experiments.exp_d2855_hierarchical_syntax import run as run_hierarchical_syntax
from ingestion import ExtractedFact, ExtractionResponse, GeminiExtractor
from web import HHRWebState


class StructuralBenchmarkExtractor(GeminiExtractor):
    def extract(self, text: str, *, source: str = "") -> tuple[ExtractionResponse, ExtractionResponse]:
        return (
            ExtractionResponse(
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
            ),
            ExtractionResponse(
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
            ),
        )

    def _api_key(self) -> str | None:
        return "fixture-key"


def run(
    *,
    seeds: tuple[int, ...] = (42, 123),
    prefix_lengths: tuple[int, ...] = (1, 2, 3, 5),
    depths: tuple[int, ...] = (2, 3),
    sentence_counts: tuple[int, ...] = (25,),
) -> dict[str, float]:
    sequence_rows = run_sequence_chain(seeds=seeds, prefix_lengths=prefix_lengths)
    sequence_summary = summarize_sequence_chain(sequence_rows)
    hierarchical_rows = run_hierarchical_syntax(dim=4096, seeds=seeds, depths=depths, sentence_counts=sentence_counts)

    prefix_k3 = next(row for row in sequence_summary if int(row["prefix_len"]) == 3)
    depth3 = next(row for row in hierarchical_rows if int(row["depth"]) == 3)

    state = HHRWebState(extractor=StructuralBenchmarkExtractor())
    doctor = state.chat({"message": "Complete this learned pattern: 'the doctor ...'"})["reply"]
    artist = state.chat({"message": "Complete this learned pattern: 'the artist ...'"})["reply"]
    pattern_scores = [
        float(doctor.get("route") == "pattern_prediction" and "treats" in str(doctor.get("text", ""))),
        float(
            artist.get("route") == "pattern_prediction"
            and "paints" in str(artist.get("text", ""))
            and "sketches" in str(artist.get("text", ""))
        ),
    ]

    return {
        "prefix_threshold_em": float(prefix_k3["mean_em"]),
        "hierarchical_depth3_acc": float(depth3["main_acc"]),
        "pattern_surface_em": float(mean(pattern_scores)),
        "breadth_score": float(mean([float(prefix_k3["mean_em"]), float(depth3["main_acc"]), mean(pattern_scores)])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()
    print(run(seeds=tuple(args.seeds)))


if __name__ == "__main__":
    main()
