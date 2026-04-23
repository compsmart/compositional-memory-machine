from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph
from hrr import SVOEncoder
from memory import AMM
from query import QueryEngine


def _prov(source: str, source_id: str, excerpt: str) -> dict[str, object]:
    return {
        "source": source,
        "source_id": source_id,
        "excerpt": excerpt,
    }


def run(*, dim: int = 2048, seeds: tuple[int, ...] = (42, 123)) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        graph = FactGraph()
        query = QueryEngine(encoder=SVOEncoder(dim=dim, seed=seed), memory=AMM(), graph=graph)

        graph.write(
            "robot",
            "location",
            "lab",
            provenance=_prov("doc_a", "doc-a", "Robot location was recorded as lab."),
        )
        graph.revise(
            "robot",
            "location",
            "field",
            provenance=_prov("doc_b", "doc-b", "Robot location was later corrected to field."),
        )
        graph.add_evidence(
            "robot",
            "location",
            "harbor",
            provenance=_prov("doc_c", "doc-c", "A competing report placed the robot at the harbor."),
        )
        graph.add_evidence(
            "sensor",
            "status",
            "offline",
            provenance=_prov("doc_d", "doc-d", "A note claimed the sensor was offline."),
        )

        current = query.ask_current_truth("robot", "location")
        history = query.ask_history("robot", "location")
        unresolved = query.ask_current_truth("sensor", "status")

        rows.append(
            {
                "seed": float(seed),
                "current_truth_em": float(current["target"] == "field"),
                "history_em": float(
                    [event["target"] for event in history["events"] if event["status"] != "evidence"]
                    == ["lab", "field"]
                ),
                "competing_evidence_em": float(current["competing_targets"] == ["harbor"]),
                "provenance_em": float(current["provenance"].get("source_id") == "doc-b"),
                "unresolved_refusal_em": float(
                    unresolved["found"] is False
                    and unresolved["unresolved"] is True
                    and unresolved["claim_count"] == 1
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds)):
        print(row)


if __name__ == "__main__":
    main()
