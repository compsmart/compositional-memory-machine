from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph
from hrr import SVOEncoder
from ingestion import ExtractedFact, TextIngestionPipeline
from memory import AMM
from query import QueryEngine


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    category: str
    description: str
    domain: str
    seed_facts: tuple[ExtractedFact, ...]
    probe_fact: ExtractedFact
    expected_canonical: str | None

    @property
    def all_facts(self) -> list[ExtractedFact]:
        return [*self.seed_facts, self.probe_fact]


def _fact(subject: str, relation: str, object_: str, excerpt: str) -> ExtractedFact:
    return ExtractedFact(
        subject=subject,
        relation=relation,
        object=object_,
        confidence=0.95,
        source="research_real_corpus",
        excerpt=excerpt,
    )


CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="worked_with_teamed_up_with",
        category="positive",
        description="Map a disjoint collaboration surface onto worked_with.",
        domain="research",
        seed_facts=(
            _fact(
                "Ada Lovelace",
                "collaborated with",
                "Charles Babbage",
                "Ada Lovelace collaborated with Charles Babbage on a research lab prototype project.",
            ),
            _fact(
                "Grace Hopper",
                "worked on with",
                "Howard Aiken",
                "Grace Hopper worked on with Howard Aiken during a systems lab prototype effort.",
            ),
        ),
        probe_fact=_fact(
            "Barbara Liskov",
            "teamed up with",
            "John McCarthy",
            "Barbara Liskov teamed up with John McCarthy on a research lab prototype project.",
        ),
        expected_canonical="worked_with",
    ),
    CorpusCase(
        case_id="published_notes_about_memo_on",
        category="positive",
        description="Map memo-style phrasing onto published_notes_about.",
        domain="archives",
        seed_facts=(
            _fact(
                "Ada Lovelace",
                "published notes about",
                "the analytical engine",
                "Ada Lovelace published notes about the analytical engine in a formal lab memorandum.",
            ),
            _fact(
                "Charles Wheatstone",
                "authored notes about",
                "telegraph circuits",
                "Charles Wheatstone authored notes about telegraph circuits in a design memorandum.",
            ),
        ),
        probe_fact=_fact(
            "Elena Torres",
            "circulated a memo on",
            "the decoder design",
            "Elena Torres circulated a memo on the decoder design after the archival review meeting.",
        ),
        expected_canonical="published_notes_about",
    ),
    CorpusCase(
        case_id="proposed_as_pitched_as",
        category="positive",
        description="Map pitched-as language onto proposed_as.",
        domain="history",
        seed_facts=(
            _fact(
                "Analytical Engine",
                "was proposed as",
                "a general-purpose computer",
                "The Analytical Engine was proposed as a general-purpose computer in the early design notes.",
            ),
            _fact(
                "Difference Engine",
                "proposed mechanical",
                "a precision calculator",
                "The Difference Engine was proposed mechanical as a precision calculator in the design brief.",
            ),
        ),
        probe_fact=_fact(
            "Analytical Loom",
            "pitched as",
            "a general-purpose research engine",
            "The Analytical Loom was pitched as a general-purpose research engine during the symposium.",
        ),
        expected_canonical="proposed_as",
    ),
    CorpusCase(
        case_id="described_outlined",
        category="positive",
        description="Map outlined onto described for a technical artifact.",
        domain="technical",
        seed_facts=(
            _fact(
                "Ada Lovelace",
                "described",
                "an algorithm for Bernoulli numbers",
                "Ada Lovelace described an algorithm for Bernoulli numbers in a technical note.",
            ),
            _fact(
                "Protocol Memo",
                "described in",
                "a layered handshake",
                "The protocol memo described in detail a layered handshake for the distributed system.",
            ),
        ),
        probe_fact=_fact(
            "Marin Chen",
            "outlined",
            "a layered protocol",
            "Marin Chen outlined a layered protocol in the workshop brief for the systems team.",
        ),
        expected_canonical="described",
    ),
    CorpusCase(
        case_id="negative_mentored",
        category="negative",
        description="Leave a mentorship-style relation unresolved when no canonical family exists.",
        domain="education",
        seed_facts=(
            _fact(
                "Ada Lovelace",
                "collaborated with",
                "Charles Babbage",
                "Ada Lovelace collaborated with Charles Babbage on a research prototype.",
            ),
            _fact(
                "Leslie Lamport",
                "described",
                "a consistency proof",
                "Leslie Lamport described a consistency proof in the seminar notes.",
            ),
        ),
        probe_fact=_fact(
            "Grace Hopper",
            "mentored",
            "Katherine Johnson",
            "Grace Hopper mentored Katherine Johnson with training feedback and career guidance.",
        ),
        expected_canonical=None,
    ),
    CorpusCase(
        case_id="negative_relocated_to",
        category="negative",
        description="Leave a relocation relation unresolved instead of forcing it into proposed_as.",
        domain="operations",
        seed_facts=(
            _fact(
                "Analytical Engine",
                "was proposed as",
                "a general-purpose computer",
                "The Analytical Engine was proposed as a general-purpose computer in the design notes.",
            ),
            _fact(
                "Ada Lovelace",
                "published notes about",
                "engine diagrams",
                "Ada Lovelace published notes about engine diagrams in an archive memo.",
            ),
        ),
        probe_fact=_fact(
            "Meridian Labs",
            "relocated to",
            "Bristol",
            "Meridian Labs relocated to Bristol after the operations center expansion.",
        ),
        expected_canonical=None,
    ),
)


def evaluate_case(case: CorpusCase, *, enable_typed_relation_fallback: bool) -> dict[str, Any]:
    encoder = SVOEncoder(dim=2048, seed=0)
    memory = AMM()
    graph = FactGraph()
    pipeline = TextIngestionPipeline(
        encoder,
        memory,
        graph,
        enable_typed_relation_fallback=enable_typed_relation_fallback,
    )
    query = QueryEngine(encoder=encoder, memory=memory, graph=graph, relation_registry=pipeline.relation_registry)

    result = pipeline.ingest_facts(case.all_facts, source="research_real_corpus", domain=case.domain)
    probe_normalized = pipeline.relation_registry.normalize(case.probe_fact.relation)
    expected_canonical = case.expected_canonical
    canonical_query_found = False
    canonical_query_confidence = 0.0
    if expected_canonical is not None:
        canonical_query = query.ask_svo(case.probe_fact.subject, expected_canonical, case.probe_fact.object)
        canonical_query_found = bool(canonical_query["found"])
        canonical_query_confidence = float(canonical_query["confidence"])
    else:
        canonical_query = None
    exact_canonical_recovery = (
        expected_canonical is not None
        and probe_normalized.canonical == expected_canonical
        and probe_normalized.resolution_source in {"seed", "pair_overlap", "typed_fallback"}
        and canonical_query is not None
        and canonical_query_found
        and str(canonical_query.get("verb", "")) == expected_canonical
        and not bool(canonical_query.get("novel_composition", True))
    )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "description": case.description,
        "fallback_enabled": enable_typed_relation_fallback,
        "expected_canonical": expected_canonical,
        "normalized_relation": probe_normalized.canonical,
        "resolution_source": probe_normalized.resolution_source,
        "typed_fallback_hits": int(result.relation_stats.get("typed_fallback_hits", 0)),
        "unresolved_relation_examples": list(result.relation_stats.get("unresolved_relation_examples", [])),
        "canonical_query_found": canonical_query_found,
        "canonical_query_confidence": canonical_query_confidence,
        "exact_canonical_recovery": exact_canonical_recovery,
        "raw_graph_target": graph.read(case.probe_fact.subject, probe_normalized.canonical),
        "passed": _case_passed(
            expected_canonical=expected_canonical,
            normalized_relation=probe_normalized.canonical,
            resolution_source=probe_normalized.resolution_source,
            exact_canonical_recovery=exact_canonical_recovery,
        ),
        "canonical_query": canonical_query,
    }


def _case_passed(
    *,
    expected_canonical: str | None,
    normalized_relation: str,
    resolution_source: str,
    exact_canonical_recovery: bool,
) -> bool:
    if expected_canonical is None:
        return normalized_relation not in {"worked_with", "published_notes_about", "proposed_as", "described"}
    return exact_canonical_recovery


def run() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for enabled in (False, True):
        for case in CASES:
            rows.append(evaluate_case(case, enable_typed_relation_fallback=enabled))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for enabled in (False, True):
        enabled_rows = [row for row in rows if bool(row["fallback_enabled"]) == enabled]
        for category in ("positive", "negative"):
            bucket = [row for row in enabled_rows if row["category"] == category]
            summary_rows.append(
                {
                    "fallback_enabled": enabled,
                    "category": category,
                    "pass_rate": mean(float(row["passed"]) for row in bucket),
                    "exact_canonical_recovery_rate": mean(float(row["exact_canonical_recovery"]) for row in bucket),
                    "typed_fallback_rate": mean(float(row["typed_fallback_hits"] > 0) for row in bucket),
                    "cases": len(bucket),
                }
            )
        summary_rows.append(
            {
                "fallback_enabled": enabled,
                "category": "overall",
                "pass_rate": mean(float(row["passed"]) for row in enabled_rows),
                "exact_canonical_recovery_rate": mean(float(row["exact_canonical_recovery"]) for row in enabled_rows),
                "typed_fallback_rate": mean(float(row["typed_fallback_hits"] > 0) for row in enabled_rows),
                "cases": len(enabled_rows),
            }
        )
    return summary_rows


def render_markdown_report(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Typed Relation Fallback Real-Corpus Validation",
        "",
        "This report validates the experimental typed relation fallback on curated",
        "corpus-style facts and benchmark-style canonical queries.",
        "",
        "## Summary",
        "",
        "| fallback | category | pass_rate | exact_canonical_recovery_rate | typed_fallback_rate | cases |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {'on' if row['fallback_enabled'] else 'off'} | {row['category']} | {row['pass_rate']:.3f} | "
            f"{row['exact_canonical_recovery_rate']:.3f} | {row['typed_fallback_rate']:.3f} | {row['cases']} |"
        )

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| case_id | fallback | category | normalized | source | exact_canonical_recovery | passed |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {'on' if row['fallback_enabled'] else 'off'} | {row['category']} | "
            f"{row['normalized_relation']} | {row['resolution_source']} | "
            f"{1 if row['exact_canonical_recovery'] else 0} | {1 if row['passed'] else 0} |"
        )

    positive_off = [row for row in rows if not row["fallback_enabled"] and row["category"] == "positive"]
    positive_on = [row for row in rows if row["fallback_enabled"] and row["category"] == "positive"]
    negative_on = [row for row in rows if row["fallback_enabled"] and row["category"] == "negative"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Positive corpus-style cases improved from `pass_rate={mean(float(row['passed']) for row in positive_off):.3f}`",
            f"  with fallback off to `pass_rate={mean(float(row['passed']) for row in positive_on):.3f}` with fallback on.",
            f"- Exact canonical recovery for positive cases improved from",
            f"  `exact_canonical_recovery_rate={mean(float(row['exact_canonical_recovery']) for row in positive_off):.3f}`",
            f"  to `exact_canonical_recovery_rate={mean(float(row['exact_canonical_recovery']) for row in positive_on):.3f}`.",
            f"- Negative safety cases with fallback on stayed at `pass_rate={mean(float(row['passed']) for row in negative_on):.3f}`,",
            "  which is the key check against over-eager relation collapse.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate typed relation fallback on curated corpus-style facts.")
    parser.add_argument("--output", choices=("summary", "json"), default="summary")
    parser.add_argument("--json-file", type=Path)
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()

    rows = run()
    summary_rows = summarize(rows)
    if args.output == "json":
        print(json.dumps(rows, indent=2))
    else:
        print(json.dumps(summary_rows, indent=2))
    if args.json_file is not None:
        args.json_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_file.write_text(json.dumps({"rows": rows, "summary": summary_rows}, indent=2), encoding="utf-8")
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(render_markdown_report(rows, summary_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
