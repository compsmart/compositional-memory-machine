from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.conversation_benchmark_cases import (  # noqa: E402
    CASE_INDEX,
    ROADMAP_SERIOUS_CASE_IDS,
    SMOKE_CASE_IDS,
    BenchmarkCase,
    BenchmarkConfig,
    run_chat_case,
    run_metric_case,
)


MetricRow = dict[str, Any]
SUMMARY_GROUPS = (
    ("overall", ("summary_type",)),
    ("track", ("summary_type", "track")),
    ("surface", ("summary_type", "surface")),
    ("category", ("summary_type", "category")),
)


def _generated_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _sort_key(row: MetricRow) -> tuple[str, str]:
    summary_type = str(row.get("summary_type", "case"))
    label = str(
        row.get("category")
        or row.get("track")
        or row.get("surface")
        or row.get("label")
        or row.get("case_id")
        or ""
    )
    return summary_type, label


def _selected_case_ids(preset: str | None, case_ids: tuple[str, ...] | None) -> tuple[str, ...]:
    if case_ids:
        return case_ids
    if preset == "smoke":
        return SMOKE_CASE_IDS
    return ROADMAP_SERIOUS_CASE_IDS


def _case_row(case: BenchmarkCase, verdict: MetricRow) -> MetricRow:
    score = float(verdict["score"])
    weight = float(case.weight)
    return {
        "summary_type": "case",
        "case_id": case.case_id,
        "category": case.category,
        "track": case.track,
        "surface": case.surface,
        "description": case.description,
        "expected_behavior": case.expected_behavior,
        "scorer_type": case.scorer_type,
        "turn_count": len(tuple(verdict.get("prompt_sequence", ()))),
        "prompt_sequence": tuple(verdict.get("prompt_sequence", ())),
        "weight": weight,
        "weighted_score": score * weight,
        "score": score,
        "pass_threshold": float(case.pass_threshold),
        "passed": bool(score >= float(case.pass_threshold)),
        "notes": str(verdict.get("notes", "")),
        "observed": verdict.get("observed", {}),
        "final_reply": verdict.get("final_reply", {}),
    }


def run(
    *,
    preset: str | None = None,
    case_ids: tuple[str, ...] | None = None,
    chat_dim: int = 2048,
    chat_seed: int = 0,
    episodic_dim: int = 2048,
    episodic_seeds: tuple[int, ...] = (42, 123, 7),
    episodic_sessions: int = 3,
    episodic_turns: int = 10,
    episodic_facts_per_turn: int = 3,
    temporal_dim: int = 2048,
    temporal_seeds: tuple[int, ...] = (42, 123),
    preload_jsonl: Path | None = None,
    preload_limit: int = 0,
) -> list[MetricRow]:
    selected_case_ids = _selected_case_ids(preset, case_ids)
    config = BenchmarkConfig(
        chat_dim=chat_dim,
        chat_seed=chat_seed,
        episodic_dim=episodic_dim,
        episodic_seeds=episodic_seeds,
        episodic_sessions=episodic_sessions,
        episodic_turns=episodic_turns,
        episodic_facts_per_turn=episodic_facts_per_turn,
        temporal_dim=temporal_dim,
        temporal_seeds=temporal_seeds,
        preload_jsonl=preload_jsonl,
        preload_limit=preload_limit,
    )
    rows: list[MetricRow] = []
    for case_id in selected_case_ids:
        case = CASE_INDEX[case_id]
        if case.metric_executor is not None:
            verdict = run_metric_case(case, config)
            prompt_sequence: tuple[str, ...] = ()
            final_reply: dict[str, Any] = {}
        else:
            chat_verdict = run_chat_case(case, config)
            verdict = chat_verdict
            prompt_sequence = case.prompts
            observed = dict(chat_verdict.observed)
            final_reply = dict(observed.get("final_reply", {}))
            verdict = {
                "score": chat_verdict.score,
                "notes": chat_verdict.notes,
                "observed": observed,
                "prompt_sequence": prompt_sequence,
                "final_reply": final_reply,
            }
        if case.metric_executor is not None:
            verdict = {
                "score": verdict.score,
                "notes": verdict.notes,
                "observed": verdict.observed,
                "prompt_sequence": prompt_sequence,
                "final_reply": final_reply,
            }
        rows.append(_case_row(case, verdict))
    return sorted(rows, key=_sort_key)


def summarize(rows: list[MetricRow]) -> list[MetricRow]:
    def build_summary(group_name: str, label: str, bucket: list[MetricRow]) -> MetricRow:
        weight_total = sum(float(row["weight"]) for row in bucket) or 1.0
        summary: MetricRow = {
            "summary_type": group_name,
            "label": label,
            "case_count": float(len(bucket)),
            "mean_score": sum(float(row["weighted_score"]) for row in bucket) / weight_total,
            "pass_rate": sum(1.0 for row in bucket if bool(row["passed"])) / len(bucket),
            "summary_key": f"{group_name}:{label}",
        }
        if group_name == "track":
            summary["track"] = label
        elif group_name == "surface":
            summary["surface"] = label
        elif group_name == "category":
            summary["category"] = label
        return summary

    summary_rows: list[MetricRow] = [build_summary("overall", "overall", rows)]
    for group_name, field in (("track", "track"), ("surface", "surface"), ("category", "category")):
        grouped: dict[str, list[MetricRow]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        for label, bucket in grouped.items():
            summary_rows.append(build_summary(group_name, label, bucket))

    return sorted(summary_rows, key=_sort_key)


def compare_summary_rows(
    current_summary_rows: list[MetricRow],
    previous_summary_rows: list[MetricRow] | None,
) -> list[MetricRow]:
    previous_by_key = {
        str(row["summary_key"]): row for row in previous_summary_rows or [] if "summary_key" in row
    }
    compared: list[MetricRow] = []
    for row in current_summary_rows:
        merged = dict(row)
        previous = previous_by_key.get(str(row["summary_key"]))
        if previous is None:
            merged["mean_score_delta"] = None
            merged["pass_rate_delta"] = None
        else:
            merged["mean_score_delta"] = float(row["mean_score"]) - float(previous["mean_score"])
            merged["pass_rate_delta"] = float(row["pass_rate"]) - float(previous["pass_rate"])
        compared.append(merged)
    return compared


def build_results_payload(
    *,
    rows: list[MetricRow],
    summary_rows: list[MetricRow],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": _generated_timestamp(),
        "config": config,
        "rows": rows,
        "summary": summary_rows,
    }


def render_markdown_report(
    rows: list[MetricRow],
    summary_rows: list[MetricRow],
    *,
    config: dict[str, Any],
    previous_summary_rows: list[MetricRow] | None = None,
) -> str:
    compared_summary = compare_summary_rows(summary_rows, previous_summary_rows)
    summary_by_key = {str(row["summary_key"]): row for row in compared_summary}
    overall = summary_by_key["overall:overall"]
    track_rows = [row for row in compared_summary if row["summary_type"] == "track"]
    surface_rows = [row for row in compared_summary if row["summary_type"] == "surface"]
    category_rows = [row for row in compared_summary if row["summary_type"] == "category"]
    lowest_rows = sorted(rows, key=lambda row: (float(row["score"]), str(row["case_id"])))[:5]

    lines = [
        "# Longitudinal Conversation Benchmark",
        "",
        f"Generated on {_generated_timestamp()}.",
        "",
        "## Configuration",
        "",
        f"- `preset={config.get('preset', 'custom')}`",
        f"- `cases={len(rows)}`",
        f"- `chat_dim={config['chat_dim']}`",
        f"- `episodic_dim={config['episodic_dim']}`",
        f"- `episodic_seeds={{{','.join(str(seed) for seed in config['episodic_seeds'])}}}`",
        f"- `temporal_dim={config['temporal_dim']}`",
        f"- `temporal_seeds={{{','.join(str(seed) for seed in config['temporal_seeds'])}}}`",
        f"- `preload_jsonl={config.get('preload_jsonl') or 'none'}`",
        f"- `preload_limit={config.get('preload_limit', 0)}`",
        "",
        "## Overall Score",
        "",
        f"- Mean score: `{float(overall['mean_score']):.3f}`",
        f"- Pass rate: `{float(overall['pass_rate']):.3f}`",
    ]
    if overall.get("mean_score_delta") is not None:
        lines.append(f"- Delta vs previous: `{float(overall['mean_score_delta']):+.3f}`")
    lines.extend(
        [
            "",
            "## Track Rollup",
            "",
            "| track | mean score | pass rate | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in track_rows:
        delta = row["mean_score_delta"]
        delta_text = "n/a" if delta is None else f"{float(delta):+.3f}"
        lines.append(
            f"| {row['track']} | {float(row['mean_score']):.3f} | {float(row['pass_rate']):.3f} | {delta_text} |"
        )

    lines.extend(
        [
            "",
            "## Surface Rollup",
            "",
            "| surface | mean score | pass rate | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in surface_rows:
        delta = row["mean_score_delta"]
        delta_text = "n/a" if delta is None else f"{float(delta):+.3f}"
        lines.append(
            f"| {row['surface']} | {float(row['mean_score']):.3f} | {float(row['pass_rate']):.3f} | {delta_text} |"
        )

    lines.extend(
        [
            "",
            "## Category Scorecard",
            "",
            "| category | mean score | pass rate | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in category_rows:
        delta = row["mean_score_delta"]
        delta_text = "n/a" if delta is None else f"{float(delta):+.3f}"
        lines.append(
            f"| {row['category']} | {float(row['mean_score']):.3f} | {float(row['pass_rate']):.3f} | {delta_text} |"
        )

    lines.extend(
        [
            "",
            "## Lowest-Scoring Cases",
            "",
        ]
    )
    for row in lowest_rows:
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                f"- Category: `{row['category']}`",
                f"- Surface: `{row['surface']}`",
                f"- Score: `{float(row['score']):.3f}`",
                f"- Expected: {row['expected_behavior']}",
                f"- Notes: {row['notes']}",
            ]
        )
        prompts = tuple(row.get("prompt_sequence", ()))
        if prompts:
            lines.append("- Prompts:")
            for prompt in prompts:
                lines.append(f"  - `{prompt}`")
        final_reply = row.get("final_reply", {})
        if isinstance(final_reply, dict) and final_reply:
            lines.append(f"- Final route: `{final_reply.get('route', '')}`")
            lines.append(f"- Final reply: `{final_reply.get('text', '')}`")
        lines.append("")

    return "\n".join(lines)


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "roadmap_serious"])
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--chat-dim", type=int, default=2048)
    parser.add_argument("--chat-seed", type=int, default=0)
    parser.add_argument("--episodic-dim", type=int, default=2048)
    parser.add_argument("--episodic-seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--episodic-sessions", type=int, default=3)
    parser.add_argument("--episodic-turns", type=int, default=10)
    parser.add_argument("--episodic-facts-per-turn", type=int, default=3)
    parser.add_argument("--temporal-dim", type=int, default=2048)
    parser.add_argument("--temporal-seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--preload-jsonl", type=Path)
    parser.add_argument("--preload-limit", type=int, default=0)
    parser.add_argument("--output", choices=["raw", "summary", "both"], default="summary")
    parser.add_argument("--results-file", type=Path)
    parser.add_argument("--report-file", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()

    rows = run(
        preset=args.preset,
        case_ids=tuple(args.case_ids) if args.case_ids else None,
        chat_dim=args.chat_dim,
        chat_seed=args.chat_seed,
        episodic_dim=args.episodic_dim,
        episodic_seeds=tuple(args.episodic_seeds),
        episodic_sessions=args.episodic_sessions,
        episodic_turns=args.episodic_turns,
        episodic_facts_per_turn=args.episodic_facts_per_turn,
        temporal_dim=args.temporal_dim,
        temporal_seeds=tuple(args.temporal_seeds),
        preload_jsonl=args.preload_jsonl,
        preload_limit=args.preload_limit,
    )
    summary_rows = summarize(rows)

    previous_summary_rows: list[MetricRow] | None = None
    compare_path = args.compare_to
    if compare_path is None and args.results_file is not None and args.results_file.exists():
        compare_path = args.results_file
    if compare_path is not None and compare_path.exists():
        previous_summary_rows = list(load_results(compare_path).get("summary", []))

    if args.output in {"raw", "both"}:
        for row in rows:
            print(row)
    if args.output in {"summary", "both"}:
        for row in compare_summary_rows(summary_rows, previous_summary_rows):
            print(row)

    config = {
        "preset": args.preset or "custom",
        "chat_dim": args.chat_dim,
        "chat_seed": args.chat_seed,
        "episodic_dim": args.episodic_dim,
        "episodic_seeds": tuple(args.episodic_seeds),
        "episodic_sessions": args.episodic_sessions,
        "episodic_turns": args.episodic_turns,
        "episodic_facts_per_turn": args.episodic_facts_per_turn,
        "temporal_dim": args.temporal_dim,
        "temporal_seeds": tuple(args.temporal_seeds),
        "preload_jsonl": str(args.preload_jsonl) if args.preload_jsonl is not None else None,
        "preload_limit": args.preload_limit,
        "case_ids": _selected_case_ids(args.preset, tuple(args.case_ids) if args.case_ids else None),
    }
    payload = build_results_payload(rows=rows, summary_rows=summary_rows, config=config)

    if args.results_file is not None:
        save_results(args.results_file, payload)
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            render_markdown_report(
                rows,
                summary_rows,
                config=config,
                previous_summary_rows=previous_summary_rows,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
