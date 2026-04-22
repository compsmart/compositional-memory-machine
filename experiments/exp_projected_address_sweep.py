from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, normalize
from hrr.datasets import fact_key, synthetic_facts
from hrr.encoder import SVOEncoder
from hrr.vectors import VectorStore
from memory.projected import ProjectedAddressIndex


KeyRows = list[tuple[str, np.ndarray, dict[str, str]]]
MetricRow = dict[str, float | str]

ROADMAP_SERIOUS_PRESET = {
    "dim": 2048,
    "addr_dims": (64, 128, 256, 512, 1024, 2048),
    "families": ("one_hot", "hrr_svo", "hrr_ngram", "continuous"),
    "seeds": (0, 1, 2),
    "items": (500, 1000, 2000),
    "probes": 200,
    "noise": (0.5, 1.0),
}
SUMMARY_GROUP_FIELDS = ("family", "dim", "addr_dim", "items", "noise")
SUMMARY_METRICS = (
    "exact_top1",
    "noisy_top1",
    "expected_candidate_rate",
    "empty_query_rate",
    "mean_candidates",
    "p95_candidates",
    "max_candidates",
    "stale_contamination",
)


def _noisy(vector: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    if noise <= 0.0:
        return vector
    return normalize(vector + noise * normalize(rng.normal(0.0, 1.0, len(vector))))


def _as_tuple(value: int | float | tuple[int, ...] | tuple[float, ...]) -> tuple[int | float, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def _sort_key(row: MetricRow) -> tuple[Any, ...]:
    return (
        str(row.get("family", "")),
        float(row.get("dim", 0.0)),
        float(row.get("items", 0.0)),
        float(row.get("noise", 0.0)),
        float(row.get("addr_dim", 0.0)),
        float(row.get("seed", -1.0)),
    )


def _format_metric(summary_row: MetricRow, metric: str) -> str:
    mean = float(summary_row[f"{metric}_mean"])
    ci95 = float(summary_row[f"{metric}_ci95"])
    return f"{mean:.3f} +/- {ci95:.3f}"


def _one_hot_rows(dim: int, n_items: int) -> KeyRows:
    if n_items > dim:
        raise ValueError("one_hot key family requires n_items <= dim")
    rows: KeyRows = []
    for idx in range(n_items):
        vector = np.zeros(dim)
        vector[idx] = 1.0
        key = f"one_hot:{idx}"
        rows.append((key, vector, {"family": "one_hot", "item": str(idx)}))
    return rows


def _hrr_svo_rows(dim: int, seed: int, n_items: int) -> KeyRows:
    encoder = SVOEncoder(dim=dim, seed=seed)
    facts = synthetic_facts(domains=5, facts_per_domain=max(1, (n_items + 4) // 5), seed=seed)[:n_items]
    return [
        (
            fact_key(domain, fact),
            encoder.encode_fact(fact),
            {
                "family": "hrr_svo",
                "domain": domain,
                "subject": fact.subject,
                "verb": fact.verb,
                "object": fact.object,
            },
        )
        for domain, fact in facts
    ]


def _hrr_ngram_rows(dim: int, seed: int, n_items: int) -> KeyRows:
    store = VectorStore(dim=dim, seed=seed)
    role_left = store.get_unitary("__SWEEP_LEFT__")
    role_right = store.get_unitary("__SWEEP_RIGHT__")
    rows: KeyRows = []
    for idx in range(n_items):
        left = f"tok{idx % 97}"
        right = f"tok{(idx * 37 + 11) % 193}"
        key = f"hrr_ngram:{left}:{right}:{idx}"
        vector = normalize(
            bind(role_left, store.get(f"tok:{left}"))
            + bind(role_right, store.get(f"tok:{right}"))
        )
        rows.append((key, vector, {"family": "hrr_ngram", "left": left, "right": right}))
    return rows


def _continuous_context_rows(dim: int, seed: int, n_items: int) -> KeyRows:
    store = VectorStore(dim=dim, seed=seed)
    rows: KeyRows = []
    for idx in range(n_items):
        topic = f"topic{idx % 31}"
        actor = f"actor{idx % 79}"
        action = f"action{(idx * 13) % 53}"
        key = f"continuous:{topic}:{actor}:{action}:{idx}"
        vector = normalize(
            0.55 * store.get(f"topic:{topic}")
            + 0.30 * store.get(f"actor:{actor}")
            + 0.15 * store.get(f"action:{action}")
        )
        rows.append((key, vector, {"family": "continuous", "topic": topic, "actor": actor, "action": action}))
    return rows


def _build_rows(family: str, dim: int, seed: int, n_items: int) -> KeyRows:
    if family == "one_hot":
        return _one_hot_rows(dim, n_items)
    if family == "hrr_svo":
        return _hrr_svo_rows(dim, seed, n_items)
    if family == "hrr_ngram":
        return _hrr_ngram_rows(dim, seed, n_items)
    if family == "continuous":
        return _continuous_context_rows(dim, seed, n_items)
    raise ValueError(f"unknown key family: {family}")


def _evaluate(
    rows: KeyRows,
    *,
    dim: int,
    addr_dim: int,
    seed: int,
    probes: int,
    noise: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + addr_dim + 17)
    sample_indices = rng.choice(len(rows), size=min(probes, len(rows)), replace=False)
    index = ProjectedAddressIndex(dim, addr_dim, seed=seed + addr_dim + 1000)
    index.build(rows)

    exact_hits = 0
    noisy_hits = 0
    expected_candidate_hits = 0
    candidate_counts: list[int] = []
    stale_rates: list[float] = []
    empty_queries = 0

    for idx in sample_indices:
        key, vector, _payload = rows[int(idx)]
        exact = index.query(vector, expected_key=key)
        if exact.key == key:
            exact_hits += 1

        noisy = index.query(_noisy(vector, rng, noise), expected_key=key)
        if noisy.key == key:
            noisy_hits += 1
        if noisy.expected_in_candidates:
            expected_candidate_hits += 1
        if noisy.candidate_count == 0:
            empty_queries += 1
            stale_rates.append(0.0)
        else:
            candidate_counts.append(noisy.candidate_count)
            stale_count = noisy.candidate_count - int(bool(noisy.expected_in_candidates))
            stale_rates.append(stale_count / noisy.candidate_count)

    probe_count = len(sample_indices)
    return {
        "exact_top1": exact_hits / probe_count,
        "noisy_top1": noisy_hits / probe_count,
        "expected_candidate_rate": expected_candidate_hits / probe_count,
        "empty_query_rate": empty_queries / probe_count,
        "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "p95_candidates": float(np.quantile(candidate_counts, 0.95)) if candidate_counts else 0.0,
        "max_candidates": float(np.max(candidate_counts)) if candidate_counts else 0.0,
        "stale_contamination": float(np.mean(stale_rates)) if stale_rates else 0.0,
    }


def run(
    *,
    dim: int = 2048,
    addr_dims: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    families: tuple[str, ...] = ("one_hot", "hrr_svo", "hrr_ngram", "continuous"),
    seeds: tuple[int, ...] = (0, 1, 2),
    n_items: int | tuple[int, ...] = 500,
    probes: int = 200,
    noise: float | tuple[float, ...] = 0.5,
) -> list[MetricRow]:
    item_counts = tuple(int(value) for value in _as_tuple(n_items))
    noise_levels = tuple(float(value) for value in _as_tuple(noise))
    rows: list[MetricRow] = []
    for family in families:
        for seed in seeds:
            for item_count in item_counts:
                family_rows = _build_rows(family, dim, seed, item_count)
                for noise_level in noise_levels:
                    for addr_dim in addr_dims:
                        metrics = _evaluate(
                            family_rows,
                            dim=dim,
                            addr_dim=addr_dim,
                            seed=seed,
                            probes=probes,
                            noise=noise_level,
                        )
                        rows.append(
                            {
                                "family": family,
                                "dim": float(dim),
                                "addr_dim": float(addr_dim),
                                "seed": float(seed),
                                "items": float(len(family_rows)),
                                "noise": float(noise_level),
                                **metrics,
                            }
                        )
    return sorted(rows, key=_sort_key)


def summarize(
    rows: list[MetricRow],
    *,
    group_fields: tuple[str, ...] = SUMMARY_GROUP_FIELDS,
) -> list[MetricRow]:
    grouped: dict[tuple[float | str, ...], list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    summary_rows: list[MetricRow] = []
    for key, bucket in grouped.items():
        summary: MetricRow = {
            field: value for field, value in zip(group_fields, key, strict=True)
        }
        summary["runs"] = float(len(bucket))
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in bucket]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_ci95"] = _ci95(values)
            summary[f"{metric}_min"] = float(np.min(values))
        summary_rows.append(summary)
    return sorted(summary_rows, key=_sort_key)


def render_markdown_report(
    summary_rows: list[MetricRow],
    *,
    dim: int,
    addr_dims: tuple[int, ...],
    families: tuple[str, ...],
    seeds: tuple[int, ...],
    item_counts: tuple[int, ...],
    probes: int,
    noise_levels: tuple[float, ...],
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Projected Address Key-Family Sweep",
        "",
        f"Generated on {generated}.",
        "",
        "## Configuration",
        "",
        f"- `dim={dim}`",
        f"- `addr_dim={{{','.join(str(value) for value in addr_dims)}}}`",
        f"- `families={{{','.join(families)}}}`",
        f"- `seeds={{{','.join(str(value) for value in seeds)}}}`",
        f"- `items={{{','.join(str(value) for value in item_counts)}}}`",
        f"- `probes={probes}`",
        f"- `noise={{{','.join(str(value) for value in noise_levels)}}}`",
        "",
        "## Aggregate Results",
        "",
        "Values are shown as mean +/- 95% CI across seeds. `mean_candidates`,",
        "`p95_candidates`, and `max_candidates` are computed over non-empty candidate",
        "sets, while `empty_query_rate` stays explicit as a separate failure mode.",
        "",
    ]

    grouped: dict[tuple[float, float], list[MetricRow]] = defaultdict(list)
    for row in summary_rows:
        grouped[(float(row["items"]), float(row["noise"]))].append(row)

    for (items, noise), bucket in sorted(grouped.items()):
        lines.extend(
            [
                f"### items={int(items)}, noise={noise:.2f}",
                "",
                "| family | addr_dim | exact top-1 | noisy top-1 | expected candidate rate | empty query rate | mean candidates | p95 candidates | max candidates | stale contamination |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(bucket, key=_sort_key):
            lines.append(
                "| "
                f"{row['family']} | "
                f"{int(float(row['addr_dim']))} | "
                f"{_format_metric(row, 'exact_top1')} | "
                f"{_format_metric(row, 'noisy_top1')} | "
                f"{_format_metric(row, 'expected_candidate_rate')} | "
                f"{_format_metric(row, 'empty_query_rate')} | "
                f"{_format_metric(row, 'mean_candidates')} | "
                f"{_format_metric(row, 'p95_candidates')} | "
                f"{_format_metric(row, 'max_candidates')} | "
                f"{_format_metric(row, 'stale_contamination')} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["roadmap_serious"])
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--addr-dims", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--families", nargs="+", default=["one_hot", "hrr_svo", "hrr_ngram", "continuous"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--items", type=int, nargs="+", default=[500])
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.5])
    parser.add_argument("--output", choices=["raw", "summary", "both"], default="raw")
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()

    config = ROADMAP_SERIOUS_PRESET if args.preset == "roadmap_serious" else {
        "dim": args.dim,
        "addr_dims": tuple(args.addr_dims),
        "families": tuple(args.families),
        "seeds": tuple(args.seeds),
        "items": tuple(args.items),
        "probes": args.probes,
        "noise": tuple(args.noise),
    }
    rows = run(
        dim=int(config["dim"]),
        addr_dims=tuple(int(value) for value in config["addr_dims"]),
        families=tuple(str(value) for value in config["families"]),
        seeds=tuple(int(value) for value in config["seeds"]),
        n_items=tuple(int(value) for value in config["items"]),
        probes=int(config["probes"]),
        noise=tuple(float(value) for value in config["noise"]),
    )
    summary_rows = summarize(rows)

    if args.output in {"raw", "both"}:
        for row in rows:
            print(row)
    if args.output in {"summary", "both"}:
        for row in summary_rows:
            print(row)

    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            render_markdown_report(
                summary_rows,
                dim=int(config["dim"]),
                addr_dims=tuple(int(value) for value in config["addr_dims"]),
                families=tuple(str(value) for value in config["families"]),
                seeds=tuple(int(value) for value in config["seeds"]),
                item_counts=tuple(int(value) for value in config["items"]),
                probes=int(config["probes"]),
                noise_levels=tuple(float(value) for value in config["noise"]),
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

