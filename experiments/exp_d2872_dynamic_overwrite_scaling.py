from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hrr.binding import bind, unbind
from hrr.vectors import VectorStore
from memory.amm import AMM

from experiments.hrr_claim_utils import bind_all, nearest_token


def _entity_state_vector(
    store: VectorStore,
    property_roles: dict[str, np.ndarray],
    assignments: dict[str, str],
) -> np.ndarray:
    return bind_all(
        [
            bind(property_roles[property_name], store.get(f"value:{value_name}"))
            for property_name, value_name in assignments.items()
        ]
    )


def _decode_value(
    entity_vector: np.ndarray,
    property_roles: dict[str, np.ndarray],
    property_name: str,
    candidates: dict[str, np.ndarray],
) -> str:
    recovered = unbind(entity_vector, property_roles[property_name])
    predicted, _score = nearest_token(recovered, candidates)
    return predicted


def _render_value(property_name: str, entity_idx: int, update_idx: int) -> str:
    return f"{property_name}_{entity_idx:03d}_{update_idx:03d}"


def _entity_assignments(entity_idx: int, update_idx: int, properties: tuple[str, ...]) -> dict[str, str]:
    return {
        property_name: _render_value(property_name, entity_idx, update_idx)
        for property_name in properties
    }


def run(
    *,
    dims: tuple[int, ...] = (256, 1024, 2048),
    entity_counts: tuple[int, ...] = (20, 50),
    update_counts: tuple[int, ...] = (5, 20),
    properties: tuple[str, ...] = ("location", "action"),
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for dim in dims:
        for entity_count in entity_counts:
            for update_count in update_counts:
                for seed in seeds:
                    store = VectorStore(dim=dim, seed=seed)
                    property_roles = {
                        property_name: store.get_unitary(f"__ROLE_{property_name.upper()}__")
                        for property_name in properties
                    }
                    candidate_vectors = {
                        f"entity:{entity_idx:03d}": {
                            property_name: {
                                _render_value(property_name, entity_idx, update_idx): store.get(
                                    f"value:{_render_value(property_name, entity_idx, update_idx)}"
                                )
                                for update_idx in range(update_count)
                            }
                            for property_name in properties
                        }
                        for entity_idx in range(entity_count)
                    }
                    conditions = {"no_reset": AMM(), "perkey_reset": AMM()}
                    final_targets: dict[str, dict[str, str]] = {}

                    for update_idx in range(update_count):
                        for entity_idx in range(entity_count):
                            entity_key = f"entity:{entity_idx:03d}"
                            assignments = _entity_assignments(entity_idx, update_idx, properties)
                            vector = _entity_state_vector(store, property_roles, assignments)
                            final_targets[entity_key] = assignments

                            conditions["no_reset"].write(entity_key, vector, assignments)
                            conditions["perkey_reset"].delete(entity_key)
                            conditions["perkey_reset"].write(entity_key, vector, assignments)

                    for condition, memory in conditions.items():
                        final_hits = 0
                        final_total = 0
                        property_hits = {property_name: 0 for property_name in properties}

                        for entity_idx in range(entity_count):
                            entity_key = f"entity:{entity_idx:03d}"
                            stored = memory.get(entity_key)
                            if stored is None:
                                continue
                            entity_memory = stored.vector
                            for property_name in properties:
                                final_total += 1
                                predicted = _decode_value(
                                    entity_memory,
                                    property_roles,
                                    property_name,
                                    candidate_vectors[entity_key][property_name],
                                )
                                hit = int(predicted == final_targets[entity_key][property_name])
                                property_hits[property_name] += hit
                                final_hits += hit

                        row = {
                            "dim": float(dim),
                            "seed": float(seed),
                            "entity_count": float(entity_count),
                            "update_count": float(update_count),
                            "property_count": float(len(properties)),
                            "mean_em": final_hits / max(final_total, 1),
                            "condition": condition,
                        }
                        for property_name in properties:
                            row[f"{property_name}_em"] = property_hits[property_name] / max(entity_count, 1)
                        rows.append(row)
    return rows


def summarize(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    grouped: dict[tuple[int, int, int, int, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["dim"]),
            int(row["entity_count"]),
            int(row["update_count"]),
            int(row["property_count"]),
            str(row["condition"]),
        )
        grouped[key].append(row)
    summary: list[dict[str, float]] = []
    for key in sorted(grouped):
        dim, entity_count, update_count, property_count, condition = key
        group = grouped[key]
        base = {
            "dim": float(dim),
            "entity_count": float(entity_count),
            "update_count": float(update_count),
            "property_count": float(property_count),
            "condition": condition,
            "runs": float(len(group)),
        }
        metric_names = [metric for metric in group[0] if metric.endswith("_em")] + ["mean_em"]
        for metric_name in metric_names:
            base[metric_name] = float(np.mean([row[metric_name] for row in group]))
        summary.append(base)
    paired: dict[tuple[int, int, int, int], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in summary:
        grid_key = (
            int(row["dim"]),
            int(row["entity_count"]),
            int(row["update_count"]),
            int(row["property_count"]),
        )
        paired[grid_key][str(row["condition"])] = row
    for row in summary:
        grid_key = (
            int(row["dim"]),
            int(row["entity_count"]),
            int(row["update_count"]),
            int(row["property_count"]),
        )
        no_reset = paired[grid_key].get("no_reset")
        perkey_reset = paired[grid_key].get("perkey_reset")
        if no_reset is None or perkey_reset is None:
            row["mean_em_delta_vs_no_reset"] = 0.0
            continue
        row["mean_em_delta_vs_no_reset"] = float(row["mean_em"] - no_reset["mean_em"])
    return summary


def _paired_conclusion(summary_rows: list[dict[str, float]]) -> list[str]:
    perkey_rows = [row for row in summary_rows if str(row["condition"]) == "perkey_reset"]
    if not perkey_rows:
        return []
    always_better = all(float(row["mean_em_delta_vs_no_reset"]) >= 0.0 for row in perkey_rows)
    best_gain = max(perkey_rows, key=lambda row: row["mean_em_delta_vs_no_reset"])
    worst_gain = min(perkey_rows, key=lambda row: row["mean_em_delta_vs_no_reset"])
    lines = [
        "## Conclusion",
        "",
        (
            f"- `perkey_reset` is {'consistently' if always_better else 'not consistently'} better than `no_reset` "
            f"across the current grid."
        ),
        (
            f"- The strongest observed gain is `+{best_gain['mean_em_delta_vs_no_reset']:.3f}` at "
            f"`dim={best_gain['dim']:.0f}`, `entities={best_gain['entity_count']:.0f}`, "
            f"`updates={best_gain['update_count']:.0f}`."
        ),
        (
            f"- The weakest observed gain is `+{worst_gain['mean_em_delta_vs_no_reset']:.3f}` at "
            f"`dim={worst_gain['dim']:.0f}`, `entities={worst_gain['entity_count']:.0f}`, "
            f"`updates={worst_gain['update_count']:.0f}`."
        ),
        "- The repo now mirrors keyed overwrite mechanics and direct reset-vs-no-reset comparison, "
        "but the absolute EM surface is still materially harsher than the lab-positive result, so this "
        "should still be treated as protocol alignment rather than numeric reproduction.",
    ]
    return lines


def render_markdown_report(summary_rows: list[dict[str, float]], *, dims: tuple[int, ...], entity_counts: tuple[int, ...], update_counts: tuple[int, ...], properties: tuple[str, ...]) -> str:
    lines = [
        "# D-2872 dynamic overwrite scaling",
        "",
        "## Configuration",
        "",
        f"- `dims={list(dims)}`",
        f"- `entity_counts={list(entity_counts)}`",
        f"- `update_counts={list(update_counts)}`",
        f"- `properties={list(properties)}`",
        "",
        "## Summary",
        "",
        "| dim | entities | updates | properties | condition | runs | mean_em | delta_vs_no_reset | location_em | action_em |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dim:.0f} | {entity_count:.0f} | {update_count:.0f} | {property_count:.0f} | {condition} | {runs:.0f} | {mean_em:.3f} | {mean_em_delta_vs_no_reset:+.3f} | {location_em:.3f} | {action_em:.3f} |".format(
                **row
            )
        )
    lines.extend(["", *_paired_conclusion(summary_rows)])
    return "\n".join(lines)


def _write_artifacts(summary_rows: list[dict[str, float]], rows: list[dict[str, float]], *, json_file: str | None, report_file: str | None, config: dict[str, Any]) -> None:
    payload = {"config": config, "rows": rows, "summary": summary_rows}
    if json_file:
        path = Path(json_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if report_file:
        path = Path(report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_markdown_report(
                summary_rows,
                dims=tuple(config["dims"]),
                entity_counts=tuple(config["entity_counts"]),
                update_counts=tuple(config["update_counts"]),
                properties=tuple(config["properties"]),
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--entities", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--updates", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--properties", nargs="+", default=["location", "action"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-file")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    rows = run(
        dims=tuple(args.dims),
        entity_counts=tuple(args.entities),
        update_counts=tuple(args.updates),
        properties=tuple(args.properties),
        seeds=tuple(args.seeds),
    )
    summary_rows = summarize(rows)
    _write_artifacts(
        summary_rows,
        rows,
        json_file=args.json_file,
        report_file=args.report_file,
        config={
            "dims": args.dims,
            "entity_counts": args.entities,
            "update_counts": args.updates,
            "properties": args.properties,
            "seeds": args.seeds,
        },
    )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
