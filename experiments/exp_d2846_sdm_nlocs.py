from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import normalize
from memory.sdm import EntropyGatedSDM


MetricRow = dict[str, float]


def _domain_prototypes(dim: int, n_domains: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [normalize(rng.normal(0.0, 1.0, dim)) for _ in range(n_domains)]


def _fact_vector(
    prototype: np.ndarray,
    dim: int,
    *,
    domain_idx: int,
    fact_idx: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + domain_idx * 100_003 + fact_idx * 97)
    content = normalize(rng.normal(0.0, 1.0, dim))
    return normalize(0.8 * prototype + 0.2 * content)


def _probe_indices(total: int, probe_samples: int) -> tuple[int, ...]:
    if total <= 0:
        return ()
    if probe_samples <= 1:
        return (0,)
    anchors = {0, total - 1}
    if probe_samples >= 3:
        anchors.add(total // 2)
    while len(anchors) < min(total, probe_samples):
        position = int(round((len(anchors) / max(probe_samples - 1, 1)) * (total - 1)))
        anchors.add(position)
    return tuple(sorted(anchors))


def _evaluate_condition(
    *,
    dim: int,
    addr_dim: int,
    n_locs: int,
    n_domains: int,
    steps_per_domain: int,
    audit_stride: int,
    probe_samples: int,
    noise: float,
    seed: int,
    gate_beta: float,
    route_top_k: int,
    fail_em_threshold: float,
) -> MetricRow:
    prototypes = _domain_prototypes(dim, n_domains, seed)
    sdm = EntropyGatedSDM(
        dim,
        addr_dim=addr_dim,
        n_locs=n_locs,
        seed=seed,
        gate_beta=gate_beta,
        route_top_k=route_top_k,
    )
    domain_keys: dict[int, list[str]] = defaultdict(list)
    domain_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    key_locations: dict[str, int] = {}
    retrieval_checks = 0
    forgetting_total = 0.0
    empty_reads = 0
    candidate_shard_hits = 0
    routed_shard_hits = 0
    entropy_sum = 0.0
    active_locations_sum = 0.0

    for domain_idx in range(n_domains):
        prototype = prototypes[domain_idx]
        for fact_idx in range(steps_per_domain):
            key = f"d{domain_idx:02d}:f{fact_idx:04d}"
            vector = _fact_vector(prototype, dim, domain_idx=domain_idx, fact_idx=fact_idx, seed=seed)
            route = sdm.write(key, vector, {"domain": domain_idx, "fact_idx": fact_idx})
            entropy_sum += route.entropy
            active_locations_sum += route.active_locations
            domain_keys[domain_idx].append(key)
            domain_vectors[domain_idx].append(vector)
            key_locations[key] = route.location

            if (fact_idx + 1) % audit_stride != 0:
                continue

            for prior_domain in range(domain_idx + 1):
                for probe_idx in _probe_indices(len(domain_keys[prior_domain]), probe_samples):
                    probe_key = domain_keys[prior_domain][probe_idx]
                    probe_vector = domain_vectors[prior_domain][probe_idx]
                    noisy = normalize(
                        probe_vector
                        + noise
                        * normalize(
                            np.random.default_rng(seed + fact_idx + prior_domain * 31 + probe_idx).normal(0.0, 1.0, dim)
                        )
                    )
                    result = sdm.query(noisy)
                    retrieval_checks += 1
                    hit = float(result.key == probe_key)
                    forgetting_total += 1.0 - hit
                    expected_location = key_locations[probe_key]
                    candidate_shard_hits += int(expected_location in result.candidate_locations)
                    routed_shard_hits += int(result.routed_location == expected_location)
                    empty_reads += int(result.key is None)

    mean_forgetting = forgetting_total / max(retrieval_checks, 1)
    retrieval_em = 1.0 - mean_forgetting
    candidate_shard_hit_rate = candidate_shard_hits / max(retrieval_checks, 1)
    routed_shard_hit_rate = routed_shard_hits / max(retrieval_checks, 1)
    return {
        "dim": float(dim),
        "addr_dim": float(addr_dim),
        "n_locs": float(n_locs),
        "seed": float(seed),
        "gate_beta": float(gate_beta),
        "route_top_k": float(route_top_k),
        "n_domains": float(n_domains),
        "steps_per_domain": float(steps_per_domain),
        "probe_samples": float(probe_samples),
        "mean_forgetting": mean_forgetting,
        "retrieval_em": retrieval_em,
        "strict_failure": float(mean_forgetting > 0.0),
        "threshold_failure": float(retrieval_em < fail_em_threshold),
        "mean_entropy": entropy_sum / max(n_domains * steps_per_domain, 1),
        "mean_active_locations": active_locations_sum / max(n_domains * steps_per_domain, 1),
        "empty_query_rate": empty_reads / max(retrieval_checks, 1),
        "candidate_shard_hit_rate": candidate_shard_hit_rate,
        "routed_shard_hit_rate": routed_shard_hit_rate,
        "candidate_read_rescue_rate": max(retrieval_em - routed_shard_hit_rate, 0.0),
        "read_path_failure_rate": max(candidate_shard_hit_rate - retrieval_em, 0.0),
        "peak_mem_mb": sdm.approx_memory_mb(),
    }


def run(
    *,
    dim: int = 2048,
    addr_dim: int = 64,
    n_locs_values: tuple[int, ...] = (16, 32, 64, 128, 256, 512),
    n_domains: int = 10,
    steps_per_domain: int = 200,
    audit_stride: int = 50,
    probe_samples: int = 3,
    noise: float = 0.15,
    seeds: tuple[int, ...] = (0, 1, 2),
    gate_betas: tuple[float, ...] = (-3.0, -2.0, -1.0),
    route_top_ks: tuple[int, ...] = (1, 3),
    fail_em_threshold: float = 0.95,
) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for n_locs in n_locs_values:
        for gate_beta in gate_betas:
            for route_top_k in route_top_ks:
                for seed in seeds:
                    rows.append(
                        _evaluate_condition(
                            dim=dim,
                            addr_dim=addr_dim,
                            n_locs=n_locs,
                            n_domains=n_domains,
                            steps_per_domain=steps_per_domain,
                            audit_stride=audit_stride,
                            probe_samples=probe_samples,
                            noise=noise,
                            seed=seed,
                            gate_beta=gate_beta,
                            route_top_k=route_top_k,
                            fail_em_threshold=fail_em_threshold,
                        )
                    )
    return rows


def summarize(rows: list[MetricRow]) -> list[MetricRow]:
    grouped: dict[tuple[int, float, int], list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["n_locs"]), float(row["gate_beta"]), int(row["route_top_k"]))].append(row)
    summary: list[MetricRow] = []
    for n_locs, gate_beta, route_top_k in sorted(grouped):
        group = grouped[(n_locs, gate_beta, route_top_k)]
        summary.append(
            {
                "n_locs": float(n_locs),
                "gate_beta": gate_beta,
                "route_top_k": float(route_top_k),
                "runs": float(len(group)),
                "mean_forgetting": float(np.mean([row["mean_forgetting"] for row in group])),
                "retrieval_em": float(np.mean([row["retrieval_em"] for row in group])),
                "strict_failures": float(sum(int(row["strict_failure"]) for row in group)),
                "threshold_failures": float(sum(int(row["threshold_failure"]) for row in group)),
                "mean_entropy": float(np.mean([row["mean_entropy"] for row in group])),
                "mean_active_locations": float(np.mean([row["mean_active_locations"] for row in group])),
                "candidate_shard_hit_rate": float(np.mean([row["candidate_shard_hit_rate"] for row in group])),
                "routed_shard_hit_rate": float(np.mean([row["routed_shard_hit_rate"] for row in group])),
                "candidate_read_rescue_rate": float(np.mean([row["candidate_read_rescue_rate"] for row in group])),
                "read_path_failure_rate": float(np.mean([row["read_path_failure_rate"] for row in group])),
                "empty_query_rate": float(np.mean([row["empty_query_rate"] for row in group])),
                "peak_mem_mb": float(np.max([row["peak_mem_mb"] for row in group])),
            }
        )
    return summary


def _report_conclusion(summary_rows: list[MetricRow], *, fail_em_threshold: float) -> list[str]:
    route_top1 = [row for row in summary_rows if int(row["route_top_k"]) == 1]
    route_topk = [row for row in summary_rows if int(row["route_top_k"]) > 1]

    best_top1 = max(route_top1, key=lambda row: (row["retrieval_em"], -row["n_locs"], -row["gate_beta"]))
    best_topk = max(route_topk, key=lambda row: (row["retrieval_em"], -row["n_locs"], -row["gate_beta"]))

    threshold_safe = [
        row
        for row in route_topk
        if float(row["retrieval_em"]) >= fail_em_threshold
    ]
    lowest_supported = min(threshold_safe, key=lambda row: row["n_locs"]) if threshold_safe else None

    lines = [
        "## Conclusion",
        "",
        (
            f"- Best routed-only condition (`route_top_k=1`) currently reaches "
            f"`retrieval_em={best_top1['retrieval_em']:.3f}` at "
            f"`n_locs={best_top1['n_locs']:.0f}`, which keeps the routing floor explicit."
        ),
        (
            f"- Best candidate-read condition (`route_top_k={best_topk['route_top_k']:.0f}`) reaches "
            f"`retrieval_em={best_topk['retrieval_em']:.3f}` with "
            f"`candidate_read_rescue_rate={best_topk['candidate_read_rescue_rate']:.3f}`."
        ),
    ]
    if lowest_supported is None:
        lines.append(
            f"- No candidate-read condition currently clears the configured numeric floor "
            f"`retrieval_em>={fail_em_threshold:.2f}`."
        )
    else:
        lines.append(
            f"- The current lowest candidate-read configuration that clears the numeric floor is "
            f"`n_locs={lowest_supported['n_locs']:.0f}` with "
            f"`route_top_k={lowest_supported['route_top_k']:.0f}` and "
            f"`gate_beta={lowest_supported['gate_beta']:.1f}`."
        )
    lines.append(
        "- The remaining mismatch is now attributable as either router miss "
        "(`routed_shard_hit_rate`) or read-path weakness after the correct shard enters the "
        "candidate set (`read_path_failure_rate`), instead of staying hidden inside a single EM number."
    )
    return lines


def render_markdown_report(
    rows: list[MetricRow],
    summary_rows: list[MetricRow],
    *,
    dim: int,
    addr_dim: int,
    n_domains: int,
    steps_per_domain: int,
    audit_stride: int,
    probe_samples: int,
    noise: float,
    gate_betas: tuple[float, ...],
    route_top_ks: tuple[int, ...],
    fail_em_threshold: float,
) -> str:
    lines = [
        "# D-2846 SDM n_locs reproduction",
        "",
        "## Configuration",
        "",
        f"- `dim={dim}`",
        f"- `addr_dim={addr_dim}`",
        f"- `n_domains={n_domains}`",
        f"- `steps_per_domain={steps_per_domain}`",
        f"- `audit_stride={audit_stride}`",
        f"- `probe_samples={probe_samples}`",
        f"- `noise={noise}`",
        f"- `gate_betas={list(gate_betas)}`",
        f"- `route_top_ks={list(route_top_ks)}`",
        f"- `fail_em_threshold={fail_em_threshold}`",
        "",
        "## Summary",
        "",
        "| n_locs | gate_beta | route_top_k | runs | retrieval_em | mean_forgetting | strict_failures | threshold_failures | candidate_shard_hit_rate | routed_shard_hit_rate | candidate_read_rescue_rate | read_path_failure_rate | peak_mem_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {n_locs:.0f} | {gate_beta:.1f} | {route_top_k:.0f} | {runs:.0f} | {retrieval_em:.3f} | {mean_forgetting:.3f} | {strict_failures:.0f} | {threshold_failures:.0f} | {candidate_shard_hit_rate:.3f} | {routed_shard_hit_rate:.3f} | {candidate_read_rescue_rate:.3f} | {read_path_failure_rate:.3f} | {peak_mem_mb:.1f} |".format(
                **row
            )
        )
    lines.extend(["", *_report_conclusion(summary_rows, fail_em_threshold=fail_em_threshold)])
    lines.extend(
        [
            "",
            "## Raw rows",
            "",
            "```json",
            json.dumps(rows, indent=2),
            "```",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(rows: list[MetricRow], summary_rows: list[MetricRow], *, json_file: str | None, report_file: str | None, config: dict[str, Any]) -> None:
    payload = {"config": config, "rows": rows, "summary": summary_rows}
    if json_file:
        path = Path(json_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if report_file:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_markdown_report(
                rows,
                summary_rows,
                dim=int(config["dim"]),
                addr_dim=int(config["addr_dim"]),
                n_domains=int(config["n_domains"]),
                steps_per_domain=int(config["steps_per_domain"]),
                audit_stride=int(config["audit_stride"]),
                probe_samples=int(config["probe_samples"]),
                noise=float(config["noise"]),
                gate_betas=tuple(float(value) for value in config["gate_betas"]),
                route_top_ks=tuple(int(value) for value in config["route_top_ks"]),
                fail_em_threshold=float(config["fail_em_threshold"]),
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--addr-dim", type=int, default=64)
    parser.add_argument("--n-locs", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--domains", type=int, default=10)
    parser.add_argument("--steps-per-domain", type=int, default=200)
    parser.add_argument("--audit-stride", type=int, default=50)
    parser.add_argument("--probe-samples", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gate-betas", type=float, nargs="+", default=[-3.0, -2.0, -1.0])
    parser.add_argument("--route-top-ks", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--fail-em-threshold", type=float, default=0.95)
    parser.add_argument("--json-file")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    rows = run(
        dim=args.dim,
        addr_dim=args.addr_dim,
        n_locs_values=tuple(args.n_locs),
        n_domains=args.domains,
        steps_per_domain=args.steps_per_domain,
        audit_stride=args.audit_stride,
        probe_samples=args.probe_samples,
        noise=args.noise,
        seeds=tuple(args.seeds),
        gate_betas=tuple(args.gate_betas),
        route_top_ks=tuple(args.route_top_ks),
        fail_em_threshold=args.fail_em_threshold,
    )
    summary_rows = summarize(rows)
    _write_artifacts(
        rows,
        summary_rows,
        json_file=args.json_file,
        report_file=args.report_file,
        config={
            "dim": args.dim,
            "addr_dim": args.addr_dim,
            "n_locs": args.n_locs,
            "n_domains": args.domains,
            "steps_per_domain": args.steps_per_domain,
            "audit_stride": args.audit_stride,
            "probe_samples": args.probe_samples,
            "noise": args.noise,
            "seeds": args.seeds,
            "gate_betas": args.gate_betas,
            "route_top_ks": args.route_top_ks,
            "fail_em_threshold": args.fail_em_threshold,
        },
    )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
