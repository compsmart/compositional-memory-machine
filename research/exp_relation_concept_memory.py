from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrr.binding import bind, normalize
from hrr.vectors import VectorStore
from memory import AMM


MAX_SUPPORT = 8


@dataclass(frozen=True)
class RelationFamily:
    canonical: str
    seed_aliases: tuple[str, ...]
    unknown_aliases: tuple[str, ...]
    subject_roles: tuple[str, ...]
    object_roles: tuple[str, ...]
    domains: tuple[str, ...]
    cues: tuple[str, ...]


@dataclass(frozen=True)
class RelationExample:
    canonical: str
    surface: str
    subject: str
    object: str
    subject_role: str
    object_role: str
    domain: str
    cues: tuple[str, ...]


FAMILIES: tuple[RelationFamily, ...] = (
    RelationFamily(
        canonical="worked_with",
        seed_aliases=("collaborated_with", "worked_alongside"),
        unknown_aliases=("teamed_up_with", "partnered_with"),
        subject_roles=("engineer", "scientist", "designer"),
        object_roles=("engineer", "scientist", "designer"),
        domains=("research", "product"),
        cues=("project", "team", "prototype"),
    ),
    RelationFamily(
        canonical="mentored",
        seed_aliases=("guided", "coached"),
        unknown_aliases=("showed_the_ropes_to",),
        subject_roles=("principal", "advisor", "senior_engineer"),
        object_roles=("intern", "student", "junior_engineer"),
        domains=("research", "education"),
        cues=("guidance", "training", "feedback"),
    ),
    RelationFamily(
        canonical="reported_to",
        seed_aliases=("answers_to", "works_under"),
        unknown_aliases=("is_managed_by",),
        subject_roles=("associate", "analyst", "engineer"),
        object_roles=("manager", "director", "lead"),
        domains=("operations", "finance", "product"),
        cues=("approval", "manager", "org_chart"),
    ),
    RelationFamily(
        canonical="located_in",
        seed_aliases=("is_based_in", "headquartered_in"),
        unknown_aliases=("operates_from",),
        subject_roles=("lab", "company", "startup"),
        object_roles=("city", "region", "campus"),
        domains=("business", "research"),
        cues=("office", "site", "hq"),
    ),
    RelationFamily(
        canonical="acquired",
        seed_aliases=("bought", "purchased"),
        unknown_aliases=("took_over",),
        subject_roles=("company", "platform", "enterprise"),
        object_roles=("startup", "vendor", "company"),
        domains=("business", "finance"),
        cues=("deal", "ownership", "merger"),
    ),
)


class RelationConceptMemory:
    def __init__(self, *, dim: int = 2048, seed: int = 0, mode: str = "typed") -> None:
        self.mode = mode
        self.store = VectorStore(dim=dim, seed=seed)
        self.memory = AMM()
        self.role_subject = self.store.get_unitary("__REL_SUBJECT__")
        self.role_object = self.store.get_unitary("__REL_OBJECT__")
        self.role_subject_role = self.store.get_unitary("__REL_SUBJECT_ROLE__")
        self.role_object_role = self.store.get_unitary("__REL_OBJECT_ROLE__")
        self.role_domain = self.store.get_unitary("__REL_DOMAIN__")
        self.role_cue = self.store.get_unitary("__REL_CUE__")

    def fit(self, examples: list[RelationExample]) -> None:
        grouped: dict[str, list[RelationExample]] = {}
        for example in examples:
            grouped.setdefault(example.canonical, []).append(example)
        for canonical, rows in grouped.items():
            centroid = self._centroid(rows)
            self.memory.write(f"relation:{canonical}", centroid, {"canonical": canonical, "count": len(rows)})

    def classify(self, examples: list[RelationExample]) -> dict[str, object]:
        probe = self._centroid(examples)
        nearest = self.memory.nearest(probe, top_k=2)
        if not nearest:
            return {"predicted": None, "score": 0.0, "margin": 0.0}
        best, best_score = nearest[0]
        second_score = nearest[1][1] if len(nearest) > 1 else 0.0
        return {
            "predicted": str(best.payload["canonical"]),
            "score": float(best_score),
            "margin": float(best_score - second_score),
        }

    def _centroid(self, examples: list[RelationExample]) -> np.ndarray:
        vectors = [self._encode_example(example) for example in examples]
        return normalize(np.mean(vectors, axis=0))

    def _encode_example(self, example: RelationExample) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.mode in {"identity", "hybrid"}:
            parts.append(bind(self.role_subject, self.store.get(f"entity:{example.subject}")))
            parts.append(bind(self.role_object, self.store.get(f"entity:{example.object}")))
        if self.mode in {"typed", "hybrid"}:
            parts.append(bind(self.role_subject_role, self.store.get(f"subject_role:{example.subject_role}")))
            parts.append(bind(self.role_object_role, self.store.get(f"object_role:{example.object_role}")))
            parts.append(bind(self.role_domain, self.store.get(f"domain:{example.domain}")))
            for cue in example.cues:
                parts.append(bind(self.role_cue, self.store.get(f"cue:{cue}")))
        return normalize(np.sum(parts, axis=0))


def _entity_name(role: str, index: int) -> str:
    return f"{role}_{index:02d}"


def _build_examples(
    *,
    seed: int,
    train_per_surface: int,
    eval_per_alias: int,
) -> tuple[list[RelationExample], dict[str, list[RelationExample]], dict[str, list[RelationExample]]]:
    rng = np.random.default_rng(seed)
    train_examples: list[RelationExample] = []
    eval_pair_reuse: dict[str, list[RelationExample]] = {}
    eval_disjoint: dict[str, list[RelationExample]] = {}

    for family_idx, family in enumerate(FAMILIES):
        surfaces = (family.canonical, *family.seed_aliases)
        train_specs: list[tuple[str, str, str, str, str]] = []
        for offset in range(train_per_surface):
            subject_role = family.subject_roles[offset % len(family.subject_roles)]
            object_role = family.object_roles[(offset + family_idx) % len(family.object_roles)]
            domain = family.domains[offset % len(family.domains)]
            cue_shift = (offset + family_idx) % len(family.cues)
            cues = (
                family.cues[cue_shift],
                family.cues[(cue_shift + 1) % len(family.cues)],
            )
            subject = _entity_name(subject_role, offset)
            object_ = _entity_name(object_role, offset + 20)
            train_specs.append((subject, object_, subject_role, object_role, domain, *cues))

        for surface_idx, surface in enumerate(surfaces):
            for offset, spec in enumerate(train_specs):
                subject, object_, subject_role, object_role, domain, cue_a, cue_b = spec
                shuffled_cues = (cue_a, cue_b) if (offset + surface_idx) % 2 == 0 else (cue_b, cue_a)
                train_examples.append(
                    RelationExample(
                        canonical=family.canonical,
                        surface=surface,
                        subject=subject,
                        object=object_,
                        subject_role=subject_role,
                        object_role=object_role,
                        domain=domain,
                        cues=shuffled_cues,
                    )
                )

        for alias_idx, alias in enumerate(family.unknown_aliases):
            pair_reuse_rows: list[RelationExample] = []
            disjoint_rows: list[RelationExample] = []
            alias_key = f"{family.canonical}:{alias}"
            for offset in range(eval_per_alias):
                train_spec = train_specs[offset % len(train_specs)]
                subject, object_, subject_role, object_role, domain, cue_a, cue_b = train_spec
                pair_reuse_rows.append(
                    RelationExample(
                        canonical=family.canonical,
                        surface=alias,
                        subject=subject,
                        object=object_,
                        subject_role=subject_role,
                        object_role=object_role,
                        domain=domain,
                        cues=(cue_a, cue_b),
                    )
                )

                subject_role = family.subject_roles[(offset + alias_idx) % len(family.subject_roles)]
                object_role = family.object_roles[(offset + family_idx + alias_idx) % len(family.object_roles)]
                domain = family.domains[(offset + alias_idx) % len(family.domains)]
                cue_shift = (offset + family_idx + alias_idx) % len(family.cues)
                cue_a = family.cues[cue_shift]
                cue_b = family.cues[(cue_shift + 1) % len(family.cues)]
                subject = _entity_name(subject_role, offset + 100 + alias_idx * 10)
                object_ = _entity_name(object_role, offset + 140 + alias_idx * 10)
                if rng.random() < 0.5:
                    cues = (cue_a, cue_b)
                else:
                    cues = (cue_b, cue_a)
                disjoint_rows.append(
                    RelationExample(
                        canonical=family.canonical,
                        surface=alias,
                        subject=subject,
                        object=object_,
                        subject_role=subject_role,
                        object_role=object_role,
                        domain=domain,
                        cues=cues,
                    )
                )
            eval_pair_reuse[alias_key] = pair_reuse_rows
            eval_disjoint[alias_key] = disjoint_rows
    return train_examples, eval_pair_reuse, eval_disjoint


def _pair_overlap_predict(
    train_examples: list[RelationExample],
    support_examples: list[RelationExample],
) -> dict[str, object]:
    overlaps: dict[str, set[tuple[str, str]]] = {}
    for example in train_examples:
        overlaps.setdefault(example.canonical, set()).add((example.subject, example.object))

    scores: dict[str, int] = {}
    for canonical, pairs in overlaps.items():
        scores[canonical] = sum((example.subject, example.object) in pairs for example in support_examples)

    best_canonical = None
    best_score = max(scores.values()) if scores else 0
    tied = [canonical for canonical, score in scores.items() if score == best_score]
    if best_score > 0 and len(tied) == 1:
        best_canonical = tied[0]
    return {
        "predicted": best_canonical,
        "score": float(best_score),
        "margin": float(best_score - sorted(scores.values(), reverse=True)[1]) if len(scores) > 1 else float(best_score),
    }


def run(
    *,
    dim: int = 2048,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    train_per_surface: int = 12,
    eval_per_alias: int = MAX_SUPPORT,
    support_sizes: tuple[int, ...] = (1, 2, 4, 8),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train_examples, eval_pair_reuse, eval_disjoint = _build_examples(
            seed=seed,
            train_per_surface=train_per_surface,
            eval_per_alias=eval_per_alias,
        )
        methods = {
            "pair_overlap": None,
            "identity_memory": RelationConceptMemory(dim=dim, seed=seed, mode="identity"),
            "typed_memory": RelationConceptMemory(dim=dim, seed=seed, mode="typed"),
            "hybrid_memory": RelationConceptMemory(dim=dim, seed=seed, mode="hybrid"),
        }
        for method_name, model in methods.items():
            if model is not None:
                model.fit(train_examples)

        for scenario, alias_examples in (
            ("pair_reuse", eval_pair_reuse),
            ("disjoint_entities", eval_disjoint),
        ):
            for support_size in support_sizes:
                for alias_key, examples in alias_examples.items():
                    support = examples[:support_size]
                    gold = support[0].canonical
                    for method_name, model in methods.items():
                        if method_name == "pair_overlap":
                            result = _pair_overlap_predict(train_examples, support)
                        else:
                            result = model.classify(support)
                        predicted = result["predicted"]
                        rows.append(
                            {
                                "seed": seed,
                                "scenario": scenario,
                                "support_size": support_size,
                                "alias_key": alias_key,
                                "gold": gold,
                                "method": method_name,
                                "predicted": predicted,
                                "correct": float(predicted == gold),
                                "unresolved": float(predicted is None),
                                "score": float(result["score"]),
                                "margin": float(result["margin"]),
                            }
                        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scenario"]), int(row["support_size"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (scenario, support_size, method), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "scenario": scenario,
                "support_size": support_size,
                "method": method,
                "accuracy": mean(float(row["correct"]) for row in group),
                "unresolved_rate": mean(float(row["unresolved"]) for row in group),
                "mean_score": mean(float(row["score"]) for row in group),
                "mean_margin": mean(float(row["margin"]) for row in group),
                "runs": len(group),
            }
        )
    return summary_rows


def render_markdown_report(
    summary_rows: list[dict[str, object]],
    *,
    dim: int,
    seeds: tuple[int, ...],
    train_per_surface: int,
    eval_per_alias: int,
) -> str:
    lines = [
        "# Relation Concept Memory Feasibility",
        "",
        "This report compares exact pair-overlap aliasing with `dax`-style relation",
        "concept memory prototypes over synthetic disjoint-relation data.",
        "",
        "## Setup",
        "",
        f"- `dim={dim}`",
        f"- `seeds={list(seeds)}`",
        f"- `train_per_surface={train_per_surface}`",
        f"- `eval_per_alias={eval_per_alias}`",
        f"- `families={len(FAMILIES)}`",
        "- Scenarios:",
        "  - `pair_reuse`: unknown alias reuses subject/object pairs seen under the canonical relation.",
        "  - `disjoint_entities`: unknown alias uses new entities but the same relation-role patterns.",
        "- Methods:",
        "  - `pair_overlap`: current exact subject/object overlap heuristic.",
        "  - `identity_memory`: HRR relation memory using only entity identities.",
        "  - `typed_memory`: HRR relation memory using subject role, object role, domain, and cue context.",
        "  - `hybrid_memory`: identity plus typed context.",
        "",
        "## Aggregate Results",
        "",
        "| scenario | support | method | accuracy | unresolved | mean_score | mean_margin |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {scenario} | {support_size} | {method} | {accuracy:.3f} | {unresolved_rate:.3f} | {mean_score:.3f} | {mean_margin:.3f} |".format(
                **row
            )
        )

    disjoint_rows = [row for row in summary_rows if row["scenario"] == "disjoint_entities"]
    best_disjoint = max(disjoint_rows, key=lambda row: (row["accuracy"], row["mean_margin"]))
    pair_overlap_disjoint = next(
        row
        for row in summary_rows
        if row["scenario"] == "disjoint_entities"
        and row["support_size"] == best_disjoint["support_size"]
        and row["method"] == "pair_overlap"
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best disjoint-setting result: `{best_disjoint['method']}` at support `{best_disjoint['support_size']}` reached",
            f"  `accuracy={best_disjoint['accuracy']:.3f}` with `unresolved_rate={best_disjoint['unresolved_rate']:.3f}`.",
            f"- At the same support, exact pair-overlap reached `accuracy={pair_overlap_disjoint['accuracy']:.3f}`",
            f"  with `unresolved_rate={pair_overlap_disjoint['unresolved_rate']:.3f}`.",
            "- If typed or hybrid relation memory clearly beats pair overlap on disjoint entities,",
            "  then a similarity-based relation-concept subsystem is worth prototyping further.",
            "- If identity-only memory stays weak on disjoint entities, that is evidence that raw",
            "  triple identities alone are not enough; the implementation would need richer context",
            "  features such as role abstractions, graph neighborhoods, or excerpt cues.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Research feasibility of relation concept memory.")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-per-surface", type=int, default=12)
    parser.add_argument("--eval-per-alias", type=int, default=MAX_SUPPORT)
    parser.add_argument("--output", choices=("summary", "json"), default="summary")
    parser.add_argument("--report-file", type=Path)
    parser.add_argument("--json-file", type=Path)
    args = parser.parse_args()

    rows = run(
        dim=args.dim,
        seeds=tuple(args.seeds),
        train_per_surface=args.train_per_surface,
        eval_per_alias=args.eval_per_alias,
    )
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
        args.report_file.write_text(
            render_markdown_report(
                summary_rows,
                dim=args.dim,
                seeds=tuple(args.seeds),
                train_per_surface=args.train_per_surface,
                eval_per_alias=args.eval_per_alias,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
