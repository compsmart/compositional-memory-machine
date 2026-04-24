from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph  # noqa: E402
from hrr import SVOEncoder  # noqa: E402
from ingestion import (  # noqa: E402
    DEFAULT_MEDICAL_QID_MAP_PATH,
    HF_STRUCTURED_WIKIPEDIA,
    HF_SUPPORTED_DATASETS,
    SqliteFactLedger,
    StructuredFactRecord,
    dataset_default_domain,
    dataset_row_to_fact_records,
    dataset_source_name,
    load_medical_wikidata_qid_map,
    write_fact_jsonl,
)
from ingestion.gemini import TextIngestionPipeline  # noqa: E402
from memory import AMM, ChunkedKGMemory  # noqa: E402


@dataclass(frozen=True)
class BatchSummary:
    batch_index: int
    input_rows: int
    mapped_facts: int
    written_facts: int
    skipped_facts: int
    next_offset: int
    domain_counts: dict[str, int]
    relation_stats: dict[str, Any]
    memory_records: int
    chunk_count: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_hf_dataset(dataset_name: str, *, split: str, config_name: str | None = None):
    if dataset_name == HF_STRUCTURED_WIKIPEDIA:
        return _load_structured_wikipedia_rows(split=split, config_name=config_name)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise RuntimeError('Install the optional HF dependency first: pip install -e ".[hf]"') from exc
    if config_name:
        return load_dataset(dataset_name, config_name, split=split, streaming=True)
    return load_dataset(dataset_name, split=split, streaming=True)


def _load_structured_wikipedia_rows(*, split: str, config_name: str | None) -> Iterable[dict[str, Any]]:
    if not config_name:
        raise ValueError("structured-wikipedia requires --config-name")
    try:
        from datasets import load_dataset_builder
        from huggingface_hub import HfFileSystem
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise RuntimeError('Install the optional HF dependency first: pip install -e ".[hf]"') from exc

    builder = load_dataset_builder(HF_STRUCTURED_WIKIPEDIA, config_name)
    data_files = getattr(builder.config, "data_files", None) or {}
    split_files = data_files.get(split)
    if not split_files:
        raise ValueError(f"no data files found for split {split!r} in {HF_STRUCTURED_WIKIPEDIA}:{config_name}")

    fs = HfFileSystem()

    def _iter_rows() -> Iterable[dict[str, Any]]:
        for hf_path in split_files:
            normalized_path = str(hf_path).removeprefix("hf://")
            with fs.open(normalized_path, "rb") as remote_handle:
                with zipfile.ZipFile(remote_handle) as archive:
                    member_names = [
                        name
                        for name in archive.namelist()
                        if name.endswith(".jsonl") and not name.startswith("__MACOSX/")
                    ]
                    for member_name in sorted(member_names, key=_structured_wikipedia_member_sort_key):
                        with archive.open(member_name, "r") as member:
                            for raw_line in member:
                                line = raw_line.decode("utf-8").strip()
                                if line:
                                    yield json.loads(line)

    return _iter_rows()


def _structured_wikipedia_member_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"_(\d+)\.jsonl$", name)
    if match is None:
        return (0, name)
    return (int(match.group(1)), name)


def _build_pipeline(*, dim: int, seed: int) -> TextIngestionPipeline:
    encoder = SVOEncoder(dim=dim, seed=seed)
    memory = AMM()
    graph = FactGraph()
    chunk_memory = ChunkedKGMemory(dim=dim, role_count=4)
    return TextIngestionPipeline(encoder, memory, graph, chunk_memory=chunk_memory)


def _batched_rows(
    rows: Iterable[dict[str, Any]],
    *,
    start_offset: int,
    batch_rows: int,
    max_total_rows: int,
):
    iterator = itertools.islice(iter(rows), start_offset, None)
    remaining = max_total_rows if max_total_rows > 0 else None
    while True:
        request_size = batch_rows if remaining is None else min(batch_rows, remaining)
        if request_size <= 0:
            return
        batch = list(itertools.islice(iterator, request_size))
        if not batch:
            return
        yield batch
        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0:
                return


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ledger_records(
    ledger: SqliteFactLedger | None,
    records: list[StructuredFactRecord],
    *,
    row_offset: int,
) -> list[StructuredFactRecord]:
    if ledger is None:
        return records
    return ledger.insert_records(records, row_offset=row_offset)


def _ingest_records_by_domain(
    records: list[StructuredFactRecord],
    *,
    dim: int,
    seed: int,
    source: str,
    default_domain: str,
) -> tuple[TextIngestionPipeline, list[StructuredFactRecord], dict[str, Any]]:
    pipeline = _build_pipeline(dim=dim, seed=seed)
    grouped_records: dict[str, list[StructuredFactRecord]] = {}
    for record in records:
        grouped_records.setdefault(record.domain or default_domain, []).append(record)

    batch_records: list[StructuredFactRecord] = []
    relation_stats_by_domain: dict[str, dict[str, Any]] = {}
    for domain, domain_records in grouped_records.items():
        result = pipeline.ingest_facts(
            [record.fact for record in domain_records],
            source=source,
            domain=domain,
        )
        relation_stats_by_domain[domain] = result.relation_stats
        emitted = [StructuredFactRecord(fact=fact, domain=domain) for fact in result.facts]
        batch_records.extend(emitted)
    return pipeline, batch_records, _merge_relation_stats(relation_stats_by_domain)


def _merge_relation_stats(per_domain: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "raw_relation_labels": 0,
        "normalized_relation_labels": 0,
        "alias_hits": 0,
        "unresolved_relation_labels": 0,
        "learned_alias_labels": 0,
        "typed_fallback_hits": 0,
        "unresolved_relation_examples": [],
        "learned_alias_examples": [],
        "per_domain": per_domain,
    }
    unresolved_examples: set[str] = set()
    learned_alias_examples: set[str] = set()
    for stats in per_domain.values():
        merged["raw_relation_labels"] += int(stats.get("raw_relation_labels", 0))
        merged["normalized_relation_labels"] += int(stats.get("normalized_relation_labels", 0))
        merged["alias_hits"] += int(stats.get("alias_hits", 0))
        merged["unresolved_relation_labels"] += int(stats.get("unresolved_relation_labels", 0))
        merged["learned_alias_labels"] += int(stats.get("learned_alias_labels", 0))
        merged["typed_fallback_hits"] += int(stats.get("typed_fallback_hits", 0))
        unresolved_examples.update(str(item) for item in stats.get("unresolved_relation_examples", []))
        learned_alias_examples.update(str(item) for item in stats.get("learned_alias_examples", []))
    merged["unresolved_relation_examples"] = sorted(unresolved_examples)
    merged["learned_alias_examples"] = sorted(learned_alias_examples)
    return merged


def run(
    *,
    dataset_name: str,
    config_name: str | None = None,
    split: str = "train",
    batch_rows: int = 1000,
    max_total_rows: int = 0,
    resume_offset: int = 0,
    output_dir: Path,
    dim: int = 2048,
    seed: int = 0,
    max_claims_per_entity: int | None = None,
    sqlite_path: Path | None = None,
    medical_only: bool = False,
    qid_allowlist: Path | None = None,
) -> dict[str, Any]:
    dataset = _load_hf_dataset(dataset_name, split=split, config_name=config_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "facts.jsonl"
    progress_path = output_dir / "progress.json"

    ledger = SqliteFactLedger(sqlite_path) if sqlite_path is not None else None
    if ledger is not None:
        saved = ledger.load_progress(dataset=dataset_name)
        if saved is not None and resume_offset == 0:
            resume_offset = int(saved["next_offset"])

    total_input_rows = 0
    total_mapped_facts = 0
    total_written_facts = 0
    batches: list[BatchSummary] = []
    next_offset = resume_offset
    default_domain = dataset_default_domain(dataset_name)
    source = dataset_source_name(dataset_name)
    qid_domain_map = (
        load_medical_wikidata_qid_map(qid_allowlist or DEFAULT_MEDICAL_QID_MAP_PATH)
        if dataset_name == "wikimedia/structured-wikipedia"
        else {}
    )

    try:
        for batch_index, batch in enumerate(
            _batched_rows(
                dataset,
                start_offset=resume_offset,
                batch_rows=batch_rows,
                max_total_rows=max_total_rows,
            ),
            start=1,
        ):
            mapped_records: list[StructuredFactRecord] = []
            for row in batch:
                mapped_records.extend(
                    dataset_row_to_fact_records(
                        dataset_name,
                        row,
                        max_claims_per_entity=max_claims_per_entity,
                        medical_only=medical_only,
                        structured_wikipedia_qid_map=qid_domain_map,
                    )
                )

            pipeline, batch_records, relation_stats = _ingest_records_by_domain(
                mapped_records,
                dim=dim,
                seed=seed,
                source=source,
                default_domain=default_domain,
            )
            accepted_records = _ledger_records(ledger, batch_records, row_offset=next_offset)
            write_fact_jsonl(jsonl_path, accepted_records)
            accepted_domain_counts: dict[str, int] = {}
            for record in accepted_records:
                accepted_domain_counts[record.domain] = accepted_domain_counts.get(record.domain, 0) + 1

            total_input_rows += len(batch)
            total_mapped_facts += len(mapped_records)
            total_written_facts += len(accepted_records)
            next_offset += len(batch)

            summary = BatchSummary(
                batch_index=batch_index,
                input_rows=len(batch),
                mapped_facts=len(mapped_records),
                written_facts=len(accepted_records),
                skipped_facts=max(0, len(batch_records) - len(accepted_records)),
                next_offset=next_offset,
                domain_counts=accepted_domain_counts,
                relation_stats=relation_stats,
                memory_records=len(pipeline.memory),
                chunk_count=len(pipeline.chunk_memory.chunks) if pipeline.chunk_memory is not None else 0,
            )
            batches.append(summary)

            progress_payload = {
                "generated_at": _timestamp(),
                "dataset": dataset_name,
                "split": split,
                "config_name": config_name,
                "batch_rows": batch_rows,
                "max_total_rows": max_total_rows,
                "resume_offset": resume_offset,
                "next_offset": next_offset,
                "medical_only": medical_only,
                "qid_allowlist": str(qid_allowlist) if qid_allowlist is not None else None,
                "dim": dim,
                "seed": seed,
                "output_dir": str(output_dir),
                "jsonl_path": str(jsonl_path),
                "sqlite_path": str(sqlite_path) if sqlite_path is not None else None,
                "input_rows": total_input_rows,
                "mapped_facts": total_mapped_facts,
                "written_facts": total_written_facts,
                "batches": [asdict(item) for item in batches],
            }
            _write_progress(progress_path, progress_payload)
            if ledger is not None:
                ledger.update_progress(
                    dataset=dataset_name,
                    split=split,
                    next_offset=next_offset,
                    written_facts=total_written_facts,
                    updated_at=_timestamp(),
                )
            print(json.dumps(asdict(summary), indent=2), flush=True)
    finally:
        if ledger is not None:
            ledger.close()

    return {
        "generated_at": _timestamp(),
        "dataset": dataset_name,
        "split": split,
        "config_name": config_name,
        "batch_rows": batch_rows,
        "max_total_rows": max_total_rows,
        "resume_offset": resume_offset,
        "next_offset": next_offset,
        "medical_only": medical_only,
        "qid_allowlist": str(qid_allowlist) if qid_allowlist is not None else None,
        "dim": dim,
        "seed": seed,
        "output_dir": str(output_dir),
        "jsonl_path": str(jsonl_path),
        "sqlite_path": str(sqlite_path) if sqlite_path is not None else None,
        "input_rows": total_input_rows,
        "mapped_facts": total_mapped_facts,
        "written_facts": total_written_facts,
        "batches": [asdict(item) for item in batches],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=HF_SUPPORTED_DATASETS)
    parser.add_argument("--config-name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-rows", type=int, default=1000)
    parser.add_argument("--max-total-rows", type=int, default=0)
    parser.add_argument("--resume-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-claims-per-entity", type=int)
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument("--medical-only", action="store_true")
    parser.add_argument("--qid-allowlist", type=Path)
    args = parser.parse_args()

    summary = run(
        dataset_name=args.dataset,
        config_name=args.config_name,
        split=args.split,
        batch_rows=args.batch_rows,
        max_total_rows=args.max_total_rows,
        resume_offset=args.resume_offset,
        output_dir=args.output_dir,
        dim=args.dim,
        seed=args.seed,
        max_claims_per_entity=args.max_claims_per_entity,
        sqlite_path=args.sqlite_path,
        medical_only=args.medical_only,
        qid_allowlist=args.qid_allowlist,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
