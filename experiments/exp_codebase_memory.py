from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgraph import FactGraph
from hrr import SVOEncoder
from ingestion import PythonCodeIngestor, TextIngestionPipeline
from memory import AMM
from query import QueryEngine


def _write_fixture(root: Path) -> None:
    pkg = root / "fixture_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers.py").write_text(
        "def format_name(name):\n"
        "    return name.strip().title()\n",
        encoding="utf-8",
    )
    (pkg / "service.py").write_text(
        "from fixture_pkg.helpers import format_name\n\n"
        "class Greeter:\n"
        "    def build(self, name):\n"
        "        return format_name(name)\n\n"
        "def greet(name):\n"
        "    helper = Greeter()\n"
        "    return helper.build(name)\n",
        encoding="utf-8",
    )


def run(*, dim: int = 2048, seed: int = 0) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        encoder = SVOEncoder(dim=dim, seed=seed)
        memory = AMM()
        graph = FactGraph()
        pipeline = TextIngestionPipeline(encoder, memory, graph)
        ingestor = PythonCodeIngestor(pipeline)
        query = QueryEngine(encoder=encoder, memory=memory, graph=graph, relation_registry=pipeline.relation_registry)

        result = ingestor.ingest_path(root / "fixture_pkg", domain="codebase")
        import_probe = query.ask_svo("fixture_pkg.service", "imports", "fixture_pkg.helpers")
        call_probe = query.ask_svo("fixture_pkg.service.greet", "calls", "helper.build")
        module_probe = query.ask_svo("fixture_pkg.service.Greeter.build", "defined_in", "fixture_pkg.service")

        return {
            "file_count": float(result.file_count),
            "fact_count": float(result.fact_count),
            "written_facts": float(result.written_facts),
            "imports_em": float(import_probe["found"] is True),
            "calls_em": float(call_probe["found"] is True),
            "defined_in_em": float(module_probe["found"] is True),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(run(dim=args.dim, seed=args.seed))


if __name__ == "__main__":
    main()
