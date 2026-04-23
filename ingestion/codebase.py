from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .gemini import ExtractedFact, TextIngestionPipeline


@dataclass(frozen=True)
class CodebaseIngestionResult:
    file_count: int
    fact_count: int
    written_facts: int
    files: list[str]


class PythonCodeIngestor:
    """Parse Python source into graph-friendly structured facts."""

    def __init__(self, pipeline: TextIngestionPipeline) -> None:
        self.pipeline = pipeline

    def extract_facts_from_file(
        self,
        path: Path,
        *,
        domain: str = "codebase",
        base_path: Path | None = None,
    ) -> list[ExtractedFact]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_name = self._module_name(path, base_path=base_path)
        facts: list[ExtractedFact] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    facts.append(self._fact(module_name, "imports", imported, path, source, node))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    facts.append(self._fact(module_name, "imports", node.module, path, source, node))
            elif isinstance(node, ast.ClassDef):
                class_name = f"{module_name}.{node.name}"
                facts.append(self._fact(class_name, "defined_in", module_name, path, source, node))
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_name = f"{class_name}.{child.name}"
                        facts.append(self._fact(class_name, "defines_method", method_name, path, source, child))
                        facts.append(self._fact(method_name, "defined_in", module_name, path, source, child))
                        facts.extend(self._call_facts(method_name, child, path, source))
            elif isinstance(node, ast.FunctionDef):
                function_name = f"{module_name}.{node.name}"
                facts.append(self._fact(function_name, "defined_in", module_name, path, source, node))
                facts.extend(self._call_facts(function_name, node, path, source))

        return facts

    def ingest_path(self, root: str | Path, *, domain: str = "codebase") -> CodebaseIngestionResult:
        base = Path(root)
        files = [path for path in sorted(base.rglob("*.py")) if path.is_file()]
        facts: list[ExtractedFact] = []
        for path in files:
            facts.extend(self.extract_facts_from_file(path, domain=domain, base_path=base))

        written = 0
        for fact in facts:
            written += int(self.pipeline.write_structured_fact(fact, source="codebase", domain=domain))
        return CodebaseIngestionResult(
            file_count=len(files),
            fact_count=len(facts),
            written_facts=written,
            files=[str(path) for path in files],
        )

    def _call_facts(
        self,
        owner: str,
        node: ast.FunctionDef,
        path: Path,
        source: str,
    ) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            callee = self._call_name(child.func)
            if callee:
                facts.append(self._fact(owner, "calls", callee, path, source, child))
        return facts

    def _fact(
        self,
        subject: str,
        relation: str,
        object_: str,
        path: Path,
        source_text: str,
        node: ast.AST,
    ) -> ExtractedFact:
        excerpt = ast.get_source_segment(source_text, node) or ""
        return ExtractedFact(
            subject=subject,
            relation=relation,
            object=object_,
            confidence=1.0,
            kind="explicit",
            source="codebase",
            source_id=str(path),
            excerpt=excerpt[:400],
            char_start=getattr(node, "col_offset", None),
            sentence_index=getattr(node, "lineno", None),
        )

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = PythonCodeIngestor._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    @staticmethod
    def _module_name(path: Path, *, base_path: Path | None = None) -> str:
        module_path = path.with_suffix("")
        if base_path is not None:
            try:
                module_path = module_path.relative_to(base_path)
            except ValueError:
                pass
        return ".".join(module_path.parts).lstrip(".")
