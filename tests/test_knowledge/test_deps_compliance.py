"""Testes de conformidade de dependências (QG-C11-09)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

FORBIDDEN_PACKAGES = ("langchain", "chromadb", "typer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
KNOWLEDGE_SRC = PROJECT_ROOT / "src" / "knowledge"


@pytest.mark.qg_c11
class TestDepsCompliance:
    """QG-C11-09: ausência de LangChain, ChromaDB e typer como dependências
    diretas ou imports nos fontes da Camada C11.
    """

    def test_pyproject_has_no_forbidden_dependencies(self) -> None:
        content = PYPROJECT_PATH.read_text(encoding="utf-8")
        data = tomllib.loads(content)

        deps = data.get("project", {}).get("dependencies", [])
        dev_deps = data.get("dependency-groups", {}).get("dev", [])
        all_deps = deps + dev_deps

        lowered = "\n".join(all_deps).lower()
        for package in FORBIDDEN_PACKAGES:
            assert package not in lowered, f"{package} encontrado em pyproject.toml"

    def test_knowledge_sources_have_no_forbidden_imports(self) -> None:
        forbidden_in_src: list[str] = []
        for path in KNOWLEDGE_SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for package in FORBIDDEN_PACKAGES:
                if f"import {package}" in text or f"from {package}" in text:
                    forbidden_in_src.append(f"{path.name}: {package}")

        assert not forbidden_in_src, f"Imports proibidos encontrados: {forbidden_in_src}"
