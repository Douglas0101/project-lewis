"""Testes de proteção dos artefatos v2.3 contra sobrescrita acidental pela research branch v2.4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E00_baseline_snapshot"


def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        raise AssertionError(f"Falha ao calcular hash de {path}: {exc}") from exc


def _load_snapshot_hashes() -> dict:
    try:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("E00 baseline snapshot ainda nao foi criado")
        with open(SNAPSHOT_DIR / "baseline_artifacts.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar snapshot E00: {exc}") from exc


def test_v2_3_artifacts_unchanged():
    """Artefatos v2.3 publicados nao devem ser alterados pela research branch."""
    snapshot = _load_snapshot_hashes()
    for name, meta in snapshot.items():
        path = PROJECT_ROOT / meta["path"]
        assert path.exists(), f"Artefato v2.3 {name} ({path}) sumiu"
        current = _sha256(path)
        expected = meta["sha256"]
        assert current == expected, (
            f"Artefato v2.3 {name} foi modificado. "
            f"Esperado {expected}, atual {current}. "
            "Use versao v2.4 para novos experimentos."
        )


def test_v2_3_publication_requires_explicit_flag():
    """Publicacao em paths v2.3 so deve ocorrer com flag explicita de migracao.

    Este teste garante que a research branch v2.4 nao publique acidentalmente
    sobre modelos/scalers/thresholds v2.3. A funcao de publicacao deve
    receber um argumento explicito (por exemplo, target_version="v2.4") e
    rejeitar target_version="v2.3" ou None.
    """
    # Verifica que nenhum codigo fonte atualmente chama funcoes de publicacao
    # com target_version v2.3 sem flag. Implementacao inicial: ausencia de
    # strings perigosas em scripts de publicacao.
    pub_scripts = [
        PROJECT_ROOT / "scripts" / "select_best_mlp_fold.py",
    ]
    for script in pub_scripts:
        if not script.exists():
            continue
        text = script.read_text(encoding="utf-8")
        assert (
            "v2.3" not in text or "target_version" in text
        ), f"{script} contem referencia v2.3 sem controle de target_version."


def test_v2_4_research_directory_isolated():
    """Diretorio de pesquisa v2.4 nao deve conter artefatos com nome v2.3."""
    research_dir = PROJECT_ROOT / "experiments" / "stage2_v2.4_research"
    if not research_dir.exists():
        pytest.skip("diretorio de research v2.4 ainda nao existe")
    for path in research_dir.rglob("*"):
        if path.is_file() and "v2.3" in path.name and path.name != "baseline_artifacts.json":
            pytest.fail(f"Arquivo de research v2.4 nao deve ter 'v2.3' no nome: {path}")
