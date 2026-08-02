"""Testes do split pareado v2 (gerador + registry de células) — herméticos.

Não usam TensorFlow nem o dataset real (884 MB): o particionamento é testado
sobre registros sintéticos; o manifesto real (se presente) é validado apenas
estruturalmente.
"""

from __future__ import annotations

import json

import pytest

from scripts.generate_paired_split import (
    MANIFEST_PATH,
    build_partitions,
)
from scripts.run_pilot_cell import PILOT_CELLS

CLASSES = ("NORM", "CD", "MI", "HYP", "STTC")


def _synthetic_records(n: int = 2000) -> list[dict]:
    """Registros sintéticos com prevalências aproximadas do Chapman."""
    import random as _random

    rng = _random.Random(7)
    prev = [0.75, 0.16, 0.29, 0.22, 0.27]
    records = []
    for i in range(n):
        mh = [1 if rng.random() < p else 0 for p in prev]
        if sum(mh) == 0:
            mh[0] = 1
        records.append({"record_name": f"JS{i:05d}", "_multihot": mh})
    return records


# 1 -------------------------------------------------------------------------
def test_generation_deterministic():
    records = _synthetic_records()
    # tolerância compatível com a amostra (partições 5% de n=2000 → σ ≈ 3,7%)
    parts_a, attempt_a, _ = build_partitions(records, max_deviation=0.05)
    parts_b, attempt_b, _ = build_partitions(records, max_deviation=0.05)
    assert attempt_a == attempt_b
    assert parts_a == parts_b


# 2 -------------------------------------------------------------------------
def test_partitions_disjoint_and_covering():
    records = _synthetic_records()
    parts, _, _ = build_partitions(records, max_deviation=0.05)
    all_names = [n for names in parts.values() for n in names]
    assert len(all_names) == len(set(all_names)), "partições não são disjuntas"
    assert set(all_names) == {r["record_name"] for r in records}
    ratios = {p: len(names) / len(records) for p, names in parts.items()}
    assert ratios["train"] == pytest.approx(0.80, abs=0.01)
    assert ratios["validation"] == pytest.approx(0.10, abs=0.01)
    assert ratios["calibration"] == pytest.approx(0.05, abs=0.01)
    assert ratios["test"] == pytest.approx(0.05, abs=0.01)


# 3 -------------------------------------------------------------------------
def test_stratification_deviation_within_tolerance():
    records = _synthetic_records(n=4000)
    parts, _, deviations = build_partitions(records, max_deviation=0.03)
    for part, dev in deviations.items():
        assert dev < 0.03, f"{part}: desvio {dev:.4f} >= tolerância"


def test_stratification_fail_closed_when_impossible():
    records = _synthetic_records()
    with pytest.raises(RuntimeError, match="nenhuma das .* tentativas"):
        build_partitions(records, max_deviation=1e-9, max_attempts=5)


# 4 -------------------------------------------------------------------------
def test_write_once_refuses_overwrite(tmp_path, monkeypatch):
    import scripts.generate_paired_split as gen

    monkeypatch.setattr(gen, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(gen, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(gen, "SHA_PATH", tmp_path / "manifest.sha256")
    monkeypatch.setattr(gen, "load_labeled_records", lambda *a, **k: _synthetic_records(2000))
    monkeypatch.setattr(gen, "_sha256_file", lambda p: "0" * 64)
    # tolerância compatível com a amostra sintética (5% de 2000 = 100 → σ ≈ 3,7%)
    monkeypatch.setattr(
        gen,
        "build_partitions",
        lambda records, **kw: build_partitions(records, max_deviation=0.05),
    )

    assert gen.main() == 0
    assert (tmp_path / "manifest.json").exists()
    # segunda execução sem FORCE → recusa (write-once)
    monkeypatch.delenv("FORCE", raising=False)
    assert gen.main() == 2
    # com FORCE=1 → regenera
    monkeypatch.setenv("FORCE", "1")
    assert gen.main() == 0


# 5 -------------------------------------------------------------------------
def test_cell_registry_matches_ablation_matrix():
    expected = {
        "c0": ("a0", "bce"),
        "c1": ("a1", "bce"),
        "c2": ("a1", "focal"),
        "c3": ("a0", "focal"),
    }
    assert set(PILOT_CELLS) == set(expected)
    for cell, (arch, loss) in expected.items():
        assert PILOT_CELLS[cell]["architecture"] == arch
        assert PILOT_CELLS[cell]["loss"] == loss


# 6 -------------------------------------------------------------------------
@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="manifesto real não gerado neste ambiente")
def test_real_manifest_structure():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["split_id"] == "chapman-record-disjoint-paired-v2"
    assert manifest["seed"] == 13
    parts = manifest["partitions"]
    all_names = [n for names in parts.values() for n in names]
    assert len(all_names) == len(set(all_names))
    assert len(all_names) == manifest["n_records"]
    for part, dev in manifest["prevalence_deviation_max"].items():
        assert dev < 0.01, f"{part}: desvio {dev}"
    for part, support in manifest["support_per_class"].items():
        assert set(support.keys()) == set(CLASSES)
