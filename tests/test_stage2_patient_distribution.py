"""Testes de integridade da distribuição de classe F no Stage 2 (E01)."""

# pyright: reportArgumentType=false

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research"
E01_DIR = RESEARCH_DIR / "E01_patient_distribution"


def _load_distribution():
    if not E01_DIR.exists():
        pytest.skip("E01 patient distribution audit ainda nao foi executado")
    return pd.read_csv(E01_DIR / "patient_class_distribution.csv")


def test_no_null_group_ids():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "stage2_multiclass.parquet")
    assert bool(df["record_id"].notna().all()), "record_id nulo encontrado"
    assert bool(df["record_id"].astype(str).str.strip().ne("").all()), "record_id vazio encontrado"


def test_group_sum_matches_global():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "stage2_multiclass.parquet")
    dist = _load_distribution()
    try:
        total_from_groups = int(dist["total_stage2"].sum())
    except Exception as exc:
        raise AssertionError(f"Falha ao somar total_stage2: {exc}") from exc
    assert total_from_groups == len(
        df
    ), f"Soma por grupo ({total_from_groups}) != total global ({len(df)})"


def test_svf_sum_matches_stage2_cardinality():
    dist = _load_distribution()
    try:
        total = int(dist["total_stage2"].sum())
        svf_sum = int(dist["S_count"].sum() + dist["V_count"].sum() + dist["F_count"].sum())
    except Exception as exc:
        raise AssertionError(f"Falha ao somar contagens S/V/F: {exc}") from exc
    assert svf_sum == total, f"S+V+F ({svf_sum}) != total Stage 2 ({total})"


def test_each_beat_has_single_stage2_class():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "stage2_multiclass.parquet")
    assert bool(df["y"].notna().all()), "label Stage 2 nulo encontrado"
    assert set(df["y"].unique()).issubset(
        {0, 1, 2}
    ), f"labels Stage 2 desconhecidos: {df['y'].unique()}"


def test_aami_mapping_traceability():
    trace = pd.read_csv(E01_DIR / "aami_traceability.csv")
    # Linhagem v2.4 (research branch E06.5): dataset congelado pré-DQ-01/02,
    # pinado por SHA-256 no preflight (stage2_multiclass.parquet = 870b386e...).
    # Totais do build v2.x congelado (backup_v2.3/training_manifest.json):
    # S=16934, V=37183, F=1044.
    # Na linhagem v3.0.0+ (relógio corrigido, DQ-01/02), V=37182 (1 beat de borda
    # descartado); se o dataset raiz for substituído por um build v3.x, este
    # teste deve ser revisitado para refletir a nova linhagem.
    expected = {
        ("S", "S", "Anormal", 0): 16934,
        ("V", "V", "Anormal", 1): 37183,
        ("F", "F", "Anormal", 2): 1044,
    }
    for (orig, mapped, stage1, stage2), count in expected.items():
        row = trace[
            (trace["original_annotation"] == orig)
            & (trace["mapped_aami_class"] == mapped)
            & (trace["stage1_target"] == stage1)
            & (trace["stage2_target"] == stage2)
        ]
        assert not row.empty, f"Mapeamento {orig} -> {stage2} nao encontrado"
        assert (
            row.iloc[0]["count"] == count
        ), f"Contagem de {orig} incorreta: {row.iloc[0]['count']} != {count}"


def test_records_208_213_reported_explicitly():
    dist = _load_distribution()
    if "208" in dist["record_id"].astype(str).values:
        row = dist[dist["record_id"].astype(str) == "208"].iloc[0]
        assert row["F_count"] > 0, "registro 208 presente mas sem F"
    if "213" in dist["record_id"].astype(str).values:
        row = dist[dist["record_id"].astype(str) == "213"].iloc[0]
        assert row["F_count"] > 0, "registro 213 presente mas sem F"


def test_f_concentration_report_exists():
    assert (E01_DIR / "f_concentration_report.json").exists()
    assert (E01_DIR / "f_concentration_report.md").exists()
    assert (E01_DIR / "patient_class_distribution.csv").exists()


def test_top_f_concentration_matches_documentation():
    """Verifica que 208 e 213 concentram a maior parte de F, sem codificar numeros fixos."""
    dist = _load_distribution()
    top2 = dist.head(2)["record_id"].astype(str).tolist()
    assert "208" in top2 and "213" in top2, f"Esperado 208 e 213 entre top 2, obtido: {top2}"
    try:
        total_f = int(dist["F_count"].sum())
        top2_f = int(dist.head(2)["F_count"].sum())
    except Exception as exc:
        raise AssertionError(f"Falha ao calcular concentracao de F: {exc}") from exc
    assert (
        top2_f / total_f >= 0.50
    ), f"Top 2 grupos concentram menos de 50% de F: {top2_f / total_f:.2%}"
