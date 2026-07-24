"""Auditoria e diagnóstico do protocolo de split do Stage 2 (E03).

Compara GroupKFold e StratifiedGroupKFold e gera split manifest + diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.split_protocol import SplitConfig, SplitProtocol, SplitterName

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("audit_stage2_split_protocol")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _split_diagnostic_score(folds: list[dict[str, Any]]) -> float:
    """Score diagnostico: penaliza overlap e folds sem classe F."""
    try:
        penalties = 0.0
        for fold in folds:
            if fold["overlap_groups"]:
                penalties += 1e6
            if fold["test_counts"].get(2, 0) == 0:
                penalties += 1000
        # Dispersao das proporcoes de F no teste
        f_props = []
        for fold in folds:
            total = sum(fold["test_counts"].values())
            f_count = fold["test_counts"].get(2, 0)
            f_props.append(f_count / total if total > 0 else 0.0)
        f_props_arr = np.asarray(f_props, dtype=np.float64)
        std_f = float(np.std(f_props_arr)) if len(f_props_arr) else 0.0
        return -penalties - std_f
    except Exception as exc:
        raise ValueError(f"Falha ao calcular diagnostic score: {exc}") from exc


def _run_split_audit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Executa auditoria comparando GroupKFold e StratifiedGroupKFold."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    diagnostics_rows = []
    for splitter_name in [SplitterName.GROUP_K_FOLD, SplitterName.STRATIFIED_GROUP_K_FOLD]:
        config = SplitConfig(
            splitter=splitter_name,
            n_splits=5,
            shuffle=splitter_name == SplitterName.STRATIFIED_GROUP_K_FOLD,
            random_state=42 if splitter_name == SplitterName.STRATIFIED_GROUP_K_FOLD else None,
        )
        protocol = SplitProtocol(config)
        manifest_path = output_dir / f"split_manifest_{splitter_name.value}.json"
        manifest = protocol.export_manifest(X=X, y=y, groups=groups, output_path=manifest_path)

        score = _split_diagnostic_score(manifest["folds"])
        results[splitter_name.value] = {
            "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "split_config_hash": manifest["split_config_hash"],
            "diagnostic_score": score,
            "folds": manifest["folds"],
        }

        for fold in manifest["folds"]:
            total_test = sum(fold["test_counts"].values())
            diagnostics_rows.append(
                {
                    "splitter": splitter_name.value,
                    "fold": fold["fold"],
                    "train_S": fold["train_counts"].get(0, 0),
                    "train_V": fold["train_counts"].get(1, 0),
                    "train_F": fold["train_counts"].get(2, 0),
                    "test_S": fold["test_counts"].get(0, 0),
                    "test_V": fold["test_counts"].get(1, 0),
                    "test_F": fold["test_counts"].get(2, 0),
                    "F_test_percentage": (
                        100.0 * fold["test_counts"].get(2, 0) / total_test
                        if total_test > 0
                        else 0.0
                    ),
                    "contains_208": "208" in [str(g) for g in fold["test_groups"]],
                    "contains_213": "213" in [str(g) for g in fold["test_groups"]],
                    "overlap_groups": len(fold["overlap_groups"]),
                    "train_groups": len(fold["train_groups"]),
                    "test_groups": len(fold["test_groups"]),
                }
            )

    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics.to_csv(output_dir / "split_diagnostics.csv", index=False)

    # Escolha: zero overlap e cobertura de F sao requisitos; score e usado para desempate
    gk_folds = results[SplitterName.GROUP_K_FOLD.value]["folds"]
    sgk_folds = results[SplitterName.STRATIFIED_GROUP_K_FOLD.value]["folds"]
    gk_ok = all(not f["overlap_groups"] for f in gk_folds)
    sgk_ok = all(not f["overlap_groups"] for f in sgk_folds)
    gk_has_f = all(f["test_counts"].get(2, 0) > 0 for f in gk_folds)
    sgk_has_f = all(f["test_counts"].get(2, 0) > 0 for f in sgk_folds)

    if gk_ok and sgk_ok and gk_has_f and sgk_has_f:
        selected = max(
            [SplitterName.GROUP_K_FOLD.value, SplitterName.STRATIFIED_GROUP_K_FOLD.value],
            key=lambda s: results[s]["diagnostic_score"],
        )
        reason = "melhor diagnostic_score com zero overlap e cobertura F"
    elif sgk_ok and sgk_has_f:
        selected = SplitterName.STRATIFIED_GROUP_K_FOLD.value
        reason = "StratifiedGroupKFold garante cobertura F e zero overlap"
    elif gk_ok and gk_has_f:
        selected = SplitterName.GROUP_K_FOLD.value
        reason = "GroupKFold garante cobertura F e zero overlap"
    else:
        selected = None
        reason = "nenhum splitter atende zero overlap e cobertura F simultaneamente"

    report = {
        "selected_splitter": selected,
        "selection_reason": reason,
        "diagnostics_csv": str((output_dir / "split_diagnostics.csv").relative_to(PROJECT_ROOT)),
        "results": results,
    }
    (output_dir / "split_diagnostics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# Auditoria do Protocolo de Split (Stage 2)",
        "",
        f"- Selecionado: **{selected}**" if selected else "- **Nenhum splitter selecionado**",
        f"- Razão: {reason}",
        "",
        "## Diagnóstico por fold",
        "",
        diagnostics.to_markdown(index=False),
    ]
    (output_dir / "split_diagnostics.md").write_text("\n".join(md_lines), encoding="utf-8")

    LOGGER.info("Split audit saved to %s", output_dir)
    LOGGER.info("Selected splitter: %s (%s)", selected, reason)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditoria e diagnóstico do protocolo de split do Stage 2."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E03_split_protocol",
    )
    args = parser.parse_args()

    try:
        npz = np.load(args.input_npz)
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64)
        groups = np.asarray(npz["groups"])
        _run_split_audit(X, y, groups, args.output_dir)
    except Exception as exc:
        LOGGER.error("Falha na auditoria de split: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
