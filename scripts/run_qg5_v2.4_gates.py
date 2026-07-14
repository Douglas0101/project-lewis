"""Executa os Quality Gates QG5 v2.4 redesenhados sobre o baseline v14.

Gera relatorio JSON e MD em experiments/stage2_v2.4_research/E04_qg5_gates/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from src.models.qg5_gates import (
    QG5CalibrationGate,
    QG5PatientwiseGate,
    QG5PublicationGate,
    QG5ReproducibilityGate,
    QG5SmokeBalancedGate,
    QG5StabilityGate,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("run_qg5_v2.4_gates")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa QG5 v2.4 redesenhado sobre baseline v14.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_mlp_features_v2.3_focal_smote_v14",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E04_qg5_gates",
    )
    args = parser.parse_args()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Carrega dados
        npz = np.load(PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz")
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64)
        groups = np.asarray(npz["groups"])

        # Thresholds publicados
        with open(PROJECT_ROOT / "models" / "stage2_thresholds_v2.3.json") as f:
            stage2_thresholds = json.load(f)["thresholds"]

        # Smoke balanced no modelo publicado
        stage2_model = tf.keras.models.load_model(
            str(PROJECT_ROOT / "models" / "stage2_float32_v2.3.keras"), compile=False
        )
        stage2_scaler = joblib.load(PROJECT_ROOT / "models" / "input_scaler_stage2_v2.3.pkl")
        smoke_gate = QG5SmokeBalancedGate()
        smoke_result = smoke_gate.evaluate(stage2_model, stage2_scaler, X, y, stage2_thresholds)

        # Patientwise nos folds v14
        patientwise_gate = QG5PatientwiseGate()
        patientwise_result = patientwise_gate.evaluate(
            args.experiment_dir, X, y, groups, stage2_thresholds
        )

        # Stability, calibration, reproducibility
        stability_gate = QG5StabilityGate()
        stability_result = stability_gate.evaluate(patientwise_result.metrics)

        calibration_gate = QG5CalibrationGate()
        calibration_result = calibration_gate.evaluate(patientwise_result.metrics)

        reproducibility_gate = QG5ReproducibilityGate()
        reproducibility_result = reproducibility_gate.evaluate(
            split_manifest_path=PROJECT_ROOT
            / "experiments"
            / "stage2_v2.4_research"
            / "E03_split_protocol"
            / "split_manifest_StratifiedGroupKFold.json",
            dataset_manifest_path=PROJECT_ROOT
            / "experiments"
            / "stage2_v2.4_research"
            / "E00_baseline_snapshot"
            / "dataset_manifest_v2.4.json",
            feature_manifest_path=PROJECT_ROOT
            / "experiments"
            / "stage2_v2.4_research"
            / "E00_baseline_snapshot"
            / "feature_manifest_v2.4.json",
            seed=42,
        )

        publication_gate = QG5PublicationGate()
        publication_gate.add(smoke_result)
        publication_gate.add(patientwise_result)
        publication_gate.add(stability_result)
        publication_gate.add(calibration_result)
        publication_gate.add(reproducibility_result)

        report = publication_gate.report()
        report["experiment_dir"] = str(args.experiment_dir.relative_to(PROJECT_ROOT))
        report["thresholds"] = stage2_thresholds

        with open(args.output_dir / "qg5_v2.4_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        md_lines = [
            "# Relatório QG5 v2.4 Redesenhado (baseline v14)",
            "",
            f"- Status: **{report['status']}**",
            f"- Pode publicar: **{report['can_publish']}**",
            "",
            "## Gates",
            "",
        ]
        for gate in report["gates"]:
            md_lines.append(f"### {gate['name']}")
            md_lines.append(f"- passou: {gate['passed']}")
            md_lines.append(f"- diagnostico apenas: {gate['diagnostic_only']}")
            md_lines.append(f"- falhas: {gate['failures']}")
            md_lines.append(f"- notas: {gate['notes']}")
            md_lines.append(f"- metricas: {json.dumps(gate['metrics'], indent=2)}")
            md_lines.append("")

        (args.output_dir / "qg5_v2.4_report.md").write_text("\n".join(md_lines), encoding="utf-8")

        LOGGER.info("QG5 v2.4 report saved to %s", args.output_dir)
        LOGGER.info("Status: %s", report["status"])
    except Exception as exc:
        LOGGER.error("Falha ao executar QG5 v2.4: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
