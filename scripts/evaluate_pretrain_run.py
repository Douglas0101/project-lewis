"""Run advanced evaluation (FASE 8) on a pretrain run directory.

Rebuilds the deterministic validation split (same seed as the run, read from
provenance.json), predicts with the saved best model, and writes
``evaluation_report.json/.md`` + ``calibration.json`` into the run dir.

Usage:
    python scripts/evaluate_pretrain_run.py [run_dir]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.keras_loader import load_keras_model  # noqa: E402
from src.models.pretrain_chapman import build_datasets  # noqa: E402
from src.models.pretrain_evaluation import (  # noqa: E402
    evaluate_predictions,
    predict_validation,
    write_evaluation_reports,
)
from scripts.validate_pretrain_artifacts import newest_run_dir  # noqa: E402

LOGGER = logging.getLogger("lewis.camada04.evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = args.run_dir or newest_run_dir()
    if run_dir is None:
        LOGGER.error("no pretrain run directory found")
        return 2
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    training = provenance["training"]
    seed = provenance["seed"]

    _, val_ds, _, val_steps = build_datasets(
        val_ratio=0.1,
        batch_size=training["batch_size"],
        segment_len=provenance["model"]["input_shape"][1],
        seed=seed,
        steps_per_epoch=1,
        validation_steps=training["validation_steps"],
    )
    model = load_keras_model(str(run_dir / "backbone_pretrained.keras"), compile=False)
    y_true, y_prob = predict_validation(model, val_ds, val_steps)
    report = evaluate_predictions(y_true, y_prob)
    write_evaluation_reports(run_dir, report)
    LOGGER.info(
        "ECE %.4f → %.4f (T=%.3f)",
        report["calibration_before"]["macro"]["ece"],
        report["temperature_scaling"]["calibration_after"]["macro"]["ece"],
        report["temperature_scaling"]["temperature"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
