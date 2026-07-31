"""Run advanced evaluation (FASE 8) on a pretrain run directory.

Rebuilds the deterministic validation split (same seed as the run, read from
provenance.json), predicts with the saved best model, and writes
``evaluation_report.json/.md`` + ``calibration.json`` into the run dir.

The calibration.json carries the T1 contract metadata (run/model identity,
split, seed, val sample count and the checkpoint SHA-256, verified fail-closed
against provenance.json) alongside the detailed calibration blocks.

Usage:
    python scripts/evaluate_pretrain_run.py [run_dir] [--n-bins 15]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_contract(
    run_dir: Path, provenance: dict, val_ratio: float, val_samples: int, sha256_model: str
) -> dict:
    training = provenance["training"]
    model = provenance["model"]
    seed = provenance["seed"]
    return {
        "run_id": provenance["run_id"],
        "model_id": f"{training['architecture']}_{training['loss']}",
        "architecture": f"{model['name']}+{training['loss']}",
        "n_params": model["params"],
        "val_samples": val_samples,
        "split_version": f"chapman-record-disjoint-val{val_ratio}-seed{seed}",
        "split_policy": provenance["dataset"]["split_policy"],
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256_model": sha256_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
        help=" número de bins do ECE/reliability (contrato T1: 15; default legado era 10)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = args.run_dir or newest_run_dir()
    if run_dir is None:
        LOGGER.error("no pretrain run directory found")
        return 2
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    training = provenance["training"]
    seed = provenance["seed"]

    model_path = run_dir / "backbone_pretrained.keras"
    sha256_model = _sha256(model_path)
    expected = provenance.get("hashes", {}).get("model_sha256")
    if expected and sha256_model != expected:
        LOGGER.error(
            "checkpoint hash mismatch: %s != provenance %s (fail-closed)",
            sha256_model,
            expected,
        )
        return 3

    _, val_ds, _, val_steps = build_datasets(
        val_ratio=args.val_ratio,
        batch_size=training["batch_size"],
        segment_len=provenance["model"]["input_shape"][1],
        seed=seed,
        steps_per_epoch=1,
        validation_steps=training["validation_steps"],
    )
    model = load_keras_model(str(model_path), compile=False)
    y_true, y_prob = predict_validation(model, val_ds, val_steps)
    report = evaluate_predictions(y_true, y_prob, n_bins=args.n_bins)
    contract = _build_contract(run_dir, provenance, args.val_ratio, len(y_prob), sha256_model)
    write_evaluation_reports(run_dir, report, contract=contract)
    ts = report["temperature_scaling"]
    LOGGER.info(
        "ECE %.4f → %.4f (T=%.3f, n_bins=%d)",
        report["calibration_before"]["macro"]["ece"],
        ts["calibration_after"]["macro"]["ece"],
        ts["temperature"],
        args.n_bins,
    )
    LOGGER.info(
        "AUC-ROC macro %.4f → %.4f (Δ=%.5f)",
        ts["auc_roc_macro_before"],
        ts["auc_roc_macro_after"],
        abs(ts["auc_roc_macro_after"] - ts["auc_roc_macro_before"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
