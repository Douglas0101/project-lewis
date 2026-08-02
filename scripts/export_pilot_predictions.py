"""Exporta predições de uma partição do split pareado v2 para npz (T10.3).

Lê o checkpoint da run (read-only), reconstrói a partição via manifesto e grava
``<run_dir>/evaluation_v2/predictions/<partition>.npz`` (y_true/y_score) + meta.
Alimenta o avaliador canônico (``--predictions`` / ``--calibration-predictions``)
para o fluxo PROSPECTIVE: T/thresholds ajustados em calibration, aplicados ao teste.

Uso:
    uv run python scripts/export_pilot_predictions.py \
        --run-dir experiments/<run> --partition validation
    # ou todas as partições de avaliação:
    uv run python scripts/export_pilot_predictions.py --run-dir experiments/<run> --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger("lewis.camada04.export_predictions")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "chapman_paired_v2" / "manifest.json"
EVAL_PARTITIONS = ("validation", "calibration", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_partition(
    run_dir: Path,
    manifest_path: Path,
    partition: str,
    batch_size: int = 64,
) -> Path:
    """Prediz ``partition`` com o checkpoint da run e grava o npz. Retorna o path."""
    import numpy as np
    from src.models.chapman_dataset import chapman_paired_datasets
    from src.models.keras_loader import load_keras_model
    from src.models.pretrain_evaluation import predict_validation

    if partition not in EVAL_PARTITIONS:
        raise ValueError(f"partição inválida '{partition}'; opções: {EVAL_PARTITIONS}")

    model_path = run_dir / "backbone_pretrained.keras"
    sha = _sha256(model_path)
    expected = None
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = provenance.get("hashes", {}).get("model_sha256")
    if expected and sha != expected:
        raise RuntimeError(f"checkpoint hash mismatch: {sha} != provenance {expected}")

    datasets = chapman_paired_datasets(manifest_path, batch_size=batch_size)
    ds_map = {"validation": datasets[1], "calibration": datasets[2], "test": datasets[3]}
    steps_map = datasets[4]

    model = load_keras_model(str(model_path), compile=False)
    y_true, y_prob = predict_validation(model, ds_map[partition], steps_map[partition])

    out_dir = run_dir / "evaluation_v2" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{partition}.npz"
    np.savez(out_path, y_true=y_true, y_score=y_prob)
    meta = {
        "partition": partition,
        "split_id": json.loads(manifest_path.read_text(encoding="utf-8"))["split_id"],
        "sha256_model": sha,
        "n_samples": int(len(y_prob)),
        "source": "export_pilot_predictions",
    }
    (out_dir / f"{partition}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    LOGGER.info("%s: %d amostras → %s", partition, len(y_prob), out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partition", choices=EVAL_PARTITIONS, default=None)
    parser.add_argument("--all", action="store_true", help="exporta as 3 partições de avaliação")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.all and args.partition is None:
        LOGGER.error("informe --partition ou --all")
        return 2
    partitions = EVAL_PARTITIONS if args.all else (args.partition,)
    for part in partitions:
        export_partition(args.run_dir, args.manifest, part, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
