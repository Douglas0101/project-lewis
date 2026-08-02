"""Exporta predições de uma partição do split pareado v2 para npz (T10.3).

Lê o checkpoint da run (read-only), reconstrói a partição via manifesto e grava
``<run_dir>/evaluation_v2/predictions/<partition>.npz`` (y_true/y_score) junto
com os IDs por amostra (RF-PRED-004: record_id, segment_id, patient_id) e meta.

Alimenta o avaliador canônico (``--predictions`` / ``--calibration-predictions``)
para o fluxo PROSPECTIVE: T/thresholds ajustados em calibration, aplicados à
partição de desenvolvimento (validation).

**Isolamento do teste (P0-03 / RF-DATA-005):** a partição ``test`` só pode ser
exportada se a run tiver ``model_freeze.json`` (ver ``src/governance``).

Uso:
    uv run python scripts/export_pilot_predictions.py \
        --run-dir experiments/<run> --partition validation
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

from src.governance.freeze_manager import is_test_authorized  # noqa: E402

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


def _iter_partition_ids(
    record_set: set,
    segment_len: int,
    catalog_path: Path,
    processed_dir: Path,
):
    """Yield ``(record_id, segment_id)`` na ordem determinística da partição.

    A iteração replica exatamente o pipeline de dados (mesmo gerador, seed=None,
    sem shuffle), garantindo alinhamento 1:1 com as predições. patient_id ==
    record_id no Chapman (1 registro ≈ 1 paciente, 10 s).
    """
    from src.models.chapman_dataset import _record_generator

    prev = None
    seg_idx = 0
    for _, _, record_name in _record_generator(
        catalog_path=catalog_path,
        processed_dir=processed_dir,
        segment_len=segment_len,
        seed=None,
    ):
        if record_name not in record_set:
            continue
        if record_name != prev:
            seg_idx = 0
            prev = record_name
        else:
            seg_idx += 1
        yield record_name, f"{record_name}#{seg_idx:03d}"


def export_partition(
    run_dir: Path,
    manifest_path: Path,
    partition: str,
    batch_size: int = 64,
) -> Path:
    """Prediz ``partition`` com o checkpoint da run e grava o npz + IDs."""
    if partition not in EVAL_PARTITIONS:
        raise ValueError(f"partição inválida '{partition}'; opções: {EVAL_PARTITIONS}")
    run_dir = Path(run_dir)
    if partition == "test" and not is_test_authorized(run_dir):
        raise RuntimeError(
            f"partição 'test' bloqueada: {run_dir} não tem model_freeze.json "
            "(RF-DATA-005 — o teste só é liberado após o freeze de seleção)"
        )

    import numpy as np
    from src.models.chapman_dataset import (
        CATALOG_PATH,
        PROCESSED_DIR,
        chapman_paired_datasets,
        load_paired_manifest,
    )
    from src.models.keras_loader import load_keras_model
    from src.models.pretrain_evaluation import predict_validation

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

    manifest = load_paired_manifest(manifest_path)
    record_set = set(manifest["partitions"][partition])
    ids = list(_iter_partition_ids(record_set, 500, CATALOG_PATH, PROCESSED_DIR))
    if len(ids) != len(y_prob):
        raise RuntimeError(
            f"desalinhamento IDs×predições: {len(ids)} ids != {len(y_prob)} predições "
            f"(partição {partition})"
        )
    record_ids = np.array([r for r, _ in ids])
    segment_ids = np.array([s for _, s in ids])

    out_dir = run_dir / "evaluation_v2" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{partition}.npz"
    np.savez(
        out_path,
        y_true=y_true,
        y_score=y_prob,
        record_ids=record_ids,
        segment_ids=segment_ids,
        patient_ids=record_ids,  # Chapman: 1 registro ≈ 1 paciente
    )
    meta = {
        "partition": partition,
        "split_id": manifest["split_id"],
        "sha256_model": sha,
        "n_samples": int(len(y_prob)),
        "n_records": len(set(record_ids.tolist())),
        "segments_per_record": round(len(y_prob) / max(len(set(record_ids.tolist())), 1), 2),
        "source": "export_pilot_predictions",
    }
    (out_dir / f"{partition}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    LOGGER.info(
        "%s: %d amostras (%d registros) → %s", partition, len(y_prob), meta["n_records"], out_path
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partition", choices=EVAL_PARTITIONS, default=None)
    parser.add_argument("--all", action="store_true", help="exporta as 3 partições de avaliação")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--runtime-profile",
        choices=["strict", "fast"],
        default="fast",
        help="perfil numérico CPU-only (PRD RF-CPU-003)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.runtime.cpu_policy import apply as apply_cpu_policy

    apply_cpu_policy(args.runtime_profile)

    if not args.all and args.partition is None:
        LOGGER.error("informe --partition ou --all")
        return 2
    partitions = EVAL_PARTITIONS if args.all else (args.partition,)
    for part in partitions:
        try:
            export_partition(args.run_dir, args.manifest, part, batch_size=args.batch_size)
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            return 4 if "bloqueada" in str(exc) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
