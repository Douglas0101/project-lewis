"""Gera o manifesto do split pareado v2 (chapman-record-disjoint-paired-v2).

Spec normativa: ``configs/ml_protocol/v2/split_paired_v2.yaml`` (matriz v2 §9).
Partições 80/10/5/5 (train/validation/calibration/test), record-disjoint,
estratificação aproximada (desvio de prevalência < 1 p.p. por superclasse por
partição), seed 13, write-once com hash detached.

Uso:
    uv run python scripts/generate_paired_split.py
    FORCE=1 uv run python scripts/generate_paired_split.py   # regenera (governança)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.chapman_labels import SCP_SUPERCLASSES, diagnosis_string_to_multihot  # noqa: E402

LOGGER = logging.getLogger("lewis.camada04.paired_split")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "dataset_catalog.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "chapman"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "chapman_paired_v2"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SHA_PATH = OUTPUT_DIR / "manifest.sha256"

SPLIT_ID = "chapman-record-disjoint-paired-v2"
SEED = 13
RATIOS = {"train": 0.80, "validation": 0.10, "calibration": 0.05, "test": 0.05}
MAX_DEVIATION = 0.01
MAX_ATTEMPTS = 200


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labeled_records(
    catalog_path: Path = CATALOG_PATH, processed_dir: Path = PROCESSED_DIR
) -> list[dict]:
    """Registros Chapman com diagnóstico mapeado e sinal processado presente."""
    records: list[dict] = []
    with catalog_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("dataset") != "chapman":
                continue
            diagnosis = str(rec.get("diagnosis", ""))
            if not diagnosis or sum(diagnosis_string_to_multihot(diagnosis)) == 0:
                continue
            name = str(rec["record_name"])
            if not (processed_dir / f"{name}_II.npy").exists():
                continue
            rec["_multihot"] = diagnosis_string_to_multihot(diagnosis)
            records.append(rec)
    return records


def _prevalence(records: list[dict]) -> list[float]:
    n = max(len(records), 1)
    return [
        sum(1 for r in records if r["_multihot"][i] > 0) / n
        for i in range(len(SCP_SUPERCLASSES))
    ]


def build_partitions(
    records: list[dict],
    seed: int = SEED,
    ratios: dict[str, float] | None = None,
    max_deviation: float = MAX_DEVIATION,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[dict[str, list[str]], int, dict[str, float]]:
    """Particiona deterministicamente até atender desvio < ``max_deviation``.

    Tentativas com seeds derivadas fixas (``seed * 1000 + attempt``) — a saída é
    reproduzível bit a bit. Falha fechada se nenhuma tentativa atender.
    Retorna ``(partitions, attempt, deviations_max)``.
    """
    ratios = ratios or dict(RATIOS)
    global_prev = _prevalence(records)
    names = [str(r["record_name"]) for r in records]
    by_name = {str(r["record_name"]): r for r in records}
    n = len(names)
    cuts: list[tuple[str, int]] = []
    start = 0
    for part, ratio in ratios.items():
        length = int(round(n * ratio))
        cuts.append((part, start + length))
        start += length
    # a última partição absorve o resto por arredondamento
    cuts[-1] = (cuts[-1][0], n)

    for attempt in range(max_attempts):
        rng = random.Random(seed * 1000 + attempt)
        shuffled = list(names)
        rng.shuffle(shuffled)
        partitions: dict[str, list[str]] = {}
        deviations: dict[str, float] = {}
        prev_cut = 0
        for part, cut in cuts:
            partitions[part] = shuffled[prev_cut:cut]
            prev_cut = cut
        ok = True
        for part, part_names in partitions.items():
            part_prev = _prevalence([by_name[x] for x in part_names])
            dev = max(abs(a - b) for a, b in zip(part_prev, global_prev))
            deviations[part] = dev
            if dev > max_deviation:
                ok = False
                break
        if ok:
            return partitions, attempt, deviations
    raise RuntimeError(
        f"nenhuma das {max_attempts} tentativas atendeu desvio < {max_deviation}"
    )


def _support_per_class(records_by_part: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        part: {
            cls: sum(1 for r in recs if r["_multihot"][i] > 0)
            for i, cls in enumerate(SCP_SUPERCLASSES)
        }
        for part, recs in records_by_part.items()
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if MANIFEST_PATH.exists() and os.environ.get("FORCE") != "1":
        LOGGER.error(
            "%s já existe (write-once). FORCE=1 para regenerar (exige governança).",
            MANIFEST_PATH,
        )
        return 2

    records = load_labeled_records()
    LOGGER.info("registros rotulados com sinal processado: %d", len(records))
    partitions, attempt, deviations = build_partitions(records)
    by_name = {str(r["record_name"]): r for r in records}
    records_by_part = {
        part: [by_name[x] for x in part_names] for part, part_names in partitions.items()
    }

    manifest = {
        "split_id": SPLIT_ID,
        "schema_version": "1.0",
        "seed": SEED,
        "attempt": attempt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": _sha256_file(CATALOG_PATH),
        "n_records": len(records),
        "ratios": RATIOS,
        "partitions": partitions,
        "support_per_class": _support_per_class(records_by_part),
        "prevalence_deviation_max": deviations,
    }
    payload = json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_bytes(payload)
    SHA_PATH.write_text(f"{_sha256_bytes(payload)}  manifest.json\n", encoding="utf-8")

    for part, part_names in partitions.items():
        LOGGER.info(
            "%s | n=%d | desvio_máx=%.4f", part, len(part_names), deviations[part]
        )
    LOGGER.info("manifesto escrito em %s (+ %s)", MANIFEST_PATH, SHA_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
