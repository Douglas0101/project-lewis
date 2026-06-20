"""Orchestrator for Camada 1 → Camada 2: raw records → processed signals.

Reads the dataset catalog, selects the canonical lead per dataset, computes
global mean/std for z-score normalization, and runs every record through
``ECGPreprocessor`` producing:

* ``data/processed/{dataset}/{record}_{lead}.npy``
* ``data/lineage/{dataset}/{record}_lineage.json``

Failures are persisted to ``data/.dlq/preprocess_failures.jsonl``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import wfdb

from . import lead_selector, preprocessor
from ._catalog import RAW_DATASETS

LOGGER = logging.getLogger("lewis.camada02.aggregator")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "dataset_catalog.jsonl"
DLQ_PATH = PROJECT_ROOT / "data" / ".dlq" / "preprocess_failures.jsonl"

# Cap the number of records used to estimate global mean/std per dataset.
# Chapman has 45k records; computing exact global stats over all of them is
# overkill and slow. 1000 records per dataset gives a stable estimate.
_GLOBAL_STATS_SAMPLE_SIZE = 1000


def _load_catalog(catalog_path: Optional[Path] = None) -> list[dict[str, Any]]:
    path = catalog_path or CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path} — run `make catalog` first")
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _catalog_records_by_dataset(
    catalog_path: Optional[Path] = None,
) -> dict[str, list[dict[str, Any]]]:
    records = _load_catalog(catalog_path)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        ds = rec.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(rec)
    return by_dataset


# Catalog uses "mitdb", but lead_selector expects "mitbih".
_LEAD_SELECTOR_NAME = {
    "mitdb": "mitbih",
    "svdb": "svdb",
    "afdb": "afdb",
    "incart": "incart",
    "chapman": "chapman",
    "ptbxl": "ptbxl",
}


def _compute_global_stats(
    dataset: str,
    records: Sequence[dict[str, Any]],
    max_samples: int = _GLOBAL_STATS_SAMPLE_SIZE,
) -> tuple[float, float]:
    """Estimate global mean/std from a stratified sample of records."""
    if not records:
        return 0.0, 1.0

    ls_name = _LEAD_SELECTOR_NAME.get(dataset, dataset)

    # Sample evenly across the dataset to avoid bias from the first folders.
    n = min(len(records), max_samples)
    if n == len(records):
        sampled = records
    else:
        indices = np.linspace(0, len(records) - 1, n, dtype=int)
        sampled = [records[int(i)] for i in indices]

    sums = 0.0
    sums_sq = 0.0
    count = 0
    skipped = 0

    for rec in sampled:
        record_name = rec["record_name"]
        raw_dir = RAW_DATASETS[dataset]
        record_path = _find_record_path(raw_dir, record_name)
        if record_path is None:
            skipped += 1
            continue

        try:
            rec_obj = wfdb.rdrecord(str(record_path), physical=True)
            sig, _ = lead_selector.select_lead(rec_obj, ls_name)
        except Exception as exc:
            LOGGER.warning(
                "Cannot load %s/%s for global stats: %s",
                dataset,
                record_name,
                exc,
            )
            skipped += 1
            continue

        sums += float(sig.sum())
        sums_sq += float((sig**2).sum())
        count += sig.size

    if count == 0:
        LOGGER.warning(
            "No samples collected for global stats of %s (skipped=%d)",
            dataset,
            skipped,
        )
        return 0.0, 1.0

    mean = sums / count
    variance = sums_sq / count - mean * mean
    std = float(np.sqrt(max(variance, 0.0)))
    if std == 0.0:
        std = 1.0

    LOGGER.info(
        "Global stats %s | mean=%.6f std=%.6f | sampled=%d/%d skipped=%d",
        dataset,
        mean,
        std,
        len(sampled) - skipped,
        len(records),
        skipped,
    )
    return mean, std


def _find_record_path(raw_dir: Path, record_name: str) -> Optional[Path]:
    """Return the base path (without extension) for a record inside raw_dir."""
    # Fast path: record at root (MIT-BIH family)
    direct = raw_dir / record_name
    if (direct.with_suffix(".hea")).exists():
        return direct

    # Recursive search (Chapman/PTB-XL)
    for hea in raw_dir.rglob(f"{record_name}.hea"):
        return hea.with_suffix("")
    return None


class ECGAggregator:
    """Orchestrates raw → processed conversion for one or more datasets."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        catalog_path: Optional[Path] = None,
        dlq_path: Optional[Path] = None,
    ):
        self.config_path = config_path
        self.catalog_path = catalog_path or CATALOG_PATH
        self.dlq_path = dlq_path or DLQ_PATH
        self._proc = preprocessor.ECGPreprocessor(
            config_path=config_path,
            dlq_path=self.dlq_path,
        )

    def run(
        self,
        datasets: Optional[Sequence[str]] = None,
        max_records: Optional[int] = None,
    ) -> dict[str, dict[str, Any]]:
        """Process all records listed in the catalog.

        Parameters
        ----------
        datasets : sequence of str, optional
            If provided, only process these datasets. Defaults to all datasets
            present in the catalog.
        max_records : int, optional
            If provided, limit each dataset to this number of records (useful
            for quick smoke tests).

        Returns
        -------
        dict
            ``{dataset: {"processed": int, "failed": int, "skipped": int}}``.
        """
        by_dataset = _catalog_records_by_dataset(self.catalog_path)
        if datasets is not None:
            datasets = [ds.lower().strip() for ds in datasets]
            by_dataset = {ds: recs for ds, recs in by_dataset.items() if ds in datasets}

        stats: dict[str, dict[str, Any]] = {}
        for dataset, records in by_dataset.items():
            if dataset not in RAW_DATASETS:
                LOGGER.warning("Dataset %s not in RAW_DATASETS — skipping", dataset)
                continue

            if max_records is not None:
                records = records[:max_records]

            mean, std = _compute_global_stats(dataset, records)
            self._proc.set_global_stats(mean, std)
            stats[dataset] = self._process_dataset(dataset, records)

        return stats

    def _process_dataset(
        self,
        dataset: str,
        records: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process every record of a single dataset."""
        raw_dir = RAW_DATASETS[dataset]
        ls_name = _LEAD_SELECTOR_NAME.get(dataset, dataset)
        processed = 0
        failed = 0
        skipped = 0

        for idx, rec in enumerate(records, start=1):
            record_name = rec["record_name"]
            fs_native = float(rec["fs"])

            # Prefer catalog source_path when available (handles Chapman JSxxxx vs Sxxxx mismatch)
            record_path: Optional[Path] = None
            source_path = rec.get("source_path")
            if source_path:
                candidate = Path(source_path).with_suffix("")
                if candidate.with_suffix(".hea").exists():
                    record_path = candidate
            if record_path is None:
                record_path = _find_record_path(raw_dir, record_name)

            if record_path is None:
                LOGGER.warning("Record not found: %s/%s", dataset, record_name)
                skipped += 1
                continue

            # Idempotency: skip if already processed with same config/checksum
            if self._proc._check_idempotency(dataset, record_name, record_path) is not None:
                continue

            try:
                rec_obj = wfdb.rdrecord(str(record_path), physical=True)
                sig, lead_name = lead_selector.select_lead(rec_obj, ls_name)

                # Read header explicitly for gain/baseline metadata
                header = wfdb.rdheader(str(record_path))
                channel = lead_selector.get_lead_index(header.sig_name, ls_name)
                gain = header.adc_gain[channel] if header.adc_gain else None
                baseline = header.adc_zero[channel] if header.adc_zero else None

                self._proc.process(
                    sig,
                    record_id=record_name,
                    dataset=dataset,
                    fs_orig=fs_native,
                    raw_path=record_path,
                    lead_name=lead_name,
                    gain=gain,
                    baseline=baseline,
                )
                processed += 1
            except Exception as exc:
                failed += 1
                LOGGER.error(
                    "Failed to process %s/%s: %s",
                    dataset,
                    record_name,
                    exc,
                )

            if idx % 100 == 0 or idx == len(records):
                LOGGER.info(
                    "%s progress: %d/%d | ok=%d failed=%d skipped=%d",
                    dataset,
                    idx,
                    len(records),
                    processed,
                    failed,
                    skipped,
                )

        return {"processed": processed, "failed": failed, "skipped": skipped}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    aggregator = ECGAggregator(config_path=PROJECT_ROOT / "config" / "preprocess_v1.0.yaml")
    stats = aggregator.run()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
