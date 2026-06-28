"""Dataset generator for Chapman-Shaoxing pre-training (SCP-ECG 5 superclasses).

Reads processed 10-second ECG records from ``data/processed/chapman/``,
maps SNOMED-CT diagnosis codes to SCP-ECG superclasses, and yields
non-overlapping 500-sample segments (1 second @ 500 Hz) with a shared
multi-label target.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.data.chapman_labels import SCP_SUPERCLASSES, diagnosis_string_to_multihot

LOGGER = logging.getLogger("lewis.camada04.chapman_dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "dataset_catalog.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "chapman"


def _load_catalog(path: Path = CATALOG_PATH) -> List[Dict[str, object]]:
    """Load JSONL catalog."""
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _find_processed_npy(record_name: str) -> Optional[Path]:
    """Locate processed Chapman signal."""
    candidate = PROCESSED_DIR / f"{record_name}_II.npy"
    return candidate if candidate.exists() else None


def _record_generator(
    catalog_path: Path,
    processed_dir: Path,
    segment_len: int = 500,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    """Yield (segment, multihot_label, record_name) for all Chapman records.

    Each 10-second record (5000 samples) is split into non-overlapping
    ``segment_len`` windows. The record's multi-label SCP-ECG superclass
    vector is attached to every segment.
    """
    records = _load_catalog(catalog_path)
    chapman_records = [r for r in records if r.get("dataset") == "chapman"]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(chapman_records)
    else:
        rng = None  # noqa: F841

    for rec in chapman_records:
        record_name = str(rec["record_name"])
        diagnosis = str(rec.get("diagnosis", ""))
        if not diagnosis:
            LOGGER.debug("Skipping %s: no diagnosis", record_name)
            continue

        y = np.array(diagnosis_string_to_multihot(diagnosis), dtype=np.float32)
        if y.sum() == 0:
            LOGGER.debug("Skipping %s: no mapped superclasses", record_name)
            continue

        npy_path = processed_dir / f"{record_name}_II.npy"
        if not npy_path.exists():
            LOGGER.warning("Processed signal not found: %s", npy_path)
            continue

        try:
            sig = np.load(npy_path).astype(np.float32)
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", npy_path, exc)
            continue

        if sig.ndim != 1:
            LOGGER.warning("Unexpected signal shape for %s: %s", record_name, sig.shape)
            continue

        n_segments = len(sig) // segment_len
        for i in range(n_segments):
            start = i * segment_len
            end = start + segment_len
            seg = sig[start:end]
            if np.any(np.isnan(seg)) or np.any(np.isinf(seg)):
                LOGGER.debug("Skipping segment %d of %s: NaN/Inf", i, record_name)
                continue
            yield seg.reshape(segment_len, 1), y, record_name


def create_chapman_dataset(
    batch_size: int = 64,
    segment_len: int = 500,
    seed: Optional[int] = 42,
    catalog_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` for Chapman-Shaoxing pre-training.

    Parameters
    ----------
    batch_size : int
        Batch size.
    segment_len : int
        Segment length (default 500 for 1000 ms @ 500 Hz).
    seed : int, optional
        Seed for catalog shuffling.
    catalog_path : Path, optional
        Path to ``dataset_catalog.jsonl``.
    processed_dir : Path, optional
        Directory with processed Chapman ``.npy`` files.

    Returns
    -------
    tf.data.Dataset
        Dataset yielding batches of ``(X, y)`` where ``X`` has shape
        ``(batch, segment_len, 1)`` and ``y`` has shape ``(batch, 5)``.
    """
    catalog_path = catalog_path or CATALOG_PATH
    processed_dir = processed_dir or PROCESSED_DIR

    def gen():
        for seg, y, _ in _record_generator(
            catalog_path=catalog_path,
            processed_dir=processed_dir,
            segment_len=segment_len,
            seed=seed,
        ):
            yield seg, y

    output_signature = (
        tf.TensorSpec(shape=(segment_len, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(len(SCP_SUPERCLASSES),), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.shuffle(buffer_size=batch_size * 10, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def chapman_train_val_split(
    val_ratio: float = 0.1,
    batch_size: int = 64,
    segment_len: int = 500,
    seed: int = 42,
    catalog_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create training and validation datasets by splitting records (not segments).

    Splitting by record avoids leaking segments of the same patient into both
    train and validation.
    """
    catalog_path = catalog_path or CATALOG_PATH
    processed_dir = processed_dir or PROCESSED_DIR

    records = _load_catalog(catalog_path)
    chapman_records = [r for r in records if r.get("dataset") == "chapman"]
    rng = random.Random(seed)
    rng.shuffle(chapman_records)

    n_val = max(1, int(len(chapman_records) * val_ratio))
    val_records = {r["record_name"] for r in chapman_records[:n_val]}
    train_records = {r["record_name"] for r in chapman_records[n_val:]}

    LOGGER.info(
        "Chapman split | train_records=%d | val_records=%d",
        len(train_records),
        len(val_records),
    )

    def make_dataset(record_set: set) -> tf.data.Dataset:
        def gen():
            for seg, y, record_name in _record_generator(
                catalog_path=catalog_path,
                processed_dir=processed_dir,
                segment_len=segment_len,
                seed=None,
            ):
                if record_name in record_set:
                    yield seg, y

        output_signature = (
            tf.TensorSpec(shape=(segment_len, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(len(SCP_SUPERCLASSES),), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return make_dataset(train_records), make_dataset(val_records)


def get_dataset_statistics(
    catalog_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    segment_len: int = 500,
) -> Dict[str, object]:
    """Return statistics about the Chapman pre-training dataset.

    Useful for sanity checks before starting a long training run.
    """
    catalog_path = catalog_path or CATALOG_PATH
    processed_dir = processed_dir or PROCESSED_DIR

    class_counts = {cls: 0 for cls in SCP_SUPERCLASSES}
    n_segments = 0

    for _, y, _ in _record_generator(
        catalog_path=catalog_path,
        processed_dir=processed_dir,
        segment_len=segment_len,
        seed=None,
    ):
        n_segments += 1
        for idx, val in enumerate(y):
            if val > 0:
                class_counts[SCP_SUPERCLASSES[idx]] += 1

    return {
        "n_segments": n_segments,
        "class_counts": class_counts,
        "superclasses": SCP_SUPERCLASSES,
    }


def chapman_generator_factory(
    split: str = "train",
    batch_size: int = 64,
    segment_len: int = 500,
    seed: int = 42,
    catalog_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Callable:
    """Factory compatible with ``pretrain_chapman``'s generator-based API.

    Returns a callable that yields (X_batch, y_batch) indefinitely.
    """
    train_ds, val_ds = chapman_train_val_split(
        val_ratio=0.1,
        batch_size=batch_size,
        segment_len=segment_len,
        seed=seed,
        catalog_path=catalog_path,
        processed_dir=processed_dir,
    )
    ds = train_ds if split == "train" else val_ds

    def generator():
        while True:
            for X, y in ds:
                yield X.numpy(), y.numpy()

    return generator
