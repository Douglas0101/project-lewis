"""Write-once ordered Stage 2 custody generation for E07R."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.stage2_research.e07r_contracts import (
    Stage2CustodyCompleteV4,
    Stage2CustodyManifestV4,
)
from src.stage2_research.integrity import hash_canonical
from src.training_integrity.contracts import CheckStatus
from src.training_integrity.integrity import (
    exclusive_publication,
    sha256_file,
    waveform_row_sha256,
    write_json_exclusive,
)
from src.training_integrity.preflight import ordered_row_binding_check

GENERATION_ID = "v3.1.0-r5-stage2-pd"
PARENT_GENERATION_ID = "advanced-training-v3.1.0-r4"
CONFIRMATORY_DATASETS = ("incart", "mitdb")
LABEL_TO_STAGE2 = {"S": 0, "V": 1, "F": 2}
REQUIRED_PARENT_COLUMNS = {
    "dataset",
    "record_id",
    "beat_idx",
    "label_aami",
    "sample_id",
    "segment_id",
    "waveform_sha256",
    "source_sampling_rate",
    "target_sampling_rate",
    "annotation_index_native",
    "annotation_index_target",
    "class_original",
    "class_canonical",
    "y",
}


@dataclass(frozen=True)
class Stage2CustodyBundle:
    """In-memory ordered derivation ready for immutable publication."""

    frame: pd.DataFrame
    waveforms: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    waveform_hashes: np.ndarray
    parent_npz_path: Path
    parent_npz_sha256: str
    parent_parquet_path: Path
    parent_parquet_sha256: str
    excluded_dataset_counts: dict[str, int]
    excluded_label_counts: dict[str, int]
    source_commit: str
    source_manifest_hash: str


def _safe_count(value: Any, context: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} is not an integer count") from error
    if count < 0:
        raise ValueError(f"{context} is negative")
    return count


def _count_true(values: np.ndarray, context: str) -> int:
    try:
        return _safe_count(np.count_nonzero(values), context)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"cannot count {context}") from error


def _validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def _sequence_hash(values: np.ndarray) -> str:
    return hash_canonical(np.asarray(values).astype(str).tolist())


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_stage2_custody_bundle(
    parent_npz_path: Path,
    parent_parquet_path: Path,
    *,
    expected_parent_npz_sha256: str,
    expected_parent_parquet_sha256: str,
    source_commit: str,
    source_manifest_hash: str,
) -> Stage2CustodyBundle:
    """Filter the ordered r4 parent into confirmatory S/V/F Stage 2 rows."""
    parent_npz_path = parent_npz_path.resolve()
    parent_parquet_path = parent_parquet_path.resolve()
    parent_npz_sha256 = _validate_hash(
        parent_npz_path,
        expected_parent_npz_sha256,
        "parent NPZ",
    )
    parent_parquet_sha256 = _validate_hash(
        parent_parquet_path,
        expected_parent_parquet_sha256,
        "parent parquet",
    )
    parent_check = ordered_row_binding_check(
        parent_npz_path,
        parent_parquet_path,
        scope="STAGE2_R5_PARENT",
    )
    if parent_check.status is not CheckStatus.PASS:
        raise ValueError(
            f"parent ordered row binding failed: {parent_check.code}: {parent_check.evidence}"
        )

    frame = pd.read_parquet(parent_parquet_path)
    missing_columns = sorted(REQUIRED_PARENT_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"parent parquet lacks custody columns: {missing_columns}")
    with np.load(parent_npz_path, allow_pickle=False) as archive:
        waveforms_parent = np.asarray(archive["X"])
        sample_ids_parent = np.asarray(archive["sample_id"]).astype(str)
        waveform_hashes_parent = np.asarray(archive["waveform_sha256"]).astype(str)
    if waveforms_parent.dtype != np.float32 or waveforms_parent.shape[1:] != (500, 1):
        raise ValueError("parent waveforms must be canonical float32 (500, 1)")

    dataset_values = frame["dataset"].astype(str).to_numpy()
    label_values = frame["label_aami"].astype(str).to_numpy()
    dataset_mask = np.isin(dataset_values, CONFIRMATORY_DATASETS)
    label_mask = np.isin(label_values, tuple(LABEL_TO_STAGE2))
    selected = dataset_mask & label_mask
    if not np.any(selected):
        raise ValueError("Stage 2 custody filter selected zero rows")

    output_frame = frame.loc[selected].copy().reset_index(drop=True)
    output_waveforms = np.ascontiguousarray(waveforms_parent[selected], dtype=np.float32)
    output_labels = np.asarray(
        [LABEL_TO_STAGE2[value] for value in label_values[selected]],
        dtype=np.int64,
    )
    output_sample_ids = np.asarray(sample_ids_parent[selected], dtype=str)
    output_waveform_hashes = np.asarray(waveform_hashes_parent[selected], dtype=str)
    output_frame["y"] = output_labels
    output_frame["stage"] = "stage2_multiclass_patient_disjoint_v4.0"

    if not np.array_equal(
        output_sample_ids,
        output_frame["sample_id"].astype(str).to_numpy(),
    ):
        raise ValueError("filtered sample_id order drifted from parent")
    if not np.array_equal(
        output_waveform_hashes,
        output_frame["waveform_sha256"].astype(str).to_numpy(),
    ):
        raise ValueError("filtered waveform hash order drifted from parent")
    recomputed_hashes = np.asarray(
        [waveform_row_sha256(row) for row in output_waveforms],
        dtype=str,
    )
    if not np.array_equal(output_waveform_hashes, recomputed_hashes):
        raise ValueError("filtered waveform bytes do not match parent digests")
    if len(set(output_sample_ids.tolist())) != len(output_sample_ids):
        raise ValueError("filtered sample_id values are not unique")
    if not np.isfinite(output_waveforms).all():
        raise ValueError("filtered waveforms contain NaN/Inf")

    excluded_dataset_counts = {
        dataset: _count_true(
            (dataset_values == dataset) & ~dataset_mask,
            f"excluded dataset {dataset}",
        )
        for dataset in sorted(set(dataset_values) - set(CONFIRMATORY_DATASETS))
    }
    excluded_label_counts = {
        label: _count_true(
            dataset_mask & (label_values == label) & ~label_mask,
            f"excluded label {label}",
        )
        for label in sorted(set(label_values) - set(LABEL_TO_STAGE2))
    }
    return Stage2CustodyBundle(
        frame=output_frame,
        waveforms=output_waveforms,
        labels=output_labels,
        sample_ids=output_sample_ids,
        waveform_hashes=output_waveform_hashes,
        parent_npz_path=parent_npz_path,
        parent_npz_sha256=parent_npz_sha256,
        parent_parquet_path=parent_parquet_path,
        parent_parquet_sha256=parent_parquet_sha256,
        excluded_dataset_counts=excluded_dataset_counts,
        excluded_label_counts=excluded_label_counts,
        source_commit=source_commit,
        source_manifest_hash=source_manifest_hash,
    )


def _write_arrays(path: Path, bundle: Stage2CustodyBundle) -> None:
    np.savez_compressed(
        path,
        X=bundle.waveforms,
        y=bundle.labels,
        sample_id=np.asarray(bundle.sample_ids, dtype=str),
        waveform_sha256=np.asarray(bundle.waveform_hashes, dtype=str),
    )


def _manifest(
    bundle: Stage2CustodyBundle,
    npz_path: Path,
    parquet_path: Path,
) -> Stage2CustodyManifestV4:
    class_counts = {
        label: _count_true(bundle.labels == index, f"class {label}")
        for label, index in LABEL_TO_STAGE2.items()
    }
    output_sha256 = {
        "stage2_multiclass.npz": sha256_file(npz_path),
        "stage2_multiclass.parquet": sha256_file(parquet_path),
    }
    output_sizes = {
        "stage2_multiclass.npz": npz_path.stat().st_size,
        "stage2_multiclass.parquet": parquet_path.stat().st_size,
    }
    payload = {
        "schema_version": "stage2-custody-v4.0",
        "generation_id": GENERATION_ID,
        "status": "AUTHORIZED_FOR_E07R_INTERNAL_TRAINING",
        "parent_generation_id": PARENT_GENERATION_ID,
        "parent_npz_path": _display_path(bundle.parent_npz_path),
        "parent_npz_sha256": bundle.parent_npz_sha256,
        "parent_parquet_path": _display_path(bundle.parent_parquet_path),
        "parent_parquet_sha256": bundle.parent_parquet_sha256,
        "parent_ordered_binding": "PASS",
        "derivation": "ordered_filter_labels_S_V_F_datasets_incart_mitdb",
        "confirmatory_datasets": CONFIRMATORY_DATASETS,
        "excluded_dataset_counts": bundle.excluded_dataset_counts,
        "excluded_label_counts": bundle.excluded_label_counts,
        "row_count": len(bundle.frame),
        "class_counts": class_counts,
        "record_count": _safe_count(
            bundle.frame.loc[:, ["dataset", "record_id"]].drop_duplicates().shape[0],
            "record count",
        ),
        "signal_shape": bundle.waveforms.shape,
        "signal_dtype": str(bundle.waveforms.dtype),
        "sample_id_sequence_hash": _sequence_hash(bundle.sample_ids),
        "waveform_hash_sequence_hash": _sequence_hash(bundle.waveform_hashes),
        "output_file_sha256": output_sha256,
        "output_size_bytes": output_sizes,
        "output_ordered_binding": "PASS",
        "source_commit": bundle.source_commit,
        "source_manifest_hash": bundle.source_manifest_hash,
        "created_at": "2026-07-26",
    }
    return Stage2CustodyManifestV4.model_validate(
        {**payload, "manifest_hash": hash_canonical(payload)}
    )


def _complete_marker(
    manifest: Stage2CustodyManifestV4,
    manifest_path: Path,
) -> Stage2CustodyCompleteV4:
    artifact_sha256 = {
        **manifest.output_file_sha256,
        "stage2_custody_manifest.json": sha256_file(manifest_path),
    }
    payload = {
        "schema_version": "stage2-custody-complete-v4.0",
        "generation_id": GENERATION_ID,
        "status": "COMPLETE",
        "manifest_hash": manifest.manifest_hash,
        "manifest_file_sha256": sha256_file(manifest_path),
        "artifact_sha256": artifact_sha256,
        "completed_at": "2026-07-26",
    }
    return Stage2CustodyCompleteV4.model_validate(
        {**payload, "marker_hash": hash_canonical(payload)}
    )


def publish_stage2_custody_generation(
    target: Path,
    bundle: Stage2CustodyBundle,
) -> tuple[Stage2CustodyManifestV4, Stage2CustodyCompleteV4]:
    """Publish an authorized Stage 2 generation once via atomic directory rename."""
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.publish.lock"
    staging = target.parent / f".{target.name}.{os.getpid()}.staging"
    try:
        with exclusive_publication(lock_path, [target]):
            staging.mkdir(mode=0o700)
            npz_path = staging / "stage2_multiclass.npz"
            parquet_path = staging / "stage2_multiclass.parquet"
            manifest_path = staging / "stage2_custody_manifest.json"
            complete_path = staging / "STAGE2_CUSTODY_COMPLETE.json"
            _write_arrays(npz_path, bundle)
            bundle.frame.to_parquet(parquet_path, index=False)
            output_check = ordered_row_binding_check(
                npz_path,
                parquet_path,
                scope="STAGE2_R5_OUTPUT",
            )
            if output_check.status is not CheckStatus.PASS:
                raise ValueError(
                    "output ordered row binding failed: "
                    f"{output_check.code}: {output_check.evidence}"
                )
            manifest = _manifest(bundle, npz_path, parquet_path)
            write_json_exclusive(manifest_path, manifest.model_dump(mode="json"))
            complete = _complete_marker(manifest, manifest_path)
            write_json_exclusive(complete_path, complete.model_dump(mode="json"))
            staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest, complete
