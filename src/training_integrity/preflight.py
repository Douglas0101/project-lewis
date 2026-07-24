"""Fail-closed preflight checks, evidence reporting, and gate publication."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import load_advanced_training_config, load_patient_identity_policy
from .contracts import (
    CheckStatus,
    DatasetRole,
    EpistemicCategory,
    FileManifest,
    IdentityStatus,
    PatientIdentityManifest,
    PatientSplitManifest,
    PreflightCheck,
    PreflightEvidenceBundle,
    PreflightReport,
    PreflightReportPublication,
    PretrainingGateMarker,
)
from .integrity import (
    afdb_episode_sample_id,
    beat_sample_id,
    hash_canonical,
    resolve_project_path,
    sha256_file,
    waveform_row_sha256,
    write_bytes_exclusive,
    write_detached_sha256,
    write_json_exclusive,
)
from .manifests import build_training_generation_manifest
from .patient_identity import (
    audit_legacy_outer_fold_leakage,
    build_patient_identity_manifest,
)
from .splits import build_patient_split_manifest, publish_split_bundle

REQUIRED_FAMILY_CUSTODY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "record_id",
    "beat_idx",
    "sample_id",
    "segment_id",
    "waveform_sha256",
    "source_sampling_rate",
    "target_sampling_rate",
    "annotation_index_native",
    "annotation_time_seconds",
    "annotation_index_target",
    "class_original",
    "class_canonical",
    "label_aami",
    "y",
)

REQUIRED_SAMPLE_LINEAGE_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "patient_id",
    "record_id",
    "beat_index",
    "segment_id",
    "sample_id",
    "waveform_sha256",
    "source_sampling_rate",
    "target_sampling_rate",
    "annotation_index_native",
    "annotation_time_seconds",
    "annotation_index_target",
    "class_original",
    "class_canonical",
    "y",
    "quality_label",
    "split",
    "fold",
)


def family_source_custody_schema_check(
    npz_path: Path,
    parquet_path: Path,
) -> PreflightCheck:
    """Require source-bound row identity on the parent beat-family artifact."""
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing_members = sorted(
                {"X", "y", "sample_id", "waveform_sha256"} - set(archive.files)
            )
        schema = pq.ParquetFile(parquet_path).schema_arrow
    except (OSError, ValueError) as error:
        return _check(
            "FAMILY_SOURCE_CUSTODY_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read parent family custody: {error}",
            "1 NPZ/parquet pair",
            "Stage 1 cannot be traced to source beats",
        )
    missing_columns = sorted(set(REQUIRED_FAMILY_CUSTODY_COLUMNS) - set(schema.names))
    if missing_members or missing_columns:
        return _check(
            "FAMILY_SOURCE_CUSTODY_INCOMPLETE",
            CheckStatus.BLOCK,
            "parent family artifact lacks source-bound row custody",
            f"{len(REQUIRED_FAMILY_CUSTODY_COLUMNS)} parquet columns and 4 NPZ members",
            "derived row hashes without parent custody are insufficient",
            details={
                "missing_npz_members": missing_members,
                "missing_parquet_columns": missing_columns,
            },
        )
    return _check(
        "FAMILY_SOURCE_CUSTODY_COMPLETE",
        CheckStatus.PASS,
        "parent family artifact exposes complete source-bound row custody",
        f"{len(REQUIRED_FAMILY_CUSTODY_COLUMNS)} parquet columns and 4 NPZ members",
        "values and source reconstruction are checked separately",
    )


def family_source_reconstruction_check(
    parquet_path: Path,
    identity: PatientIdentityManifest,
) -> PreflightCheck:
    """Reconstruct every eligible beat from bound annotations and processed signals."""
    from src.data.segmenter import ECGSegmenter
    from src.features.pipeline import (
        TARGET_FS,
        _find_processed_npy,
        _load_raw_annotation_custody,
    )

    try:
        frame = pd.read_parquet(
            parquet_path,
            columns=list(REQUIRED_FAMILY_CUSTODY_COLUMNS),
        )
    except (OSError, ValueError, KeyError) as error:
        return _check(
            "FAMILY_SOURCE_RECONSTRUCTION_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read parent family values: {error}",
            "1 parquet artifact",
            "source reconstruction cannot be evaluated",
        )
    issues: Counter[str] = Counter()
    expected_record_keys = {
        record.record_key
        for record in identity.records
        if record.role is not DatasetRole.RHYTHM_EXPLORATORY
    }
    observed_record_keys = set(frame["dataset"].astype(str) + "/" + frame["record_id"].astype(str))
    issues["record_population_mismatch"] = len(
        expected_record_keys.symmetric_difference(observed_record_keys)
    )
    segmenter = ECGSegmenter(fs=TARGET_FS, window_ms=1000.0, min_window_ms=600.0)
    label_to_int = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
    for record_key in sorted(expected_record_keys & observed_record_keys):
        dataset_id, record_id = record_key.split("/", 1)
        custody = _load_raw_annotation_custody(record_id, dataset_id)
        processed_path = _find_processed_npy(record_id, dataset_id)
        if custody is None or processed_path is None:
            issues["source_record_unreadable"] += 1
            continue
        try:
            signal = np.load(processed_path, allow_pickle=False).astype(np.float32)
        except (OSError, ValueError):
            issues["source_record_unreadable"] += 1
            continue
        target_samples = np.asarray(custody.target_samples, dtype=np.int64)
        native_samples = np.asarray(custody.native_samples, dtype=np.int64)
        original_symbols = np.asarray(custody.original_symbols).astype(str)
        canonical_labels = np.asarray(custody.canonical_labels)
        source_annotation_indices = np.arange(len(target_samples), dtype=np.int64)
        in_range = (target_samples >= 0) & (target_samples < len(signal))
        n_out_of_range = _safe_count(
            (~in_range).sum(),
            context="out-of-range source annotations",
        )
        if n_out_of_range:
            if n_out_of_range / len(target_samples) > 0.01:
                issues["source_annotation_range_invalid"] += 1
                continue
            target_samples = target_samples[in_range]
            native_samples = native_samples[in_range]
            original_symbols = original_symbols[in_range]
            canonical_labels = canonical_labels[in_range]
            source_annotation_indices = source_annotation_indices[in_range]
        waveforms, labels, metadata = segmenter.segment_with_labels(
            signal,
            target_samples,
            canonical_labels,
            rr_intervals_ms=None,
        )
        kept_indices = np.asarray(
            metadata.get("kept_indices", np.arange(len(waveforms))), dtype=np.int64
        )
        source_beat_indices = source_annotation_indices[kept_indices]
        observed = frame[
            (frame["dataset"].astype(str) == dataset_id)
            & (frame["record_id"].astype(str) == record_id)
        ]
        if len(observed) != len(source_beat_indices):
            issues["source_beat_count_mismatch"] += 1
            continue
        expected_sample_ids: list[str] = []
        expected_hashes: list[str] = []
        expected_original: list[str] = []
        expected_canonical: list[str] = []
        expected_y: list[int] = []
        expected_native: list[int] = []
        expected_target: list[int] = []
        for waveform, filtered_beat_index, source_beat_index, label in zip(
            waveforms,
            kept_indices,
            source_beat_indices,
            labels,
            strict=True,
        ):
            filtered_beat = _safe_count(
                filtered_beat_index,
                context="filtered reconstructed beat index",
            )
            beat = _safe_count(source_beat_index, context="source reconstructed beat index")
            target_index = _safe_count(
                target_samples[filtered_beat], context="reconstructed target index"
            )
            label_text = str(label)
            expected_sample_ids.append(beat_sample_id(dataset_id, record_id, beat, target_index))
            waveform_array = np.asarray(waveform, dtype=np.float32)
            if waveform_array.shape == (_safe_count(TARGET_FS, context="target rate"),):
                waveform_array = waveform_array[..., np.newaxis]
            elif waveform_array.shape != (
                _safe_count(TARGET_FS, context="target rate"),
                1,
            ):
                issues["source_waveform_shape_invalid"] += 1
            expected_hashes.append(waveform_row_sha256(waveform_array))
            expected_original.append(str(original_symbols[filtered_beat]))
            expected_canonical.append(
                {"F": "FUSION", "Q": "Q_OR_UNKNOWN"}.get(label_text, label_text)
            )
            expected_y.append(label_to_int[label_text])
            expected_native.append(
                _safe_count(
                    native_samples[filtered_beat],
                    context="reconstructed native index",
                )
            )
            expected_target.append(target_index)
        comparisons = (
            observed["beat_idx"].to_numpy() == source_beat_indices,
            observed["sample_id"].astype(str).to_numpy() == np.asarray(expected_sample_ids),
            observed["segment_id"].astype(str).to_numpy() == np.asarray(expected_sample_ids),
            observed["waveform_sha256"].astype(str).to_numpy() == np.asarray(expected_hashes),
            observed["annotation_index_native"].to_numpy() == np.asarray(expected_native),
            observed["annotation_index_target"].to_numpy() == np.asarray(expected_target),
            observed["class_original"].astype(str).to_numpy() == np.asarray(expected_original),
            observed["class_canonical"].astype(str).to_numpy() == np.asarray(expected_canonical),
            observed["label_aami"].astype(str).to_numpy()
            == np.asarray([str(label) for label in labels]),
            observed["y"].to_numpy() == np.asarray(expected_y),
        )
        if not all(bool(np.asarray(comparison).all()) for comparison in comparisons):
            issues["source_row_mismatch"] += 1
        source_rates = pd.to_numeric(observed["source_sampling_rate"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        target_rates = pd.to_numeric(observed["target_sampling_rate"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        times = pd.to_numeric(observed["annotation_time_seconds"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        expected_times = (
            np.asarray(expected_native, dtype=np.float64) / custody.source_sampling_rate
        )
        if not (
            np.all(source_rates == custody.source_sampling_rate)
            and np.all(target_rates == TARGET_FS)
            and np.allclose(times, expected_times, rtol=0.0, atol=0.001)
        ):
            issues["source_clock_mismatch"] += 1
    nonzero = {key: value for key, value in issues.items() if value}
    if nonzero:
        return _check(
            "FAMILY_SOURCE_RECONSTRUCTION_INVALID",
            CheckStatus.BLOCK,
            "parent beat-family rows differ from bound annotations or processed signals",
            f"{len(frame)} rows",
            "missing, extra, relabeled, or replaced source beats are not repaired",
            details={"issue_counts": nonzero},
        )
    return _check(
        "FAMILY_SOURCE_RECONSTRUCTION_VALIDATED",
        CheckStatus.PASS,
        "every parent beat row and waveform reconstructs from bound source evidence",
        f"{len(frame)} rows",
        "processed-signal generation remains a separately hashed preprocessing claim",
    )


def stage1_parent_binding_check(
    family_parquet: Path,
    stage1_parquet: Path,
) -> PreflightCheck:
    """Require Stage 1 to be the exact deterministic transform of its parent rows."""
    try:
        parent = pd.read_parquet(
            family_parquet,
            columns=[
                "sample_id",
                "waveform_sha256",
                "label_aami",
                "class_canonical",
                "y",
            ],
        )
        stage1 = pd.read_parquet(
            stage1_parquet,
            columns=["sample_id", "waveform_sha256", "class_canonical", "y"],
        )
    except (OSError, ValueError, KeyError) as error:
        return _check(
            "STAGE1_PARENT_BINDING_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read Stage 1 parent binding: {error}",
            "1 parent/child parquet pair",
            "derived population completeness cannot be evaluated",
        )
    conflicting_hashes = {
        str(waveform_hash)
        for waveform_hash, group in parent.groupby("waveform_sha256", sort=False)
        if len(group) > 1 and group["label_aami"].astype(str).nunique() > 1
    }
    expected = parent[
        ~parent["waveform_sha256"].astype(str).isin(conflicting_hashes)
        & (parent["class_canonical"].astype(str) != "Q_OR_UNKNOWN")
    ].copy()
    expected_y = np.where(expected["class_canonical"].astype(str).to_numpy() == "N", 0, 1)
    aligned = (
        len(stage1) == len(expected)
        and np.array_equal(
            stage1["sample_id"].astype(str).to_numpy(),
            expected["sample_id"].astype(str).to_numpy(),
        )
        and np.array_equal(
            stage1["waveform_sha256"].astype(str).to_numpy(),
            expected["waveform_sha256"].astype(str).to_numpy(),
        )
        and np.array_equal(
            stage1["class_canonical"].astype(str).to_numpy(),
            expected["class_canonical"].astype(str).to_numpy(),
        )
        and np.array_equal(stage1["y"].to_numpy(), expected_y)
    )
    if not aligned:
        return _check(
            "STAGE1_PARENT_BINDING_INVALID",
            CheckStatus.BLOCK,
            "Stage 1 is not the exact Q-excluded, conflict-deduplicated parent transform",
            f"{len(expected)} expected rows / {len(stage1)} observed rows",
            "missing, extra, reordered, relabeled, or replaced rows are not repaired",
        )
    return _check(
        "STAGE1_PARENT_BINDING_VALIDATED",
        CheckStatus.PASS,
        "Stage 1 exactly matches the deterministic parent population transform",
        f"{len(stage1)} rows",
        "the parent source reconstruction is a separate required check",
    )


def afdb_source_reconstruction_check(
    parquet_path: Path,
    *,
    raw_dir: Path,
    processed_dir: Path,
) -> PreflightCheck:
    """Reconstruct the complete AFDB episode population and waveform hashes."""
    from src.features.afdb_rhythm import (
        EPISODE_LEN,
        _load_rhythm_intervals_500,
    )

    try:
        observed = pd.read_parquet(
            parquet_path,
            columns=[
                "record_id",
                "sample_id",
                "waveform_sha256",
                "interval_start_native",
                "interval_end_native",
                "interval_start_target",
                "interval_end_target",
                "rhythm_original",
                "rhythm_canonical",
            ],
        )
    except (OSError, ValueError, KeyError) as error:
        return _check(
            "AFDB_SOURCE_RECONSTRUCTION_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read AFDB source reconstruction fields: {error}",
            "1 parquet artifact",
            "episode population completeness cannot be evaluated",
        )
    expected_rows: list[tuple[object, ...]] = []
    try:
        for signal_path in sorted(processed_dir.glob("*_ECG1.npy")):
            record_id = signal_path.name.split("_", 1)[0]
            base = raw_dir / record_id
            if not base.with_suffix(".hea").is_file() or not base.with_suffix(".atr").is_file():
                continue
            signal = np.load(signal_path, allow_pickle=False).astype(np.float32)
            intervals = _load_rhythm_intervals_500(base, len(signal))
            for start_target in range(0, len(signal) - EPISODE_LEN + 1, EPISODE_LEN):
                end_target = start_target + EPISODE_LEN
                interval = next(
                    (
                        candidate
                        for candidate in intervals
                        if candidate[2] <= start_target and end_target <= candidate[3]
                    ),
                    None,
                )
                if interval is None:
                    continue
                (
                    interval_start_native,
                    interval_end_native,
                    interval_start_target,
                    interval_end_target,
                    original,
                    canonical,
                ) = interval
                episode_index = start_target // EPISODE_LEN
                sample_id = afdb_episode_sample_id(
                    record_id, episode_index, start_target, end_target
                )
                waveform = signal[start_target:end_target].reshape(EPISODE_LEN, 1)
                expected_rows.append(
                    (
                        record_id,
                        sample_id,
                        waveform_row_sha256(waveform),
                        interval_start_native,
                        interval_end_native,
                        interval_start_target,
                        interval_end_target,
                        original,
                        canonical,
                    )
                )
    except (OSError, ValueError) as error:
        return _check(
            "AFDB_SOURCE_RECONSTRUCTION_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot reconstruct AFDB source episodes: {error}",
            f"{len(expected_rows)} episodes reconstructed before failure",
            "partial reconstruction cannot authorize rhythm evidence",
        )
    observed_rows = [tuple(row) for row in observed.itertuples(index=False, name=None)]
    if observed_rows != expected_rows:
        return _check(
            "AFDB_SOURCE_RECONSTRUCTION_INVALID",
            CheckStatus.BLOCK,
            "AFDB episode rows differ from the complete deterministic source reconstruction",
            f"{len(expected_rows)} expected / {len(observed_rows)} observed episodes",
            "subsets, extra episodes, altered windows, and reordered rows are rejected",
        )
    return _check(
        "AFDB_SOURCE_RECONSTRUCTION_VALIDATED",
        CheckStatus.PASS,
        "every eligible AFDB episode and waveform reconstructs from bound source evidence",
        f"{len(expected_rows)} episodes",
        "AFDB patient identity remains a separate requirement",
    )


def sample_lineage_schema_check(path: Path) -> PreflightCheck:
    """Require the complete per-sample custody schema before training."""
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except (OSError, ValueError) as error:
        return PreflightCheck(
            code="SAMPLE_LINEAGE_UNREADABLE",
            status=CheckStatus.BLOCK,
            epistemic_category=EpistemicCategory.OBSERVED,
            evidence=f"cannot read sample lineage schema: {error}",
            denominator="1 parquet artifact",
            limitation="sample custody cannot be evaluated",
            details={"path": str(path)},
        )
    present = set(schema.names)
    missing = tuple(column for column in REQUIRED_SAMPLE_LINEAGE_COLUMNS if column not in present)
    if missing:
        return PreflightCheck(
            code="SAMPLE_LINEAGE_INCOMPLETE",
            status=CheckStatus.BLOCK,
            epistemic_category=EpistemicCategory.OBSERVED,
            evidence=f"{len(missing)} required custody columns are absent",
            denominator=f"{len(REQUIRED_SAMPLE_LINEAGE_COLUMNS)} required columns",
            limitation="record-level lineage cannot replace sample-level dual-clock custody",
            details={"path": str(path), "missing_columns": list(missing)},
        )
    return PreflightCheck(
        code="SAMPLE_LINEAGE_COMPLETE",
        status=CheckStatus.PASS,
        epistemic_category=EpistemicCategory.OBSERVED,
        evidence="all required sample custody columns are present",
        denominator=f"{len(REQUIRED_SAMPLE_LINEAGE_COLUMNS)} required columns",
        limitation="schema presence alone does not prove row values",
        details={"path": str(path), "missing_columns": []},
    )


def _safe_count(value: Any, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid count for {context}: {value!r}") from error


def _is_integral_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(value) and float(value).is_integer())
    except (TypeError, ValueError, OverflowError):
        return False


def sample_lineage_value_check(
    path: Path,
    identity: PatientIdentityManifest,
    split: PatientSplitManifest,
) -> PreflightCheck:
    """Validate every dual-clock, ontology, identity, and split custody value."""
    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_SAMPLE_LINEAGE_COLUMNS))
    except (OSError, ValueError, KeyError) as error:
        return _check(
            "SAMPLE_LINEAGE_VALUES_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read sample lineage values: {error}",
            "1 parquet artifact",
            "sample custody values cannot be evaluated",
        )
    if frame.empty:
        return _check(
            "SAMPLE_LINEAGE_VALUES_INVALID",
            CheckStatus.BLOCK,
            "sample lineage is empty",
            "0 rows",
            "no training population is available",
        )

    issues: Counter[str] = Counter()
    issues["null_required_values"] = _safe_count(
        frame.isna().any(axis=1).sum(), context="null required values"
    )
    segment_ids = frame["segment_id"].astype(str)
    issues["empty_segment_ids"] = _safe_count(
        (segment_ids.str.len() == 0).sum(), context="empty segment IDs"
    )
    issues["duplicate_segment_ids"] = _safe_count(
        segment_ids.duplicated(keep=False).sum(), context="duplicate segment IDs"
    )
    sample_ids = frame["sample_id"].astype(str)
    issues["segment_sample_identity_mismatch"] = _safe_count(
        (segment_ids != sample_ids).sum(), context="segment/sample identity mismatches"
    )

    numeric_columns = (
        "beat_index",
        "source_sampling_rate",
        "target_sampling_rate",
        "annotation_index_native",
        "annotation_time_seconds",
        "annotation_index_target",
        "fold",
    )
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for column in numeric_columns
    }
    finite_rows = np.ones(len(frame), dtype=bool)
    for values in numeric.values():
        finite_rows &= np.isfinite(values)
    issues["nonfinite_numeric_values"] = _safe_count(
        (~finite_rows).sum(), context="nonfinite numeric values"
    )
    valid_numeric = finite_rows & (numeric["source_sampling_rate"] > 0)
    issues["invalid_source_sampling_rate"] = _safe_count(
        (finite_rows & (numeric["source_sampling_rate"] <= 0)).sum(),
        context="invalid source sampling rates",
    )
    target_500 = numeric["target_sampling_rate"] == 500.0
    issues["invalid_target_sampling_rate"] = _safe_count(
        (finite_rows & ~target_500).sum(), context="invalid target sampling rates"
    )
    nonnegative_indices = (numeric["annotation_index_native"] >= 0) & (
        numeric["annotation_index_target"] >= 0
    )
    integral_indices = (
        numeric["annotation_index_native"] == np.floor(numeric["annotation_index_native"])
    ) & (numeric["annotation_index_target"] == np.floor(numeric["annotation_index_target"]))
    issues["invalid_annotation_indices"] = _safe_count(
        (finite_rows & (~nonnegative_indices | ~integral_indices)).sum(),
        context="invalid annotation indices",
    )
    valid_beat_indices = (numeric["beat_index"] >= 0) & (
        numeric["beat_index"] == np.floor(numeric["beat_index"])
    )
    issues["invalid_beat_indices"] = _safe_count(
        (finite_rows & ~valid_beat_indices).sum(), context="invalid beat indices"
    )
    expected_sample_ids: list[str] = []
    for row in frame.loc[
        :, ["dataset_id", "record_id", "beat_index", "annotation_index_target"]
    ].itertuples(index=False, name=None):
        dataset_id, record_id, beat_index, target_index = row
        try:
            expected_sample_ids.append(
                beat_sample_id(
                    str(dataset_id),
                    str(record_id),
                    int(beat_index),
                    int(target_index),
                )
            )
        except (TypeError, ValueError, OverflowError):
            expected_sample_ids.append("INVALID")
    issues["sample_identity_mismatch"] = _safe_count(
        (sample_ids.to_numpy() != np.asarray(expected_sample_ids)).sum(),
        context="beat sample identity mismatches",
    )
    if valid_numeric.any():
        expected_target = np.rint(
            numeric["annotation_index_native"]
            * numeric["target_sampling_rate"]
            / numeric["source_sampling_rate"]
        )
        issues["annotation_clock_mismatch"] = _safe_count(
            (valid_numeric & (expected_target != numeric["annotation_index_target"])).sum(),
            context="annotation clock mismatches",
        )
        expected_time = numeric["annotation_index_native"] / numeric["source_sampling_rate"]
        tolerance = np.divide(
            0.5,
            numeric["target_sampling_rate"],
            out=np.full(len(frame), np.nan),
            where=numeric["target_sampling_rate"] > 0,
        )
        time_match = np.abs(numeric["annotation_time_seconds"] - expected_time) <= (
            tolerance + 1e-12
        )
        issues["annotation_time_mismatch"] = _safe_count(
            (valid_numeric & ~time_match).sum(), context="annotation time mismatches"
        )

    from src.features.ontology_v3 import BEAT_MAP_V3

    ontology_mismatch = 0
    for original, canonical in frame.loc[:, ["class_original", "class_canonical"]].itertuples(
        index=False, name=None
    ):
        mapping = BEAT_MAP_V3.get(str(original))
        if mapping is None or mapping[0] != str(canonical):
            ontology_mismatch += 1
    issues["ontology_mismatch"] = ontology_mismatch
    canonical = frame["class_canonical"].astype(str).to_numpy()
    allowed_stage1_classes = {"N", "S", "V", "FUSION"}
    issues["unsupported_stage1_class"] = _safe_count(
        (~np.isin(canonical, tuple(allowed_stage1_classes))).sum(),
        context="unsupported Stage 1 classes",
    )
    binary_labels = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=np.float64)
    valid_binary = np.isfinite(binary_labels) & np.isin(binary_labels, (0.0, 1.0))
    issues["invalid_binary_label"] = _safe_count(
        (~valid_binary).sum(), context="invalid binary labels"
    )
    expected_binary = np.where(canonical == "N", 0.0, 1.0)
    issues["binary_label_semantic_mismatch"] = _safe_count(
        (valid_binary & (binary_labels != expected_binary)).sum(),
        context="binary label semantic mismatches",
    )
    allowed_quality = {"VALID", "FLATLINE", "CLIP", "OFF_CENTER", "MULTIPLE_FLAGS"}
    quality = frame["quality_label"].astype(str)
    issues["invalid_quality_label"] = _safe_count(
        (~quality.isin(allowed_quality)).sum(), context="invalid quality labels"
    )

    identity_by_key = {record.record_key: record for record in identity.records}
    patient_fold = {
        patient_id: fold.fold for fold in split.folds for patient_id in fold.outer_test_patient_ids
    }
    record_fold = {
        record_key: fold.fold for fold in split.folds for record_key in fold.outer_test_record_keys
    }
    record_contracts = frame.loc[
        :, ["dataset_id", "record_id", "patient_id", "split", "fold"]
    ].drop_duplicates()
    observed_record_keys: set[str] = set()
    for dataset_id, record_id, patient_id, split_name, fold_value in record_contracts.itertuples(
        index=False, name=None
    ):
        record_key = f"{dataset_id}/{record_id}"
        observed_record_keys.add(record_key)
        record = identity_by_key.get(record_key)
        if record is None:
            issues["record_without_identity"] += 1
            continue
        fold = pd.to_numeric(pd.Series([fold_value]), errors="coerce").iloc[0]
        if record.role is DatasetRole.CONFIRMATORY_CORE:
            patient_id_text = str(patient_id)
            if patient_id_text != record.patient_id:
                issues["patient_identity_mismatch"] += 1
            expected_patient_fold = patient_fold.get(patient_id_text)
            expected_record_fold = record_fold.get(record_key)
            valid_fold = _is_integral_number(fold)
            if (
                str(split_name) != "outer_test"
                or not valid_fold
                or expected_patient_fold is None
                or expected_record_fold is None
                or fold != expected_patient_fold
                or fold != expected_record_fold
            ):
                issues["confirmatory_split_mismatch"] += 1
        else:
            expected_split = (
                "domain_sensitivity"
                if record.role is DatasetRole.DOMAIN_SENSITIVITY
                else "rhythm_exploratory"
            )
            if str(patient_id) != "UNKNOWN_OR_UNVERIFIED":
                issues["unverified_patient_assertion"] += 1
            if str(split_name) != expected_split or fold != -1:
                issues["quarantine_split_mismatch"] += 1
    expected_beat_records = {
        record.record_key
        for record in identity.records
        if record.role is not DatasetRole.RHYTHM_EXPLORATORY
    }
    issues["identity_records_missing_from_lineage"] = len(
        expected_beat_records - observed_record_keys
    )

    nonzero = {key: value for key, value in issues.items() if value}
    if nonzero:
        return _check(
            "SAMPLE_LINEAGE_VALUES_INVALID",
            CheckStatus.BLOCK,
            "sample lineage violates one or more custody invariants",
            f"{len(frame)} rows",
            "invalid rows are not repaired, dropped, or relabeled by preflight",
            details={"issue_counts": nonzero},
        )
    return _check(
        "SAMPLE_LINEAGE_VALUES_VALIDATED",
        CheckStatus.PASS,
        "all sample identity, clock, ontology, quality, and split values satisfy the contract",
        f"{len(frame)} rows",
        "validation does not establish external clinical validity",
        details={"rows": len(frame)},
    )


REQUIRED_AFDB_LINEAGE_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "record_id",
    "patient_id",
    "episode_idx",
    "segment_id",
    "sample_id",
    "waveform_sha256",
    "source_sampling_rate",
    "target_sampling_rate",
    "start_sample_native",
    "end_sample_native",
    "start_time_seconds",
    "end_time_seconds",
    "start_sample_target",
    "end_sample_target",
    "interval_start_native",
    "interval_end_native",
    "interval_start_target",
    "interval_end_target",
    "rhythm_original",
    "rhythm_canonical",
    "split",
    "fold",
)


def afdb_lineage_schema_check(path: Path) -> PreflightCheck:
    """Require episode-level AFDB source and dual-clock custody."""
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except (OSError, ValueError) as error:
        return _check(
            "AFDB_LINEAGE_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read AFDB lineage schema: {error}",
            "1 parquet artifact",
            "AFDB source custody cannot be evaluated",
        )
    missing = tuple(
        column for column in REQUIRED_AFDB_LINEAGE_COLUMNS if column not in schema.names
    )
    if missing:
        return _check(
            "AFDB_LINEAGE_INCOMPLETE",
            CheckStatus.BLOCK,
            f"{len(missing)} required AFDB custody columns are absent",
            f"{len(REQUIRED_AFDB_LINEAGE_COLUMNS)} required columns",
            "episode hashes alone do not prove source-record or clock custody",
            details={"missing_columns": list(missing)},
        )
    return _check(
        "AFDB_LINEAGE_COMPLETE",
        CheckStatus.PASS,
        "all required AFDB episode custody columns are present",
        f"{len(REQUIRED_AFDB_LINEAGE_COLUMNS)} required columns",
        "schema presence alone does not prove row values",
    )


def afdb_lineage_value_check(
    path: Path,
    identity: PatientIdentityManifest,
    *,
    raw_dir: Path,
) -> PreflightCheck:
    """Validate AFDB episode source identity, clocks, ontology, and quarantine."""
    from src.features.ontology_v3 import AFDB_RHYTHM_MAP_V3

    from .integrity import afdb_episode_sample_id

    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_AFDB_LINEAGE_COLUMNS))
    except (OSError, ValueError, KeyError) as error:
        return _check(
            "AFDB_LINEAGE_VALUES_UNREADABLE",
            CheckStatus.BLOCK,
            f"cannot read AFDB lineage values: {error}",
            "1 parquet artifact",
            "AFDB source custody cannot be evaluated",
        )
    if frame.empty:
        return _check(
            "AFDB_LINEAGE_VALUES_INVALID",
            CheckStatus.BLOCK,
            "AFDB episode lineage is empty",
            "0 rows",
            "no rhythm population is available",
        )

    issues: Counter[str] = Counter()
    issues["null_required_values"] = _safe_count(
        frame.isna().any(axis=1).sum(), context="AFDB null required values"
    )
    sample_ids = frame["sample_id"].astype(str)
    segment_ids = frame["segment_id"].astype(str)
    issues["duplicate_sample_ids"] = _safe_count(
        sample_ids.duplicated(keep=False).sum(), context="AFDB duplicate sample IDs"
    )
    issues["segment_sample_identity_mismatch"] = _safe_count(
        (segment_ids != sample_ids).sum(), context="AFDB segment/sample identity mismatches"
    )

    numeric_columns = (
        "episode_idx",
        "source_sampling_rate",
        "target_sampling_rate",
        "start_sample_native",
        "end_sample_native",
        "start_time_seconds",
        "end_time_seconds",
        "start_sample_target",
        "end_sample_target",
        "interval_start_native",
        "interval_end_native",
        "interval_start_target",
        "interval_end_target",
        "fold",
    )
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for column in numeric_columns
    }
    finite = np.ones(len(frame), dtype=bool)
    for values in numeric.values():
        finite &= np.isfinite(values)
    issues["nonfinite_numeric_values"] = _safe_count(
        (~finite).sum(), context="AFDB nonfinite numeric values"
    )
    rates_valid = (numeric["source_sampling_rate"] == 250.0) & (
        numeric["target_sampling_rate"] == 500.0
    )
    issues["sampling_rate_mismatch"] = _safe_count(
        (finite & ~rates_valid).sum(), context="AFDB sampling-rate mismatches"
    )
    target_length = numeric["end_sample_target"] - numeric["start_sample_target"]
    native_length = numeric["end_sample_native"] - numeric["start_sample_native"]
    integral_boundaries = np.ones(len(frame), dtype=bool)
    for column in (
        "episode_idx",
        "start_sample_native",
        "end_sample_native",
        "start_sample_target",
        "end_sample_target",
        "interval_start_native",
        "interval_end_native",
        "interval_start_target",
        "interval_end_target",
    ):
        integral_boundaries &= numeric[column] == np.floor(numeric[column])
    boundary_valid = (
        integral_boundaries
        & (target_length == 5000.0)
        & (native_length == 2500.0)
        & (numeric["start_sample_target"] >= 0)
        & (numeric["start_sample_native"] >= 0)
        & (numeric["episode_idx"] >= 0)
        & (numeric["start_sample_target"] == numeric["episode_idx"] * 5000.0)
    )
    interval_contains_episode = (
        (numeric["interval_start_native"] <= numeric["start_sample_native"])
        & (numeric["end_sample_native"] <= numeric["interval_end_native"])
        & (numeric["interval_start_target"] <= numeric["start_sample_target"])
        & (numeric["end_sample_target"] <= numeric["interval_end_target"])
    )
    issues["episode_boundary_mismatch"] = _safe_count(
        (finite & (~boundary_valid | ~interval_contains_episode)).sum(),
        context="AFDB episode boundary mismatches",
    )
    expected_start_target = np.rint(
        numeric["start_sample_native"]
        * numeric["target_sampling_rate"]
        / numeric["source_sampling_rate"]
    )
    expected_end_target = np.rint(
        numeric["end_sample_native"]
        * numeric["target_sampling_rate"]
        / numeric["source_sampling_rate"]
    )
    expected_interval_start_target = np.rint(
        numeric["interval_start_native"]
        * numeric["target_sampling_rate"]
        / numeric["source_sampling_rate"]
    )
    expected_interval_end_target = np.rint(
        numeric["interval_end_native"]
        * numeric["target_sampling_rate"]
        / numeric["source_sampling_rate"]
    )
    clock_valid = (
        (expected_start_target == numeric["start_sample_target"])
        & (expected_end_target == numeric["end_sample_target"])
        & (expected_interval_start_target == numeric["interval_start_target"])
        & (expected_interval_end_target == numeric["interval_end_target"])
    )
    time_valid = np.isclose(
        numeric["start_time_seconds"],
        numeric["start_sample_native"] / numeric["source_sampling_rate"],
        rtol=0.0,
        atol=0.001,
    ) & np.isclose(
        numeric["end_time_seconds"],
        numeric["end_sample_native"] / numeric["source_sampling_rate"],
        rtol=0.0,
        atol=0.001,
    )
    issues["episode_clock_mismatch"] = _safe_count(
        (finite & (~clock_valid | ~time_valid)).sum(),
        context="AFDB episode clock mismatches",
    )

    expected_ids: list[str] = []
    for row in frame.loc[
        :, ["record_id", "episode_idx", "start_sample_target", "end_sample_target"]
    ].itertuples(index=False, name=None):
        record_id, episode_idx, start_target, end_target = row
        try:
            expected_ids.append(
                afdb_episode_sample_id(
                    str(record_id),
                    int(episode_idx),
                    int(start_target),
                    int(end_target),
                )
            )
        except (TypeError, ValueError, OverflowError):
            expected_ids.append("INVALID")
    issues["sample_identity_mismatch"] = _safe_count(
        (sample_ids.to_numpy() != np.asarray(expected_ids)).sum(),
        context="AFDB sample identity mismatches",
    )

    rhythm_mismatch = 0
    for original, canonical in frame.loc[:, ["rhythm_original", "rhythm_canonical"]].itertuples(
        index=False, name=None
    ):
        expected = AFDB_RHYTHM_MAP_V3.get(str(original), "OTHER_RHYTHM")
        if expected != str(canonical):
            rhythm_mismatch += 1
    issues["rhythm_ontology_mismatch"] = rhythm_mismatch

    from src.features.afdb_rhythm import _load_rhythm_intervals_500

    identity_by_key = {record.record_key: record for record in identity.records}
    source_intervals: dict[str, set[tuple[int, int, int, int, str, str]]] = {}
    for record_id in sorted(frame["record_id"].astype(str).unique().tolist()):
        record = identity_by_key.get(f"afdb/{record_id}")
        if record is None:
            continue
        try:
            source_intervals[record_id] = set(_load_rhythm_intervals_500(raw_dir / record_id))
        except (OSError, ValueError) as error:
            issues["source_annotation_unreadable"] += 1
            source_intervals[record_id] = set()
            del error
    for row in frame.loc[
        :,
        [
            "record_id",
            "interval_start_native",
            "interval_end_native",
            "interval_start_target",
            "interval_end_target",
            "rhythm_original",
            "rhythm_canonical",
        ],
    ].itertuples(index=False, name=None):
        (
            record_id,
            start_native,
            end_native,
            start_target,
            end_target,
            original,
            canonical,
        ) = row
        try:
            claimed = (
                int(start_native),
                int(end_native),
                int(start_target),
                int(end_target),
                str(original),
                str(canonical),
            )
        except (TypeError, ValueError, OverflowError):
            issues["source_interval_mismatch"] += 1
            continue
        if claimed not in source_intervals.get(str(record_id), set()):
            issues["source_interval_mismatch"] += 1

    observed_records: set[str] = set()
    contracts = frame.loc[
        :, ["dataset_id", "record_id", "patient_id", "split", "fold"]
    ].drop_duplicates()
    for dataset_id, record_id, patient_id, split_name, fold_value in contracts.itertuples(
        index=False, name=None
    ):
        record_key = f"{dataset_id}/{record_id}"
        observed_records.add(record_key)
        record = identity_by_key.get(record_key)
        if (
            dataset_id != "afdb"
            or record is None
            or record.role is not DatasetRole.RHYTHM_EXPLORATORY
        ):
            issues["invalid_source_record"] += 1
            continue
        if record.identity_status is IdentityStatus.IDENTITY_VERIFIED:
            if str(patient_id) != record.patient_id:
                issues["verified_patient_identity_mismatch"] += 1
        elif str(patient_id) != "UNKNOWN_OR_UNVERIFIED":
            issues["unverified_patient_assertion"] += 1
        fold = pd.to_numeric(pd.Series([fold_value]), errors="coerce").iloc[0]
        if str(split_name) != "rhythm_exploratory" or fold != -1:
            issues["quarantine_split_mismatch"] += 1
    issues["episode_records_without_identity"] = len(observed_records - set(identity_by_key))

    nonzero = {key: value for key, value in issues.items() if value}
    if nonzero:
        return _check(
            "AFDB_LINEAGE_VALUES_INVALID",
            CheckStatus.BLOCK,
            "AFDB episode lineage violates one or more custody invariants",
            f"{len(frame)} rows",
            "invalid rows are not repaired or relabeled by preflight",
            details={"issue_counts": nonzero},
        )
    return _check(
        "AFDB_LINEAGE_VALUES_VALIDATED",
        CheckStatus.PASS,
        "all AFDB episode source, clock, ontology, and quarantine values satisfy the contract",
        f"{len(frame)} rows",
        "AFDB patient identity remains a separate evidence requirement",
        details={"rows": len(frame)},
    )


def ordered_row_binding_check(
    npz_path: Path,
    parquet_path: Path,
    *,
    scope: str = "",
    expected_shape: tuple[int, int] = (500, 1),
    parquet_label_column: str = "y",
) -> PreflightCheck:
    """Recompute and verify ordered sample, label, and waveform identities."""
    code_prefix = f"{scope}_" if scope else ""
    required_members = {"X", "y", "sample_id", "waveform_sha256"}
    required_columns = {parquet_label_column, "sample_id", "waveform_sha256"}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing_members = sorted(required_members - set(archive.files))
            if missing_members:
                return PreflightCheck(
                    code=f"{code_prefix}ORDERED_ROW_BINDING_INCOMPLETE",
                    status=CheckStatus.BLOCK,
                    epistemic_category=EpistemicCategory.OBSERVED,
                    evidence="NPZ lacks ordered sample identity or waveform digests",
                    denominator=f"{len(required_members)} required NPZ members",
                    limitation="same-class waveform permutations cannot be detected",
                    details={"missing_npz_members": missing_members},
                )
            waveforms = np.asarray(archive["X"])
            sample_ids_npz = np.asarray(archive["sample_id"]).astype(str)
            waveform_hashes_npz = np.asarray(archive["waveform_sha256"]).astype(str)
            labels_npz = np.asarray(archive["y"])
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        missing_columns = sorted(required_columns - schema_names)
        if missing_columns:
            return PreflightCheck(
                code=f"{code_prefix}ORDERED_ROW_BINDING_INCOMPLETE",
                status=CheckStatus.BLOCK,
                epistemic_category=EpistemicCategory.OBSERVED,
                evidence="parquet lacks ordered sample identity or waveform digests",
                denominator=f"{len(required_columns)} required parquet columns",
                limitation="NPZ and parquet row order cannot be authenticated",
                details={"missing_parquet_columns": missing_columns},
            )
        frame = pd.read_parquet(
            parquet_path,
            columns=["sample_id", "waveform_sha256", parquet_label_column],
        )
    except (OSError, ValueError, KeyError) as error:
        return PreflightCheck(
            code=f"{code_prefix}ORDERED_ROW_BINDING_UNREADABLE",
            status=CheckStatus.BLOCK,
            epistemic_category=EpistemicCategory.OBSERVED,
            evidence=f"cannot validate ordered NPZ/parquet binding: {error}",
            denominator="1 NPZ/parquet pair",
            limitation="ordered row identity cannot be evaluated",
            details={},
        )

    sample_ids_parquet = frame["sample_id"].astype(str).to_numpy()
    waveform_hashes_parquet = frame["waveform_sha256"].astype(str).to_numpy()
    labels_parquet = frame[parquet_label_column].to_numpy()
    hash_format_valid = all(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in waveform_hashes_npz
    )
    shape_valid = waveforms.ndim == 3 and waveforms.shape[1:] == expected_shape
    finite = bool(np.isfinite(waveforms).all())
    computed_hashes = np.array([waveform_row_sha256(row) for row in waveforms])
    aligned = (
        len(waveforms) == len(frame)
        and len(labels_npz) == len(frame)
        and len(sample_ids_npz) == len(frame)
        and len(waveform_hashes_npz) == len(frame)
        and len(set(sample_ids_npz.tolist())) == len(sample_ids_npz)
        and all(value != "" for value in sample_ids_npz)
        and hash_format_valid
        and shape_valid
        and finite
        and np.array_equal(sample_ids_npz, sample_ids_parquet)
        and np.array_equal(waveform_hashes_npz, waveform_hashes_parquet)
        and np.array_equal(waveform_hashes_npz, computed_hashes)
        and np.array_equal(labels_npz, labels_parquet)
    )
    if not aligned:
        return PreflightCheck(
            code=f"{code_prefix}ORDERED_ROW_BINDING_MISMATCH",
            status=CheckStatus.BLOCK,
            epistemic_category=EpistemicCategory.OBSERVED,
            evidence="ordered sample, label, or recomputed waveform identities differ",
            denominator=f"{len(frame)} parquet rows",
            limitation="the first mismatch is not used to repair or reorder data",
            details={
                "npz_rows": len(waveforms),
                "parquet_rows": len(frame),
                "shape_valid": shape_valid,
                "finite": finite,
            },
        )
    return PreflightCheck(
        code=f"{code_prefix}ORDERED_ROW_BINDING_VALIDATED",
        status=CheckStatus.PASS,
        epistemic_category=EpistemicCategory.OBSERVED,
        evidence="ordered sample IDs, labels, and recomputed waveform digests match",
        denominator=f"{len(frame)} rows",
        limitation="validation is bound to canonical float32 waveform bytes",
        details={"rows": len(frame)},
    )


REQUIRED_GATE_PASS_CHECK_CODES = frozenset(
    {
        "CONFIRMATORY_PATIENT_IDENTITY_VALIDATED",
        "PATIENT_SPLIT_V3_1_VALIDATED",
        "FAMILY_SOURCE_CUSTODY_COMPLETE",
        "FAMILY_ORDERED_ROW_BINDING_VALIDATED",
        "FAMILY_SOURCE_RECONSTRUCTION_VALIDATED",
        "SAMPLE_LINEAGE_COMPLETE",
        "SAMPLE_LINEAGE_VALUES_VALIDATED",
        "ORDERED_ROW_BINDING_VALIDATED",
        "STAGE1_PARENT_BINDING_VALIDATED",
        "AFDB_LINEAGE_COMPLETE",
        "AFDB_LINEAGE_VALUES_VALIDATED",
        "AFDB_ORDERED_ROW_BINDING_VALIDATED",
        "AFDB_SOURCE_RECONSTRUCTION_VALIDATED",
        "INPUT_SNAPSHOT_STABLE",
        "AFDB_PATIENT_IDENTITY_VALIDATED",
        "EXACT_INPUT_BYTES_BOUND",
    }
)
REQUIRED_GATE_CHECK_CODES = REQUIRED_GATE_PASS_CHECK_CODES | frozenset(
    {
        "SESSION_AUTHORIZATION_RECORDED",
        "UNVERIFIED_DATASETS_QUARANTINED",
        "LEGACY_SPLIT_PATIENT_LEAKAGE_DETECTED",
        "EXTERNAL_VALIDATION_REQUIRED",
    }
)
REQUIRED_COMPONENT_KEYS = frozenset(
    {
        "raw_data_manifest",
        "annotation_manifest",
        "processed_data_manifest",
        "preprocessing_manifest",
        "source_manifest",
        "ontology_manifest",
        "training_config_manifest",
        "identity_source",
        "git",
        "environment",
        "feature_schema",
    }
)
_FILE_COMPONENTS = {
    "raw_data_manifest": ("raw-data", "raw_data_hash", "payload"),
    "annotation_manifest": ("annotations", "annotation_hash", "payload"),
    "processed_data_manifest": ("processed-data", "processed_data_hash", "payload"),
    "preprocessing_manifest": ("preprocessing", "preprocessing_hash", "payload"),
    "source_manifest": ("research-source", "source_revision", "source"),
    "ontology_manifest": ("ontology-source", "ontology_hash", "single-file"),
    "training_config_manifest": ("training-config", "training_config_hash", "single-file"),
}


def _component_payload(bundle: PreflightEvidenceBundle) -> dict[str, Any]:
    payload = bundle.model_dump(mode="json")["component_evidence"]
    if not isinstance(payload, dict):
        raise ValueError("component evidence is not a JSON object")
    return payload


def _validated_component_manifests(
    bundle: PreflightEvidenceBundle,
) -> dict[str, FileManifest]:
    component_payload = _component_payload(bundle)
    if set(component_payload) != REQUIRED_COMPONENT_KEYS:
        missing = sorted(REQUIRED_COMPONENT_KEYS - set(component_payload))
        extra = sorted(set(component_payload) - REQUIRED_COMPONENT_KEYS)
        raise ValueError(f"component evidence keys mismatch: missing={missing}, extra={extra}")
    manifests: dict[str, FileManifest] = {}
    for key, (expected_category, _, _) in _FILE_COMPONENTS.items():
        manifest = FileManifest.model_validate_json(
            json.dumps(component_payload[key], sort_keys=True)
        )
        if manifest.category != expected_category:
            raise ValueError(f"component category mismatch for {key}")
        manifest_payload = {
            "schema_version": manifest.schema_version,
            "category": manifest.category,
            "files": [file.model_dump(mode="json") for file in manifest.files],
        }
        expected_payload_hash = hash_canonical(
            f"file-manifest:{manifest.category}", manifest_payload
        )
        if manifest.payload_hash != expected_payload_hash:
            raise ValueError(f"file manifest payload hash mismatch for {key}")
        manifests[key] = manifest
    return manifests


def verify_complete_component_snapshot(
    project_root: Path,
    bundle: PreflightEvidenceBundle,
) -> None:
    """Revalidate all evidence-bound files immediately before gate publication."""
    manifests = _validated_component_manifests(bundle)
    generation = bundle.generation_manifest
    components = _component_payload(bundle)
    identity_source = components["identity_source"]
    if not isinstance(identity_source, dict):
        raise ValueError("identity source evidence is not a JSON object")
    expected_identity_source_hash = hash_canonical("patient-identity-source", identity_source)
    if bundle.patient_identity_manifest.source_data_hash != expected_identity_source_hash:
        raise ValueError("patient identity manifest does not bind its source artifacts")
    processed_hashes = {file.sha256 for file in manifests["processed_data_manifest"].files}
    identity_source_hashes = set(identity_source.values())
    if not identity_source_hashes or not identity_source_hashes <= processed_hashes:
        raise ValueError("patient identity source artifacts are absent from processed data")
    for key, (_, generation_field, binding_kind) in _FILE_COMPONENTS.items():
        manifest = manifests[key]
        bound_hash = getattr(generation, generation_field)
        if binding_kind == "payload" and manifest.payload_hash != bound_hash:
            raise ValueError(f"generation hash mismatch for {key}")
        if binding_kind == "single-file":
            if len(manifest.files) != 1 or manifest.files[0].sha256 != bound_hash:
                raise ValueError(f"single-file generation hash mismatch for {key}")
        for file in manifest.files:
            path = resolve_project_path(project_root, file.project_relative_path)
            if path.stat().st_size != file.size_bytes or sha256_file(path) != file.sha256:
                raise RuntimeError(f"evidence-bound file changed: {file.project_relative_path}")

    source_payload = {
        "git": components["git"],
        "files": manifests["source_manifest"].model_dump(mode="json"),
    }
    if hash_canonical("research-source-snapshot", source_payload) != generation.source_revision:
        raise ValueError("source revision does not bind source manifest and git evidence")
    if (
        hash_canonical("research-environment", components["environment"])
        != generation.environment_hash
    ):
        raise ValueError("environment hash does not bind environment evidence")
    if (
        hash_canonical("feature-schema-v3.1.0", components["feature_schema"])
        != generation.feature_schema_hash
    ):
        raise ValueError("feature schema hash does not bind feature-schema evidence")


def _validate_gate_eligibility(
    project_root: Path,
    bundle: PreflightEvidenceBundle,
) -> PreflightEvidenceBundle:
    validated = PreflightEvidenceBundle.model_validate_json(bundle.model_dump_json())
    report = validated.preflight_report
    if not report.training_allowed or report.final_state != "PRETRAINING_GATE_PASS":
        raise ValueError("blocked preflight cannot publish PRETRAINING_GATE_PASS")
    codes = [check.code for check in report.checks]
    if len(codes) != len(set(codes)):
        raise ValueError("preflight report contains duplicate check codes")
    missing = sorted(REQUIRED_GATE_CHECK_CODES - set(codes))
    if missing:
        raise ValueError(f"preflight report lacks required checks: {missing}")
    statuses = {check.code: check.status for check in report.checks}
    failed_required = sorted(
        code
        for code in REQUIRED_GATE_PASS_CHECK_CODES
        if statuses.get(code) is not CheckStatus.PASS
    )
    if failed_required:
        raise ValueError(f"required pretraining checks did not pass: {failed_required}")
    verify_complete_component_snapshot(project_root, validated)
    return validated


def finalize_preflight_report(
    *,
    generation_id: str,
    checks: tuple[PreflightCheck, ...],
) -> PreflightReport:
    """Derive the only allowed training decision from the canonical check set."""
    finalized_checks = list(checks)
    blockers = tuple(check.code for check in finalized_checks if check.status is CheckStatus.BLOCK)
    if not blockers:
        codes = [check.code for check in finalized_checks]
        missing = sorted(REQUIRED_GATE_CHECK_CODES - set(codes))
        duplicates = sorted(code for code, count in Counter(codes).items() if count > 1)
        status_by_code = {check.code: check.status for check in finalized_checks}
        failed_required = sorted(
            code
            for code in REQUIRED_GATE_PASS_CHECK_CODES
            if status_by_code.get(code) is not CheckStatus.PASS
        )
        if missing or duplicates or failed_required:
            finalized_checks.append(
                _check(
                    "PREFLIGHT_CHECK_SET_INCOMPLETE",
                    CheckStatus.BLOCK,
                    "canonical preflight checks are missing or duplicated",
                    f"{len(REQUIRED_GATE_CHECK_CODES)} required check codes",
                    "a partial report can never authorize training",
                    category=EpistemicCategory.NOT_SUPPORTED,
                    details={
                        "missing": missing,
                        "duplicates": duplicates,
                        "failed_required": failed_required,
                    },
                )
            )
            blockers = ("PREFLIGHT_CHECK_SET_INCOMPLETE",)
    checks = tuple(finalized_checks)
    allowed = not blockers
    return PreflightReport(
        schema_version="training-preflight-v3.1.0",
        generation_id=generation_id,
        checks=checks,
        training_allowed=allowed,
        final_state="PRETRAINING_GATE_PASS" if allowed else "REVIEW_REQUIRED",
        blocking_codes=blockers,
    )


def _comparable_bundle(bundle: PreflightEvidenceBundle) -> dict[str, Any]:
    payload = bundle.model_dump(mode="json")
    payload.pop("generated_at_utc", None)
    return payload


def verify_canonical_preflight_execution(
    config_path: Path,
    bundle: PreflightEvidenceBundle,
) -> None:
    """Recompute every canonical check; caller-authored PASS claims are rejected."""
    recomputed = run_project_preflight(
        config_path,
        publish_splits=False,
        write_reports=False,
    )
    if _comparable_bundle(recomputed) != _comparable_bundle(bundle):
        raise ValueError("supplied bundle differs from canonical preflight recomputation")


def publish_report_bundle(
    report_json: Path,
    report_markdown: Path,
    completion_marker: Path,
    bundle: PreflightEvidenceBundle,
    *,
    config_path: Path,
    project_root: Path,
) -> None:
    """Commit a canonically recomputed write-once report bundle."""
    verify_canonical_preflight_execution(config_path, bundle)
    verify_complete_component_snapshot(project_root, bundle)
    write_json_exclusive(report_json, bundle)
    write_detached_sha256(report_json)
    write_bytes_exclusive(report_markdown, _render_markdown(bundle).encode("utf-8"))
    write_detached_sha256(report_markdown)
    publication = PreflightReportPublication(
        schema_version="preflight-report-publication-v3.1.0",
        generation_id=bundle.generation_manifest.generation_id,
        evidence_bundle_hash=hash_canonical("training-preflight-evidence", bundle),
        report_json_sha256=sha256_file(report_json),
        report_markdown_sha256=sha256_file(report_markdown),
        status="REPORT_BUNDLE_COMPLETE",
    )
    write_json_exclusive(completion_marker, publication)
    write_detached_sha256(completion_marker)


def publish_pretraining_gate(
    path: Path,
    bundle: PreflightEvidenceBundle,
    *,
    config_path: Path,
    project_root: Path,
) -> None:
    """Publish a write-once marker after complete snapshot revalidation."""
    report = bundle.preflight_report
    if not report.training_allowed or report.final_state != "PRETRAINING_GATE_PASS":
        raise ValueError("blocked preflight cannot publish PRETRAINING_GATE_PASS")
    verify_canonical_preflight_execution(config_path, bundle)
    bundle = _validate_gate_eligibility(project_root.resolve(), bundle)
    report = bundle.preflight_report
    marker = PretrainingGateMarker(
        schema_version="pretraining-gate-v3.1.0",
        generation_id=report.generation_id,
        evidence_bundle_hash=hash_canonical("training-preflight-evidence", bundle),
        generation_manifest_hash=hash_canonical("training-generation", bundle.generation_manifest),
        preflight_report_hash=hash_canonical("training-preflight", report),
        status="TRAINING_PROVENANCE_VALIDATED",
    )
    write_json_exclusive(path, marker)
    write_detached_sha256(path)


def _check(
    code: str,
    status: CheckStatus,
    evidence: str,
    denominator: str,
    limitation: str,
    *,
    category: EpistemicCategory = EpistemicCategory.OBSERVED,
    details: dict[str, object] | None = None,
) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        status=status,
        epistemic_category=category,
        evidence=evidence,
        denominator=denominator,
        limitation=limitation,
        details=details or {},
    )


def _render_markdown(bundle: PreflightEvidenceBundle) -> str:
    report = bundle.preflight_report
    generation = bundle.generation_manifest
    lines = [
        "# `training_preflight_v3.1`",
        "",
        f"**Gerado em:** {bundle.generated_at_utc}  ",
        f"**Geração:** `{generation.generation_id}`  ",
        f"**Estado:** `{report.final_state}`  ",
        f"**Treinamento permitido:** `{str(report.training_allowed).lower()}`",
        "",
        "## Checks",
        "",
        "| Código | Status | Categoria | Evidência | Denominador | Limitação |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for check in report.checks:
        values = (
            check.code,
            check.status.value,
            check.epistemic_category.value,
            check.evidence,
            check.denominator,
            check.limitation,
        )
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## Identidade da geração",
            "",
            "```text",
            f"raw_data_hash={generation.raw_data_hash}",
            f"annotation_hash={generation.annotation_hash}",
            f"processed_data_hash={generation.processed_data_hash}",
            f"ontology_hash={generation.ontology_hash}",
            f"preprocessing_hash={generation.preprocessing_hash}",
            f"feature_schema_hash={generation.feature_schema_hash}",
            f"patient_split_hash={generation.patient_split_hash}",
            f"training_config_hash={generation.training_config_hash}",
            f"source_revision={generation.source_revision}",
            f"environment_hash={generation.environment_hash}",
            "```",
            "",
            "## Decisão",
            "",
        ]
    )
    if report.training_allowed:
        lines.append("`PRETRAINING_GATE_PASS`")
    else:
        lines.extend(
            [
                "```text",
                "TRAINING_BLOCKED_BY_DATA_PROVENANCE",
                "REVIEW_REQUIRED",
                "```",
                "",
                "Nenhuma célula de treinamento pode iniciar.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_project_preflight(
    config_path: Path,
    *,
    publish_splits: bool,
    write_reports: bool,
) -> PreflightEvidenceBundle:
    """Execute the canonical evidence-only preflight without training."""
    resolved_config_path = config_path.resolve()
    config_hash_before = sha256_file(resolved_config_path)
    config, project_root = load_advanced_training_config(resolved_config_path)
    if config_hash_before != sha256_file(resolved_config_path):
        raise RuntimeError("training config changed while being loaded")
    if not resolved_config_path.is_relative_to(project_root):
        raise ValueError("training config must be contained by the project root")
    if config.required_sample_lineage_columns != REQUIRED_SAMPLE_LINEAGE_COLUMNS:
        raise ValueError("training config sample-lineage contract differs from canonical v3.1")
    policy_path = resolve_project_path(project_root, config.identity_policy)
    policy_hash_before = sha256_file(policy_path)
    policy = load_patient_identity_policy(policy_path)
    if policy_hash_before != sha256_file(policy_path):
        raise RuntimeError("patient identity policy changed while being loaded")
    afdb_policies = tuple(item for item in policy.datasets if item.dataset_id == "afdb")
    if len(afdb_policies) != 1:
        raise ValueError("patient identity policy must define AFDB exactly once")
    afdb_raw_dir = resolve_project_path(project_root, afdb_policies[0].raw_dir)
    family_npz = resolve_project_path(project_root, config.family_npz)
    family_path = resolve_project_path(project_root, config.family_parquet)
    rhythm_path = resolve_project_path(project_root, config.afdb_rhythm_parquet)
    rhythm_npz = resolve_project_path(project_root, config.afdb_rhythm_npz)
    stage1_npz = resolve_project_path(project_root, config.stage1_npz)
    stage1_parquet = resolve_project_path(project_root, config.stage1_parquet)
    config_relative = resolved_config_path.relative_to(project_root).as_posix()
    protected_inputs = {
        config_relative: resolved_config_path,
        config.identity_policy: policy_path,
        config.family_npz: family_npz,
        config.family_parquet: family_path,
        config.stage1_npz: stage1_npz,
        config.stage1_parquet: stage1_parquet,
        config.afdb_rhythm_npz: rhythm_npz,
        config.afdb_rhythm_parquet: rhythm_path,
    }
    input_hashes_before = {
        relative: sha256_file(path) for relative, path in protected_inputs.items()
    }

    family = pd.read_parquet(
        family_path,
        columns=["dataset", "record_id", "label_aami"],
    )
    rhythm_records = pd.read_parquet(rhythm_path, columns=["record_id"])
    rhythm_records = rhythm_records.assign(dataset="afdb")
    identity_records = pd.concat(
        [family.loc[:, ["dataset", "record_id"]], rhythm_records],
        ignore_index=True,
    )
    identity_source_evidence = {
        "family_parquet_sha256": sha256_file(family_path),
        "rhythm_parquet_sha256": sha256_file(rhythm_path),
    }
    identity_source_hash = hash_canonical("patient-identity-source", identity_source_evidence)
    identity = build_patient_identity_manifest(
        identity_records,
        project_root=project_root,
        policy=policy,
        source_data_hash=identity_source_hash,
    )
    legacy = audit_legacy_outer_fold_leakage(
        identity,
        split_dir=resolve_project_path(project_root, config.legacy_split_dir),
    )
    split = build_patient_split_manifest(
        family,
        identity,
        split_version="3.1.0",
        n_splits=config.n_splits,
        random_state=config.split_random_state,
    )
    generation, component_evidence = build_training_generation_manifest(
        project_root=project_root,
        config_path=resolved_config_path,
        config=config,
        identity=identity,
        split=split,
        policy=policy,
    )
    component_evidence["identity_source"] = identity_source_evidence
    input_hashes_after = {
        relative: sha256_file(path) for relative, path in protected_inputs.items()
    }
    drifted_inputs = tuple(
        sorted(
            relative
            for relative in protected_inputs
            if input_hashes_before[relative] != input_hashes_after[relative]
        )
    )
    input_stability_check = _check(
        "INPUT_SNAPSHOT_STABLE" if not drifted_inputs else "INPUT_SNAPSHOT_DRIFT",
        CheckStatus.PASS if not drifted_inputs else CheckStatus.BLOCK,
        (
            "all consumed input bytes remained stable during preflight"
            if not drifted_inputs
            else "one or more consumed inputs changed during preflight"
        ),
        f"{len(protected_inputs)} input artifacts",
        "filesystem metadata detects concurrent change but not malicious timestamp forgery",
        category=EpistemicCategory.DERIVED_MATHEMATICALLY,
        details={"drifted_inputs": list(drifted_inputs)},
    )

    quarantine_counts = Counter(
        record.dataset_id for record in identity.records if record.patient_group_id is None
    )
    afdb_identity_records = tuple(
        record for record in identity.records if record.dataset_id == "afdb"
    )
    afdb_identity_verified = bool(afdb_identity_records) and all(
        record.identity_status is IdentityStatus.IDENTITY_VERIFIED
        for record in afdb_identity_records
    )
    afdb_identity_check = _check(
        (
            "AFDB_PATIENT_IDENTITY_VALIDATED"
            if afdb_identity_verified
            else "AFDB_PATIENT_IDENTITY_UNVERIFIED"
        ),
        CheckStatus.PASS if afdb_identity_verified else CheckStatus.BLOCK,
        (
            "all AFDB rhythm records have authenticated patient mappings"
            if afdb_identity_verified
            else "AFDB rhythm records lack authenticated patient mappings"
        ),
        f"{len(afdb_identity_records)} AFDB records",
        "family D rhythm output remains exploratory until identity evidence is resolved",
        category=(
            EpistemicCategory.DERIVED_MATHEMATICALLY
            if afdb_identity_verified
            else EpistemicCategory.NOT_SUPPORTED
        ),
    )
    family_schema = family_source_custody_schema_check(family_npz, family_path)
    family_binding = ordered_row_binding_check(
        family_npz,
        family_path,
        scope="FAMILY",
    )
    family_checks: tuple[PreflightCheck, ...] = (family_schema, family_binding)
    if family_schema.status is CheckStatus.PASS and family_binding.status is CheckStatus.PASS:
        family_checks += (family_source_reconstruction_check(family_path, identity),)
    else:
        family_checks += (
            _check(
                "FAMILY_SOURCE_RECONSTRUCTION_NOT_EVALUATED",
                CheckStatus.BLOCK,
                "parent source reconstruction prerequisites did not pass",
                "1 parent NPZ/parquet pair",
                "source population and waveform custody remain unproven",
                category=EpistemicCategory.NOT_SUPPORTED,
            ),
        )

    lineage_schema = sample_lineage_schema_check(stage1_parquet)
    lineage_checks: tuple[PreflightCheck, ...] = (lineage_schema,)
    if lineage_schema.status is CheckStatus.PASS:
        lineage_checks += (sample_lineage_value_check(stage1_parquet, identity, split),)
    stage1_binding = ordered_row_binding_check(stage1_npz, stage1_parquet)
    stage1_parent_checks: tuple[PreflightCheck, ...] = ()
    if (
        family_checks[-1].code == "FAMILY_SOURCE_RECONSTRUCTION_VALIDATED"
        and lineage_checks[-1].code == "SAMPLE_LINEAGE_VALUES_VALIDATED"
        and stage1_binding.status is CheckStatus.PASS
    ):
        stage1_parent_checks = (stage1_parent_binding_check(family_path, stage1_parquet),)
    else:
        stage1_parent_checks = (
            _check(
                "STAGE1_PARENT_BINDING_NOT_EVALUATED",
                CheckStatus.BLOCK,
                "Stage 1 parent-binding prerequisites did not pass",
                "1 parent/child artifact pair",
                "Stage 1 population completeness remains unproven",
                category=EpistemicCategory.NOT_SUPPORTED,
            ),
        )
    afdb_lineage_schema = afdb_lineage_schema_check(rhythm_path)
    afdb_lineage_checks: tuple[PreflightCheck, ...] = (afdb_lineage_schema,)
    if afdb_lineage_schema.status is CheckStatus.PASS:
        afdb_lineage_checks += (
            afdb_lineage_value_check(rhythm_path, identity, raw_dir=afdb_raw_dir),
        )
    afdb_binding = ordered_row_binding_check(
        rhythm_npz,
        rhythm_path,
        scope="AFDB",
        expected_shape=(5000, 1),
        parquet_label_column="rhythm_canonical",
    )
    afdb_source_checks: tuple[PreflightCheck, ...] = ()
    if (
        afdb_lineage_checks[-1].code == "AFDB_LINEAGE_VALUES_VALIDATED"
        and afdb_binding.status is CheckStatus.PASS
    ):
        afdb_source_checks = (
            afdb_source_reconstruction_check(
                rhythm_path,
                raw_dir=afdb_raw_dir,
                processed_dir=project_root / "data" / "processed" / "afdb",
            ),
        )
    else:
        afdb_source_checks = (
            _check(
                "AFDB_SOURCE_RECONSTRUCTION_NOT_EVALUATED",
                CheckStatus.BLOCK,
                "AFDB source-reconstruction prerequisites did not pass",
                "1 AFDB NPZ/parquet pair",
                "episode population and source waveform custody remain unproven",
                category=EpistemicCategory.NOT_SUPPORTED,
            ),
        )
    checks = (
        _check(
            "SESSION_AUTHORIZATION_RECORDED",
            CheckStatus.WARN,
            "project owner authorized continued research with existing project data",
            "1 interactive authorization statement",
            "approver name and authenticated identity were not supplied; promotion remains blocked",
        ),
        _check(
            "CONFIRMATORY_PATIENT_IDENTITY_VALIDATED",
            CheckStatus.PASS,
            "INCART header groups and MITDB documented subject groups are complete",
            f"{identity.confirmatory_record_count} records / "
            f"{identity.confirmatory_patient_count} patient groups",
            "SVDB and AFDB are excluded from the confirmatory patient-wise core",
            category=EpistemicCategory.DERIVED_MATHEMATICALLY,
            details={
                "confirmatory_records": identity.confirmatory_record_count,
                "confirmatory_patients": identity.confirmatory_patient_count,
            },
        ),
        _check(
            "UNVERIFIED_DATASETS_QUARANTINED",
            CheckStatus.WARN,
            "unresolved record-to-patient mappings are assigned non-confirmatory roles",
            f"{sum(quarantine_counts.values())} records",
            "record-clustered patient metrics are not supported for quarantined datasets",
            category=EpistemicCategory.DERIVED_MATHEMATICALLY,
            details={"records_by_dataset": dict(sorted(quarantine_counts.items()))},
        ),
        _check(
            "LEGACY_SPLIT_PATIENT_LEAKAGE_DETECTED",
            CheckStatus.WARN,
            f"{legacy.cross_fold_patient_count} known patients cross legacy outer folds",
            f"{legacy.checked_patient_count} verified patient groups checked",
            "legacy v3 splits and matrix remain invalid and are never repaired in place",
            category=EpistemicCategory.DERIVED_MATHEMATICALLY,
            details=legacy.model_dump(mode="json"),
        ),
        _check(
            "PATIENT_SPLIT_V3_1_VALIDATED",
            CheckStatus.PASS,
            "each confirmatory patient and record is assigned to exactly one outer test fold",
            f"{identity.confirmatory_patient_count} patients across 5 folds",
            "inner/calibration/threshold partitions are not generated in this task",
            category=EpistemicCategory.DERIVED_MATHEMATICALLY,
            details={
                "fold_patient_counts": [len(fold.outer_test_patient_ids) for fold in split.folds],
                "fold_sample_counts": [fold.n_samples for fold in split.folds],
            },
        ),
        *family_checks,
        *lineage_checks,
        stage1_binding,
        *stage1_parent_checks,
        *afdb_lineage_checks,
        afdb_binding,
        *afdb_source_checks,
        input_stability_check,
        afdb_identity_check,
        _check(
            "EXACT_INPUT_BYTES_BOUND",
            CheckStatus.PASS,
            "raw, annotation, processed, ontology, preprocessing, config, source, "
            "and environment bytes are content-addressed",
            "10 required generation identities",
            "hashing does not repair missing row-level custody",
            category=EpistemicCategory.DERIVED_MATHEMATICALLY,
        ),
        _check(
            "EXTERNAL_VALIDATION_REQUIRED",
            CheckStatus.WARN,
            "no genuinely untouched external validation source is authenticated",
            "0 external sources",
            "internal transport analysis cannot be renamed external validation",
            category=EpistemicCategory.NOT_SUPPORTED,
        ),
    )
    report = finalize_preflight_report(generation_id=config.generation_id, checks=checks)
    bundle = PreflightEvidenceBundle(
        schema_version="training-preflight-evidence-v3.1.0",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        generation_manifest=generation,
        patient_identity_manifest=identity,
        patient_split_manifest=split,
        legacy_leakage_audit=legacy,
        preflight_report=report,
        component_evidence=component_evidence,
    )

    verify_complete_component_snapshot(project_root, bundle)

    if publish_splits:
        split_prerequisites = {
            "PATIENT_SPLIT_V3_1_VALIDATED",
            "FAMILY_SOURCE_CUSTODY_COMPLETE",
            "FAMILY_ORDERED_ROW_BINDING_VALIDATED",
            "FAMILY_SOURCE_RECONSTRUCTION_VALIDATED",
        }
        passed_codes = {check.code for check in report.checks if check.status is CheckStatus.PASS}
        missing_split_prerequisites = sorted(split_prerequisites - passed_codes)
        if missing_split_prerequisites:
            raise ValueError(
                "split publication prerequisites did not pass: " f"{missing_split_prerequisites}"
            )
        publish_split_bundle(
            resolve_project_path(project_root, config.split_output_dir),
            identity=identity,
            split=split,
        )
    if write_reports:
        report_json = resolve_project_path(project_root, config.report_json)
        report_markdown = resolve_project_path(project_root, config.report_markdown)
        completion_marker = resolve_project_path(project_root, config.report_completion_marker)
        publish_report_bundle(
            report_json,
            report_markdown,
            completion_marker,
            bundle,
            config_path=resolved_config_path,
            project_root=project_root,
        )
        if report.training_allowed:
            publish_pretraining_gate(
                resolve_project_path(project_root, config.pretraining_gate_marker),
                bundle,
                config_path=resolved_config_path,
                project_root=project_root,
            )
    return bundle
