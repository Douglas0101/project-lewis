"""Pydantic schemas for training-ready data artifacts.

Validates contracts for processed signals, beat records, and the consolidated
training dataset manifest. Used by ``scripts/audit_training_data.py`` and the
feature pipeline to guarantee data quality before model training.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums / constrained literals
# ---------------------------------------------------------------------------

DatasetName = Literal["chapman", "mitdb", "svdb", "afdb", "incart", "ptbxl"]
AAMIClass = Literal["N", "S", "V", "F", "Q"]
WFDBBeatSymbol = Literal[
    "N",
    "L",
    "R",
    "e",
    "j",
    "V",
    "E",
    "A",
    "a",
    "J",
    "S",
    "F",
    "/",
    "f",
    "Q",
    "|",
]

# ---------------------------------------------------------------------------
# Processed signal (Camada 2 output)
# ---------------------------------------------------------------------------


class ProcessedSignalRecord(BaseModel):
    """One fully pre-processed ECG record ready for segmentation."""

    record_id: str = Field(..., min_length=1, description="Canonical record identifier")
    dataset: DatasetName
    fs: float = Field(500.0, gt=0, description="Sampling frequency after resampling")
    shape: Tuple[int, ...] = Field(..., description="Shape of the processed .npy array")
    dtype: str = Field(..., pattern=r"^(float32|float64)$")
    raw_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA256 hex digest of the raw .dat/.hea used for lineage",
    )
    lineage_path: str = Field(..., description="Path to the corresponding lineage.json")
    npy_path: str = Field(..., description="Path to the processed .npy signal")
    input_range_mV: Optional[Tuple[float, float]] = Field(
        None, description="Raw signal range in mV (if available)"
    )
    output_range_mV: Tuple[float, float] = Field(..., description="Processed signal range in mV")
    mean: float = Field(..., description="Sample mean of the processed signal")
    std: float = Field(..., gt=0, description="Sample std of the processed signal")
    duration_sec: float = Field(..., gt=0, description="Signal duration in seconds")
    config_version: str = Field("1.0.0", description="Pre-processing config version")

    @field_validator("shape")
    @classmethod
    def _shape_is_1d(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(v) != 1:
            raise ValueError("Processed signal must be 1-D (shape length 1)")
        if v[0] <= 0:
            raise ValueError("Signal length must be positive")
        return v

    @field_validator("input_range_mV", "output_range_mV")
    @classmethod
    def _range_ordered(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError("Range min must be <= max")
        return v


# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------


class TemporalFeatures(BaseModel):
    """RR-interval based features for a single beat."""

    rr_prev: float = Field(..., ge=0, description="Previous RR interval in ms")
    rr_next: float = Field(..., ge=0, description="Next RR interval in ms")
    rr_ratio: float = Field(..., ge=0, description="rr_prev / rr_next")
    rr_local_mean: float = Field(..., ge=0, description="Local 5-beat RR mean in ms")
    rr_local_std: float = Field(..., ge=0, description="Local 5-beat RR std in ms")
    rmssd: float = Field(..., ge=0, description="Root mean square of successive RR differences")
    heart_rate: float = Field(..., ge=0, description="Instantaneous heart rate in BPM")


class MorphologicalFeatures(BaseModel):
    """Morphological features for a single beat segment."""

    r_amplitude: float = Field(..., description="R-peak amplitude in mV")
    q_depth: float = Field(..., description="Q-wave depth (minimum) in mV")
    t_amplitude: float = Field(..., description="T-wave peak amplitude in mV")
    qrs_width_ms: float = Field(
        ...,
        ge=0,
        le=500,
        description="QRS width in ms (NaN represented as 0 for pydantic)",
    )
    qrs_area: float = Field(..., ge=0, description="Absolute QRS area in mV·s")
    st_slope_mV_s: float = Field(..., description="ST slope from J+60ms to J+80ms in mV/s")
    j_point: int = Field(..., ge=0, description="QRS offset sample index within segment")


# ---------------------------------------------------------------------------
# Beat record (Camada 3 output)
# ---------------------------------------------------------------------------


class BeatRecord(BaseModel):
    """One segmented beat with labels and extracted features."""

    record_id: str = Field(..., min_length=1)
    beat_idx: int = Field(..., ge=0)
    dataset: DatasetName
    segment_shape: Tuple[int, ...] = Field(..., description="Shape of the segment array")
    label_wfdb: WFDBBeatSymbol
    label_aami: AAMIClass
    r_peak_sample: int = Field(..., ge=0, description="R-peak index within the full record")
    r_peak_in_segment: int = Field(..., ge=0, description="R-peak index within the segment")
    temporal: TemporalFeatures
    morph: MorphologicalFeatures
    augmentation_applied: bool = False
    augmentation_methods: List[str] = Field(default_factory=list)
    lineage_path: str = Field(..., description="Lineage of the parent processed record")

    @field_validator("segment_shape")
    @classmethod
    def _segment_shape_ok(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(v) not in (1, 2):
            raise ValueError("Segment must be 1-D or 2-D (samples, [channel])")
        if v[0] not in (300, 500):
            raise ValueError("Segment length must be 300 or 500 samples")
        return v


# ---------------------------------------------------------------------------
# Training dataset manifest
# ---------------------------------------------------------------------------


class DatasetStats(BaseModel):
    """Aggregate statistics for one dataset in the training manifest."""

    n_records: int = Field(..., ge=0)
    n_beats: int = Field(..., ge=0)
    class_distribution: Dict[AAMIClass, int] = Field(
        default_factory=lambda: {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}  # type: ignore[arg-type]
    )
    mean_segment_amplitude: Optional[float] = None
    std_segment_amplitude: Optional[float] = None
    pct_flatline_beats: Optional[float] = Field(None, ge=0, le=100)


class QualityFlags(BaseModel):
    """Boolean quality flags for the whole training dataset."""

    no_nan_inf: bool = True
    no_flatline_records: bool = True
    all_lineage_valid: bool = True
    all_checksums_match: bool = True
    aami_labels_valid: bool = True
    pii_free: bool = True
    group_kfold_feasible: bool = True
    class_balance_reported: bool = True


class TrainingDatasetManifest(BaseModel):
    """Canonical manifest describing a training-ready dataset."""

    version: str = Field("1.0.0", description="Manifest schema version")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config_version: str = Field("1.0.0", description="Pre-processing config version")
    datasets_included: List[DatasetName]
    n_records: int = Field(..., ge=0)
    n_beats: int = Field(..., ge=0)
    per_dataset: Dict[DatasetName, DatasetStats]
    global_class_distribution: Dict[AAMIClass, int]
    quality_flags: QualityFlags = Field(default_factory=QualityFlags)
    features_schema_version: str = "1.0.0"
    notes: Optional[str] = None

    @field_validator("global_class_distribution")
    @classmethod
    def _all_classes_present(cls, v: Dict[str, int]) -> Dict[str, int]:
        for c in ("N", "S", "V", "F", "Q"):
            if c not in v:
                v[c] = 0
        return v


# ---------------------------------------------------------------------------
# Audit report schemas
# ---------------------------------------------------------------------------


class AuditCheck(BaseModel):
    """Result of a single audit check."""

    category: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    status: Literal["PASS", "FAIL", "WARNING"]
    count: int = Field(0, ge=0)
    details: Optional[str] = None


class TrainingDataAuditReport(BaseModel):
    """Consolidated report emitted by ``scripts/audit_training_data.py``."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config_version: str = "1.0.0"
    overall_status: Literal["PASS", "FAIL"] = "PASS"
    n_records_inspected: int = Field(0, ge=0)
    n_beats_inspected: int = Field(0, ge=0)
    checks: List[AuditCheck] = Field(default_factory=list)
    dataset_summaries: Dict[DatasetName, Dict[str, Any]] = Field(default_factory=dict)
    anomaly_records: List[str] = Field(
        default_factory=list,
        description="Record IDs flagged as anomalous (no PII)",
    )
    dlq_path: Optional[str] = None
