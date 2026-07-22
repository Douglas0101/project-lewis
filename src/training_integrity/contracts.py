"""Strict immutable contracts for advanced-training provenance artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

HashString = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SafeIdentifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$"),
]
RecordIdentifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$"),
]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_deep_thaw(item) for item in value)
    return value


class StrictFrozenModel(BaseModel):
    """Pydantic base used by every persisted training-integrity contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetRole(StrEnum):
    """Permitted scientific role for a dataset under the v3.1 identity policy."""

    CONFIRMATORY_CORE = "CONFIRMATORY_CORE"
    DOMAIN_SENSITIVITY = "DOMAIN_SENSITIVITY"
    RHYTHM_EXPLORATORY = "RHYTHM_EXPLORATORY"


class IdentityStatus(StrEnum):
    """Evidence status of a record-to-patient assertion."""

    IDENTITY_VERIFIED = "IDENTITY_VERIFIED"
    IDENTITY_UNVERIFIED = "IDENTITY_UNVERIFIED"


class IdentityMethod(StrEnum):
    """Allowed non-PII methods for constructing patient groups."""

    INCART_HEADER_PATIENT = "INCART_HEADER_PATIENT"
    MITDB_DOCUMENTED_SUBJECT = "MITDB_DOCUMENTED_SUBJECT"
    UNRESOLVED = "UNRESOLVED"


class EpistemicCategory(StrEnum):
    """Only epistemic labels permitted by the master protocol."""

    OBSERVED = "OBSERVED"
    DERIVED_MATHEMATICALLY = "DERIVED_MATHEMATICALLY"
    SUPPORTED_INFERENCE = "SUPPORTED_INFERENCE"
    HYPOTHESIS_REQUIRING_TEST = "HYPOTHESIS_REQUIRING_TEST"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class CheckStatus(StrEnum):
    """Fail-closed preflight check state."""

    PASS = "PASS"  # nosec B105 - scientific status label, not a credential
    WARN = "WARN"
    BLOCK = "BLOCK"


class DatasetIdentityPolicy(StrictFrozenModel):
    """Dataset-specific patient identity evidence and scientific role."""

    dataset_id: SafeIdentifier
    role: DatasetRole
    method: IdentityMethod
    raw_dir: str = Field(min_length=1)
    expected_records: int = Field(ge=1)
    expected_patients: int | None = Field(default=None, ge=1)
    same_patient_record_groups: tuple[tuple[RecordIdentifier, ...], ...] = ()
    evidence_ref: str = Field(min_length=1)

    @field_validator("raw_dir")
    @classmethod
    def validate_raw_dir(cls, value: str) -> str:
        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("raw_dir must be a contained project-relative directory")
        return value

    @model_validator(mode="after")
    def validate_evidence_policy(self) -> "DatasetIdentityPolicy":
        canonical_methods = {
            "incart": IdentityMethod.INCART_HEADER_PATIENT,
            "mitdb": IdentityMethod.MITDB_DOCUMENTED_SUBJECT,
            "svdb": IdentityMethod.UNRESOLVED,
            "afdb": IdentityMethod.UNRESOLVED,
        }
        expected_method = canonical_methods.get(self.dataset_id)
        if expected_method is not None and self.method is not expected_method:
            raise ValueError(
                f"identity method {self.method.value} is not valid for {self.dataset_id}"
            )
        if self.method is IdentityMethod.UNRESOLVED:
            if self.role is DatasetRole.CONFIRMATORY_CORE:
                raise ValueError("unresolved identity cannot be confirmatory core")
            if self.expected_patients is not None:
                raise ValueError("unresolved identity cannot declare expected patients")
        elif self.expected_patients is None:
            raise ValueError("verified identity policy requires expected_patients")
        flattened = [record for group in self.same_patient_record_groups for record in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("same-patient record groups overlap")
        if any(len(group) < 2 for group in self.same_patient_record_groups):
            raise ValueError("same-patient record groups require at least two records")
        return self


class PatientIdentityPolicy(StrictFrozenModel):
    """Frozen policy selecting identity evidence for every dataset."""

    schema_version: Literal["patient-identity-policy-v3.1.0"]
    datasets: tuple[DatasetIdentityPolicy, ...]

    @model_validator(mode="after")
    def validate_unique_datasets(self) -> "PatientIdentityPolicy":
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("duplicate dataset identity policy")
        return self


class PatientIdentityRecord(StrictFrozenModel):
    """One evidence-scoped record-to-patient mapping."""

    dataset_id: SafeIdentifier
    record_id: RecordIdentifier
    patient_id: SafeIdentifier | None
    patient_group_id: SafeIdentifier | None
    role: DatasetRole
    identity_status: IdentityStatus
    evidence_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_role_and_identity(self) -> "PatientIdentityRecord":
        if self.role is DatasetRole.CONFIRMATORY_CORE:
            if self.identity_status is not IdentityStatus.IDENTITY_VERIFIED:
                raise ValueError("confirmatory core requires verified patient identity")
            if self.patient_id is None or self.patient_group_id is None:
                raise ValueError("confirmatory core requires patient identifiers")
        if self.identity_status is IdentityStatus.IDENTITY_VERIFIED:
            if self.patient_id is None or self.patient_group_id is None:
                raise ValueError("verified identity requires patient identifiers")
        elif self.patient_id is not None or self.patient_group_id is not None:
            raise ValueError("unverified identity cannot assert a patient identifier")
        return self

    @property
    def record_key(self) -> str:
        return f"{self.dataset_id}/{self.record_id}"


class PatientIdentityManifest(StrictFrozenModel):
    """Complete deterministic identity mapping for one source dataset generation."""

    schema_version: Literal["patient-identity-v3.1.0"]
    source_data_hash: HashString
    records: tuple[PatientIdentityRecord, ...]
    confirmatory_patient_count: int = Field(ge=0)
    confirmatory_record_count: int = Field(ge=0)
    quarantined_record_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts_and_uniqueness(self) -> "PatientIdentityManifest":
        keys = [record.record_key for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate patient identity record")
        core = [record for record in self.records if record.role is DatasetRole.CONFIRMATORY_CORE]
        quarantine = [
            record for record in self.records if record.role is not DatasetRole.CONFIRMATORY_CORE
        ]
        patients = {record.patient_group_id for record in core}
        if None in patients:
            raise ValueError("confirmatory patient group cannot be null")
        if self.confirmatory_patient_count != len(patients):
            raise ValueError("confirmatory_patient_count mismatch")
        if self.confirmatory_record_count != len(core):
            raise ValueError("confirmatory_record_count mismatch")
        if self.quarantined_record_count != len(quarantine):
            raise ValueError("quarantined_record_count mismatch")
        return self


class LegacyLeakageAudit(StrictFrozenModel):
    """Observed known-patient overlap in a legacy outer-fold collection."""

    checked_patient_count: int = Field(ge=0)
    cross_fold_patient_count: int = Field(ge=0)
    cross_fold_patients: Mapping[SafeIdentifier, tuple[int, ...]]

    @field_validator("cross_fold_patients", mode="after")
    @classmethod
    def freeze_cross_fold_patients(cls, value: Mapping[str, tuple[int, ...]]) -> Mapping[str, Any]:
        return _deep_freeze(value)

    @field_serializer("cross_fold_patients")
    def serialize_cross_fold_patients(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)


class FoldAssignment(StrictFrozenModel):
    """One immutable outer-test patient assignment."""

    fold: int = Field(ge=0, le=4)
    outer_test_patient_ids: tuple[SafeIdentifier, ...]
    outer_test_record_keys: tuple[str, ...]
    n_samples: int = Field(ge=1)
    class_counts: Mapping[str, int]
    dataset_counts: Mapping[str, int]

    @field_validator("class_counts", "dataset_counts", mode="after")
    @classmethod
    def freeze_counts(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return _deep_freeze(value)

    @field_serializer("class_counts", "dataset_counts")
    def serialize_counts(self, value: Mapping[str, int]) -> dict[str, int]:
        return _deep_thaw(value)


class PatientSplitManifest(StrictFrozenModel):
    """Patient-disjoint split assignment with explicit dataset quarantine."""

    schema_version: Literal["patient-split-v3.1.0"]
    split_version: Literal["3.1.0"]
    splitter: Literal["StratifiedGroupKFold"] = "StratifiedGroupKFold"
    n_splits: Literal[5] = 5
    random_state: int = Field(ge=0)
    source_data_hash: HashString
    patient_identity_hash: HashString
    core_dataset_ids: tuple[SafeIdentifier, ...]
    quarantine_dataset_ids: tuple[SafeIdentifier, ...]
    folds: tuple[FoldAssignment, ...]
    quarantined_records: tuple[str, ...]

    @model_validator(mode="after")
    def validate_disjoint_outer_folds(self) -> "PatientSplitManifest":
        if {fold.fold for fold in self.folds} != set(range(self.n_splits)):
            raise ValueError("outer folds must be exactly 0..4")
        patient_ids = [patient for fold in self.folds for patient in fold.outer_test_patient_ids]
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("patient appears in more than one outer fold")
        record_keys = [record for fold in self.folds for record in fold.outer_test_record_keys]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError("record appears in more than one outer fold")
        if set(record_keys) & set(self.quarantined_records):
            raise ValueError("quarantined record appears in confirmatory outer folds")
        return self


class HashedFile(StrictFrozenModel):
    """Byte identity for a project-relative file."""

    project_relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: HashString


class FileManifest(StrictFrozenModel):
    """Deterministic collection of exact input files."""

    schema_version: Literal["file-manifest-v3.1.0"]
    category: SafeIdentifier
    files: tuple[HashedFile, ...]
    payload_hash: HashString


class AdvancedTrainingConfig(StrictFrozenModel):
    """Frozen paths and protocol constants for the v3.1 preflight."""

    schema_version: Literal["advanced-training-config-v3.1.0"]
    generation_id: SafeIdentifier
    project_root: str
    family_npz: str
    family_parquet: str
    stage1_npz: str
    stage1_parquet: str
    afdb_rhythm_npz: str
    afdb_rhythm_parquet: str
    identity_policy: str
    legacy_split_dir: str
    split_output_dir: str
    report_json: str
    report_markdown: str
    report_completion_marker: str
    pretraining_gate_marker: str
    families: tuple[Literal["A", "B", "C", "D"], ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    n_splits: Literal[5] = 5
    split_random_state: int = Field(ge=0)
    target_sampling_rate: Literal[500] = 500
    input_shape: tuple[Literal[500], Literal[1]] = (500, 1)
    feature_columns: tuple[str, ...]
    quality_heads: tuple[str, ...]
    preprocessing_files: tuple[str, ...]
    processed_signal_globs: tuple[str, ...]
    source_globs: tuple[str, ...]
    required_sample_lineage_columns: tuple[str, ...]
    research_execution_authorized: Literal[True] = True
    promotion_authorized: Literal[False] = False

    @field_validator(
        "project_root",
        "family_npz",
        "family_parquet",
        "stage1_npz",
        "stage1_parquet",
        "afdb_rhythm_npz",
        "afdb_rhythm_parquet",
        "identity_policy",
        "legacy_split_dir",
        "split_output_dir",
        "report_json",
        "report_markdown",
        "report_completion_marker",
        "pretraining_gate_marker",
    )
    @classmethod
    def validate_relative_paths(cls, value: str) -> str:
        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("configured paths must remain project-relative")
        return value

    @model_validator(mode="after")
    def validate_protocol_matrix(self) -> "AdvancedTrainingConfig":
        if self.families != ("A", "B", "C", "D"):
            raise ValueError("families must be exactly A, B, C, D")
        if self.folds != (0, 1, 2, 3, 4):
            raise ValueError("folds must be exactly 0..4")
        if self.seeds != (17, 29, 43, 71, 101):
            raise ValueError("seeds must be exactly 17, 29, 43, 71, 101")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("feature columns must be unique")
        return self


class TrainingGenerationManifest(StrictFrozenModel):
    """Required ten-hash identity for one advanced-training generation."""

    schema_version: Literal["training-generation-v3.1.0"]
    generation_id: SafeIdentifier
    raw_data_hash: HashString
    annotation_hash: HashString
    processed_data_hash: HashString
    ontology_hash: HashString
    preprocessing_hash: HashString
    feature_schema_hash: HashString
    patient_split_hash: HashString
    training_config_hash: HashString
    source_revision: HashString
    environment_hash: HashString
    research_execution_authorized: bool
    promotion_authorized: Literal[False] = False


class PreflightCheck(StrictFrozenModel):
    """One evidence-labeled preflight conclusion."""

    code: SafeIdentifier
    status: CheckStatus
    epistemic_category: EpistemicCategory
    evidence: str = Field(min_length=1)
    denominator: str = Field(min_length=1)
    limitation: str = Field(min_length=1)
    details: Mapping[str, Any]

    @field_validator("details", mode="after")
    @classmethod
    def freeze_details(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _deep_freeze(value)

    @field_serializer("details")
    def serialize_details(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)


class PreflightReport(StrictFrozenModel):
    """Fail-closed aggregate decision for a proposed generation."""

    schema_version: Literal["training-preflight-v3.1.0"]
    generation_id: SafeIdentifier
    checks: tuple[PreflightCheck, ...]
    training_allowed: bool
    final_state: Literal["PRETRAINING_GATE_PASS", "REVIEW_REQUIRED"]
    blocking_codes: tuple[SafeIdentifier, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> "PreflightReport":
        blockers = tuple(check.code for check in self.checks if check.status is CheckStatus.BLOCK)
        if self.blocking_codes != blockers:
            raise ValueError("blocking_codes mismatch")
        expected_allowed = not blockers
        if self.training_allowed != expected_allowed:
            raise ValueError("training_allowed mismatch")
        expected_state = "PRETRAINING_GATE_PASS" if expected_allowed else "REVIEW_REQUIRED"
        if self.final_state != expected_state:
            raise ValueError("final_state mismatch")
        return self


class PreflightEvidenceBundle(StrictFrozenModel):
    """Machine-readable evidence emitted by the canonical v3.1 preflight."""

    schema_version: Literal["training-preflight-evidence-v3.1.0"]
    generated_at_utc: str = Field(min_length=20)
    generation_manifest: TrainingGenerationManifest
    patient_identity_manifest: PatientIdentityManifest
    patient_split_manifest: PatientSplitManifest
    legacy_leakage_audit: LegacyLeakageAudit
    preflight_report: PreflightReport
    component_evidence: Mapping[str, Any]

    @field_validator("component_evidence", mode="after")
    @classmethod
    def freeze_component_evidence(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _deep_freeze(value)

    @field_serializer("component_evidence")
    def serialize_component_evidence(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)

    @model_validator(mode="after")
    def validate_cross_component_identity(self) -> "PreflightEvidenceBundle":
        generation_id = self.generation_manifest.generation_id
        if self.preflight_report.generation_id != generation_id:
            raise ValueError("preflight report generation_id mismatch")
        if (
            self.patient_split_manifest.source_data_hash
            != self.patient_identity_manifest.source_data_hash
        ):
            raise ValueError("patient split and identity source hashes differ")

        from .integrity import hash_canonical

        expected_identity_hash = hash_canonical("patient-identity", self.patient_identity_manifest)
        if self.patient_split_manifest.patient_identity_hash != expected_identity_hash:
            raise ValueError("patient split does not bind the patient identity manifest")
        expected_split_hash = hash_canonical("patient-split", self.patient_split_manifest)
        if self.generation_manifest.patient_split_hash != expected_split_hash:
            raise ValueError("generation manifest does not bind the patient split manifest")

        from .splits import validate_split_identity_consistency

        validate_split_identity_consistency(
            self.patient_identity_manifest,
            self.patient_split_manifest,
        )
        return self


class SplitBundlePublication(StrictFrozenModel):
    """Commit marker written last for an immutable patient split bundle."""

    schema_version: Literal["patient-split-publication-v3.1.0"]
    patient_identity_hash: HashString
    patient_split_hash: HashString
    status: Literal["SPLIT_BUNDLE_COMPLETE"]


class PreflightReportPublication(StrictFrozenModel):
    """Commit marker written only after the complete report bundle exists."""

    schema_version: Literal["preflight-report-publication-v3.1.0"]
    generation_id: SafeIdentifier
    evidence_bundle_hash: HashString
    report_json_sha256: HashString
    report_markdown_sha256: HashString
    status: Literal["REPORT_BUNDLE_COMPLETE"]


class PretrainingGateMarker(StrictFrozenModel):
    """Write-once marker bound to the full preflight evidence bundle."""

    schema_version: Literal["pretraining-gate-v3.1.0"]
    generation_id: SafeIdentifier
    evidence_bundle_hash: HashString
    generation_manifest_hash: HashString
    preflight_report_hash: HashString
    requires_revalidation_at_consumption: Literal[True] = True
    status: Literal["TRAINING_PROVENANCE_VALIDATED"]
