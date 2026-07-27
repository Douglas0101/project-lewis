"""Strict contracts for E07R patient-disjoint identity and split evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from src.stage2_research.integrity import hash_canonical

HashString = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
IdentityStatusV4 = Literal[
    "OFFICIAL",
    "IDENTITY_VERIFIED",
    "IDENTITY_UNVERIFIED_GROUPED_CONSERVATIVELY",
]


class FrozenModel(BaseModel):
    """Immutable, extra-forbid evidence model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage2PatientRecordV4(FrozenModel):
    """One dataset-scoped identity assertion and split barrier."""

    dataset: Literal["incart", "mitdb", "svdb"]
    record_id: str = Field(min_length=1)
    patient_id: str | None
    partition_barrier_id: str = Field(min_length=1)
    identity_status: IdentityStatusV4
    evidence_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity_semantics(self) -> Stage2PatientRecordV4:
        unverified = self.identity_status == "IDENTITY_UNVERIFIED_GROUPED_CONSERVATIVELY"
        if unverified and self.patient_id is not None:
            raise ValueError("unverified identity cannot claim a biological patient_id")
        if not unverified and self.patient_id is None:
            raise ValueError("verified identity requires patient_id")
        if self.patient_id is not None and self.partition_barrier_id != self.patient_id:
            raise ValueError("verified patient and partition barrier must agree")
        return self


class Stage2PatientMappingV4(FrozenModel):
    """Complete patient-group mapping for the frozen Stage 2 parquet."""

    schema_version: Literal["stage2-patient-identity-v4.0"]
    stage2_parquet_sha256: HashString
    upstream_identity_sha256: HashString
    mitdb_source_url: str
    mapping_policy: Literal["official_verified_or_conservative_single_group_for_unverified_dataset"]
    records: tuple[Stage2PatientRecordV4, ...]
    record_count: int = Field(ge=1)
    verified_patient_count: int = Field(ge=1)
    partition_barrier_count: int = Field(ge=1)
    dataset_record_counts: dict[str, int]
    dataset_verified_patient_counts: dict[str, int]
    dataset_partition_barrier_counts: dict[str, int]
    mapping_hash: HashString

    @model_validator(mode="after")
    def validate_mapping(self) -> Stage2PatientMappingV4:
        keys = [(record.dataset, record.record_id) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate dataset/record identity mapping")
        if self.record_count != len(self.records):
            raise ValueError("record_count does not match records")
        patients = {record.patient_id for record in self.records if record.patient_id is not None}
        barriers = {record.partition_barrier_id for record in self.records}
        if self.verified_patient_count != len(patients):
            raise ValueError("verified_patient_count does not match records")
        if self.partition_barrier_count != len(barriers):
            raise ValueError("partition_barrier_count does not match records")
        observed_records: dict[str, int] = {}
        observed_patients: dict[str, int] = {}
        observed_barriers: dict[str, int] = {}
        for dataset in {record.dataset for record in self.records}:
            subset = [record for record in self.records if record.dataset == dataset]
            observed_records[dataset] = len(subset)
            observed_patients[dataset] = len(
                {record.patient_id for record in subset if record.patient_id is not None}
            )
            observed_barriers[dataset] = len({record.partition_barrier_id for record in subset})
        if observed_records != self.dataset_record_counts:
            raise ValueError("dataset_record_counts mismatch")
        if observed_patients != self.dataset_verified_patient_counts:
            raise ValueError("dataset_verified_patient_counts mismatch")
        if observed_barriers != self.dataset_partition_barrier_counts:
            raise ValueError("dataset_partition_barrier_counts mismatch")
        payload = self.model_dump(mode="json", exclude={"mapping_hash"})
        if hash_canonical(payload) != self.mapping_hash:
            raise ValueError("mapping_hash mismatch")
        return self


class PatientSplitPartitionV4(FrozenModel):
    """One index partition with patient and record evidence."""

    indices: tuple[int, ...]
    patient_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    class_counts: dict[str, int]
    f_208: int = Field(ge=0)
    f_213: int = Field(ge=0)
    f_outside_208_213: int = Field(ge=0)
    n_samples: int = Field(ge=0)
    n_patients: int = Field(ge=0)
    n_records: int = Field(ge=0)
    indices_hash: HashString
    patient_ids_hash: HashString
    record_ids_hash: HashString

    @model_validator(mode="after")
    def validate_counts(self) -> PatientSplitPartitionV4:
        if self.n_samples != len(self.indices):
            raise ValueError("partition n_samples mismatch")
        if self.n_patients != len(self.patient_ids):
            raise ValueError("partition n_patients mismatch")
        if self.n_records != len(self.record_ids):
            raise ValueError("partition n_records mismatch")
        if set(self.class_counts) != {"S", "V", "F"}:
            raise ValueError("partition class counts must be S/V/F")
        if sum(self.class_counts.values()) != self.n_samples:
            raise ValueError("partition class counts do not sum to n_samples")
        return self


class PatientOuterFoldV4(FrozenModel):
    """One patient-disjoint outer fold."""

    fold: int = Field(ge=1, le=5)
    random_state: int
    train: PatientSplitPartitionV4
    test: PatientSplitPartitionV4
    patient_overlap: tuple[str, ...] = ()
    record_overlap: tuple[str, ...] = ()


class PatientOuterFoldsV4(FrozenModel):
    """Complete patient-disjoint outer split collection."""

    schema_version: Literal["stage2-outer-patient-disjoint-v4.0"]
    split_version: Literal["v4.0-patient-disjoint"]
    dataset_binding_hash: HashString
    patient_mapping_hash: HashString
    folds: tuple[PatientOuterFoldV4, ...]
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> PatientOuterFoldsV4:
        if len(self.folds) != 5:
            raise ValueError("outer split must contain five folds")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("outer manifest hash mismatch")
        return self


class PatientInnerFoldV4(FrozenModel):
    """One candidate inner train/validation split inside an outer train."""

    outer_fold: int = Field(ge=1, le=5)
    inner_fold: int = Field(ge=1, le=4)
    random_state: int
    selected_for_training: bool
    train: PatientSplitPartitionV4
    validation: PatientSplitPartitionV4
    outer_test_patient_ids: tuple[str, ...]
    outer_test_record_ids: tuple[str, ...]
    train_validation_patient_overlap: tuple[str, ...] = ()
    train_outer_test_patient_overlap: tuple[str, ...] = ()
    validation_outer_test_patient_overlap: tuple[str, ...] = ()
    train_validation_record_overlap: tuple[str, ...] = ()


class PatientInnerFoldsV4(FrozenModel):
    """All four inner candidates for every outer fold."""

    schema_version: Literal["stage2-inner-patient-disjoint-v4.0"]
    split_version: Literal["v4.0-patient-disjoint"]
    dataset_binding_hash: HashString
    patient_mapping_hash: HashString
    outer_manifest_hash: HashString
    folds: tuple[PatientInnerFoldV4, ...]
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> PatientInnerFoldsV4:
        if len(self.folds) != 20:
            raise ValueError("inner split must contain twenty folds")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("inner manifest hash mismatch")
        return self


class SplitLeakageReportV4(FrozenModel):
    """Fail-closed overlap audit for all outer and inner splits."""

    split_version: Literal["v4.0-patient-disjoint"]
    patient_disjoint: bool
    record_disjoint: bool
    outer_folds_checked: int = Field(ge=0)
    inner_folds_checked: int = Field(ge=0)
    patient_overlap_found: bool
    record_overlap_found: bool
    known_group_201_202_respected: bool
    structural_zero_folds: tuple[str, ...]
    low_support_folds: tuple[str, ...]
    status: Literal["PASS", "FAIL"]
    report_hash: HashString

    @model_validator(mode="after")
    def validate_decision(self) -> SplitLeakageReportV4:
        should_pass = (
            self.patient_disjoint
            and self.record_disjoint
            and not self.patient_overlap_found
            and not self.record_overlap_found
            and self.known_group_201_202_respected
            and self.outer_folds_checked == 5
            and self.inner_folds_checked == 20
        )
        if (self.status == "PASS") != should_pass:
            raise ValueError("leakage report decision mismatch")
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        if hash_canonical(payload) != self.report_hash:
            raise ValueError("leakage report hash mismatch")
        return self


class FoldStatisticV4(FrozenModel):
    """One outer-fold composition row."""

    fold: int = Field(ge=1, le=5)
    train_samples: int = Field(ge=0)
    test_samples: int = Field(ge=0)
    train_patients: int = Field(ge=0)
    test_patients: int = Field(ge=0)
    train_records: int = Field(ge=0)
    test_records: int = Field(ge=0)
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    test_f_patients: int = Field(ge=0)
    contains_svdb_conservative_group_in_test: bool


class FoldStatisticsV4(FrozenModel):
    """Composition evidence for all outer folds."""

    schema_version: Literal["stage2-fold-statistics-v4.0"]
    rows: tuple[FoldStatisticV4, ...]
    statistics_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> FoldStatisticsV4:
        if len(self.rows) != 5:
            raise ValueError("fold statistics must contain five rows")
        payload = self.model_dump(mode="json", exclude={"statistics_hash"})
        if hash_canonical(payload) != self.statistics_hash:
            raise ValueError("fold statistics hash mismatch")
        return self


class PatientDisjointSplitManifestV4(FrozenModel):
    """Top-level binding for all patient-disjoint split evidence."""

    schema_version: Literal["stage2-patient-disjoint-split-bundle-v4.0"]
    split_version: Literal["v4.0-patient-disjoint"]
    algorithm: Literal["StratifiedGroupKFold"]
    outer_folds: int = Field(ge=2)
    inner_folds_per_outer: int = Field(ge=2)
    random_state: int
    source_manifest_hash: HashString
    dataset_binding_hash: HashString
    patient_mapping_hash: HashString
    outer_manifest_hash: HashString
    inner_manifest_hash: HashString
    leakage_report_hash: HashString
    fold_statistics_hash: HashString
    stage2_outer_adapter_hash: HashString
    stage2_inner_adapter_hash: HashString
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> PatientDisjointSplitManifestV4:
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("split bundle manifest hash mismatch")
        return self


class E07RPreauthorizationManifestV4(FrozenModel):
    """Strict autonomous-governance authorization for internal E07R work."""

    schema_version: Literal["e07r-preauthorization-v1"]
    stage: Literal["E07R"]
    date: Literal["2026-07-26"]
    governance_mode: Literal["AUTONOMOUS_PREAUTHORIZED"]
    responsible: Literal["AUTONOMOUS_GOVERNANCE_PREAUTH"]
    human_intervention_required: Literal[False]
    publication_authorized: Literal[False]
    model_promotion_authorized: Literal[False]
    gate_relaxation_authorized: Literal[False]
    preauthorized_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    source_physionet: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    source_quote_sha256: HashString
    known_patient_groups: tuple[dict[str, Any], ...]
    legacy_hashes: dict[str, HashString]
    new_artifact_hashes: dict[str, HashString]
    signature: dict[str, str]
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> E07RPreauthorizationManifestV4:
        required_actions = {
            "regenerate_versioned_stage2_custody_generation_from_r4_parent",
            "generate_versioned_patient_disjoint_outer_and_inner_splits",
            "create_freeze_manifest_and_write_guards",
            "rerun_e065pd_100_cells",
            "run_e07pd_150_cells_after_valid_e065pd",
        }
        if not required_actions.issubset(self.preauthorized_actions):
            raise ValueError("preauthorization omits required E07R actions")
        required_forbidden = {
            "external_publication",
            "write_or_promote_to_models",
            "relax_scientific_gates",
            "allow_patient_leakage",
            "overwrite_frozen_legacy_artifacts",
        }
        if not required_forbidden.issubset(self.forbidden_actions):
            raise ValueError("preauthorization omits required prohibitions")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("preauthorization manifest hash mismatch")
        return self


class E07RProducerAttestationV4(FrozenModel):
    """Additive attestation binding the r5 wrapper omitted by its internal hash."""

    schema_version: Literal["e07r-r5-producer-attestation-v1"]
    generation_id: Literal["v3.1.0-r5-stage2-pd"]
    custody_manifest_hash: HashString
    producer_file_sha256: dict[str, HashString]
    date: Literal["2026-07-26"]
    attestation_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> E07RProducerAttestationV4:
        required = {
            "scripts/build_stage2_patient_disjoint_v4.py",
            "src/stage2_research/stage2_custody.py",
            "src/stage2_research/e07r_contracts.py",
        }
        if not required.issubset(self.producer_file_sha256):
            raise ValueError("r5 producer attestation is incomplete")
        payload = self.model_dump(mode="json", exclude={"attestation_hash"})
        if hash_canonical(payload) != self.attestation_hash:
            raise ValueError("r5 producer attestation hash mismatch")
        return self


class E07REvidenceCompleteV4(FrozenModel):
    """Final cross-directory completion marker for E07R remediation evidence."""

    schema_version: Literal["e07r-evidence-complete-v4.0"]
    status: Literal["COMPLETE"]
    date: Literal["2026-07-26"]
    custody_manifest_hash: HashString
    patient_mapping_hash: HashString
    split_manifest_hash: HashString
    leakage_report_hash: HashString
    quarantine_manifest_hash: HashString
    preauthorization_manifest_hash: HashString
    pd_protocol_manifest_hash: HashString
    producer_attestation_hash: HashString
    artifact_file_sha256: dict[str, HashString]
    completion_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> E07REvidenceCompleteV4:
        if len(self.artifact_file_sha256) < 10:
            raise ValueError("E07R evidence completion inventory is incomplete")
        payload = self.model_dump(mode="json", exclude={"completion_hash"})
        if hash_canonical(payload) != self.completion_hash:
            raise ValueError("E07R evidence completion hash mismatch")
        return self


class E07RPDProtocolManifestV4(FrozenModel):
    """Frozen E06.5-PD/E07-PD data, hyperparameter, gate, and source contract."""

    schema_version: Literal["e07r-pd-protocol-v4.0"]
    status: Literal["FROZEN"]
    date: Literal["2026-07-26"]
    stage2_npz_sha256: HashString
    stage2_parquet_sha256: HashString
    full_npz_sha256: HashString
    full_parquet_sha256: HashString
    patient_mapping_hash: HashString
    split_manifest_hash: HashString
    candidates: tuple[Literal["baseline", "H6", "H11", "H12"], ...]
    samplers: tuple[
        Literal[
            "pd_s0_natural",
            "pd_s1_f_target",
            "pd_s2_patient_uniform_capped",
            "pd_s3_patient_sqrt_capped",
            "pd_s4_focal_gentle",
            "pd_s5_smote_feature",
        ],
        ...,
    ]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    profile: Literal["audit"]
    deterministic: Literal[True]
    device: Literal["cpu"]
    f_target_fraction: float
    patient_cap_multiplier: float
    f1_f_gate: float
    primary_target: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    source_file_sha256: dict[str, HashString]
    source_manifest_hash: HashString
    base_config_sha256: HashString
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_protocol(self) -> E07RPDProtocolManifestV4:
        if self.candidates != ("baseline", "H6", "H11", "H12"):
            raise ValueError("PD candidates are not canonical")
        if self.samplers != (
            "pd_s0_natural",
            "pd_s1_f_target",
            "pd_s2_patient_uniform_capped",
            "pd_s3_patient_sqrt_capped",
            "pd_s4_focal_gentle",
            "pd_s5_smote_feature",
        ):
            raise ValueError("PD samplers are not canonical")
        if self.folds != (1, 2, 3, 4, 5):
            raise ValueError("PD folds are not canonical")
        if self.seeds != (17, 29, 43, 71, 101):
            raise ValueError("PD seeds are not canonical")
        if self.f_target_fraction != 0.125 or self.patient_cap_multiplier != 2.0:
            raise ValueError("PD sampler targets changed")
        if self.f1_f_gate != 0.15 or self.primary_target != 0.50:
            raise ValueError("PD scientific gates changed")
        if self.bootstrap_repetitions != 10_000 or self.bootstrap_seed != 42:
            raise ValueError("PD bootstrap contract changed")
        if hash_canonical(self.source_file_sha256) != self.source_manifest_hash:
            raise ValueError("PD source manifest hash mismatch")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("PD protocol manifest hash mismatch")
        return self


class E065PDSelectionV4(FrozenModel):
    """Frozen H*-PD decision derived only after all 100 E06.5-PD cells."""

    schema_version: Literal["e06-5-pd-selection-v4.0"]
    status: Literal["VALID_H_STAR_PD", "NO_VALID_CANDIDATE"]
    selected_candidate: Literal["H6"] | None
    experiment_id: str = Field(min_length=1)
    completed_cells: Literal[100]
    protocol_manifest_hash: HashString
    source_done_hashes: dict[str, HashString]
    aggregate_metrics: dict[str, dict[str, float]]
    h6_minus_baseline_f1_f: float
    h6_minus_baseline_ci95: tuple[float, float]
    f1_f_gate: float
    primary_target: float
    decision_reasons: tuple[str, ...]
    model_reference: dict[str, str] | None
    selection_hash: HashString

    @model_validator(mode="after")
    def validate_selection(self) -> E065PDSelectionV4:
        if self.f1_f_gate != 0.15 or self.primary_target != 0.50:
            raise ValueError("E06.5-PD scientific gates changed")
        valid = self.status == "VALID_H_STAR_PD"
        if valid != (self.selected_candidate == "H6" and self.model_reference is not None):
            raise ValueError("E06.5-PD selection status/reference mismatch")
        payload = self.model_dump(mode="json", exclude={"selection_hash"})
        if hash_canonical(payload) != self.selection_hash:
            raise ValueError("E06.5-PD selection hash mismatch")
        return self


class E07PDResultV4(FrozenModel):
    """Complete 150-cell E07-PD scientific result without promotion authority."""

    schema_version: Literal["e07-pd-result-v4.0"]
    status: Literal["COMPLETE"]
    experiment_id: str = Field(min_length=1)
    completed_cells: Literal[150]
    protocol_manifest_hash: HashString
    h_star_selection_hash: HashString
    source_done_hashes: dict[str, HashString]
    aggregate_metrics: dict[str, dict[str, float]]
    comparisons_vs_s0: dict[str, dict[str, Any]]
    ranking: tuple[str, ...]
    selected_sampler: str | None
    f1_f_gate: float
    primary_target: float
    publication_authorized: Literal[False]
    model_promotion_authorized: Literal[False]
    result_hash: HashString

    @model_validator(mode="after")
    def validate_result(self) -> E07PDResultV4:
        required = {
            "pd_s0_natural",
            "pd_s1_f_target",
            "pd_s2_patient_uniform_capped",
            "pd_s3_patient_sqrt_capped",
            "pd_s4_focal_gentle",
            "pd_s5_smote_feature",
        }
        if set(self.aggregate_metrics) != required or set(self.ranking) != required:
            raise ValueError("E07-PD sampler result set is incomplete")
        if len(self.source_done_hashes) != 150:
            raise ValueError("E07-PD DONE evidence count is not 150")
        if self.f1_f_gate != 0.15 or self.primary_target != 0.50:
            raise ValueError("E07-PD scientific gates changed")
        if self.selected_sampler is not None and self.selected_sampler not in required:
            raise ValueError("E07-PD selected sampler is unknown")
        payload = self.model_dump(mode="json", exclude={"result_hash"})
        if hash_canonical(payload) != self.result_hash:
            raise ValueError("E07-PD result hash mismatch")
        return self


class E07RFreezePinV4(FrozenModel):
    """One content-addressed path in the E07R immutable baseline."""

    artifact_path: str = Field(min_length=1)
    sha256: HashString
    size_bytes: int = Field(ge=0)
    role: Literal[
        "CUSTODY",
        "IDENTITY",
        "SPLIT",
        "GOVERNANCE",
        "QUARANTINE",
        "SOURCE",
        "LEGACY_SENTINEL",
    ]
    enforce_read_only: bool

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("freeze artifact path must be project-relative")
        return value


class E07RFreezeManifestV4(FrozenModel):
    """Write-once E07R baseline binding data, identity, splits, governance, and code."""

    schema_version: Literal["e07r-freeze-v4.0"]
    stage: Literal["E07R"]
    status: Literal["FROZEN"]
    date: Literal["2026-07-26"]
    split_version: Literal["v4.0-patient-disjoint"]
    custody_manifest_hash: HashString
    patient_mapping_hash: HashString
    split_manifest_hash: HashString
    preauthorization_manifest_hash: HashString
    source_manifest_hash: HashString
    pins: tuple[E07RFreezePinV4, ...]
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> E07RFreezeManifestV4:
        if not self.pins:
            raise ValueError("freeze manifest must contain pins")
        paths = [pin.artifact_path for pin in self.pins]
        if len(paths) != len(set(paths)):
            raise ValueError("freeze pin paths are duplicated")
        required_roles = {
            "CUSTODY",
            "IDENTITY",
            "SPLIT",
            "GOVERNANCE",
            "QUARANTINE",
            "SOURCE",
            "LEGACY_SENTINEL",
        }
        roles = {pin.role for pin in self.pins}
        if roles != required_roles:
            raise ValueError("freeze manifest does not cover every required pin role")
        source_payload = {
            pin.artifact_path: pin.sha256 for pin in self.pins if pin.role == "SOURCE"
        }
        if not source_payload or hash_canonical(source_payload) != self.source_manifest_hash:
            raise ValueError("freeze source manifest hash mismatch")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("freeze manifest hash mismatch")
        return self


class E07RIntegrityCheckV4(FrozenModel):
    """One fail-closed preflight check."""

    code: str = Field(pattern=r"^[A-Z0-9_]+$")
    status: Literal["PASS", "BLOCKED"]
    evidence: str = Field(min_length=1)


class E07RPreflightReportV4(FrozenModel):
    """Auditable integrity result emitted before each E06.5-PD/E07-PD workflow."""

    schema_version: Literal["e07r-preflight-v4.0"]
    stage: Literal["E07R"]
    workflow: Literal["FREEZE_VALIDATION", "E06_5_PD", "E07_PD"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    split_version: Literal["v4.0-patient-disjoint"]
    freeze_manifest_hash: HashString
    checks: tuple[E07RIntegrityCheckV4, ...]
    status: Literal["PASS", "BLOCKED"]
    report_hash: HashString

    @model_validator(mode="after")
    def validate_report(self) -> E07RPreflightReportV4:
        expected_status = (
            "PASS" if self.checks and all(c.status == "PASS" for c in self.checks) else "BLOCKED"
        )
        if self.status != expected_status:
            raise ValueError("preflight status does not reflect checks")
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        if hash_canonical(payload) != self.report_hash:
            raise ValueError("preflight report hash mismatch")
        return self


class E07RIntegrityViolationV4(FrozenModel):
    """Append-only evidence for a blocked write or split-policy violation."""

    schema_version: Literal["e07r-integrity-violation-v4.0"]
    event_type: Literal[
        "FORBIDDEN_WRITE",
        "LEGACY_SPLIT_USE",
        "MODEL_PROMOTION_ATTEMPT",
        "FROZEN_ARTIFACT_MUTATION",
    ]
    timestamp_utc: str = Field(min_length=20)
    workflow: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempted_path: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    freeze_manifest_hash: HashString
    event_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> E07RIntegrityViolationV4:
        payload = self.model_dump(mode="json", exclude={"event_hash"})
        if hash_canonical(payload) != self.event_hash:
            raise ValueError("integrity violation event hash mismatch")
        return self


class FrozenArtifactReferenceV4(FrozenModel):
    """One immutable artifact referenced by quarantine or freeze evidence."""

    artifact_path: str = Field(min_length=1)
    sha256: HashString
    size_bytes: int = Field(ge=0)


class LegacySplitQuarantineManifestV4(FrozenModel):
    """Additive marker that makes record-disjoint splits inactive without deletion."""

    schema_version: Literal["legacy-split-quarantine-v4.0"]
    status: Literal["QUARANTINED_NOT_DELETED"]
    reason: Literal["PATIENT_LEAKAGE_RECORD_DISJOINT_ONLY"]
    date: Literal["2026-07-26"]
    active_for_e07r: Literal[False]
    replacement_split_version: Literal["v4.0-patient-disjoint"]
    artifacts: tuple[FrozenArtifactReferenceV4, ...]
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> LegacySplitQuarantineManifestV4:
        if not self.artifacts:
            raise ValueError("quarantine manifest must reference legacy artifacts")
        paths = [artifact.artifact_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("quarantine artifact paths are duplicated")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("quarantine manifest hash mismatch")
        return self


class Stage2CustodyManifestV4(FrozenModel):
    """Authorized ordered Stage 2 derivation from the immutable r4 parent."""

    schema_version: Literal["stage2-custody-v4.0"]
    generation_id: Literal["v3.1.0-r5-stage2-pd"]
    status: Literal["AUTHORIZED_FOR_E07R_INTERNAL_TRAINING"]
    parent_generation_id: Literal["advanced-training-v3.1.0-r4"]
    parent_npz_path: str
    parent_npz_sha256: HashString
    parent_parquet_path: str
    parent_parquet_sha256: HashString
    parent_ordered_binding: Literal["PASS"]
    derivation: Literal["ordered_filter_labels_S_V_F_datasets_incart_mitdb"]
    confirmatory_datasets: tuple[Literal["incart", "mitdb"], ...]
    excluded_dataset_counts: dict[str, int]
    excluded_label_counts: dict[str, int]
    row_count: int = Field(ge=1)
    class_counts: dict[str, int]
    record_count: int = Field(ge=1)
    signal_shape: tuple[int, int, int]
    signal_dtype: Literal["float32"]
    sample_id_sequence_hash: HashString
    waveform_hash_sequence_hash: HashString
    output_file_sha256: dict[str, HashString]
    output_size_bytes: dict[str, int]
    output_ordered_binding: Literal["PASS"]
    source_commit: str
    source_manifest_hash: HashString
    created_at: Literal["2026-07-26"]
    manifest_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> Stage2CustodyManifestV4:
        if self.confirmatory_datasets != ("incart", "mitdb"):
            raise ValueError("custody datasets must be exactly INCART+MITDB")
        if set(self.class_counts) != {"S", "V", "F"}:
            raise ValueError("custody classes must be exactly S/V/F")
        if sum(self.class_counts.values()) != self.row_count:
            raise ValueError("custody class counts do not sum to row_count")
        if self.signal_shape != (self.row_count, 500, 1):
            raise ValueError("custody signal shape mismatch")
        expected_outputs = {"stage2_multiclass.npz", "stage2_multiclass.parquet"}
        if set(self.output_file_sha256) != expected_outputs:
            raise ValueError("custody output hash keys are not exact")
        if set(self.output_size_bytes) != expected_outputs:
            raise ValueError("custody output size keys are not exact")
        if any(size <= 0 for size in self.output_size_bytes.values()):
            raise ValueError("custody output sizes must be positive")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        if hash_canonical(payload) != self.manifest_hash:
            raise ValueError("custody manifest hash mismatch")
        return self


class Stage2CustodyCompleteV4(FrozenModel):
    """Completion marker binding the final manifest bytes and outputs."""

    schema_version: Literal["stage2-custody-complete-v4.0"]
    generation_id: Literal["v3.1.0-r5-stage2-pd"]
    status: Literal["COMPLETE"]
    manifest_hash: HashString
    manifest_file_sha256: HashString
    artifact_sha256: dict[str, HashString]
    completed_at: Literal["2026-07-26"]
    marker_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> Stage2CustodyCompleteV4:
        expected_outputs = {
            "stage2_multiclass.npz",
            "stage2_multiclass.parquet",
            "stage2_custody_manifest.json",
        }
        if set(self.artifact_sha256) != expected_outputs:
            raise ValueError("custody completion artifact keys are not exact")
        payload = self.model_dump(mode="json", exclude={"marker_hash"})
        if hash_canonical(payload) != self.marker_hash:
            raise ValueError("custody completion marker hash mismatch")
        return self


class MitdbPatientMappingDocument(FrozenModel):
    """Required official MIT-BIH mapping evidence document."""

    dataset: Literal["MIT-BIH Arrhythmia Database"]
    source: Literal["PhysioNet official documentation"]
    source_url: str
    access_date: Literal["2026-07-26"]
    mapping_policy: Literal["official_evidence_required"]
    default_rule: Literal["record_id_equals_patient_id_unless_official_evidence"]
    patient_groups: tuple[dict[str, Any], ...]
    offline_provenance: bool
    cached_source_path: str
    mapping_hash: HashString

    @model_validator(mode="after")
    def validate_hash(self) -> MitdbPatientMappingDocument:
        payload = self.model_dump(mode="json", exclude={"mapping_hash"})
        if hash_canonical(payload) != self.mapping_hash:
            raise ValueError("MIT-BIH mapping hash mismatch")
        return self
