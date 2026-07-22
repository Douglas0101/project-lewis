"""Validated contracts for the canonical Stage 2 research CLI."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.e06_protocol import E06EvaluationContract


class ExitCode(IntEnum):
    """Process exit codes defined by the Stage 2 research contract."""

    PASS = 0
    ARGUMENT_ERROR = 2
    BLOCKED_PRECONDITION = 3
    REGRESSION = 4
    INVALID_EXPERIMENT = 5
    DATA_INTEGRITY = 6
    LEAKAGE = 7
    INCOMPATIBLE_ARTIFACT = 8
    INTERRUPTED_RESUMABLE = 9
    TRAINING_FAILURE = 10
    EVALUATION_FAILURE = 11
    SCIENTIFIC_GATE_NOT_MET = 12


class ResearchError(RuntimeError):
    """Expected CLI failure carrying a stable exit code."""

    def __init__(
        self,
        message: str,
        exit_code: ExitCode,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.details = details or {}


HashString = str
StageName = Literal["e06.5", "fold-audit", "e07", "e08", "all"]
ProfileName = Literal["smoke", "screening", "audit", "performance"]
CandidateName = Literal["baseline", "H6", "H11", "H12"]
SamplerName = Literal[
    "natural",
    "random_oversampling",
    "patient_uniform",
    "patient_sqrt",
    "smote",
]
REQUIRED_RUN_ARTIFACTS = (
    "environment.json",
    "config_resolved.json",
    "metrics.json",
    "predictions.parquet",
    "confusion_matrix.json",
    "group_metrics.csv",
    "training_history.csv",
    "checkpoint.keras",
    "stdout.log",
    "stderr.log",
    "checkpoint.md",
    "preprocessing_manifest.json",
    "sampling_manifest.json",
    "method_manifest.json",
)

MethodName = Literal[
    "ce_control",
    "crt_patient_aware",
    "logit_adjustment",
    "balanced_softmax",
    "ldam_drw",
    "focal_legacy",
]


class FrozenModel(BaseModel):
    """Strict immutable base model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HashedPath(FrozenModel):
    """A project-relative input with an immutable SHA-256 expectation."""

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetConfig(FrozenModel):
    """All immutable Stage 2 inputs."""

    stage2_npz: HashedPath
    stage2_parquet: HashedPath
    full_npz: HashedPath
    full_parquet: HashedPath


class CandidateConfig(FrozenModel):
    """Frozen representation candidate."""

    name: CandidateName
    fusion_template_count: int = Field(ge=0, le=32)
    feature_families: tuple[str, ...]
    complexity_rank: int = Field(ge=0)
    rr_context_source: Literal["stage2_filtered"] = "stage2_filtered"

    @model_validator(mode="after")
    def validate_candidate(self) -> CandidateConfig:
        expected = {"baseline": 0, "H6": 8, "H11": 16, "H12": 24}
        if self.fusion_template_count != expected[self.name]:
            raise ValueError(f"{self.name} requires fusion_template_count={expected[self.name]}")
        if self.name == "baseline" and self.feature_families != ("base16",):
            raise ValueError("baseline must contain only base16")
        if self.name != "baseline" and self.feature_families != (
            "base16",
            "causal_rr_h3",
            "class_templates_h5",
        ):
            raise ValueError(f"{self.name} feature family contract changed")
        return self


class ProfileConfig(FrozenModel):
    """Execution budget and determinism profile."""

    max_epochs: int = Field(ge=1)
    patience: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    deterministic: bool
    max_parallel: int = Field(ge=1)
    publication_eligible: bool


class GateConfig(FrozenModel):
    """Research and publication gates."""

    publication_f1_f: float = Field(default=0.50, ge=0.0, le=1.0)
    material_gain_outside_208_213: float = Field(default=0.05, ge=0.0)
    minimum_macro_f1: float = Field(default=0.45, ge=0.0, le=1.0)
    research_baseline_f1_f: float = Field(default=0.18, ge=0.0, le=1.0)
    research_candidate_f1_f: float = Field(default=0.25, ge=0.0, le=1.0)
    architecture_candidate_f1_f: float = Field(default=0.30, ge=0.0, le=1.0)


class E07Config(FrozenModel):
    """Frozen sampling-stage matrix."""

    samplers: tuple[SamplerName, ...]
    screening_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    smote_k_neighbors: int = Field(default=5, ge=1)


class E08Config(FrozenModel):
    """Frozen classifier/long-tail matrix."""

    methods: tuple[MethodName, ...]
    screening_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    logit_adjustment_tau: float = Field(default=1.0, ge=0.0)
    ldam_max_margin: float = Field(default=0.5, gt=0.0)
    ldam_drw_epoch_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    focal_alpha: tuple[float, float, float] = (0.20, 0.15, 3.00)
    focal_gamma: float = Field(default=2.0, ge=0.0)
    focal_class_weight: tuple[float, float, float] = (1.0, 1.0, 8.0)


class ResourceConfig(FrozenModel):
    """Preflight resource assumptions."""

    minimum_free_disk_gib: float = Field(default=2.0, gt=0.0)
    estimated_mib_per_run: float = Field(default=2.0, gt=0.0)


class ResearchConfig(FrozenModel):
    """Top-level canonical Stage 2 research configuration."""

    schema_version: Literal["stage2-research-v1"] = "stage2-research-v1"
    project_root: Path = Path(".")
    output_root: Path = Path("experiments/stage2_v2.4_research")
    datasets: DatasetConfig
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: dict[CandidateName, CandidateConfig]
    folds: tuple[int, ...] = (1, 2, 3, 4, 5)
    seeds: tuple[int, ...] = (17, 29, 43, 71, 101)
    profiles: dict[ProfileName, ProfileConfig]
    gates: GateConfig = GateConfig()
    e07: E07Config
    e08: E08Config
    resources: ResourceConfig = ResourceConfig()
    split_contract: E06EvaluationContract = E06EvaluationContract()
    default_experiment_ids: dict[str, str]
    validation_commands: tuple[tuple[str, ...], ...] = ()

    @model_validator(mode="after")
    def validate_matrix(self) -> ResearchConfig:
        if tuple(sorted(set(self.folds))) != self.folds or self.folds != (1, 2, 3, 4, 5):
            raise ValueError("folds must be exactly (1,2,3,4,5)")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("seeds must be non-empty and unique")
        required_candidates = {"baseline", "H6", "H11", "H12"}
        if set(self.candidates) != required_candidates:
            raise ValueError("candidate matrix must contain baseline,H6,H11,H12")
        for key, candidate in self.candidates.items():
            if key != candidate.name:
                raise ValueError(f"candidate key/name mismatch for {key}")
        for required_profile in ("smoke", "screening", "audit", "performance"):
            if required_profile not in self.profiles:
                raise ValueError(f"missing profile: {required_profile}")
        audit = self.profiles["audit"]
        if not audit.deterministic or audit.max_parallel != 1:
            raise ValueError("audit profile must be deterministic with max_parallel=1")
        if self.profiles["smoke"].publication_eligible:
            raise ValueError("smoke profile cannot be publication eligible")
        required_ids = {
            "e065_smoke",
            "e065_audit",
            "e07_screening",
            "e07_audit",
            "e08_screening",
            "e08_audit",
        }
        if set(self.default_experiment_ids) != required_ids:
            raise ValueError("default_experiment_ids are incomplete")
        return self


class EnvironmentManifest(BaseModel):
    """Runtime identity recorded for every run."""

    model_config = ConfigDict(extra="forbid")

    python_version: str
    tensorflow_version: str
    keras_version: str
    numpy_version: str
    platform: str
    device: str
    deterministic_requested: bool
    deterministic_enabled: bool
    pythonhashseed: str
    numpy_seed: int
    tensorflow_seed: int
    keras_seed: int
    split_random_state: int
    sampler_random_state: int
    started_monotonic_seconds: float


RunStatus = Literal["RUNNING", "PASS", "FAILED", "INTERRUPTED"]


class RunManifest(BaseModel):
    """Required immutable identity plus mutable lifecycle fields for one cell."""

    model_config = ConfigDict(extra="forbid")

    experiment_stage: str
    experiment_id: str
    candidate: str
    fold: int = Field(ge=1, le=5)
    seed: int = Field(ge=0)
    model_seed: int = Field(ge=0)
    git_head: str
    git_dirty: bool
    dataset_manifest_hash: HashString
    split_manifest_hash: HashString
    feature_manifest_hash: HashString
    config_hash: HashString
    preflight_hash: HashString
    source_manifest_hash: HashString
    runtime_identity_hash: HashString
    uv_lock_hash: HashString
    python_version: str
    tensorflow_version: str
    keras_version: str
    device: str
    deterministic: bool
    sampling: str
    loss: str
    architecture: str
    early_stopping_source: Literal["inner_validation"] = "inner_validation"
    outer_test_used_for_selection: Literal[False] = False
    started_at: str
    finished_at: str
    status: RunStatus
    profile: ProfileName
    publication_eligible: bool
    split_random_state: int
    sampler_random_state: int
    artifact_hashes: dict[str, HashString] = Field(default_factory=dict)


class DoneMarker(FrozenModel):
    """Completion marker created only after artifact verification."""

    run_manifest_hash: HashString
    config_hash: HashString
    completed_at: str
    artifact_hashes: dict[str, HashString]


class SplitPartition(FrozenModel):
    """One frozen split partition with evidence."""

    indices: tuple[int, ...]
    groups: tuple[str, ...]
    class_counts: dict[str, int]
    f_208: int
    f_213: int
    f_outside_208_213: int
    indices_hash: HashString
    groups_hash: HashString


class OuterFoldManifest(FrozenModel):
    """One outer fold."""

    fold: int = Field(ge=1, le=5)
    train: SplitPartition
    test: SplitPartition
    overlap_groups: tuple[str, ...] = ()


class InnerFoldManifest(FrozenModel):
    """One nested inner train/validation split."""

    fold: int = Field(ge=1, le=5)
    train: SplitPartition
    validation: SplitPartition
    outer_test_groups: tuple[str, ...]
    train_validation_overlap: tuple[str, ...] = ()
    validation_outer_test_overlap: tuple[str, ...] = ()


class SplitManifest(FrozenModel):
    """Canonical persisted split collection."""

    schema_version: Literal["stage2-splits-v2.4"] = "stage2-splits-v2.4"
    dataset_manifest_hash: HashString
    splitter: Literal["StratifiedGroupKFold"] = "StratifiedGroupKFold"
    split_random_state: int
    outer_folds: tuple[OuterFoldManifest, ...]
    manifest_hash: HashString


class InnerSplitManifest(FrozenModel):
    """Canonical persisted inner split collection."""

    schema_version: Literal["stage2-inner-splits-v2.4"] = "stage2-inner-splits-v2.4"
    dataset_manifest_hash: HashString
    outer_split_manifest_hash: HashString
    split_random_state: int
    inner_folds: tuple[InnerFoldManifest, ...]
    manifest_hash: HashString


class RunCell(FrozenModel):
    """One deterministic stage/candidate/fold/seed run cell."""

    stage: str
    experiment_id: str
    candidate: str
    fold: int = Field(ge=1, le=5)
    seed: int = Field(ge=0)
    profile: ProfileName
    run_dir: Path
    status: Literal["PLANNED", "RESUMABLE", "DONE", "INCOMPATIBLE"]
    dependencies: tuple[str, ...] = ()


class SelectionManifest(FrozenModel):
    """Frozen stage selection."""

    stage: str
    selected_name: str
    selected_feature_manifest_hash: str
    selection_policy_hash: str
    source_experiment_id: str
    metrics: dict[str, Any]
    created_at: str
    manifest_hash: HashString
