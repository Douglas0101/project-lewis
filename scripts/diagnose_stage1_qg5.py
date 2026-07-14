"""Generate a read-only forensic trace of the QG5 Stage 1 inference path.

The diagnostic is intentionally disabled unless ``STAGE1_DIAGNOSTIC=1``.
It never writes model, scaler, threshold, dataset, or full ECG signal values.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.two_stage_pipeline import (  # noqa: E402
    TwoStageInferencePipeline,
)

MAX_SAMPLES_PER_STAGE = 2048
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "baseline"
TRACE_PATH = OUTPUT_DIR / "stage1_qg5_trace.json"
SAMPLES_PATH = OUTPUT_DIR / "stage1_qg5_samples.csv"


class StrictModel(BaseModel):
    """Forbid accidental, undocumented fields in forensic artifacts."""

    model_config = ConfigDict(extra="forbid")


class ArrayStats(StrictModel):
    """Technical statistics for one pipeline transformation."""

    file: str
    function: str
    line: int
    input_shape: list[int] | None
    output_shape: list[int]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    std: float
    nan_count: int
    inf_count: int


class ArtifactRef(StrictModel):
    """Immutable identity of an artifact used during inference."""

    path: str
    sha256: str


class SourceRef(StrictModel):
    """Immutable identity and available contract of a source NPZ."""

    path: str
    sha256: str
    keys: list[str]
    has_record_id: bool
    has_group_id: bool


class Confusion(StrictModel):
    """Manual binary confusion matrix for 1=Anormal."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


class IntegrityChecks(StrictModel):
    """Alignment and cardinality checks for the selected subset."""

    stable_sample_ids: bool
    duplicate_sample_ids: int
    duplicate_selected_sources: int
    samples_selected: int
    targets: int
    outputs: int
    predictions: int
    order_preserved: bool
    samples_lost: int
    canonical_score_max_abs_delta: float
    canonical_threshold_matches: bool
    canonical_decision_disagreement: int


class Stage1Trace(StrictModel):
    """Validated top-level schema of the Stage 1 trace."""

    schema_version: str = Field(pattern=r"^1\.1$")
    diagnostic_only: bool
    positive_label: int
    positive_name: str
    selection_method: str
    model: ArtifactRef
    scaler: ArtifactRef
    threshold_artifact: ArtifactRef
    threshold_used: float
    sources: list[SourceRef]
    transformations: list[ArrayStats]
    confusion: Confusion
    recall_abnormal: float
    precision_abnormal: float
    f1_abnormal: float
    specificity: float
    false_negative_rate: float
    balanced_accuracy: float
    average_precision: float
    positive_prevalence: float
    ap_lift: float
    ap_lift_interpretation: str
    predicted_abnormal_rate: float
    minimum_true_positives_for_gate: int
    additional_true_positives_needed: int
    integrity: IntegrityChecks
    provenance_warnings: list[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _line(function: Any) -> int:
    try:
        return int(inspect.getsourcelines(function)[1])
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Unable to locate source line for {function!r}") from error


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not an integer") from error


def _to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except (TypeError, ValueError) as error:
        raise ValueError("Unable to calculate average precision") from error


def ap_reference(average_precision: float, y_true: np.ndarray) -> tuple[float, float, str]:
    """Compare AP with the positive prevalence baseline without approving a model."""
    positive_prevalence = _to_float(np.mean(y_true == 1), "positive prevalence")
    ap_lift = average_precision - positive_prevalence
    if ap_lift > 1e-12:
        interpretation = "ABOVE_PREVALENCE_REFERENCE"
    elif ap_lift < -1e-12:
        interpretation = "BELOW_PREVALENCE_REFERENCE"
    else:
        interpretation = "NEAR_PREVALENCE_REFERENCE"
    return positive_prevalence, ap_lift, interpretation


def _stats(
    array: np.ndarray,
    *,
    file: Path,
    function: str,
    line: int,
    input_shape: tuple[int, ...] | None,
) -> ArrayStats:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{function} produced no finite values")
    try:
        return ArrayStats(
            file=str(file.resolve()),
            function=function,
            line=line,
            input_shape=list(input_shape) if input_shape is not None else None,
            output_shape=list(values.shape),
            dtype=str(values.dtype),
            minimum=float(finite.min()),
            maximum=float(finite.max()),
            mean=float(finite.mean()),
            std=float(finite.std()),
            nan_count=int(np.isnan(values).sum()),
            inf_count=int(np.isinf(values).sum()),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unable to summarize transformation {function}") from error


def _select_qg5_subset() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[SourceRef]]:
    """Reproduce the QG5 selection while retaining stable source indices."""
    stage1_path = PROJECT_ROOT / "data" / "features" / "stage1_binary.npz"
    stage2_path = PROJECT_ROOT / "data" / "features" / "stage2_multiclass.npz"
    sources: list[SourceRef] = []

    with (
        np.load(stage1_path, allow_pickle=False) as stage1,
        np.load(stage2_path, allow_pickle=False) as stage2,
    ):
        for path, data in ((stage1_path, stage1), (stage2_path, stage2)):
            sources.append(
                SourceRef(
                    path=str(path.resolve()),
                    sha256=_sha256_file(path),
                    keys=sorted(data.files),
                    has_record_id="record_id" in data.files,
                    has_group_id="group_id" in data.files,
                )
            )

        n_normal = max(1, MAX_SAMPLES_PER_STAGE // 16)
        n_per_abnormal = max(1, (MAX_SAMPLES_PER_STAGE - n_normal) // 3)
        selections = [
            ("stage1_binary", "N", 0, stage1, np.where(stage1["y"] == 0)[0][:n_normal]),
            ("stage2_multiclass", "S", 1, stage2, np.where(stage2["y"] == 0)[0][:n_per_abnormal]),
            ("stage2_multiclass", "V", 2, stage2, np.where(stage2["y"] == 1)[0][:n_per_abnormal]),
            ("stage2_multiclass", "F", 3, stage2, np.where(stage2["y"] == 2)[0][:n_per_abnormal]),
        ]

        arrays: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for dataset, original_label, integrated_label, data, indices in selections:
            arrays.append(data["X"][indices])
            labels.append(np.full(len(indices), integrated_label, dtype=np.int64))
            for source_index in indices.tolist():
                try:
                    stable_index = int(source_index)
                except (TypeError, ValueError) as error:
                    raise ValueError("NPZ source index is not an integer") from error
                metadata.append(
                    {
                        "sample_id": f"{dataset}:{original_label}:{stable_index}",
                        "record_id": None,
                        "group_id": None,
                        "dataset": dataset,
                        "source_index": stable_index,
                        "true_label_original": original_label,
                    }
                )

        x_values = np.concatenate(arrays, axis=0).astype(np.float32)
        y_values = np.concatenate(labels)
        permutation = np.random.default_rng(42).permutation(len(y_values))
        try:
            shuffled_metadata = [metadata[int(index)] for index in permutation]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("QG5 permutation does not align with metadata") from error
        return x_values[permutation], y_values[permutation], shuffled_metadata, sources


def _safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(OUTPUT_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Diagnostic output escapes the approved directory: {resolved}") from error
    return resolved


def _write_csv_atomic(rows: list[dict[str, Any]]) -> None:
    output = _safe_output_path(SAMPLES_PATH)
    temporary = _safe_output_path(output.with_suffix(output.suffix + ".tmp"))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def _write_json_atomic(trace: Stage1Trace) -> None:
    output = _safe_output_path(TRACE_PATH)
    temporary = _safe_output_path(output.with_suffix(output.suffix + ".tmp"))
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(trace.model_dump_json(indent=2) + "\n")
    temporary.replace(output)


def generate_trace() -> Stage1Trace:
    """Execute the Stage 1 diagnostic without changing inference decisions."""
    x_raw, y_aami, metadata, sources = _select_qg5_subset()
    pipeline = TwoStageInferencePipeline.from_directory(
        PROJECT_ROOT / "models", use_quantized=False
    ).load()
    if pipeline.stage1_model is None:
        raise RuntimeError("Stage 1 model was not loaded")

    x_scaled = pipeline._normalize(x_raw, pipeline.stage1_scaler)
    raw_output = pipeline._forward(pipeline.stage1_model, x_scaled)
    if raw_output.ndim != 2 or raw_output.shape[1] != 2:
        raise ValueError(f"Expected Stage 1 output shape (n, 2), received {raw_output.shape}")

    interpreted_probability = raw_output[:, 1]
    try:
        threshold = float(pipeline.stage1_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 1 threshold is not numeric") from error
    y_pred = (interpreted_probability >= threshold).astype(np.int64)
    y_true = (y_aami != 0).astype(np.int64)

    canonical_result = pipeline.predict(x_raw)
    canonical_scores = np.asarray(canonical_result["stage1_score"], dtype=np.float64)
    canonical_pred = np.asarray(
        [_to_int(label != "N", "canonical prediction") for label in canonical_result["class"]],
        dtype=np.int64,
    )
    canonical_delta = np.abs(canonical_scores - interpreted_probability)

    try:
        true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
        false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
        false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
        true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    except (TypeError, ValueError) as error:
        raise ValueError("Unable to calculate the manual confusion matrix") from error
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    false_negative_rate = false_negative / (true_positive + false_negative)
    balanced_accuracy = (recall + specificity) / 2
    average_precision = _average_precision(y_true, interpreted_probability)
    positive_prevalence, ap_lift, ap_lift_interpretation = ap_reference(average_precision, y_true)
    minimum_true_positives = math.ceil(0.30 * (true_positive + false_negative))
    additional_true_positives = max(0, minimum_true_positives - true_positive)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(metadata):
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "record_id": "",
                "group_id": "",
                "dataset": sample["dataset"],
                "source_index": sample["source_index"],
                "true_label_original": sample["true_label_original"],
                "true_label_stage1": _to_int(y_true[index], "true Stage 1 label"),
                "raw_feature_hash": _sha256_array(x_raw[index]),
                "scaled_feature_hash": _sha256_array(x_scaled[index]),
                "raw_output": json.dumps(raw_output[index].tolist(), separators=(",", ":")),
                "interpreted_probability": repr(
                    _to_float(interpreted_probability[index], "interpreted probability")
                ),
                "threshold": repr(threshold),
                "predicted_label": _to_int(y_pred[index], "predicted Stage 1 label"),
                "correct": bool(y_pred[index] == y_true[index]),
            }
        )

    pipeline_file = PROJECT_ROOT / "src" / "inference" / "two_stage_pipeline.py"
    test_file = PROJECT_ROOT / "tests" / "test_two_stage_qg5.py"
    transformations = [
        _stats(
            x_raw,
            file=test_file,
            function="_load_combined_test_subset",
            line=44,
            input_shape=None,
        ),
        _stats(
            x_scaled,
            file=pipeline_file,
            function="TwoStageInferencePipeline._normalize",
            line=_line(TwoStageInferencePipeline._normalize),
            input_shape=x_raw.shape,
        ),
        _stats(
            raw_output,
            file=pipeline_file,
            function="TwoStageInferencePipeline._forward",
            line=_line(TwoStageInferencePipeline._forward),
            input_shape=x_scaled.shape,
        ),
        _stats(
            interpreted_probability,
            file=pipeline_file,
            function="TwoStageInferencePipeline._run_stage1 score_anormal",
            line=_line(TwoStageInferencePipeline._run_stage1),
            input_shape=raw_output.shape,
        ),
        _stats(
            y_pred,
            file=pipeline_file,
            function="TwoStageInferencePipeline._run_stage1 threshold decision",
            line=_line(TwoStageInferencePipeline._run_stage1),
            input_shape=interpreted_probability.shape,
        ),
    ]

    sample_ids = [str(row["sample_id"]) for row in rows]
    selected_sources = [
        (str(row["dataset"]), _to_int(row["source_index"], "selected source index")) for row in rows
    ]
    model_path = Path(pipeline.stage1_model_path).resolve()
    scaler_path = Path(pipeline.stage1_scaler_path).resolve()
    if pipeline.stage1_threshold_path is None:
        raise ValueError("Stage 1 threshold artifact path is missing")
    threshold_path = Path(pipeline.stage1_threshold_path).resolve()
    trace = Stage1Trace(
        schema_version="1.1",
        diagnostic_only=True,
        positive_label=1,
        positive_name="Anormal",
        selection_method=(
            "Exact deterministic QG5 selection: first 128 N and first 640 each of S/V/F, "
            "then numpy default_rng(42) permutation"
        ),
        model=ArtifactRef(path=str(model_path), sha256=_sha256_file(model_path)),
        scaler=ArtifactRef(path=str(scaler_path), sha256=_sha256_file(scaler_path)),
        threshold_artifact=ArtifactRef(
            path=str(threshold_path), sha256=_sha256_file(threshold_path)
        ),
        threshold_used=threshold,
        sources=sources,
        transformations=transformations,
        confusion=Confusion(
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
        ),
        recall_abnormal=recall,
        precision_abnormal=precision,
        f1_abnormal=f1,
        specificity=specificity,
        false_negative_rate=false_negative_rate,
        balanced_accuracy=balanced_accuracy,
        average_precision=average_precision,
        positive_prevalence=positive_prevalence,
        ap_lift=ap_lift,
        ap_lift_interpretation=ap_lift_interpretation,
        predicted_abnormal_rate=_to_float(y_pred.mean(), "predicted abnormal rate"),
        minimum_true_positives_for_gate=minimum_true_positives,
        additional_true_positives_needed=additional_true_positives,
        integrity=IntegrityChecks(
            stable_sample_ids=True,
            duplicate_sample_ids=len(sample_ids) - len(set(sample_ids)),
            duplicate_selected_sources=len(selected_sources) - len(set(selected_sources)),
            samples_selected=len(x_raw),
            targets=len(y_true),
            outputs=len(raw_output),
            predictions=len(y_pred),
            order_preserved=True,
            samples_lost=len(x_raw) - len(raw_output),
            canonical_score_max_abs_delta=_to_float(
                canonical_delta.max(), "canonical score maximum absolute delta"
            ),
            canonical_threshold_matches=bool(canonical_result["stage1_threshold"] == threshold),
            canonical_decision_disagreement=_to_int(
                (canonical_pred != y_pred).sum(), "canonical decision disagreement"
            ),
        ),
        provenance_warnings=[
            "Source NPZ files contain no record_id or group_id arrays; those CSV fields "
            "are blank.",
            "Stable sample_id is derived from dataset, original class, and immutable " "NPZ index.",
            "QG5 subset uses first class occurrences and is enriched to 93.75% abnormal; "
            "it is not a patient-wise generalization estimate.",
            "raw_feature_hash refers to the complete 500x1 model input; no ECG samples "
            "are saved.",
            "Interpreted probability is output column 1 as implemented; activation "
            "semantics remain to be proven in R02/R07.",
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(rows)
    _write_json_atomic(trace)
    return trace


def main() -> int:
    """Run only under explicit diagnostic opt-in."""
    if os.environ.get("STAGE1_DIAGNOSTIC") != "1":
        print("STAGE1_DIAGNOSTIC is not 1; no diagnostic artifacts were written.")
        return 2
    trace = generate_trace()
    print(f"Wrote {TRACE_PATH}")
    print(f"Wrote {SAMPLES_PATH}")
    print(
        "Stage 1: "
        f"recall={trace.recall_abnormal:.10f} "
        f"precision={trace.precision_abnormal:.10f} "
        f"threshold={trace.threshold_used:.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
