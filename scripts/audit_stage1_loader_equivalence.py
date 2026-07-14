"""Orchestrate independent R03 loader lanes and compare their outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
THRESHOLD = 0.5800000000000001
LANE_SCRIPT = PROJECT_ROOT / "scripts" / "stage1_loader_lane.py"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionComparison(StrictModel):
    prediction_shape_reference: list[int]
    prediction_shape_helper: list[int]
    prediction_dtype_reference: str
    prediction_dtype_helper: str
    nan_count_reference: int
    nan_count_helper: int
    inf_count_reference: int
    inf_count_helper: int
    max_abs_prediction_delta: float
    mean_abs_prediction_delta: float
    p99_abs_prediction_delta: float
    argmax_disagreement_count: int
    threshold_0_58_disagreement_count: int
    compile_true_max_abs_prediction_delta: float
    compile_true_mean_abs_prediction_delta: float
    compile_true_argmax_disagreement_count: int
    compile_true_threshold_disagreement_count: int
    compile_optimizer_restored: bool
    compile_optimizer_class: str | None
    compile_loss: str | None
    structural_equivalence: bool
    weight_tensor_count_reference: int
    weight_tensor_count_helper: int
    all_weight_shapes_equal: bool
    all_weight_dtypes_equal: bool
    all_weights_array_equal: bool
    max_abs_weight_delta: float
    criteria_passed: bool


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


def _run_lane(lane: str, fixture: Path, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / f"{lane}_loader_result.json"
    arrays_path = output_dir / f"{lane}_lane_arrays.npz"
    if result_path.exists() or arrays_path.exists():
        raise FileExistsError(f"R03 {lane} lane evidence already exists; refusing overwrite")
    command = [
        sys.executable,
        str(LANE_SCRIPT),
        "--lane",
        lane,
        "--model",
        str(MODEL_PATH),
        "--fixture",
        str(fixture),
        "--result",
        str(result_path),
        "--arrays",
        str(arrays_path),
    ]
    environment = os.environ.copy()
    environment["TF_USE_LEGACY_KERAS"] = "0"
    environment["KERAS_BACKEND"] = "tensorflow"
    process = subprocess.run(command, text=True, capture_output=True, env=environment)
    (output_dir / f"{lane}_lane_process.txt").write_text(
        process.stdout + process.stderr + f"\n[exit_code={process.returncode}]\n",
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(f"R03 {lane} lane failed; see process artifact")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid R03 {lane} lane result") from error
    if not isinstance(result, dict):
        raise ValueError(f"R03 {lane} lane result is not an object")
    return result


def _weight_rows(
    reference_result: dict[str, Any],
    helper_result: dict[str, Any],
    reference_arrays: Any,
    helper_arrays: Any,
) -> tuple[list[dict[str, Any]], float, bool, bool, bool]:
    reference_metadata = reference_result["structure"]["weight_tensors"]
    helper_metadata = helper_result["structure"]["weight_tensors"]
    rows: list[dict[str, Any]] = []
    max_delta_all = 0.0
    all_shapes = len(reference_metadata) == len(helper_metadata)
    all_dtypes = all_shapes
    all_equal = all_shapes
    for index, (reference, helper) in enumerate(
        zip(reference_metadata, helper_metadata, strict=True)
    ):
        key = f"weight_{index:03d}"
        left = reference_arrays[key]
        right = helper_arrays[key]
        shape_equal = left.shape == right.shape
        dtype_equal = left.dtype == right.dtype
        if shape_equal:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            max_delta = _to_float(delta.max(initial=0.0), "maximum weight delta")
            mean_delta = _to_float(delta.mean(), "mean weight delta") if delta.size else 0.0
            equal = bool(np.array_equal(left, right))
            close = bool(np.allclose(left, right, rtol=0.0, atol=1e-7))
        else:
            max_delta = math.inf
            mean_delta = math.inf
            equal = False
            close = False
        max_delta_all = max(max_delta_all, max_delta)
        all_shapes = all_shapes and shape_equal
        all_dtypes = all_dtypes and dtype_equal
        all_equal = all_equal and equal
        rows.append(
            {
                "tensor_index": index,
                "layer_name": reference["layer_name"],
                "shape_reference": json.dumps(reference["shape"]),
                "shape_helper": json.dumps(helper["shape"]),
                "dtype_reference": reference["dtype"],
                "dtype_helper": helper["dtype"],
                "array_equal": equal,
                "allclose": close,
                "max_abs_delta": max_delta,
                "mean_abs_delta": mean_delta,
            }
        )
    return rows, max_delta_all, all_shapes, all_dtypes, all_equal


def _delta_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    if left.shape != right.shape:
        return math.inf, math.inf, math.inf
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return (
        _to_float(delta.max(initial=0.0), "maximum prediction delta"),
        _to_float(delta.mean(), "mean prediction delta") if delta.size else 0.0,
        _to_float(np.percentile(delta, 99), "p99 prediction delta") if delta.size else 0.0,
    )


def audit(fixture: Path, output_dir: Path) -> PredictionComparison:
    """Launch clean lanes, compare evidence, and publish R03 raw results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_result = _run_lane("reference", fixture, output_dir)
    helper_result = _run_lane("helper", fixture, output_dir)
    with (
        np.load(output_dir / "reference_lane_arrays.npz", allow_pickle=False) as reference_arrays,
        np.load(output_dir / "helper_lane_arrays.npz", allow_pickle=False) as helper_arrays,
    ):
        weight_rows, max_weight_delta, shapes_equal, dtypes_equal, weights_equal = _weight_rows(
            reference_result,
            helper_result,
            reference_arrays,
            helper_arrays,
        )
        reference_prediction = reference_arrays["prediction"]
        helper_prediction = helper_arrays["prediction"]
        compiled_prediction = reference_arrays["prediction_compile_true"]
        maximum, mean, p99 = _delta_metrics(reference_prediction, helper_prediction)
        compile_max, compile_mean, _ = _delta_metrics(reference_prediction, compiled_prediction)

        argmax_disagreement = _to_int(
            np.sum(np.argmax(reference_prediction, axis=1) != np.argmax(helper_prediction, axis=1)),
            "argmax disagreement count",
        )
        threshold_disagreement = _to_int(
            np.sum(
                (reference_prediction[:, 1] >= THRESHOLD) != (helper_prediction[:, 1] >= THRESHOLD)
            ),
            "threshold disagreement count",
        )
        compile_argmax_disagreement = _to_int(
            np.sum(
                np.argmax(reference_prediction, axis=1) != np.argmax(compiled_prediction, axis=1)
            ),
            "compile argmax disagreement count",
        )
        compile_threshold_disagreement = _to_int(
            np.sum(
                (reference_prediction[:, 1] >= THRESHOLD)
                != (compiled_prediction[:, 1] >= THRESHOLD)
            ),
            "compile threshold disagreement count",
        )

    structural_keys = (
        "model_type",
        "model_module",
        "input_shape",
        "output_shape",
        "parameter_count",
        "layer_count",
        "layer_names",
        "layer_classes",
        "layer_modules",
        "weight_tensor_count",
        "dtype_policy",
        "trainable_variable_count",
        "non_trainable_variable_count",
    )
    structural_equivalence = all(
        reference_result["structure"][key] == helper_result["structure"][key]
        for key in structural_keys
    )
    compile_data = reference_result["compile_true"] or {}
    criteria = all(
        (
            helper_result["artifact_family_detected"] == "KERAS_3_STANDALONE",
            helper_result["selected_loader"] == "keras.saving.load_model",
            bool(helper_result["safe_mode"]),
            structural_equivalence,
            shapes_equal,
            dtypes_equal,
            max_weight_delta <= 1e-7,
            maximum <= 1e-7,
            argmax_disagreement == 0,
            threshold_disagreement == 0,
            compile_max <= 1e-7,
            compile_argmax_disagreement == 0,
            compile_threshold_disagreement == 0,
            bool(compile_data.get("optimizer_restored")),
        )
    )
    result = PredictionComparison(
        prediction_shape_reference=list(reference_prediction.shape),
        prediction_shape_helper=list(helper_prediction.shape),
        prediction_dtype_reference=str(reference_prediction.dtype),
        prediction_dtype_helper=str(helper_prediction.dtype),
        nan_count_reference=_to_int(np.isnan(reference_prediction).sum(), "reference NaNs"),
        nan_count_helper=_to_int(np.isnan(helper_prediction).sum(), "helper NaNs"),
        inf_count_reference=_to_int(np.isinf(reference_prediction).sum(), "reference Infs"),
        inf_count_helper=_to_int(np.isinf(helper_prediction).sum(), "helper Infs"),
        max_abs_prediction_delta=maximum,
        mean_abs_prediction_delta=mean,
        p99_abs_prediction_delta=p99,
        argmax_disagreement_count=argmax_disagreement,
        threshold_0_58_disagreement_count=threshold_disagreement,
        compile_true_max_abs_prediction_delta=compile_max,
        compile_true_mean_abs_prediction_delta=compile_mean,
        compile_true_argmax_disagreement_count=compile_argmax_disagreement,
        compile_true_threshold_disagreement_count=compile_threshold_disagreement,
        compile_optimizer_restored=bool(compile_data.get("optimizer_restored", False)),
        compile_optimizer_class=compile_data.get("optimizer_class"),
        compile_loss=compile_data.get("loss"),
        structural_equivalence=structural_equivalence,
        weight_tensor_count_reference=len(reference_result["structure"]["weight_tensors"]),
        weight_tensor_count_helper=len(helper_result["structure"]["weight_tensors"]),
        all_weight_shapes_equal=shapes_equal,
        all_weight_dtypes_equal=dtypes_equal,
        all_weights_array_equal=weights_equal,
        max_abs_weight_delta=max_weight_delta,
        criteria_passed=criteria,
    )
    with (output_dir / "weight_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)
    (output_dir / "prediction_comparison.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.fixture, args.output_dir)
    print(result.model_dump_json(indent=2))
    return 0 if result.criteria_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
