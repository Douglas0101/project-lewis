"""Independent subprocess lane for Stage 1 Keras loader equivalence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TensorMetadata(StrictModel):
    tensor_index: int
    layer_name: str
    layer_tensor_index: int
    shape: list[int]
    dtype: str
    trainable: bool


class ModelStructure(StrictModel):
    model_type: str
    model_module: str
    input_shape: list[int | None]
    output_shape: list[int | None]
    parameter_count: int
    layer_count: int
    layer_names: list[str]
    layer_classes: list[str]
    layer_modules: list[str]
    weight_tensor_count: int
    weight_tensors: list[TensorMetadata]
    dtype_policy: str
    trainable_variable_count: int
    non_trainable_variable_count: int


class LaneResult(StrictModel):
    lane: str
    artifact_family_detected: str
    selected_loader: str
    compile: bool
    safe_mode: bool
    custom_objects: list[str]
    model_path_resolved: str
    model_sha256: str
    fixture_path_resolved: str
    fixture_sha256: str
    structure: ModelStructure
    prediction_shape: list[int]
    prediction_dtype: str
    prediction_nan_count: int
    prediction_inf_count: int
    compile_true: dict[str, Any] | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not an integer") from error


def _shape(value: Any) -> list[int | None]:
    shape = value[0] if isinstance(value, list) and len(value) == 1 else value
    return [_to_int(item, "shape dimension") if item is not None else None for item in shape]


def _snapshot_structure(model: Any) -> tuple[ModelStructure, list[np.ndarray]]:
    arrays: list[np.ndarray] = []
    metadata: list[TensorMetadata] = []
    tensor_index = 0
    trainable_ids = {id(variable) for variable in model.trainable_variables}
    for layer in model.layers:
        for layer_index, variable in enumerate(layer.weights):
            value = np.asarray(variable.numpy())
            arrays.append(value)
            metadata.append(
                TensorMetadata(
                    tensor_index=tensor_index,
                    layer_name=str(layer.name),
                    layer_tensor_index=layer_index,
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    trainable=id(variable) in trainable_ids,
                )
            )
            tensor_index += 1

    return (
        ModelStructure(
            model_type=type(model).__name__,
            model_module=type(model).__module__,
            input_shape=_shape(model.input_shape),
            output_shape=_shape(model.output_shape),
            parameter_count=_to_int(model.count_params(), "parameter count"),
            layer_count=len(model.layers),
            layer_names=[str(layer.name) for layer in model.layers],
            layer_classes=[type(layer).__name__ for layer in model.layers],
            layer_modules=[type(layer).__module__ for layer in model.layers],
            weight_tensor_count=len(arrays),
            weight_tensors=metadata,
            dtype_policy=str(model.dtype_policy.name),
            trainable_variable_count=len(model.trainable_variables),
            non_trainable_variable_count=len(model.non_trainable_variables),
        ),
        arrays,
    )


def _load_model(lane: str, model_path: Path) -> tuple[Any, dict[str, Any]]:
    if lane == "reference":
        import keras

        model = keras.saving.load_model(
            model_path,
            compile=False,
            safe_mode=True,
        )
        audit = {
            "artifact_family_detected": "KERAS_3_STANDALONE",
            "selected_loader": "keras.saving.load_model",
            "compile": False,
            "safe_mode": True,
            "custom_objects": [],
            "model_path_resolved": str(model_path),
            "model_sha256": _sha256_file(model_path),
        }
        return model, audit

    from src.models.keras_loader import inspect_loader_selection, load_keras_model

    decision = inspect_loader_selection(model_path, compile=False)
    return load_keras_model(model_path, compile=False), decision.model_dump()


def _compile_true_snapshot(
    model_path: Path, x_values: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    import keras

    model: Any = keras.saving.load_model(
        model_path,
        compile=True,
        safe_mode=True,
    )
    prediction = np.asarray(model.predict(x_values, verbose=0))
    optimizer = getattr(model, "optimizer", None)
    return (
        {
            "optimizer_restored": optimizer is not None,
            "optimizer_class": type(optimizer).__name__ if optimizer is not None else None,
            "loss": str(model.loss),
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": str(prediction.dtype),
        },
        prediction,
    )


def run_lane(
    lane: str,
    model_path: Path,
    fixture_path: Path,
    result_path: Path,
    arrays_path: Path,
) -> LaneResult:
    """Load one independent model and persist its immutable observations."""
    model_resolved = model_path.resolve(strict=True)
    fixture_resolved = fixture_path.resolve(strict=True)
    with np.load(fixture_resolved, allow_pickle=False) as fixture:
        x_values = fixture["X_scaled"].astype(np.float32, copy=False)

    model, audit = _load_model(lane, model_resolved)
    structure, weights = _snapshot_structure(model)
    prediction = np.asarray(model.predict(x_values, verbose=0))
    compile_true = None
    compiled_prediction = np.empty((0,), dtype=np.float32)
    if lane == "reference":
        compile_true, compiled_prediction = _compile_true_snapshot(model_resolved, x_values)

    arrays: dict[str, np.ndarray] = {"prediction": prediction}
    if lane == "reference":
        arrays["prediction_compile_true"] = compiled_prediction
    for index, weight in enumerate(weights):
        arrays[f"weight_{index:03d}"] = weight
    savez: Any = np.savez
    savez(arrays_path, **arrays)

    result = LaneResult(
        lane=lane,
        artifact_family_detected=str(audit["artifact_family_detected"]),
        selected_loader=str(audit["selected_loader"]),
        compile=bool(audit["compile"]),
        safe_mode=bool(audit["safe_mode"]),
        custom_objects=list(audit["custom_objects"]),
        model_path_resolved=str(audit["model_path_resolved"]),
        model_sha256=str(audit["model_sha256"]),
        fixture_path_resolved=str(fixture_resolved),
        fixture_sha256=_sha256_file(fixture_resolved),
        structure=structure,
        prediction_shape=list(prediction.shape),
        prediction_dtype=str(prediction.dtype),
        prediction_nan_count=_to_int(np.isnan(prediction).sum(), "prediction NaN count"),
        prediction_inf_count=_to_int(np.isinf(prediction).sum(), "prediction Inf count"),
        compile_true=compile_true,
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("reference", "helper"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()
    run_lane(args.lane, args.model, args.fixture, args.result, args.arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
