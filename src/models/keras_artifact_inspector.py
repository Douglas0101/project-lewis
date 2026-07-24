"""Read-only structural inspection for Keras v3 ``.keras`` archives."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Forbid undocumented fields in forensic inspection reports."""

    model_config = ConfigDict(extra="forbid")


class ArchiveMember(StrictModel):
    """Identity of one immutable ZIP member."""

    name: str
    size: int
    compressed_size: int
    sha256: str


class LayerContract(StrictModel):
    """Serialized structural contract for one Keras layer."""

    index: int
    name: str
    module: str
    class_name: str
    dtype: str | None
    batch_shape: list[int | None] | None
    units: int | None
    activation: str | None


class WeightDataset(StrictModel):
    """Shape and dtype of a model or optimizer HDF5 dataset."""

    name: str
    shape: list[int]
    dtype: str
    element_count: int


class KerasArtifactInspection(StrictModel):
    """Validated report produced without deserializing the Keras model."""

    schema_version: str = Field(pattern=r"^1\.0$")
    inspection_mode: str
    model_path: str
    model_sha256_before: str
    model_sha256_after: str
    model_size: int
    archive_members: list[ArchiveMember]
    metadata: dict[str, Any]
    config_sha256: str
    metadata_sha256: str
    weights_sha256: str
    top_level_module: str
    top_level_class_name: str
    registered_name: str | None
    keras_family: str
    model_name: str
    input_layers: list[Any]
    output_layers: list[Any]
    input_shape: list[int | None]
    input_dtype: str
    output_shape: list[int | None]
    output_units: int
    output_activation: str
    output_domain: str
    layer_count: int
    layers: list[LayerContract]
    batch_normalization_layers: list[str]
    dropout_layers: list[str]
    lambda_layers: list[str]
    custom_object_references: list[str]
    serialized_reference_counts: dict[str, int]
    compile_optimizer: str | None
    compile_loss: str | None
    compile_metrics: list[Any]
    model_weight_count: int
    model_parameter_count: int
    weight_datasets: list[WeightDataset]
    label_mapping_serialized: bool


def sha256_file(path: Path) -> str:
    """Return SHA-256 without mutating the file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} is not an integer") from error


def _dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        config = value.get("config", {})
        if isinstance(config, dict):
            name = config.get("name")
            return str(name) if name is not None else None
    return str(value)


def _read_json_member(archive: zipfile.ZipFile, name: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = archive.read(name)
        parsed = json.loads(raw)
    except KeyError as error:
        raise ValueError(f"Missing required Keras archive member: {name}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in Keras archive member: {name}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object in {name}")
    return raw, parsed


def _layer_contracts(config: dict[str, Any]) -> list[LayerContract]:
    model_config = config.get("config", {})
    serialized_layers = model_config.get("layers", []) if isinstance(model_config, dict) else []
    if not isinstance(serialized_layers, list):
        raise ValueError("Serialized model layers are not a list")

    layers: list[LayerContract] = []
    for index, serialized in enumerate(serialized_layers):
        if not isinstance(serialized, dict):
            raise ValueError("Serialized layer is not an object")
        layer_config = serialized.get("config", {})
        if not isinstance(layer_config, dict):
            raise ValueError("Serialized layer config is not an object")
        batch_shape = layer_config.get("batch_shape")
        normalized_shape = list(batch_shape) if isinstance(batch_shape, list) else None
        units = layer_config.get("units")
        layers.append(
            LayerContract(
                index=index,
                name=str(layer_config.get("name", serialized.get("name", ""))),
                module=str(serialized.get("module", "")),
                class_name=str(serialized.get("class_name", "")),
                dtype=_dtype_name(layer_config.get("dtype")),
                batch_shape=normalized_shape,
                units=_as_int(units, "layer units") if units is not None else None,
                activation=(
                    str(layer_config["activation"])
                    if layer_config.get("activation") is not None
                    else None
                ),
            )
        )
    if not layers:
        raise ValueError("Keras archive contains no layers")
    return layers


def _weight_datasets(content: bytes) -> list[WeightDataset]:
    datasets: list[WeightDataset] = []
    try:
        with h5py.File(io.BytesIO(content), "r") as weights:

            def collect(name: str, value: Any) -> None:
                if not isinstance(value, h5py.Dataset):
                    return
                element_count = _as_int(np.prod(value.shape), "weight element count")
                datasets.append(
                    WeightDataset(
                        name=name,
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        element_count=element_count,
                    )
                )

            weights.visititems(collect)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid model.weights.h5 member") from error
    return datasets


def _custom_references(layers: list[LayerContract]) -> list[str]:
    references: list[str] = []
    allowed_modules = {"keras", "keras.layers"}
    for layer in layers:
        if layer.class_name == "Lambda":
            references.append(f"{layer.name}:Lambda")
        elif layer.module not in allowed_modules:
            references.append(f"{layer.name}:{layer.module}.{layer.class_name}")
    return references


def inspect_keras_archive(path: Path) -> KerasArtifactInspection:
    """Inspect a Keras ZIP structurally without calling any model loader."""
    resolved = path.resolve(strict=True)
    before = sha256_file(resolved)
    try:
        with zipfile.ZipFile(resolved, "r") as archive:
            metadata_raw, metadata = _read_json_member(archive, "metadata.json")
            config_raw, config = _read_json_member(archive, "config.json")
            try:
                weights_raw = archive.read("model.weights.h5")
            except KeyError as error:
                raise ValueError(
                    "Missing required Keras archive member: model.weights.h5"
                ) from error
            members = [
                ArchiveMember(
                    name=info.filename,
                    size=info.file_size,
                    compressed_size=info.compress_size,
                    sha256=_sha256_bytes(archive.read(info.filename)),
                )
                for info in archive.infolist()
            ]
    except zipfile.BadZipFile as error:
        raise ValueError(f"Not a valid Keras ZIP archive: {resolved}") from error

    layers = _layer_contracts(config)
    input_layer = next((layer for layer in layers if layer.class_name == "InputLayer"), None)
    if input_layer is None or input_layer.batch_shape is None or input_layer.dtype is None:
        raise ValueError("InputLayer shape or dtype is missing")
    output_layer = layers[-1]
    if output_layer.units is None or output_layer.activation is None:
        raise ValueError("Final layer units or activation is missing")

    model_config = config.get("config", {})
    compile_config = config.get("compile_config", {})
    if not isinstance(model_config, dict) or not isinstance(compile_config, dict):
        raise ValueError("Invalid model or compile configuration")
    optimizer = compile_config.get("optimizer")
    optimizer_name = None
    if isinstance(optimizer, dict) and optimizer.get("class_name") is not None:
        optimizer_name = str(optimizer["class_name"])
    metrics = compile_config.get("metrics", [])
    if not isinstance(metrics, list):
        metrics = [metrics]

    serialized = config_raw.decode("utf-8")
    reference_counts = {
        token: serialized.count(token)
        for token in (
            "keras.src",
            "tf_keras.src",
            "tensorflow.keras",
            "tensorflow.python.keras",
            "custom_objects",
        )
    }
    weights = _weight_datasets(weights_raw)
    model_weights = [item for item in weights if item.name.startswith("layers/")]
    top_module = str(config.get("module", ""))
    keras_version = str(metadata.get("keras_version", ""))
    keras_family = (
        "KERAS_3_STANDALONE"
        if top_module.startswith("keras.src") and keras_version.startswith("3.")
        else "UNKNOWN"
    )
    output_domain = (
        "probabilities_sum_to_one" if output_layer.activation == "softmax" else "unknown"
    )
    after = sha256_file(resolved)

    return KerasArtifactInspection(
        schema_version="1.0",
        inspection_mode="zip_only_no_model_deserialization",
        model_path=str(resolved),
        model_sha256_before=before,
        model_sha256_after=after,
        model_size=resolved.stat().st_size,
        archive_members=members,
        metadata=metadata,
        config_sha256=_sha256_bytes(config_raw),
        metadata_sha256=_sha256_bytes(metadata_raw),
        weights_sha256=_sha256_bytes(weights_raw),
        top_level_module=top_module,
        top_level_class_name=str(config.get("class_name", "")),
        registered_name=(
            str(config["registered_name"]) if config.get("registered_name") is not None else None
        ),
        keras_family=keras_family,
        model_name=str(model_config.get("name", "")),
        input_layers=list(model_config.get("input_layers", [])),
        output_layers=list(model_config.get("output_layers", [])),
        input_shape=input_layer.batch_shape,
        input_dtype=input_layer.dtype,
        output_shape=[None, output_layer.units],
        output_units=output_layer.units,
        output_activation=output_layer.activation,
        output_domain=output_domain,
        layer_count=len(layers),
        layers=layers,
        batch_normalization_layers=[
            layer.name for layer in layers if layer.class_name == "BatchNormalization"
        ],
        dropout_layers=[layer.name for layer in layers if layer.class_name == "Dropout"],
        lambda_layers=[layer.name for layer in layers if layer.class_name == "Lambda"],
        custom_object_references=_custom_references(layers),
        serialized_reference_counts=reference_counts,
        compile_optimizer=optimizer_name,
        compile_loss=(
            str(compile_config["loss"]) if compile_config.get("loss") is not None else None
        ),
        compile_metrics=metrics,
        model_weight_count=len(model_weights),
        model_parameter_count=sum(item.element_count for item in model_weights),
        weight_datasets=weights,
        label_mapping_serialized=False,
    )
