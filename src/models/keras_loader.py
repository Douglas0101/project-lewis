"""Family-aware, safe loader for Keras ``.keras`` artifacts.

The project contains Keras 3 standalone and may also encounter legacy
``tf_keras`` archives produced by pruning/QAT tooling. The serialized top-level
module selects the compatible loader; Keras 3 archives always use
``keras.saving.load_model(..., safe_mode=True)``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import keras
import tensorflow as tf
from pydantic import BaseModel, ConfigDict


class LoaderDecision(BaseModel):
    """Auditable loader decision made before model deserialization."""

    model_config = ConfigDict(extra="forbid")

    artifact_family_detected: str
    serialized_module: str
    selected_loader: str
    compile: bool
    safe_mode: bool
    custom_objects: list[str]
    model_path_resolved: str
    model_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialized_module(path: Path) -> str:
    """Read the top-level module without deserializing executable objects."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            config = json.loads(archive.read("config.json"))
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid Keras archive config: {path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Keras archive config is not an object: {path}")
    return str(config.get("module", ""))


def _serialized_family(path: str | Path) -> str:
    """Return the normalized serialized family used by compatibility callers."""
    module = _serialized_module(Path(path).resolve(strict=True))
    if "tf_keras" in module:
        return "tf_keras"
    if "keras.src" in module or module.startswith("keras."):
        return "keras"
    return "unknown"


def inspect_loader_selection(
    path: str | Path,
    *,
    compile: bool = False,
    custom_objects: dict[str, Any] | None = None,
) -> LoaderDecision:
    """Return the exact family, loader, safety mode, path, and artifact hash."""
    resolved = Path(path).resolve(strict=True)
    module = _serialized_module(resolved)
    if "tf_keras" in module:
        family = "TF_KERAS_LEGACY"
        selected_loader = "tf.keras.models.load_model"
    elif "keras.src" in module or module.startswith("keras."):
        family = "KERAS_3_STANDALONE"
        selected_loader = "keras.saving.load_model"
    else:
        family = "UNKNOWN"
        selected_loader = "keras.saving.load_model"

    return LoaderDecision(
        artifact_family_detected=family,
        serialized_module=module,
        selected_loader=selected_loader,
        compile=compile,
        safe_mode=True,
        custom_objects=sorted(custom_objects) if custom_objects is not None else [],
        model_path_resolved=str(resolved),
        model_sha256=_sha256_file(resolved),
    )


def load_keras_model(
    path: str | Path,
    *,
    compile: bool = False,
    custom_objects: dict[str, Any] | None = None,
) -> keras.Model:
    """Load a Keras archive using its serialized family and safe mode.

    Keras 3 standalone is the safe default for unknown families. Legacy
    ``tf_keras`` routing is retained for archives whose top-level module
    explicitly contains ``tf_keras``.
    """
    decision = inspect_loader_selection(
        path,
        compile=compile,
        custom_objects=custom_objects,
    )
    kwargs: dict[str, Any] = {
        "compile": decision.compile,
        "safe_mode": decision.safe_mode,
    }
    if custom_objects is not None:
        kwargs["custom_objects"] = custom_objects

    if decision.artifact_family_detected == "TF_KERAS_LEGACY":
        return tf.keras.models.load_model(  # type: ignore[return-value]
            decision.model_path_resolved,
            **kwargs,
        )

    return cast(
        keras.Model,
        keras.saving.load_model(
            decision.model_path_resolved,
            **kwargs,
        ),
    )
