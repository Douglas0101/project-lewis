"""Validação de manifests para modelos e scalers do Project-Lewis.

Garante que artefatos de inferência só sejam carregados quando o dataset e o
esquema de features forem compatíveis com os manifests registrados.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("lewis.inference.manifest_validator")


class ManifestValidationError(ValueError):
    """Levantado quando manifest é incompatível ou ausente."""


def _sha256_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_feature_schema_hash(feature_names: list[str]) -> str:
    """Hash canônico da ordem e nomes das features."""
    return _sha256_string(json.dumps(feature_names, sort_keys=False, ensure_ascii=True))


def load_manifest(path: Path) -> dict[str, Any]:
    """Carrega um manifest JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"Manifest nao encontrado: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"Manifest invalido: {path}: {exc}") from exc


def validate_feature_schema(
    artifact_manifest: dict[str, Any],
    expected_schema_hash: str,
    artifact_name: str = "artefato",
) -> None:
    """Verifica se o feature_schema_hash do artefato bate com o esperado."""
    artifact_hash = artifact_manifest.get("feature_schema_hash")
    if artifact_hash is None:
        raise ManifestValidationError(
            f"{artifact_name}: feature_schema_hash ausente no manifest. "
            "Regenere o artefato com o schema v2.4."
        )
    if artifact_hash != expected_schema_hash:
        raise ManifestValidationError(
            f"{artifact_name}: feature_schema_hash incompativel. "
            f"Esperado {expected_schema_hash}, obtido {artifact_hash}."
        )


def validate_dataset_manifest(
    artifact_manifest: dict[str, Any],
    expected_dataset_hash: str,
    artifact_name: str = "artefato",
) -> None:
    """Verifica se o dataset_manifest_hash do artefato bate com o esperado."""
    artifact_hash = artifact_manifest.get("dataset_manifest_hash")
    if artifact_hash is None:
        raise ManifestValidationError(
            f"{artifact_name}: dataset_manifest_hash ausente no manifest."
        )
    if artifact_hash != expected_dataset_hash:
        raise ManifestValidationError(
            f"{artifact_name}: dataset_manifest_hash incompativel. "
            f"Esperado {expected_dataset_hash}, obtido {artifact_hash}."
        )


def load_and_validate_manifest(
    manifest_path: Path,
    expected_feature_schema_hash: str | None,
    expected_dataset_hash: str | None,
    artifact_name: str = "artefato",
) -> dict[str, Any]:
    """Carrega manifest e valida hashes."""
    manifest = load_manifest(manifest_path)
    if expected_feature_schema_hash is not None:
        validate_feature_schema(manifest, expected_feature_schema_hash, artifact_name)
    if expected_dataset_hash is not None:
        validate_dataset_manifest(manifest, expected_dataset_hash, artifact_name)
    return manifest


def write_manifest(
    path: Path,
    feature_names: list[str],
    dataset_manifest_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Escreve um manifesto canônico para um artefato."""
    manifest = {
        "feature_schema_hash": compute_feature_schema_hash(feature_names),
        "dataset_manifest_hash": dataset_manifest_hash,
        "feature_names": feature_names,
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
