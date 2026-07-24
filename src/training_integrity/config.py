"""Strict YAML loading for advanced-training integrity configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .contracts import (
    AdvancedTrainingConfig,
    DatasetRole,
    IdentityMethod,
    PatientIdentityPolicy,
)


def _reject_duplicate_keys(node: yaml.Node | None) -> None:
    """Reject duplicate scalar keys before safe construction."""
    if node is None:
        return
    if isinstance(node, yaml.MappingNode):
        seen: set[tuple[str, str]] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                raise ValueError("integrity config mapping keys must be scalar")
            key = (key_node.tag, key_node.value)
            if key in seen:
                raise ValueError(f"duplicate integrity config key: {key_node.value}")
            seen.add(key)
            _reject_duplicate_keys(value_node)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _reject_duplicate_keys(child)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        syntax_tree = yaml.compose(text, Loader=yaml.SafeLoader)
        _reject_duplicate_keys(syntax_tree)
        payload = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load integrity config: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"integrity config must be a mapping: {path}")
    return payload


def _freeze_sequences(value: Any) -> Any:
    """Convert YAML sequences to tuples before strict Pydantic validation."""
    if isinstance(value, list):
        return tuple(_freeze_sequences(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_sequences(item) for key, item in value.items()}
    return value


def load_advanced_training_config(path: Path) -> tuple[AdvancedTrainingConfig, Path]:
    """Validate the frozen training config and resolve its project root."""
    config_path = path.resolve()
    try:
        config = AdvancedTrainingConfig.model_validate(_freeze_sequences(_load_yaml(config_path)))
    except ValidationError as error:
        raise ValueError(f"invalid advanced training config: {path}") from error
    base = config_path.parent.parent
    project_root = (base / config.project_root).resolve()
    if not (project_root / "pyproject.toml").is_file():
        raise ValueError(f"invalid project root in training config: {project_root}")
    return config, project_root


def load_patient_identity_policy(path: Path) -> PatientIdentityPolicy:
    """Validate a non-PII record-to-patient evidence policy."""
    try:
        payload = _freeze_sequences(_load_yaml(path.resolve()))
        datasets = []
        for raw_dataset in payload.get("datasets", ()):
            if not isinstance(raw_dataset, dict):
                raise ValueError("patient identity dataset policy must be a mapping")
            dataset = dict(raw_dataset)
            role_value = dataset.get("role")
            method_value = dataset.get("method")
            if not isinstance(role_value, str) or not isinstance(method_value, str):
                raise ValueError("patient identity role and method must be strings")
            dataset["role"] = DatasetRole(role_value)
            dataset["method"] = IdentityMethod(method_value)
            datasets.append(dataset)
        payload["datasets"] = tuple(datasets)
        return PatientIdentityPolicy.model_validate(payload)
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise ValueError(f"invalid patient identity policy: {path}") from error
