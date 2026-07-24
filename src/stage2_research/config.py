"""Configuration loading and canonical hashing for Stage 2 research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.stage2_research.contracts import (
    DatasetConfig,
    ExitCode,
    HashedPath,
    ResearchConfig,
    ResearchError,
)
from src.stage2_research.integrity import hash_canonical


def _raw_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResearchError(
            f"cannot read config: {path}",
            ExitCode.BLOCKED_PRECONDITION,
        ) from error
    try:
        parsed: Any
        if path.suffix.lower() == ".json":
            parsed = json.loads(raw)
        else:
            parsed = yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ResearchError(
            f"invalid config syntax: {path}",
            ExitCode.ARGUMENT_ERROR,
        ) from error
    if not isinstance(parsed, dict):
        raise ResearchError("research config must be a mapping", ExitCode.ARGUMENT_ERROR)
    return parsed


def _resolve_hashed_path(value: HashedPath, project_root: Path) -> HashedPath:
    path = value.path
    resolved = path if path.is_absolute() else project_root / path
    return value.model_copy(update={"path": resolved.resolve()})


def load_research_config(
    path: Path,
    *,
    output_root_override: Path | None = None,
) -> ResearchConfig:
    """Load YAML/JSON through Pydantic and resolve project-relative paths."""
    config_path = path.resolve()
    raw = _raw_config(config_path)
    try:
        parsed = ResearchConfig.model_validate(raw)
    except ValueError as error:
        raise ResearchError(
            f"invalid Stage 2 research config: {error}",
            ExitCode.ARGUMENT_ERROR,
        ) from error

    root_value = parsed.project_root
    if not root_value.is_absolute():
        root_value = (config_path.parent.parent / root_value).resolve()
    project_root = root_value.resolve()
    datasets = DatasetConfig(
        stage2_npz=_resolve_hashed_path(parsed.datasets.stage2_npz, project_root),
        stage2_parquet=_resolve_hashed_path(parsed.datasets.stage2_parquet, project_root),
        full_npz=_resolve_hashed_path(parsed.datasets.full_npz, project_root),
        full_parquet=_resolve_hashed_path(parsed.datasets.full_parquet, project_root),
    )
    output_value = output_root_override or parsed.output_root
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (project_root / output_value).resolve()
    )
    return parsed.model_copy(
        update={
            "project_root": project_root,
            "output_root": output_root,
            "datasets": datasets,
        }
    )


def config_hash(config: ResearchConfig) -> str:
    """Hash resolved semantic configuration."""
    return hash_canonical(config.model_dump(mode="json"))
