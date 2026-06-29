"""JSON Schemas for Project-Lewis Camada 1 artifacts.

Defines the canonical schemas for:

* ``src/data/checksums.json`` — ZIP integrity manifest (§7.1 of Camada-01 spec).
* ``data/catalog/dataset_catalog.jsonl`` — WFDB metadata catalog (§10.2).
* ``data/.dlq/failed_downloads.jsonl`` — Dead Letter Queue entry (§9.3).

Also provides a lazy validation helper that uses :mod:`jsonschema` when
available, and falls back to a minimal required-keys check otherwise.
"""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER = logging.getLogger("lewis.camada01")

CHECKSUMS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://project-lewis.local/schemas/checksums.schema.json",
    "title": "DatasetChecksums",
    "type": "object",
    "required": ["version", "generated_at", "datasets"],
    "properties": {
        "version": {"const": "1.1"},
        "generated_at": {"type": "string", "format": "date-time"},
        "datasets": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mitdb", "svdb", "afdb", "incartdb"],
            "properties": {
                "chapman_shaoxing": {"$ref": "#/$defs/entry"},
                "ptbxl": {"$ref": "#/$defs/entry"},
                "mitdb": {"$ref": "#/$defs/entry"},
                "svdb": {"$ref": "#/$defs/entry"},
                "afdb": {"$ref": "#/$defs/entry"},
                "incartdb": {"$ref": "#/$defs/entry"},
            },
        },
        "mirrors": {"$ref": "#/$defs/mirrors"},
    },
    "$defs": {
        "entry": {
            "type": "object",
            "required": ["sha256", "size_bytes", "source", "verified_at"],
            "properties": {
                "sha256": {
                    "oneOf": [
                        {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                        {"type": "null"},
                    ]
                },
                "size_bytes": {
                    "oneOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"},
                    ]
                },
                "source": {
                    "enum": [
                        "physionet",
                        "kagglehub",
                        "figshare",
                        "mirror",
                        "local_mirror",
                    ]
                },
                "url": {"type": "string", "format": "uri"},
                "path": {"type": "string"},
                "verified_at": {"type": "string", "format": "date-time"},
                "verified_by": {"type": "string"},
                "note": {"type": "string"},
            },
        },
        "mirror_entry": {
            "type": "object",
            "required": ["sha256", "size_bytes", "path"],
            "properties": {
                "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                "size_bytes": {"type": "integer", "minimum": 0},
                "path": {"type": "string"},
            },
        },
        "mirrors": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "chapman_shaoxing": {"$ref": "#/$defs/mirror_entry"},
                "mitbih_family_zip": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/mirror_entry"},
                },
            },
        },
    },
}

CATALOG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://project-lewis.local/schemas/catalog.schema.json",
    "title": "DatasetCatalogLine",
    "description": "Schema for one line of data/catalog/dataset_catalog.jsonl",
    "type": "object",
    "required": [
        "record_name",
        "dataset",
        "fs",
        "n_sig",
        "sig_len",
        "duration_sec",
        "units",
        "gains",
        "sig_name",
        "source_path",
    ],
    "properties": {
        "record_name": {"type": "string"},
        "dataset": {"enum": ["chapman", "mitdb", "svdb", "afdb", "incart", "ptbxl"]},
        "fs": {"type": "integer", "minimum": 1},
        "n_sig": {"type": "integer", "minimum": 1},
        "sig_len": {"type": "integer", "minimum": 0},
        "duration_sec": {"type": "number", "minimum": 0},
        "units": {"type": "array", "items": {"type": "string"}},
        "gains": {"type": "array", "items": {"type": "integer"}},
        "sig_name": {"type": "array", "items": {"type": "string"}},
        "baselines": {"type": "array"},
        "adc_res": {"type": "array"},
        "adc_zero": {"type": "array"},
        "initial_value": {"type": "array"},
        "checksum": {"type": "array"},
        "comments": {"type": "array", "items": {"type": "string"}},
        "age": {"type": ["integer", "null"]},
        "sex": {"type": ["string", "null"], "enum": ["M", "F", None]},
        "diagnosis": {"type": ["string", "null"]},
        "source_path": {"type": "string"},
    },
}

DLQ_LINE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://project-lewis.local/schemas/dlq_line.schema.json",
    "title": "DLQEntry",
    "type": "object",
    "required": ["event", "ts", "url", "attempts"],
    "properties": {
        "event": {"const": "download_failed"},
        "ts": {"type": "string", "format": "date-time"},
        "url": {"type": "string"},
        "dest": {"type": "string"},
        "attempts": {"type": "integer", "minimum": 1},
        "source": {"type": "string"},
        "error": {"type": "string"},
    },
}

try:
    import jsonschema as _jsonschema  # type: ignore

    _HAS_JSONSCHEMA = True
except ImportError:
    _jsonschema = None  # type: ignore[assignment]
    _HAS_JSONSCHEMA = False


def _get_jsonschema():
    """Return the imported ``jsonschema`` module or ``None`` if not available."""
    return _jsonschema


def _fallback_required(payload: dict[str, Any], required: set[str]) -> bool:
    return required.issubset(payload.keys())


def validate_catalog_line(line: str) -> bool:
    """Validate a single JSONL line against :data:`CATALOG_SCHEMA`."""
    try:
        obj: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        return False
    if _HAS_JSONSCHEMA and _jsonschema is not None:
        try:
            _jsonschema.validate(obj, CATALOG_SCHEMA)
            return True
        except _jsonschema.ValidationError as exc:
            LOGGER.debug("catalog line schema fail: %s", exc)
            return False
    required = {
        "record_name",
        "dataset",
        "fs",
        "n_sig",
        "sig_len",
        "duration_sec",
        "units",
        "gains",
        "sig_name",
        "source_path",
    }
    return _fallback_required(obj, required)


def validate_dlq_line(line: str) -> bool:
    """Validate a single JSONL line against :data:`DLQ_LINE_SCHEMA`."""
    try:
        obj: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        return False
    if _HAS_JSONSCHEMA and _jsonschema is not None:
        try:
            _jsonschema.validate(obj, DLQ_LINE_SCHEMA)
            return True
        except _jsonschema.ValidationError as exc:
            LOGGER.debug("dlq line schema fail: %s", exc)
            return False
    return _fallback_required(obj, {"event", "ts", "url", "attempts"})


def validate_checksums_manifest(payload: dict[str, Any]) -> bool:
    """Validate an in-memory ``checksums.json`` document."""
    if _HAS_JSONSCHEMA and _jsonschema is not None:
        try:
            _jsonschema.validate(payload, CHECKSUMS_SCHEMA)
            return True
        except _jsonschema.ValidationError as exc:
            LOGGER.debug("checksums manifest schema fail: %s", exc)
            return False
    if not {"version", "generated_at", "datasets"}.issubset(payload.keys()):
        return False
    datasets = payload.get("datasets", {})
    return {"mitdb", "svdb", "afdb", "incartdb"}.issubset(datasets.keys())


def has_jsonschema() -> bool:
    """Return True when the optional ``jsonschema`` package is importable."""
    return _HAS_JSONSCHEMA
