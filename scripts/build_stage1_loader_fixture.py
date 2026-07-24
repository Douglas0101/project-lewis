"""Build the deterministic, immutable shared fixture for R03 loader lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import keras
import numpy as np
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_stage1_qg5 import _select_qg5_subset  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "input_scaler_stage1_v2.0.pkl"
THRESHOLD = 0.5800000000000001
SELECTION_POLICY = (
    "Deterministic union of first true Normal, first true Anormal, highest positive score, "
    "lowest negative score, nearest score below/above 0.58, and nearest scores to global "
    "p01/p25/p50/p75/p99; duplicate indices are merged and roles retained."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FixtureSample(StrictModel):
    fixture_index: int
    qg5_subset_index: int
    sample_id: str
    true_label_stage1: int
    reference_score: float
    roles: list[str]


class FixtureManifest(StrictModel):
    schema_version: str
    shape: list[int]
    dtype: str
    sha256: str
    array_sha256: str
    scaler_sha256: str
    model_sha256: str
    threshold: float
    selection_policy: str
    samples: list[FixtureSample]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


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


def _add_role(roles: dict[int, list[str]], index: Any, role: str) -> None:
    normalized = _to_int(index, f"index for role {role}")
    roles.setdefault(normalized, []).append(role)


def _selected_roles(y_true: np.ndarray, scores: np.ndarray) -> dict[int, list[str]]:
    roles: dict[int, list[str]] = {}
    normal = np.flatnonzero(y_true == 0)
    abnormal = np.flatnonzero(y_true == 1)
    below = np.flatnonzero(scores < THRESHOLD)
    above = np.flatnonzero(scores >= THRESHOLD)
    if not normal.size or not abnormal.size or not below.size or not above.size:
        raise ValueError("QG5 subset cannot satisfy the R03 fixture policy")

    _add_role(roles, normal[0], "true_normal")
    _add_role(roles, abnormal[0], "true_abnormal")
    _add_role(roles, np.argmax(scores), "predicted_positive_highest_score")
    _add_role(roles, np.argmin(scores), "predicted_negative_lowest_score")
    _add_role(roles, below[np.argmax(scores[below])], "immediately_below_threshold")
    _add_role(roles, above[np.argmin(scores[above])], "immediately_above_threshold")
    for quantile in (0.01, 0.25, 0.50, 0.75, 0.99):
        target = np.quantile(scores, quantile)
        nearest = np.argmin(np.abs(scores - target))
        _add_role(roles, nearest, f"score_quantile_p{round(quantile * 100):02d}")
    return roles


def build_fixture(output_path: Path, manifest_path: Path) -> FixtureManifest:
    """Create the fixture once; refuse to overwrite any existing evidence."""
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("R03 fixture or manifest already exists; refusing overwrite")

    x_raw, y_aami, metadata, _ = _select_qg5_subset()
    scaler = joblib.load(SCALER_PATH)
    n_samples, sequence_length, channels = x_raw.shape
    x_scaled_all = scaler.transform(x_raw.reshape(-1, channels)).reshape(
        n_samples, sequence_length, channels
    )
    model: Any = keras.saving.load_model(MODEL_PATH, compile=False, safe_mode=True)
    output = np.asarray(model.predict(x_scaled_all, verbose=0))
    scores = output[:, 1]
    y_true = (y_aami != 0).astype(np.int64)
    roles = _selected_roles(y_true, scores)
    selected = np.array(sorted(roles), dtype=np.int64)

    x_scaled = x_scaled_all[selected].astype(np.float32, copy=False)
    fixture_arrays: dict[str, np.ndarray] = {
        "X_scaled": x_scaled,
        "y_true": y_true[selected],
        "qg5_subset_index": selected,
        "reference_score": scores[selected].astype(np.float32, copy=False),
        "sample_id": np.asarray([metadata[index]["sample_id"] for index in selected]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savez: Any = np.savez
    savez(output_path, **fixture_arrays)

    samples = [
        FixtureSample(
            fixture_index=fixture_index,
            qg5_subset_index=_to_int(qg5_index, "QG5 subset index"),
            sample_id=str(metadata[qg5_index]["sample_id"]),
            true_label_stage1=_to_int(y_true[qg5_index], "true Stage 1 label"),
            reference_score=_to_float(scores[qg5_index], "reference score"),
            roles=roles[qg5_index],
        )
        for fixture_index, qg5_index in enumerate(selected.tolist())
    ]
    manifest = FixtureManifest(
        schema_version="1.0",
        shape=list(x_scaled.shape),
        dtype=str(x_scaled.dtype),
        sha256=_sha256_file(output_path),
        array_sha256=_sha256_array(x_scaled),
        scaler_sha256=_sha256_file(SCALER_PATH),
        model_sha256=_sha256_file(MODEL_PATH),
        threshold=THRESHOLD,
        selection_policy=SELECTION_POLICY,
        samples=samples,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_fixture(args.output, args.manifest)
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
