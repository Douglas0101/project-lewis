"""Orchestrate R04 independent inference-mode audits across subprocesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.models.keras_loader import load_keras_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
FIXTURE_PATH = (
    PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R03" / "loader_fixture.npz"
)
THRESHOLD = 0.5800000000000001


def _run(mode: str, output: Path, seed: int = 42) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/stage1_inference_mode_instance.py",
        "--model-path",
        str(MODEL_PATH),
        "--fixture",
        str(FIXTURE_PATH),
        "--mode",
        mode,
        "--output",
        str(output),
        "--seed",
        str(seed),
    ]
    process = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"R04 {mode} instance failed:\n{process.stdout}\n{process.stderr}")
    try:
        return json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid R04 {mode} instance result") from error


def _load_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data["X_scaled"].astype(np.float32, copy=False)


def _predict_with_current_model(x_values: np.ndarray) -> np.ndarray:
    """Reference prediction using the project's helper in the current process."""
    model = load_keras_model(MODEL_PATH, compile=False)
    return np.asarray(model.predict(x_values, verbose=0))  # type: ignore[reportArgumentType]


def _delta(left: np.ndarray, right: np.ndarray, label: str) -> dict[str, Any]:
    try:
        delta = np.abs(left - right)
        return {
            "label": label,
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "max_abs_delta": float(np.max(delta)),
            "mean_abs_delta": float(np.mean(delta)),
            "p99_abs_delta": float(np.percentile(delta, 99)),
            "array_equal": bool(np.array_equal(left, right)),
            "argmax_disagreement_count": int(
                np.sum(np.argmax(left, axis=1) != np.argmax(right, axis=1))
            ),
            "threshold_disagreement_count": int(
                np.sum((left[:, 1] >= THRESHOLD) != (right[:, 1] >= THRESHOLD))
            ),
        }
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to compare outputs for {label}") from exc


def _hash_trainable(model: Any) -> str:
    digest = hashlib.sha256()
    for variable in model.trainable_variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


def _hash_non_trainable(model: Any) -> str:
    digest = hashlib.sha256()
    for variable in model.non_trainable_variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


def _model_state_hashes() -> dict[str, Any]:
    model = load_keras_model(MODEL_PATH, compile=False)
    before_trainable = _hash_trainable(model)
    before_non_trainable = _hash_non_trainable(model)
    x_values = _load_array(FIXTURE_PATH)
    _ = np.asarray(model.predict(x_values, verbose=0))  # type: ignore[reportArgumentType]
    after_trainable = _hash_trainable(model)
    after_non_trainable = _hash_non_trainable(model)
    return {
        "trainable_weight_hash_before": before_trainable,
        "trainable_weight_hash_after": after_trainable,
        "non_trainable_weight_hash_before": before_non_trainable,
        "non_trainable_weight_hash_after": after_non_trainable,
        "trainable_weights_immutable": before_trainable == after_trainable,
        "non_trainable_weights_changed": before_non_trainable != after_non_trainable,
    }


def _cross_process_determinism(output_dir: Path, seed: int) -> dict[str, Any]:
    """Run two independent subprocesses with same seed and compare first training=True."""
    a_path = output_dir / "det_reload_a.json"
    b_path = output_dir / "det_reload_b.json"
    a = _run("deterministic_reload", a_path, seed=seed)
    b = _run("deterministic_reload", b_path, seed=seed)
    # We do not have raw arrays, so compare SHA-256 and note it is diagnostic.
    return {
        "seed": seed,
        "process_a_first_training_true_sha256": a["run"]["first_training_true"]["sha256"],
        "process_b_first_training_true_sha256": b["run"]["first_training_true"]["sha256"],
        "identical": a["run"]["first_training_true"]["sha256"]
        == b["run"]["first_training_true"]["sha256"],
    }


def _compare_predict_vs_training_false(
    predict_result: dict[str, Any], training_false_result: dict[str, Any]
) -> dict[str, Any]:
    """Compare first output from predict and training=False lanes."""
    # We need raw arrays. Re-run reference modes locally to capture them.
    x_values = _load_array(FIXTURE_PATH)
    model = load_keras_model(MODEL_PATH, compile=False)
    p_ref = np.asarray(model.predict(x_values, verbose=0))  # type: ignore[reportArgumentType]
    tf_ref = np.asarray(model(x_values, training=False))
    return {
        "predict_reproducible": all(p["array_equal"] for p in predict_result["run"]["pairwise"]),
        "training_false_reproducible": all(
            p["array_equal"] for p in training_false_result["run"]["pairwise"]
        ),
        "predict_vs_training_false": _delta(p_ref, tf_ref, "predict_vs_training_false"),
        "trainable_weight_hash_before_predict": predict_result["trainable_weight_hash_before"],
        "trainable_weight_hash_after_predict": predict_result["trainable_weight_hash_after"],
        "trainable_weight_hash_before_training_false": training_false_result[
            "trainable_weight_hash_before"
        ],
        "trainable_weight_hash_after_training_false": training_false_result[
            "trainable_weight_hash_after"
        ],
    }


def _rng_state_transitions(
    predict_result: dict[str, Any],
    training_false_result: dict[str, Any],
    training_true_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "predict": {
            "initial": predict_result["initial_rng_state"],
            "final": predict_result["final_rng_state"],
            "changed": predict_result["initial_rng_state"] != predict_result["final_rng_state"],
        },
        "training_false": {
            "initial": training_false_result["initial_rng_state"],
            "final": training_false_result["final_rng_state"],
            "changed": training_false_result["initial_rng_state"]
            != training_false_result["final_rng_state"],
        },
        "training_true": {
            "samples": [
                {
                    "state_before_changed": sample["state_before"] != sample["state_after"],
                    "state_after_sha256": (
                        sample["state_after"].get("state", {}).get("sha256")
                        if isinstance(sample["state_after"], dict)
                        else None
                    ),
                }
                for sample in training_true_result["run"]["samples"]
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R04",
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_result = _run("predict", output_dir / "predict_reproducibility.json")
    training_false_result = _run(
        "training_false", output_dir / "training_false_reproducibility.json"
    )
    training_true_result = _run("training_true", output_dir / "training_true_stochasticity.json")

    cross_process = _cross_process_determinism(output_dir, seed=42)

    model_state = _model_state_hashes()
    (output_dir / "model_state_hashes.json").write_text(
        json.dumps(model_state, indent=2) + "\n", encoding="utf-8"
    )

    predict_vs_tf = _compare_predict_vs_training_false(predict_result, training_false_result)
    (output_dir / "predict_vs_training_false.json").write_text(
        json.dumps(predict_vs_tf, indent=2) + "\n", encoding="utf-8"
    )

    rng_transitions = _rng_state_transitions(
        predict_result, training_false_result, training_true_result
    )
    combined_rng = {
        "cross_process_determinism": cross_process,
        "intra_process_transitions": rng_transitions,
    }
    (output_dir / "rng_state_transitions.json").write_text(
        json.dumps(combined_rng, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "predict_reproducible": all(p["array_equal"] for p in predict_result["run"]["pairwise"]),
        "training_false_reproducible": all(
            p["array_equal"] for p in training_false_result["run"]["pairwise"]
        ),
        "predict_vs_training_false_equal": predict_vs_tf["predict_vs_training_false"][
            "array_equal"
        ],
        "predict_vs_training_false_max_delta": predict_vs_tf["predict_vs_training_false"][
            "max_abs_delta"
        ],
        "predict_argmax_disagreements": predict_vs_tf["predict_vs_training_false"][
            "argmax_disagreement_count"
        ],
        "predict_threshold_disagreements": predict_vs_tf["predict_vs_training_false"][
            "threshold_disagreement_count"
        ],
        "training_true_weights_immutable": training_true_result["trainable_weights_immutable"],
        "training_true_any_rng_state_changed": training_true_result["run"]["any_rng_state_changed"],
        "model_state_trainable_weights_immutable": model_state["trainable_weights_immutable"],
        "rng_not_advanced_in_predict": not rng_transitions["predict"]["changed"],
        "rng_not_advanced_in_training_false": not rng_transitions["training_false"]["changed"],
        "cross_process_first_training_true_identical": cross_process["identical"],
    }
    (output_dir / "inference_mode_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
