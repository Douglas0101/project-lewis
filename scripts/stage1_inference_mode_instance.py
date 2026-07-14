"""Run one independent Stage 1 inference instance for R04 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import keras
import numpy as np

THRESHOLD = 0.5800000000000001


def _sha256_trainable(model: keras.Model) -> str:
    digest = hashlib.sha256()
    for variable in model.trainable_variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


def _sha256_non_trainable(model: keras.Model) -> str:
    digest = hashlib.sha256()
    for variable in model.non_trainable_variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


def _dropout_seed_generator(model: keras.Model) -> keras.layers.Dropout | None:
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dropout):
            return layer
    return None


def _capture_rng_state(model: keras.Model) -> dict[str, Any] | None:
    dropout = _dropout_seed_generator(model)
    if dropout is None:
        return None
    seed_gen = getattr(dropout, "seed_generator", None)
    if seed_gen is None:
        return None
    state_info: dict[str, Any] = {}
    for attr in ("state", "_state", "seed", "_seed", "counter"):
        try:
            value = getattr(seed_gen, attr, None)
        except Exception as exc:  # noqa: BLE001
            state_info[attr] = f"error: {exc}"
            continue
        if value is None:
            state_info[attr] = None
            continue
        if hasattr(value, "numpy"):
            arr = value.numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            try:
                arr = np.asarray(value)
            except Exception as exc:  # noqa: BLE001
                state_info[attr] = f"unconvertible: {exc}"
                continue
        if isinstance(arr, np.ndarray) and arr.size:
            state_info[attr] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "values": arr.tolist() if arr.size <= 10 else "...large...",
            }
        else:
            state_info[attr] = arr.tolist() if hasattr(arr, "tolist") else str(arr)
    return state_info


def _run_predict(model: keras.Model, x_values: np.ndarray) -> dict[str, Any]:
    outputs = [
        np.asarray(model.predict(x_values, verbose=0))  # type: ignore[reportArgumentType]
        for _ in range(3)
    ]
    return _compare_outputs(outputs, "predict")


def _run_training_false(model: keras.Model, x_values: np.ndarray) -> dict[str, Any]:
    outputs = [np.asarray(model(x_values, training=False)) for _ in range(3)]
    return _compare_outputs(outputs, "training_false")


def _run_training_true(model: keras.Model, x_values: np.ndarray) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    for _ in range(3):
        state_before = _capture_rng_state(model)
        weight_hash_before = _sha256_trainable(model)
        output: Any = np.asarray(model(x_values, training=True))
        state_after = _capture_rng_state(model)
        weight_hash_after = _sha256_trainable(model)
        samples.append(
            {
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "state_before": state_before,
                "state_after": state_after,
                "weight_hash_before": weight_hash_before,
                "weight_hash_after": weight_hash_after,
            }
        )
        outputs.append(output)
    pairwise = []
    for i in range(3):
        for j in range(i + 1, 3):
            pairwise.append(
                _delta_pair(outputs=[outputs[i], outputs[j]], label=f"training_true_{i+1}_vs_{j+1}")
            )
    return {
        "mode": "training_true",
        "samples": samples,
        "pairwise": pairwise,
        "trainable_weights_immutable": all(
            s["weight_hash_before"] == s["weight_hash_after"] for s in samples
        ),
        "any_rng_state_changed": any(s["state_before"] != s["state_after"] for s in samples),
    }


def _delta_pair(outputs: list[np.ndarray], label: str) -> dict[str, Any]:
    try:
        left, right = outputs[0], outputs[1]
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


def _compare_outputs(outputs: list[np.ndarray], mode: str) -> dict[str, Any]:
    try:
        pairwise = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                pairwise.append(_delta_pair([outputs[i], outputs[j]], f"{mode}_{i+1}_vs_{j+1}"))
        return {
            "mode": mode,
            "outputs_shape": [list(o.shape) for o in outputs],
            "outputs_dtype": [str(o.dtype) for o in outputs],
            "nan_counts": [int(np.sum(np.isnan(o))) for o in outputs],
            "inf_counts": [int(np.sum(np.isinf(o))) for o in outputs],
            "pairwise": pairwise,
        }
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to compare outputs for mode {mode}") from exc


def _run_deterministic_reload(model_path: Path, x_values: np.ndarray, seed: int) -> dict[str, Any]:
    keras.utils.set_random_seed(seed)
    model: Any = cast(Any, keras.saving.load_model(model_path, compile=False, safe_mode=True))
    output: Any = np.asarray(model(x_values, training=True))
    state_before = _capture_rng_state(model)
    return {
        "mode": "deterministic_reload",
        "seed": seed,
        "environment_seed": seed,
        "first_training_true": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        },
        "initial_rng_state": state_before,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["predict", "training_false", "training_true", "deterministic_reload"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

    model: Any = keras.saving.load_model(args.model_path, compile=False, safe_mode=True)
    with np.load(args.fixture, allow_pickle=False) as data:
        x_values = data["X_scaled"].astype(np.float32, copy=False)

    result: dict[str, Any] = {
        "mode": args.mode,
        "model_path": str(args.model_path.resolve()),
        "fixture_path": str(args.fixture.resolve()),
        "fixture_shape": list(x_values.shape),
        "fixture_dtype": str(x_values.dtype),
        "trainable_weight_hash_before": _sha256_trainable(model),
        "non_trainable_weight_hash_before": _sha256_non_trainable(model),
        "initial_rng_state": _capture_rng_state(model),
    }

    if args.mode == "predict":
        result["run"] = _run_predict(model, x_values)
    elif args.mode == "training_false":
        result["run"] = _run_training_false(model, x_values)
    elif args.mode == "training_true":
        result["run"] = _run_training_true(model, x_values)
    elif args.mode == "deterministic_reload":
        result["run"] = _run_deterministic_reload(args.model_path, x_values, args.seed)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    result["trainable_weight_hash_after"] = _sha256_trainable(model)
    result["non_trainable_weight_hash_after"] = _sha256_non_trainable(model)
    result["final_rng_state"] = _capture_rng_state(model)
    result["trainable_weights_immutable"] = (
        result["trainable_weight_hash_before"] == result["trainable_weight_hash_after"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
