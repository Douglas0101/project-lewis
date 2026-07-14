"""Inspect Dropout layer and seed generator state for R04 forensic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import keras
import numpy as np


def _capture_seed_generator_state(seed_gen: Any) -> dict[str, Any]:
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
                "values": arr.tolist() if arr.size <= 10 else "...large...",
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
        else:
            state_info[attr] = arr.tolist() if hasattr(arr, "tolist") else str(arr)
    return state_info


def inspect_dropout(model_path: Path, output_path: Path) -> dict[str, Any]:
    """Load model and capture Dropout layer, seed generator and variable state."""
    model = keras.saving.load_model(model_path, compile=False, safe_mode=True)
    dropout_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dropout)]
    if len(dropout_layers) != 1:
        raise ValueError(f"Expected exactly 1 Dropout layer; found {len(dropout_layers)}")
    dropout = dropout_layers[0]

    config = dropout.get_config()
    inventory: dict[str, Any] = {
        "layer_name": dropout.name,
        "layer_index": model.layers.index(dropout),
        "rate": config.get("rate"),
        "seed": config.get("seed"),
        "noise_shape": config.get("noise_shape"),
        "trainable": bool(dropout.trainable),
        "dtype": str(dropout.dtype),
        "module": type(dropout).__module__,
        "class_name": type(dropout).__name__,
        "config": config,
    }

    seed_gen = getattr(dropout, "seed_generator", None)
    inventory["has_seed_generator"] = seed_gen is not None
    inventory["seed_generator_type"] = (
        f"{type(seed_gen).__module__}.{type(seed_gen).__name__}" if seed_gen is not None else None
    )
    if seed_gen is not None:
        inventory["seed_generator_state"] = _capture_seed_generator_state(seed_gen)
    else:
        inventory["seed_generator_state"] = None

    nt_vars = model.non_trainable_variables
    inventory["non_trainable_variable_count"] = len(nt_vars)
    inventory["non_trainable_variables"] = []
    for variable in nt_vars:
        arr = variable.numpy() if hasattr(variable, "numpy") else np.asarray(variable)
        inventory["non_trainable_variables"].append(
            {
                "name": str(variable.name) if hasattr(variable, "name") else str(variable),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
        )

    t_vars = model.trainable_variables
    inventory["trainable_variable_count"] = len(t_vars)
    inventory["trainable_variables"] = [
        {
            "name": str(v.name),
            "shape": list(v.shape),
            "dtype": str(v.dtype),
        }
        for v in t_vars
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect_dropout(args.model_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
