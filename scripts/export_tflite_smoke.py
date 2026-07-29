"""TFLite export smoke for pretrain backbones (FASE 6/10).

Converts the newest (or given) run's ``backbone_pretrained.keras`` to TFLite
float32 and INT8 (representative dataset = synthetic z-score noise — this is
a *smoke*, not QG6), checks the 64 KB FlatBuffer budget, and runs one
interpreter inference.

Usage:
    python scripts/export_tflite_smoke.py [run_dir]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.keras_loader import load_keras_model  # noqa: E402
from scripts.validate_pretrain_artifacts import newest_run_dir  # noqa: E402

LOGGER = logging.getLogger("lewis.camada04.export_smoke")

MAX_FLATBUFFER_KB = 64


def _representative_dataset(input_len: int, n: int = 100):
    rng = np.random.default_rng(0)
    for _ in range(n):
        yield [rng.normal(0, 1, (1, input_len, 1)).astype(np.float32)]


def convert(model, *, quantize: bool, input_len: int) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset(input_len)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return converter.convert()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = args.run_dir or newest_run_dir()
    if run_dir is None:
        LOGGER.error("no pretrain run directory found")
        return 2
    model_path = run_dir / "backbone_pretrained.keras"
    model = load_keras_model(str(model_path), compile=False)
    input_len = int(model.input_shape[1])

    ok = True
    for tag, quantize in (("float32", False), ("int8", True)):
        blob = convert(model, quantize=quantize, input_len=input_len)
        size_kb = len(blob) / 1024
        out = run_dir / f"backbone_pretrained_{tag}.tflite"
        out.write_bytes(blob)
        status = "OK" if size_kb <= MAX_FLATBUFFER_KB * (4 if not quantize else 1) else "OVER"
        if quantize and size_kb > MAX_FLATBUFFER_KB:
            ok = False
        LOGGER.info("%s: %.1f KB -> %s [%s]", tag, size_kb, out.name, status)

    import tensorflow as tf

    blob = (run_dir / "backbone_pretrained_int8.tflite").read_bytes()
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    x = np.zeros((1, input_len, 1), dtype=np.float32)
    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    LOGGER.info("interpreter inference OK | output shape=%s", output.shape)

    if not ok:
        LOGGER.error("INT8 FlatBuffer exceeds %d KB budget", MAX_FLATBUFFER_KB)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
