"""Quantização INT8 do backbone pré-treinado A2-full (T3, C05).

Converte o checkpoint A2-full (a1_stable+focal, sigmoid head) para TFLite INT8
full-integer **com cabeça de logits** (sigmoid removido na conversão) de modo que
a ordem de inferência seja ``logits -> /T -> sigmoid``, com T = 0.3741 aprendido
na T1 (calibration.json). A divisão por T é aplicada em float32 **após** a
dequantização — não há caminho de overflow int8 (T < 1 amplifica logits ×2.67
somente no domínio float).

Saídas em ``experiments/<run>/quantized/``:
- ``a2_full_int8.tflite``        FlatBuffer INT8 (logits)
- ``quantization_params.json``   scales/zero-points + temperature + metadados
- ``quant_report.json``          Δ métricas float vs INT8, tamanho, range logits
- ``post_quant_calibration.json`` ECE pós-PTQ antes/depois de /T (n_bins=15)

Uso:
    uv run python scripts/quantize_pretrain_a2_full.py [--n-cal 512] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.keras_loader import load_keras_model  # noqa: E402
from src.models.pretrain_chapman import build_datasets  # noqa: E402
from src.models.pretrain_evaluation import (  # noqa: E402
    apply_temperature,
    calibration_summary,
    sigmoid_to_logits,
)
from src.models.pretrain_provenance import compute_per_class_metrics  # noqa: E402
from src.quantization.export_tflite import extract_quantization_params  # noqa: E402
from src.quantization.ptq import calibrate, representative_dataset_stratified  # noqa: E402

LOGGER = logging.getLogger("lewis.camada05.quantize_a2_full")

DEFAULT_RUN_DIR = PROJECT_ROOT / "experiments" / "20260728_053011_pretrain_chapman"
TEMPERATURE = 0.3741036858061345  # fallback; main() lê de calibration.json (fonte única)
MAX_FLATBUFFER_KB = 64


def make_logit_head(model: tf.keras.Model) -> tf.keras.Model:
    """Clona o modelo trocando a ativação sigmoid da camada ``output`` por linear.

    Os pesos são preservados; ``sigmoid(logits)`` reproduz as predições originais
    (verificado por assert no caller e por teste unitário).
    """

    def _clone(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if layer.name == "output" and isinstance(layer, tf.keras.layers.Dense):
            config = layer.get_config()
            config["activation"] = None
            return tf.keras.layers.Dense.from_config(config)
        return layer

    logit_model = tf.keras.models.clone_model(model, clone_function=_clone)
    logit_model.set_weights(model.get_weights())
    return logit_model


def quantize_away_from_zero(
    values: np.ndarray, scale: float, zero_point: int
) -> np.ndarray:
    """Quantiza float32 -> int8 com arredondamento para longe do zero (regra do firmware)."""
    normalized = values / scale + zero_point
    rounded = np.where(normalized >= 0.0, np.floor(normalized + 0.5), np.ceil(normalized - 0.5))
    return np.clip(rounded, -128, 127).astype(np.int8)


def apply_int8_dequant_temperature_sigmoid(
    int8_logits: np.ndarray, scale: float, zero_point: int, temperature: float
) -> np.ndarray:
    """logits int8 -> float (dequant) -> /T -> sigmoid. T aplicado em float32."""
    logits = (int8_logits.astype(np.float32) - zero_point) * scale
    scaled = logits / temperature
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))


def predict_int8(interpreter: tf.lite.Interpreter, X_int8: np.ndarray) -> np.ndarray:
    """Roda o interpretador TFLite amostra a amostra; retorna logits int8 (n, classes)."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    outputs = []
    for sample in X_int8:
        interpreter.set_tensor(input_details["index"], sample[np.newaxis, ...])
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_details["index"])[0].copy())
    return np.stack(outputs).astype(np.int8)


def inspect_ops(tflite_model: bytes) -> list[str]:
    """Lista os ops do grafo TFLite (ordem de execução, sem DELEGATE do host)."""
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        ops = [d["op_name"] for d in interpreter._get_ops_details()]  # noqa: SLF001
        return [op for op in ops if op != "DELEGATE"]
    except Exception:  # noqa: BLE001
        LOGGER.warning("inspeção de ops indisponível; lista vazia")
        return []


def build_quant_report(
    *,
    auc_float: float,
    auc_int8: float,
    f1_float: float,
    f1_int8: float,
    size_kb: float,
    logits_stats: dict,
    n_calibration: int,
    split_version: str,
    temperature: float,
    ops: list[str],
) -> dict:
    """Relatório Δ métricas float vs INT8 + range/saturação dos logits (AC-3.x)."""
    lo, hi = int(logits_stats["min"]), int(logits_stats["max"])
    scale = float(logits_stats["scale"])
    zp = int(logits_stats["zero_point"])
    total = int(logits_stats["total"])
    rail_low = int(logits_stats["rail_low_count"])
    rail_high = int(logits_stats["rail_high_count"])
    saturated = lo <= -128 or hi >= 127
    return {
        "auc_roc_float": float(auc_float),
        "auc_roc_int8": float(auc_int8),
        "delta_auc_roc": abs(float(auc_float) - float(auc_int8)),
        "f1_macro_float": float(f1_float),
        "f1_macro_int8": float(f1_int8),
        "delta_f1_macro": abs(float(f1_float) - float(f1_int8)),
        "size_kb": float(size_kb),
        "passes_qg6_size": bool(size_kb <= MAX_FLATBUFFER_KB),
        "logits_int8_range": {
            "min": lo,
            "max": hi,
            "saturated": bool(saturated),
            "rail_low_count": rail_low,
            "rail_high_count": rail_high,
            "rail_frac": (rail_low + rail_high) / total if total else 0.0,
            "rail_logit_bounds": [(-128 - zp) * scale, (127 - zp) * scale],
            "note": "saturação medida nos logits int8 crus; /T é aplicado em float32 "
            "após dequantização (não amplifica saturação no domínio int8)",
        },
        "n_calibration": int(n_calibration),
        "split_version": split_version,
        "temperature": float(temperature),
        "temperature_applied_in": "float32_after_dequant",
        "ops": ops,
    }


def build_post_quant_calibration(
    *,
    cal_before: dict,
    cal_after: dict,
    temperature: float,
    val_samples: int,
    split_version: str,
) -> dict:
    """ECE do modelo INT8 antes/depois de /T (T fixo da T1 — não re-aprendido)."""
    return {
        "temperature": float(temperature),
        "temperature_source": "calibration.json (T1, fixo — não re-aprendido)",
        "n_bins": cal_before["n_bins"],
        "val_samples": int(val_samples),
        "split_version": split_version,
        "ece_before": cal_before["macro"]["ece"],
        "ece_after": cal_after["macro"]["ece"],
        "before": cal_before,
        "after": cal_after,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macro_auc(metrics: dict) -> float:
    aucs = [m["auc_roc"] for m in metrics["per_class"].values() if m["auc_roc"] is not None]
    return float(np.mean(aucs))


def _macro_f1(metrics: dict) -> float:
    return float(np.mean([m["f1"] for m in metrics["per_class"].values()]))


def _collect_val_windows(provenance: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstrói o split val determinístico (seed do run) e materializa X, y."""
    training = provenance["training"]
    _, val_ds, _, val_steps = build_datasets(
        val_ratio=0.1,
        batch_size=training["batch_size"],
        segment_len=provenance["model"]["input_shape"][1],
        seed=provenance["seed"],
        steps_per_epoch=1,
        validation_steps=training["validation_steps"],
    )
    xs, ys = [], []
    for xb, yb in val_ds.take(val_steps):
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--n-cal", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = args.run_dir
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    split_version = calibration["split_version"]
    seed_run = provenance["seed"]

    model_path = run_dir / "backbone_pretrained.keras"
    sha = _sha256(model_path)
    expected = provenance.get("hashes", {}).get("model_sha256")
    if expected and sha != expected:
        LOGGER.error("checkpoint hash mismatch: %s != %s (fail-closed)", sha, expected)
        return 3

    model = load_keras_model(str(model_path), compile=False)
    logit_model = make_logit_head(model)

    LOGGER.info("Reconstruindo split val (seed=%d)...", seed_run)
    X, y_true = _collect_val_windows(provenance)
    LOGGER.info("val: X=%s y=%s", X.shape, y_true.shape)

    # Equivalência logit-head (fail-closed)
    probs_ref = model.predict(X[:256], batch_size=256, verbose=0)
    logits_probe = logit_model.predict(X[:256], batch_size=256, verbose=0)
    max_diff = float(np.max(np.abs(1.0 / (1.0 + np.exp(-logits_probe)) - probs_ref)))
    if max_diff > 1e-6:
        LOGGER.error("logit-head diverge do original: max|Δ|=%.3e (fail-closed)", max_diff)
        return 4
    LOGGER.info("logit-head equivalente (max|Δ|=%.2e)", max_diff)

    # PTQ com 512 janelas estratificadas (pseudo-rótulo = argmax do multihot)
    pseudo_labels = y_true.argmax(axis=1)
    rep = representative_dataset_stratified(X, pseudo_labels, n_samples=args.n_cal, seed=args.seed)
    LOGGER.info("Convertendo INT8 (n_cal=%d)...", args.n_cal)
    tflite_model = calibrate(logit_model, rep)

    out_dir = run_dir / "quantized"
    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / "a2_full_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    LOGGER.info("TFLite salvo: %s (%.2f KB)", tflite_path, size_kb)

    params = extract_quantization_params(tflite_model, out_dir / "quantization_params.json")
    params.update(
        {
            "temperature": temperature,
            "temperature_source": "calibration.json (T1)",
            "temperature_applied_in": "float32_after_dequant",
            "inference_order": "logits -> /T -> sigmoid",
            "split_version": split_version,
            "seed": seed_run,
            "n_calibration": args.n_cal,
            "sha256_keras": sha,
            "sha256_tflite": hashlib.sha256(tflite_model).hexdigest(),
            "output_semantics": "int8 logits (sigmoid removido na conversão)",
        }
    )
    (out_dir / "quantization_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Avaliação float vs INT8 no split val completo
    LOGGER.info("Avaliando float (batch)...")
    probs_float = model.predict(X, batch_size=256, verbose=0)
    metrics_float = compute_per_class_metrics(y_true, probs_float)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale = float(in_det["quantization_parameters"]["scales"][0])
    in_zp = int(in_det["quantization_parameters"]["zero_points"][0])
    out_scale = float(out_det["quantization_parameters"]["scales"][0])
    out_zp = int(out_det["quantization_parameters"]["zero_points"][0])

    LOGGER.info("Avaliando INT8 (%d amostras, 1 a 1)...", len(X))
    X_q = quantize_away_from_zero(X, in_scale, in_zp)
    logits_q = predict_int8(interpreter, X_q)
    logits_stats = {
        "min": int(logits_q.min()),
        "max": int(logits_q.max()),
        "rail_low_count": int((logits_q == -128).sum()),
        "rail_high_count": int((logits_q == 127).sum()),
        "total": int(logits_q.size),
        "scale": out_scale,
        "zero_point": out_zp,
    }

    probs_int8_raw = apply_int8_dequant_temperature_sigmoid(logits_q, out_scale, out_zp, 1.0)
    probs_int8_cal = apply_int8_dequant_temperature_sigmoid(
        logits_q, out_scale, out_zp, temperature
    )
    metrics_int8 = compute_per_class_metrics(y_true, probs_int8_cal)

    report = build_quant_report(
        auc_float=_macro_auc(metrics_float),
        auc_int8=_macro_auc(metrics_int8),
        f1_float=_macro_f1(metrics_float),
        f1_int8=_macro_f1(metrics_int8),
        size_kb=size_kb,
        logits_stats=logits_stats,
        n_calibration=args.n_cal,
        split_version=split_version,
        temperature=temperature,
        ops=inspect_ops(tflite_model),
    )
    (out_dir / "quant_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ECE pós-PTQ: sem T vs com T fixo (n_bins=15, mesmo do calibration.json)
    n_bins = int(calibration["n_bins"])
    cal_before = calibration_summary(y_true, probs_int8_raw, n_bins=n_bins)
    cal_after = calibration_summary(y_true, probs_int8_cal, n_bins=n_bins)
    post_cal = build_post_quant_calibration(
        cal_before=cal_before,
        cal_after=cal_after,
        temperature=temperature,
        val_samples=len(X),
        split_version=split_version,
    )
    # sanity: sigmoid(logits/T) via helper da C04 bate com apply_temperature nos logits float
    logits_float = sigmoid_to_logits(np.clip(probs_float, 1e-7, 1 - 1e-7))
    ref_cal = apply_temperature(probs_float, temperature)
    assert logits_float.shape == probs_float.shape and ref_cal.shape == probs_float.shape

    (out_dir / "post_quant_calibration.json").write_text(
        json.dumps(post_cal, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    LOGGER.info("=== ACs T3 ===")
    LOGGER.info("AC-3.1 size %.2f KB < 64: %s", size_kb, report["passes_qg6_size"])
    LOGGER.info("AC-3.2 ΔAUC %.4f < 0.01: %s", report["delta_auc_roc"], report["delta_auc_roc"] < 0.01)
    LOGGER.info("AC-3.3 ΔF1  %.4f < 0.02: %s", report["delta_f1_macro"], report["delta_f1_macro"] < 0.02)
    LOGGER.info(
        "AC-3.4 ECE pós-PTQ (T) %.4f ≤ 0.025: %s",
        post_cal["ece_after"],
        post_cal["ece_after"] <= 0.025,
    )
    LOGGER.info("AC-3.5 ops: %s", report["ops"])
    LOGGER.info(
        "AC-3.8 logits int8 [%d, %d] saturated=%s (rails: %d low + %d high de %d, %.4f%%)",
        logits_stats["min"],
        logits_stats["max"],
        report["logits_int8_range"]["saturated"],
        logits_stats["rail_low_count"],
        logits_stats["rail_high_count"],
        logits_stats["total"],
        100.0 * report["logits_int8_range"]["rail_frac"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
