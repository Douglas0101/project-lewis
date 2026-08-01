"""Tests for A2-full INT8 quantization pipeline (T3, C05).

Covers the logit-head conversion, firmware-compatible int8 quantization,
temperature application order (logits -> /T -> sigmoid, in float) and the
schema of the report artifacts written to ``experiments/<run>/quantized/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.quantize_pretrain_a2_full import (  # noqa: E402
    TEMPERATURE,
    apply_int8_dequant_temperature_sigmoid,
    build_post_quant_calibration,
    build_quant_report,
    make_logit_head,
    predict_int8,
    quantize_away_from_zero,
)


def _tiny_sigmoid_model(num_classes: int = 5) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(500, 1), name="input")
    x = tf.keras.layers.Conv1D(4, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def test_make_logit_head_preserves_predictions():
    rng = np.random.default_rng(42)
    model = _tiny_sigmoid_model()
    X = rng.normal(size=(8, 500, 1)).astype(np.float32)

    logit_model = make_logit_head(model)
    logits = logit_model.predict(X, verbose=0)
    probs_ref = model.predict(X, verbose=0)
    probs_from_logits = 1.0 / (1.0 + np.exp(-logits))

    assert logit_model.get_layer("output").activation == tf.keras.activations.linear
    np.testing.assert_allclose(probs_from_logits, probs_ref, atol=1e-6)


def test_quantize_away_from_zero_matches_firmware_rule():
    values = np.array([0.4, 0.5, -0.5, -0.6, 1.5, -1.5, 127.9, -200.0], dtype=np.float32)
    q = quantize_away_from_zero(values, scale=1.0, zero_point=0)
    # half away from zero: 0.5->1, -0.5->-1, 1.5->2, -1.5->-2; clip [-128, 127]
    np.testing.assert_array_equal(q, np.array([0, 1, -1, -1, 2, -2, 127, -128], dtype=np.int8))


def test_apply_int8_dequant_temperature_sigmoid_order():
    int8_logits = np.array([[10, -20, 0, 5, -3]], dtype=np.int8)
    scale, zp, temperature = 0.25, 2, 0.5
    probs = apply_int8_dequant_temperature_sigmoid(int8_logits, scale, zp, temperature)
    logits_f = (int8_logits.astype(np.float32) - zp) * scale
    expected = 1.0 / (1.0 + np.exp(-(logits_f / temperature)))
    np.testing.assert_allclose(probs, expected, atol=1e-6)
    # T < 1 afia: probabilidade máxima maior que com T = 1
    probs_t1 = apply_int8_dequant_temperature_sigmoid(int8_logits, scale, zp, 1.0)
    assert probs.max() > probs_t1.max()


def test_build_quant_report_schema():
    report = build_quant_report(
        auc_float=0.8639,
        auc_int8=0.8630,
        f1_float=0.60,
        f1_int8=0.595,
        size_kb=40.1,
        logits_stats={
            "min": -110,
            "max": 118,
            "rail_low_count": 0,
            "rail_high_count": 0,
            "total": 1000,
            "scale": 0.034,
            "zero_point": -15,
        },
        n_calibration=512,
        split_version="chapman-record-disjoint-val0.1-seed13",
        temperature=TEMPERATURE,
        ops=["CONV_2D", "FULLY_CONNECTED"],
    )
    for key in (
        "auc_roc_float",
        "auc_roc_int8",
        "delta_auc_roc",
        "f1_macro_float",
        "f1_macro_int8",
        "delta_f1_macro",
        "size_kb",
        "passes_qg6_size",
        "logits_int8_range",
        "n_calibration",
        "split_version",
        "temperature",
        "ops",
        "temperature_applied_in",
    ):
        assert key in report, key
    assert report["delta_auc_roc"] == pytest.approx(abs(0.8639 - 0.8630))
    assert report["passes_qg6_size"] is True
    rng = report["logits_int8_range"]
    assert rng["min"] == -110 and rng["max"] == 118 and rng["saturated"] is False
    assert rng["rail_low_count"] == 0 and rng["rail_high_count"] == 0
    assert rng["rail_logit_bounds"] == [
        pytest.approx((-128 + 15) * 0.034),
        pytest.approx((127 + 15) * 0.034),
    ]
    assert report["temperature_applied_in"] == "float32_after_dequant"


def test_build_quant_report_flags_saturation():
    report = build_quant_report(
        auc_float=0.86,
        auc_int8=0.86,
        f1_float=0.6,
        f1_int8=0.6,
        size_kb=40.0,
        logits_stats={
            "min": -128,
            "max": 127,
            "rail_low_count": 12,
            "rail_high_count": 34,
            "total": 1000,
            "scale": 0.034,
            "zero_point": -15,
        },
        n_calibration=512,
        split_version="s",
        temperature=TEMPERATURE,
        ops=[],
    )
    rng = report["logits_int8_range"]
    assert rng["saturated"] is True
    assert rng["rail_low_count"] == 12 and rng["rail_high_count"] == 34
    assert rng["rail_frac"] == pytest.approx(46 / 1000)


def test_build_post_quant_calibration_schema():
    from src.models.pretrain_evaluation import calibration_summary

    rng = np.random.default_rng(7)
    y_prob = rng.uniform(size=(300, 5))
    y_true = (rng.uniform(size=(300, 5)) < 0.3).astype(int)
    cal = calibration_summary(y_true, y_prob, n_bins=15)
    out = build_post_quant_calibration(
        cal_before=cal,
        cal_after=cal,
        temperature=TEMPERATURE,
        val_samples=300,
        split_version="chapman-record-disjoint-val0.1-seed13",
    )
    assert out["temperature"] == TEMPERATURE
    assert out["n_bins"] == 15
    assert out["val_samples"] == 300
    assert 0.0 <= out["ece_before"] <= 1.0
    assert 0.0 <= out["ece_after"] <= 1.0
    assert out["temperature_source"] == "calibration.json (T1, fixo — não re-aprendido)"


@pytest.mark.slow
def test_end_to_end_tiny_model_int8_roundtrip(tmp_path):
    from src.quantization.ptq import calibrate, representative_dataset_random

    rng = np.random.default_rng(3)
    model = _tiny_sigmoid_model()
    logit_model = make_logit_head(model)
    X = rng.normal(size=(64, 500, 1)).astype(np.float32)
    rep = representative_dataset_random(X, n_samples=16, seed=42)

    tflite_bytes = calibrate(logit_model, rep)
    (tmp_path / "tiny_int8.tflite").write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    assert in_det["dtype"] == np.int8 and out_det["dtype"] == np.int8

    in_scale, in_zp = in_det["quantization_parameters"]["scales"][0], in_det[
        "quantization_parameters"
    ]["zero_points"][0]
    X_q = quantize_away_from_zero(X, in_scale, in_zp)
    logits_q = predict_int8(interpreter, X_q)
    assert logits_q.shape == (64, 5) and logits_q.dtype == np.int8

    out_scale, out_zp = out_det["quantization_parameters"]["scales"][0], out_det[
        "quantization_parameters"
    ]["zero_points"][0]
    probs = apply_int8_dequant_temperature_sigmoid(logits_q, out_scale, out_zp, TEMPERATURE)
    assert probs.shape == (64, 5)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
