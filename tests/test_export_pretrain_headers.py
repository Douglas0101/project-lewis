"""Tests for the A2-full firmware header export (T4, C08)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_pretrain_a2_full_headers import (  # noqa: E402
    TEMPERATURE,
    build_config_header,
    export_headers,
)

PARAMS = {
    "input": {"scale": 0.0784313753247261, "zero_point": -1, "dtype": "int8"},
    "output": {"scale": 0.033894360065460205, "zero_point": -15, "dtype": "int8"},
    "temperature": TEMPERATURE,
    "sha256_tflite": "",
}


def test_build_config_header_content():
    content = build_config_header(
        temperature=TEMPERATURE,
        sha256_tflite="ab" * 32,
        run_id="20260728_053011_pretrain_chapman",
        version="3.1.0",
    )
    assert "PRETRAIN_A2_FULL_TEMPERATURE = 0.3741036858f" in content
    assert "PRETRAIN_A2_FULL_INPUT_LEN 500" in content
    assert "PRETRAIN_A2_FULL_OUTPUT_LEN 5" in content
    assert ("ab" * 32) in content
    assert "20260728_053011_pretrain_chapman" in content
    assert "#ifndef PRETRAIN_A2_FULL_CONFIG_H" in content


def test_export_headers_writes_three_files(tmp_path):
    import hashlib

    tflite_bytes = bytes(range(256)) * 10
    params = dict(PARAMS, sha256_tflite=hashlib.sha256(tflite_bytes).hexdigest())
    written = export_headers(tflite_bytes, params, tmp_path)

    model_h = tmp_path / "pretrain_a2_full_int8.h"
    quant_h = tmp_path / "pretrain_a2_full_quant_params.h"
    config_h = tmp_path / "pretrain_a2_full_config.h"
    assert set(written) == {model_h, quant_h, config_h}
    assert "pretrain_a2_full_int8_tflite" in model_h.read_text(encoding="utf-8")
    quant_text = quant_h.read_text(encoding="utf-8")
    assert "PRETRAIN_A2_FULL_QUANT_PARAMS_INPUT_SCALE 0.0784313753f" in quant_text
    assert "PRETRAIN_A2_FULL_QUANT_PARAMS_OUTPUT_ZERO_POINT -15" in quant_text
    assert "PRETRAIN_A2_FULL_TEMPERATURE = 0.3741036858f" in config_h.read_text(
        encoding="utf-8"
    )


def test_export_headers_rejects_sha_mismatch(tmp_path):
    params = dict(PARAMS, sha256_tflite="00" * 32)
    with pytest.raises(ValueError, match="sha256"):
        export_headers(b"\x01\x02\x03", params, tmp_path)
