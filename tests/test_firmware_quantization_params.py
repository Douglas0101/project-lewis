"""Tests for firmware quantization-parameter contract parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.quantization.firmware_params import load_firmware_quantization_params

FIRMWARE_HEADER = Path("firmware/src/ml/quantization_params.h")


def test_deployed_firmware_quantization_params_are_valid() -> None:
    params = load_firmware_quantization_params(FIRMWARE_HEADER)

    assert params.input.scale > 0.0
    assert -128 <= params.input.zero_point <= 127
    assert params.output.scale > 0.0
    assert -128 <= params.output.zero_point <= 127


def test_missing_firmware_quantization_macro_is_rejected(tmp_path: Path) -> None:
    header = tmp_path / "quantization_params.h"
    header.write_text(
        "\n".join(
            [
                "#define LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE 0.1f",
                "#define LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT -1",
                "#define LEWIS_QUANTIZATION_PARAMS_OUTPUT_SCALE 0.2f",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing firmware quantization macros"):
        load_firmware_quantization_params(header)
