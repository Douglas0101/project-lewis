"""Parse and validate quantization parameters deployed in firmware headers."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TensorQuantizationParams(BaseModel):
    """Validated INT8 affine-quantization parameters for one tensor."""

    model_config = ConfigDict(frozen=True)

    scale: float = Field(gt=0.0)
    zero_point: int = Field(ge=-128, le=127)
    dtype: str = "int8"


class FirmwareQuantizationParams(BaseModel):
    """Input and output parameters compiled into the firmware."""

    model_config = ConfigDict(frozen=True)

    input: TensorQuantizationParams
    output: TensorQuantizationParams


_DEFINE_PATTERN = re.compile(
    r"^#define\s+(?P<name>LEWIS_QUANTIZATION_PARAMS_(?:INPUT|OUTPUT)_"
    r"(?:SCALE|ZERO_POINT))\s+(?P<value>[-+0-9.eE]+)f?\s*$",
    re.MULTILINE,
)


def load_firmware_quantization_params(header_path: Path) -> FirmwareQuantizationParams:
    """Load deployed parameters from ``quantization_params.h``.

    Raises
    ------
    ValueError
        If one of the four required macros is absent or invalid.
    """
    values = {
        match.group("name"): match.group("value")
        for match in _DEFINE_PATTERN.finditer(header_path.read_text(encoding="utf-8"))
    }
    required = {
        "LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE",
        "LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT",
        "LEWIS_QUANTIZATION_PARAMS_OUTPUT_SCALE",
        "LEWIS_QUANTIZATION_PARAMS_OUTPUT_ZERO_POINT",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"Missing firmware quantization macros: {missing}")

    try:
        input_scale = float(values["LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE"])
        input_zero_point = int(values["LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT"])
        output_scale = float(values["LEWIS_QUANTIZATION_PARAMS_OUTPUT_SCALE"])
        output_zero_point = int(values["LEWIS_QUANTIZATION_PARAMS_OUTPUT_ZERO_POINT"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid firmware quantization macro value: {exc}") from exc

    return FirmwareQuantizationParams(
        input=TensorQuantizationParams(scale=input_scale, zero_point=input_zero_point),
        output=TensorQuantizationParams(scale=output_scale, zero_point=output_zero_point),
    )
