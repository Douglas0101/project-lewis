"""Replica Python deterministica do firmware/src/dsp/adc_stub.c.

Usada pelos testes de bit-exatidao para gerar o mesmo sinal de ECG de teste
produzido pelo C, sem depender de bibliotecas matematicas que introduziriam
diferencas de arredondamento.
"""

from __future__ import annotations

import numpy as np


# Tabela de seno * 256 do firmware (fases 0..63).
_SINE_TABLE = [
    0, 6, 13, 19, 25, 31, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92,
    98, 104, 109, 115, 121, 126, 132, 137, 142, 147, 152, 157, 162, 167, 171, 176,
    180, 184, 188, 192, 196, 199, 203, 206, 209, 212, 215, 218, 220, 223, 225, 227,
    229, 231, 233, 234, 236, 237, 238, 239, 240, 241, 241, 242, 242, 242, 243, 243,
]


def _lcg_next(state: int) -> int:
    """LCG deterministico de 32 bits usado pelo adc_stub.c."""
    return ((state * 1103515245) + 12345) & 0xFFFFFFFF


def _sin_lut(phase_256: int) -> int:
    """Seno aproximado por lookup table (escala * 256)."""
    quadrant = (phase_256 >> 6) & 0x3
    idx = phase_256 & 0x3F
    if quadrant == 0:
        value = _SINE_TABLE[idx]
    elif quadrant == 1:
        value = _SINE_TABLE[63 - idx]
    elif quadrant == 2:
        value = -_SINE_TABLE[idx]
    else:
        value = -_SINE_TABLE[63 - idx]
    return value


def _exp_approx(x: float) -> float:
    """Aproximacao de exp(-x) usada pelo adc_stub.c."""
    if x <= 0.0:
        return 1.0
    if x > 5.0:
        return 0.0
    term = 1.0
    total = 1.0
    for i in range(1, 9):
        term *= -x / float(i)
        total += term
    return total


def adc_stub_get_beat(idx: int) -> np.ndarray:
    """Replica exata de lewis_adc_stub_get_beat em C.

    Retorna array int8 de 500 amostras do batimento sintetico ``idx``.
    """
    state = (42 + idx * 7) & 0xFFFFFFFF
    out = np.empty(500, dtype=np.int8)
    for i in range(500):
        t = i / 500.0
        phase = (i * 256) // 500
        qrs = 0.8 * _exp_approx(-200.0 * (t - 0.5) * (t - 0.5))
        baseline = 0.05 * (_sin_lut(phase // 3) / 256.0)
        state = _lcg_next(state)
        r = state % 101
        noise = (float(r) - 50.0) / 500.0
        sample = qrs + baseline + noise
        value = int(sample * 40.0)
        if value > 127:
            value = 127
        elif value < -128:
            value = -128
        out[i] = value
    return out


def adc_stub_get_sine(amplitude_mv: float) -> np.ndarray:
    """Replica exata de lewis_adc_stub_get_sine em C."""
    out = np.empty(500, dtype=np.int8)
    for i in range(500):
        phase = (i * 256) // 500
        sample = amplitude_mv * (_sin_lut(phase) / 256.0)
        value = int(sample * 40.0)
        if value > 127:
            value = 127
        elif value < -128:
            value = -128
        out[i] = value
    return out
