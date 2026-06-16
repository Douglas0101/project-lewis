"""Referencia Python para os filtros DSP do firmware.

Gerado automaticamente por scripts/generate_filter_coeffs.py.
Contem os mesmos coeficientes SOS causais usados em C e uma implementacao
biquad transposed direct-form II equivalente.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

FS = 500.0
LOWCUT = 0.5
HIGHCUT = 40.0
BANDPASS_ORDER = 4
NOTCH_FREQ = 60.0
NOTCH_BW = 2.0

BANDPASS_COEFFS_FLAT = np.array(
    [
        # section 0
        2.138798733e-03,
        4.277597465e-03,
        2.138798733e-03,
        -1.227875879e00,
        3.935230602e-01,
        # section 1
        1.000000000e00,
        2.000000000e00,
        1.000000000e00,
        -1.486663673e00,
        6.949675580e-01,
        # section 2
        1.000000000e00,
        -2.000000000e00,
        1.000000000e00,
        -1.988215471e00,
        9.882564157e-01,
        # section 3
        1.000000000e00,
        -2.000000000e00,
        1.000000000e00,
        -1.995246749e00,
        9.952864068e-01,
    ],
    dtype=np.float32,
)
BANDPASS_COEFFS = BANDPASS_COEFFS_FLAT.reshape(-1, 5)

NOTCH_COEFFS_FLAT = np.array(
    [
        # section 0
        9.875889381e-01,
        -1.439842705e00,
        9.875889381e-01,
        -1.439842705e00,
        9.751778762e-01,
    ],
    dtype=np.float32,
)
NOTCH_COEFFS = NOTCH_COEFFS_FLAT.reshape(-1, 5)


def _biquad_sample(x: float, coeffs: np.ndarray, state: np.ndarray) -> float:
    """Processa uma amostra em uma secao biquad (transposed DF-II)."""
    b0, b1, b2, a1, a2 = coeffs
    d0, d1 = state[0], state[1]
    y = b0 * x + d0
    state[0] = b1 * x - a1 * y + d1
    state[1] = b2 * x - a2 * y
    return y


def _cascade_sample(x: float, coeffs: np.ndarray, states: np.ndarray) -> float:
    """Processa uma amostra em cascata de biquads."""
    n_sections = coeffs.shape[0]
    y = x
    for i in range(n_sections):
        y = _biquad_sample(y, coeffs[i], states[i])
    return y


def cascade_process_block(
    input_arr: np.ndarray,
    coeffs: np.ndarray,
    states: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Processa um bloco em cascata de biquads.

    Returns
    -------
    output, states
    """
    n = len(input_arr)
    out = np.empty(n, dtype=np.float32)
    n_sections = coeffs.shape[0]
    if states is None:
        states = np.zeros((n_sections, 2), dtype=np.float32)
    for i in range(n):
        out[i] = _cascade_sample(float(input_arr[i]), coeffs, states)
    return out, states


def bandpass_filter(
    x: np.ndarray,
    states: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtro passa-banda 0.5-40 Hz (causal)."""
    return cascade_process_block(x, BANDPASS_COEFFS, states)


def notch_filter(
    x: np.ndarray,
    states: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtro notch 60 Hz (causal)."""
    return cascade_process_block(x, NOTCH_COEFFS, states)


def filter_chain(
    x: np.ndarray,
    bp_states: np.ndarray | None = None,
    notch_states: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pipeline bandpass -> notch (causal)."""
    y, bp_states = bandpass_filter(x, bp_states)
    y, notch_states = notch_filter(y, notch_states)
    return y, bp_states, notch_states


def design_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Reproduz o projeto dos coeficientes (para auditoria)."""
    bp_sos = signal.iirfilter(
        N=BANDPASS_ORDER,
        Wn=[LOWCUT, HIGHCUT],
        fs=FS,
        btype="band",
        ftype="butter",
        output="sos",
    )
    q = NOTCH_FREQ / NOTCH_BW
    b, a = signal.iirnotch(w0=NOTCH_FREQ, Q=q, fs=FS)
    notch_sos = signal.tf2sos(b, a)
    return bp_sos.astype(np.float32), notch_sos.astype(np.float32)
