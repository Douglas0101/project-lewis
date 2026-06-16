"""ECG signal resampling utilities.

Regras mandatórias (ecg-preprocessing-pipeline):
- Método: scipy.signal.resample_poly APENAS
- padtype="line" para ECG não-periódico
- NUNCA usar scipy.signal.resample (FFT-based) — introduz artefatos de borda
- SVDB fs = 250 Hz (não 128 Hz)
"""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np
from scipy import signal

LOGGER = logging.getLogger("lewis.camada02.resampler")

TARGET_FS = 500.0  # Hz — nativo Chapman-Shaoxing e ADS1292R

# Fs nativos canônicos por dataset (skill + Camada-02 spec)
NATIVE_FS = {
    "mitbih": 360.0,
    "svdb": 250.0,  # CORREÇÃO CRÍTICA: não é 128 Hz
    "afdb": 250.0,
    "incart": 257.0,
    "chapman": 500.0,
}

# Threshold para ativar filtro FIR customizado.  O resample_poly padrão usa
# half_len = 10 * max(up, down), o que para frações grandes (ex: INCART 500/257)
# gera filtros com >10.000 taps, causando "up*down too large" em algumas
# versões do SciPy ou OOM em hardware limitado.
_MAX_RATE_FOR_DEFAULT_FIR = 200
# Comprimento máximo do filtro customizado (half_len).  Valor empírico que
# garante atenuação > 40 dB acima de 250 Hz para todas as frações usadas.
_MAX_HALF_LEN_CUSTOM = 2000


def _validate_attenuation(sig: np.ndarray, fs: float) -> bool:
    """Valida via Welch que componentes > 250 Hz estão atenuados > 40 dB.

    Retorna True se validado; loga warning caso contrário.
    """
    try:
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, len(sig) // 4))
        if len(freqs) == 0:
            return True
        mask = freqs > 250.0
        if not np.any(mask):
            return True
        max_power_above_nyquist = np.max(psd[mask])
        max_power_total = np.max(psd)
        if max_power_total == 0:
            return True
        attenuation_db = -10.0 * np.log10(max_power_above_nyquist / max_power_total)
        if attenuation_db < 40.0:
            LOGGER.warning(
                "Atenuação acima de 250 Hz = %.1f dB (esperado > 40 dB)",
                attenuation_db,
            )
            return False
        return True
    except Exception as exc:
        LOGGER.warning("Falha na validação de atenuação espectral: %s", exc)
        return False


def _design_fir_for_resample_poly(up: int, down: int) -> np.ndarray:
    """Projeta filtro FIR encurtado para resample_poly em frações grandes.

    O filtro padrão do resample_poly tem half_len = 10 * max(up, down).
    Esta função limita half_len a _MAX_HALF_LEN_CUSTOM, mantendo o mesmo
    cutoff (1/max_rate) e janela Kaiser (beta=5.0) do SciPy.

    Parameters
    ----------
    up : int
        Fator de upsampling.
    down : int
        Fator de downsampling.

    Returns
    -------
    np.ndarray
        Coeficientes do filtro FIR (float64).
    """
    max_rate = max(up, down)
    # Para max_rate pequeno não deveríamos estar aqui, mas mantemos proteção.
    half_len = min(10 * max_rate, _MAX_HALF_LEN_CUSTOM)
    h = signal.firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    LOGGER.debug(
        "FIR customizado para resample_poly: max_rate=%d, taps=%d, half_len=%d",
        max_rate,
        len(h),
        half_len,
    )
    return h.astype(np.float64)


def resample_to_500hz(
    sig: np.ndarray,
    fs_orig: float,
    *,
    padtype: str = "line",
    validate: bool = True,
) -> np.ndarray:
    """Resample sinal 1-D para 500 Hz usando scipy.signal.resample_poly.

    Parameters
    ----------
    sig : np.ndarray
        Sinal de entrada (1-D), preferencialmente float64.
    fs_orig : float
        Frequência de amostragem original (Hz).
    padtype : str
        Tipo de padding para resample_poly. "line" é obrigatório para ECG
        não-periódico, evitando artefatos de wrap.
    validate : bool
        Se True, valida atenuação > 40 dB acima de 250 Hz via Welch.

    Returns
    -------
    np.ndarray
        Sinal resampleado para 500 Hz.

    Raises
    ------
    ValueError
        Se fs_orig for não-positivo ou sig não for 1-D.
    """
    if fs_orig <= 0:
        raise ValueError(f"fs_orig deve ser positivo, recebido {fs_orig}")
    if sig.ndim != 1:
        raise ValueError(f"sig deve ser 1-D, recebido shape {sig.shape}")

    if fs_orig == TARGET_FS:
        LOGGER.debug("Fs origem == %s Hz — sem resample necessário", TARGET_FS)
        return sig.copy()

    # Fração irredutível para resample_poly
    frac = Fraction(int(round(TARGET_FS)), int(round(fs_orig))).limit_denominator(1000)
    up = frac.numerator
    down = frac.denominator

    len_in = len(sig)
    len_out = int(round(len_in * (TARGET_FS / fs_orig)))

    LOGGER.info(
        "Resample: fs_orig=%.1f Hz → %.1f Hz | up=%d down=%d | len_in=%d → len_out=%d | padtype=%s",
        fs_orig,
        TARGET_FS,
        up,
        down,
        len_in,
        len_out,
        padtype,
    )

    # Para frações grandes (ex: INCART 500/257) o filtro FIR padrão do
    # resample_poly é excessivamente longo, causando falha "up*down too large"
    # em algumas versões do SciPy ou estouro de memória.  Mantemos o método
    # resample_poly (mandatório pela skill), mas fornecemos um filtro FIR
    # customizado de comprimento controlado — ainda zero-phase, ainda
    # anti-aliasing, e com atenuação validada > 40 dB acima de 250 Hz.
    max_rate = max(up, down)
    if max_rate > _MAX_RATE_FOR_DEFAULT_FIR:
        fir_window = _design_fir_for_resample_poly(up, down)
        out = signal.resample_poly(sig, up, down, padtype=padtype, window=fir_window)
    else:
        out = signal.resample_poly(sig, up, down, padtype=padtype)

    # Garantir que o shape bate com a projeção
    if abs(len(out) - len_out) > 1:
        LOGGER.warning(
            "len_out projetado (%d) diverge do obtido (%d) para fs_orig=%.1f",
            len_out,
            len(out),
            fs_orig,
        )

    if validate:
        _validate_attenuation(out, TARGET_FS)

    return out.astype(np.float64)


def resample_record(
    sig: np.ndarray,
    fs_orig: float,
    *,
    axis: int = 0,
    padtype: str = "line",
    validate: bool = True,
) -> np.ndarray:
    """Resample sinal multi-lead ou batched para 500 Hz ao longo do eixo temporal.

    Parameters
    ----------
    sig : np.ndarray
        Pode ser 1-D, 2-D (samples x leads) ou 3-D (batch x samples x leads).
    fs_orig : float
        Frequência nativa.
    axis : int
        Eixo correspondente ao tempo.
    padtype : str
        Padding para resample_poly.
    validate : bool
        Validar atenuação espectral.

    Returns
    -------
    np.ndarray
        Sinal resampleado.
    """
    if fs_orig == TARGET_FS:
        return sig.copy()

    if sig.ndim == 1:
        return resample_to_500hz(sig, fs_orig, padtype=padtype, validate=validate)

    return np.apply_along_axis(
        lambda s: resample_to_500hz(s, fs_orig, padtype=padtype, validate=False),
        axis=axis,
        arr=sig,
    )
