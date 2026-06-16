"""Testes para o resampler — Quality Gate QG1 (Camada-02)."""

import numpy as np
import pytest
from scipy import signal

from src.data.resampler import (
    TARGET_FS,
    _design_fir_for_resample_poly,
    _validate_attenuation,
    resample_to_500hz,
)


@pytest.mark.qg1
class TestResamplePolyFallback:
    """Valida que frações grandes (ex: INCART 257→500) não quebram."""

    def test_incart_257hz_does_not_raise(self):
        """INCART: 257 Hz → 500 Hz deve completar sem exceção."""
        # 30 minutos de sinal sintético a 257 Hz
        sig = np.random.randn(257 * 60 * 30).astype(np.float64)
        out = resample_to_500hz(sig, 257.0, validate=True)
        assert out.dtype == np.float64
        assert len(out) > 0

    def test_incart_attenuation_above_250hz(self):
        """Atenuação pós-resample deve ser > 40 dB acima de 250 Hz."""
        t = np.linspace(0, 10, 2570, endpoint=False)
        sig = (
            np.sin(2 * np.pi * 1.0 * t)
            + 0.5 * np.sin(2 * np.pi * 5.0 * t)
            + 0.8 * np.sin(2 * np.pi * 15.0 * t)
            + 0.3 * np.sin(2 * np.pi * 30.0 * t)
        )
        out = resample_to_500hz(sig, 257.0, validate=False)
        assert _validate_attenuation(out, TARGET_FS)

    def test_custom_fir_half_len_capped(self):
        """Filtro customizado deve ter half_len limitado a 2000."""
        h = _design_fir_for_resample_poly(500, 257)
        assert len(h) == 2 * 2000 + 1
        assert h.dtype == np.float64

    def test_custom_fir_attenuation(self):
        """Filtro customizado deve rejeitar > 40 dB acima de 250 Hz."""
        h = _design_fir_for_resample_poly(500, 257)
        w, resp = signal.freqz(h, worN=8192)
        # Frequências em Hz no sinal upsampled (fs = 257*500 = 128500 Hz)
        freqs_hz = w * (257 * 500) / (2 * np.pi)
        mask = freqs_hz > 250.0
        att_db = -20 * np.log10(np.max(np.abs(resp[mask])))
        assert att_db > 40.0

    def test_mitbih_360hz_uses_default_path(self):
        """MIT-BIH: fração pequena (25/18) deve usar caminho padrão."""
        sig = np.random.randn(3600).astype(np.float64)
        out = resample_to_500hz(sig, 360.0, validate=False)
        assert out.dtype == np.float64
        assert len(out) == int(round(3600 * 500 / 360))


@pytest.mark.qg1
class TestResampleShapeAndType:
    """Valida shape, dtype e comportamento básico."""

    def test_no_resample_when_fs_equals_target(self):
        sig = np.array([1.0, 2.0, 3.0])
        out = resample_to_500hz(sig, 500.0)
        np.testing.assert_array_equal(out, sig)

    def test_raises_on_invalid_fs(self):
        with pytest.raises(ValueError, match="positivo"):
            resample_to_500hz(np.zeros(10), 0.0)

    def test_raises_on_multidim(self):
        with pytest.raises(ValueError, match="1-D"):
            resample_to_500hz(np.zeros((10, 2)), 360.0)
