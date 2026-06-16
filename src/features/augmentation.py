"""ECG data augmentation — aplicável APENAS no treino do fine-tuning.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-03 spec §3.7):
- jitter: ruído Gaussiano σ = 1% do std do sinal
- baseline_wander: senoide 0.05–0.5 Hz, amp < 0.2 mV
- powerline_noise: 50/60 Hz + harmônicos, amp < 0.05 mV
- time_warp: stretch 0.95–1.05× via scipy.interpolate
- Aplicar APENAS no treino do fine-tuning (MIT-BIH+)
- NUNCA no pré-treino (Chapman) ou teste
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import interpolate

LOGGER = logging.getLogger("lewis.camada03.augmentation")


class ECGAugmenter:
    """Apply controlled augmentation to ECG segments.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def jitter(self, x: np.ndarray, std_factor: float = 0.01) -> np.ndarray:
        """Add Gaussian noise proportional to signal std.

        Parameters
        ----------
        x : np.ndarray
            Signal segment (1-D).
        std_factor : float
            Noise std as fraction of signal std. Default 0.01 (1%).

        Returns
        -------
        np.ndarray
            Augmented signal.
        """
        noise = self.rng.normal(0, std_factor * float(np.std(x)), size=x.shape)
        return (x + noise).astype(x.dtype, copy=False)

    def baseline_wander(
        self,
        x: np.ndarray,
        fs: float = 500.0,
        freq_range: tuple[float, float] = (0.05, 0.5),
        amp_range: tuple[float, float] = (0.05, 0.2),
    ) -> np.ndarray:
        """Add low-frequency sinusoid simulating respiration.

        Parameters
        ----------
        x : np.ndarray
            Signal segment.
        fs : float
            Sampling frequency.
        freq_range : tuple
            Frequency range in Hz. Default (0.05, 0.5).
        amp_range : tuple
            Amplitude range in mV. Default (0.05, 0.2).

        Returns
        -------
        np.ndarray
            Augmented signal.
        """
        t = np.arange(len(x)) / fs
        freq = self.rng.uniform(*freq_range)
        phase = self.rng.uniform(0, 2 * np.pi)
        amplitude = self.rng.uniform(*amp_range)
        wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return x + wander

    def powerline_noise(
        self,
        x: np.ndarray,
        fs: float = 500.0,
        freq: float = 60.0,
        amp_range: tuple[float, float] = (0.02, 0.05),
    ) -> np.ndarray:
        """Add powerline interference (50/60 Hz) + harmonics.

        Parameters
        ----------
        x : np.ndarray
            Signal segment.
        fs : float
            Sampling frequency.
        freq : float
            Powerline frequency (50 or 60 Hz).
        amp_range : tuple
            Amplitude range in mV. Default (0.02, 0.05).

        Returns
        -------
        np.ndarray
            Augmented signal.
        """
        t = np.arange(len(x)) / fs
        amplitude = self.rng.uniform(*amp_range)
        harmonic = self.rng.choice([1, 2, 3])
        noise = amplitude * np.sin(2 * np.pi * freq * harmonic * t)
        return x + noise

    def time_warp(self, x: np.ndarray, max_stretch: float = 0.05) -> np.ndarray:
        """Slightly stretch/compress signal along time axis.

        Parameters
        ----------
        x : np.ndarray
            Signal segment.
        max_stretch : float
            Max relative stretch. Default 0.05 (±5%).

        Returns
        -------
        np.ndarray
            Augmented signal with original length.
        """
        old_len = len(x)
        stretch = self.rng.uniform(1.0 - max_stretch, 1.0 + max_stretch)
        new_len = max(2, int(old_len * stretch))

        # Interpolate to stretched length
        f = interpolate.interp1d(
            np.arange(old_len),
            x,
            kind="cubic",
            fill_value="extrapolate",
        )
        x_warped = f(np.linspace(0, old_len - 1, new_len))

        # Interpolate back to original length
        f2 = interpolate.interp1d(
            np.arange(new_len),
            x_warped,
            kind="cubic",
            fill_value="extrapolate",
        )
        return f2(np.linspace(0, new_len - 1, old_len))

    def apply(
        self,
        x: np.ndarray,
        fs: float = 500.0,
        p: float = 0.5,
        methods: Optional[list[str]] = None,
        *,
        stage: str = "train",
    ) -> np.ndarray:
        """Apply augmentations with probability p each.

        Parameters
        ----------
        x : np.ndarray
            Signal segment.
        fs : float
            Sampling frequency.
        p : float
            Probability of applying each augmentation. Default 0.5.
        methods : list[str], optional
            Which methods to apply. Default: ["jitter", "baseline", "powerline", "warp"].
        stage : str
            Pipeline stage. Augmentation is only allowed for ``"train"`` (fine-tuning).
            Raises ``ValueError`` for ``"pretrain"``, ``"test"``, ``"val"`` etc.

        Returns
        -------
        np.ndarray
            Augmented signal.
        """
        if stage != "train":
            raise ValueError(
                f"Augmentation is only allowed during training (stage='train'), "
                f"got stage={stage!r}"
            )

        if methods is None:
            methods = ["jitter", "baseline", "powerline", "warp"]

        if "jitter" in methods and self.rng.random() < p:
            x = self.jitter(x)
        if "baseline" in methods and self.rng.random() < p:
            x = self.baseline_wander(x, fs)
        if "powerline" in methods and self.rng.random() < p:
            x = self.powerline_noise(x, fs)
        if "warp" in methods and self.rng.random() < p:
            x = self.time_warp(x)

        return x
