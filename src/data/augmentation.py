"""Rotinas de data augmentation para sinais ECG.

Todas as transformações são aplicadas no domínio do tempo e preservam o
comprimento do segmento (500 amostras @ 500Hz), garantindo compatibilidade
com o backbone TFLM.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def oversample_class(
    X: np.ndarray,
    y: np.ndarray,
    class_idx: int,
    factor: int,
    augment_fn: Callable[..., np.ndarray] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample uma classe aplicando transformações leves.

    Parameters
    ----------
    X : np.ndarray
        Sinais, shape (n, 500, channels).
    y : np.ndarray
        Labels inteiros, shape (n,).
    class_idx : int
        Classe a ser oversampled.
    factor : int
        Fator de multiplicação (factor=5 gera 5x mais amostras da classe).
    augment_fn : callable, optional
        Função ``X_aug = augment_fn(X)`` aplicada a cada cópia. Se None,
        apenas replica as amostras (a randomização deve ocorrer depois,
        ex.: via tf.data.map).
    seed : int, optional
        Seed para RNG quando ``augment_fn`` usa numpy.

    Returns
    -------
    tuple
        (X_oversampled, y_oversampled).
    """
    if factor <= 1:
        return X, y

    mask = y == class_idx
    n_minority = int(mask.sum())
    if n_minority == 0:
        return X, y

    X_min = X[mask]
    y_min = y[mask]

    rng = np.random.default_rng(seed)
    augmented_blocks: list[np.ndarray] = []
    for _ in range(factor - 1):
        block = X_min.copy()
        if augment_fn is not None:
            block = augment_fn(block, rng=rng)
        augmented_blocks.append(block)

    X_aug = np.concatenate([X] + augmented_blocks, axis=0)
    y_aug = np.concatenate([y] + [y_min] * (factor - 1), axis=0)
    return X_aug, y_aug


def oversample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample múltiplas classes com configurações específicas por classe.

    Parameters
    ----------
    X : np.ndarray
        Sinais, shape (n, 500, channels).
    y : np.ndarray
        Labels inteiros, shape (n,).
    config : dict
        Mapa ``{class_idx: {"factor": int, "methods": list, "intensity": str}}``.
        ``methods`` pode conter ``jitter``, ``time_warp``, ``baseline_wander``,
        ``powerline_noise``, ``amplitude_scale``.
        ``intensity`` pode ser ``low``, ``medium``, ``high``.
    seed : int, optional
        Seed para RNG.

    Returns
    -------
    tuple
        (X_oversampled, y_oversampled).
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    y_out = y.copy()

    from typing import cast

    intensity_params: dict[str, dict[str, Any]] = {
        "low": {
            "scale_range": (0.95, 1.05),
            "shift_max": 5,
            "noise_std": 0.005,
            "baseline_amp": 0.05,
            "powerline_amp": 0.01,
            "time_warp_stretch": 0.02,
        },
        "medium": {
            "scale_range": (0.9, 1.1),
            "shift_max": 10,
            "noise_std": 0.01,
            "baseline_amp": 0.1,
            "powerline_amp": 0.02,
            "time_warp_stretch": 0.04,
        },
        "high": {
            "scale_range": (0.85, 1.15),
            "shift_max": 15,
            "noise_std": 0.02,
            "baseline_amp": 0.2,
            "powerline_amp": 0.03,
            "time_warp_stretch": 0.06,
        },
    }

    def _get_float(key: str) -> float:
        return float(cast(float, params[key]))

    def _get_tuple(key: str) -> tuple[float, float]:
        value = cast(tuple[float, float], params[key])
        return (float(value[0]), float(value[1]))

    for class_idx_str, class_cfg in config.items():
        class_idx = int(class_idx_str)
        factor = int(class_cfg.get("factor", 1))
        if factor <= 1:
            continue
        methods = class_cfg.get("methods", ["jitter"])
        intensity = class_cfg.get("intensity", "medium")
        params = intensity_params.get(intensity, intensity_params["medium"])

        mask = y_out == class_idx
        n_minority = int(mask.sum())
        if n_minority == 0:
            continue

        X_min = X_out[mask]
        y_min = y_out[mask]

        augmented_blocks: list[np.ndarray] = []
        for _ in range(factor - 1):
            block = X_min.copy()
            for method in methods:
                if method == "jitter":
                    block = add_gaussian_noise(block, std_factor=_get_float("noise_std"), rng=rng)
                elif method == "time_warp":
                    block = time_warp(block, max_stretch=_get_float("time_warp_stretch"), rng=rng)
                elif method == "baseline_wander":
                    block = add_baseline_wander(
                        block, amp_range=_get_float("baseline_amp"), rng=rng
                    )
                elif method == "powerline_noise":
                    block = add_powerline_noise(
                        block, amp_range=_get_float("powerline_amp"), rng=rng
                    )
                elif method == "amplitude_scale":
                    block = amplitude_scale(block, scale_range=_get_tuple("scale_range"), rng=rng)
            augmented_blocks.append(block)

        X_out = np.concatenate([X_out] + augmented_blocks, axis=0)
        y_out = np.concatenate([y_out] + [y_min] * (factor - 1), axis=0)

    return X_out, y_out


def add_gaussian_noise(
    X: np.ndarray,
    std_factor: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Adiciona ruído gaussiano proporcional ao desvio padrão do sinal."""
    if rng is None:
        rng = np.random.default_rng()
    signal_std = float(np.std(X))
    if signal_std == 0:
        return X.copy()
    noise = rng.normal(0.0, std_factor * signal_std, size=X.shape)
    return X + noise


def amplitude_scale(
    X: np.ndarray,
    scale_range: tuple[float, float] = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Escala a amplitude de cada amostra independentemente."""
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = scale_range
    scales = np.asarray(rng.uniform(lo, hi, size=X.shape[0]), dtype=np.float32)
    return X * scales[:, np.newaxis, np.newaxis]


def time_warp(
    X: np.ndarray,
    max_stretch: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Aplica warping temporal leve preservando o comprimento do sinal."""
    if rng is None:
        rng = np.random.default_rng()
    n, length, channels = X.shape
    out = np.empty_like(X)
    for i in range(n):
        stretch = rng.uniform(-max_stretch, max_stretch)
        src_idx = np.linspace(0, length - 1, length) * (1.0 + stretch)
        src_idx = np.clip(src_idx, 0, length - 1)
        for c in range(channels):
            out[i, :, c] = np.interp(np.arange(length), src_idx, X[i, :, c])
    return out


def add_baseline_wander(
    X: np.ndarray,
    amp_range: float = 0.1,
    freq_hz: float = 0.3,
    fs: float = 500.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Adiciona derivação de baseline sinusoidal de baixa frequência."""
    if rng is None:
        rng = np.random.default_rng()
    n, length, channels = X.shape
    t = np.arange(length) / fs
    out = X.copy()
    for i in range(n):
        amp = rng.uniform(-amp_range, amp_range)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        wander = amp * np.sin(2.0 * np.pi * freq_hz * t + phase)
        out[i] += wander[:, np.newaxis]
    return out


def add_powerline_noise(
    X: np.ndarray,
    amp_range: float = 0.02,
    freq_hz: float = 60.0,
    fs: float = 500.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Adiciona ruído de linha de energia sinusoidal."""
    if rng is None:
        rng = np.random.default_rng()
    n, length, channels = X.shape
    t = np.arange(length) / fs
    out = X.copy()
    for i in range(n):
        amp = rng.uniform(-amp_range, amp_range)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        noise = amp * np.sin(2.0 * np.pi * freq_hz * t + phase)
        out[i] += noise[:, np.newaxis]
    return out


def ecg_time_augment(
    X: np.ndarray,
    rng: np.random.Generator | None = None,
    scale_range: tuple[float, float] = (0.9, 1.1),
    shift_max: int = 10,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Augmentation leve no tempo para batimentos ECG.

    Transformações:
    - Escala de amplitude uniforme em [scale_range].
    - Deslocamento circular de até ±shift_max amostras.
    - Ruído gaussiano com std=noise_std * std(X).

    Parameters
    ----------
    X : np.ndarray
        Batch de sinais, shape (n, 500, channels).
    rng : np.random.Generator, optional
        Gerador de números aleatórios.
    scale_range : tuple[float, float]
        Intervalo de escala de amplitude.
    shift_max : int
        Máximo deslocamento circular em amostras.
    noise_std : float
        Desvio padrão do ruído relativo ao desvio padrão do sinal.

    Returns
    -------
    np.ndarray
        Sinais aumentados com o mesmo shape de ``X``.
    """
    if rng is None:
        rng = np.random.default_rng()

    X_aug = X.copy()
    n = X_aug.shape[0]

    # Escala de amplitude por amostra
    scales = np.asarray(rng.uniform(*scale_range, size=n))
    X_aug = X_aug * scales[:, np.newaxis, np.newaxis]

    # Deslocamento circular por amostra
    shifts = np.asarray(rng.integers(-shift_max, shift_max + 1, size=n))
    for i, shift in enumerate(shifts):
        if shift != 0:
            X_aug[i] = np.roll(X_aug[i], int(shift), axis=0)

    # Ruído gaussiano proporcional ao desvio padrão do sinal
    signal_std = float(np.std(X_aug))
    if signal_std > 0:
        noise = rng.normal(0.0, noise_std * signal_std, size=X_aug.shape)
        X_aug = X_aug + noise

    return X_aug
