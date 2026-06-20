"""ECG beat segmentation — window extraction centered on R-peaks.

Regras mandatórias (ecg-preprocessing-pipeline + spec v1.1 §4.3):
- Janela padrão: 1000 ms @ 500 Hz → 500 amostras, R próximo ao centro
  (DECISÃO_ARQUITETURAL: modelo 1D-CNN espera input shape (500, 1))
- Fallback: 600 ms @ 500 Hz → 300 amostras (RR < 600 ms, para evitar overlap)
- Sem padding com zeros — bordas insuficientes → descartar batimento
- NUNCA padding com zeros (introduz descontinuidades artificiais)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("lewis.camada02.segmenter")


class ECGSegmenter:
    """Segmenta batimentos ECG em janelas centradas nos R-peaks.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Default 500.0.
    window_ms : float
        Standard window duration in milliseconds. Default 1000.0.
    min_window_ms : float
        Minimum window duration for tachycardia (RR < min_window_ms).
        Default 600.0.
    """

    def __init__(
        self,
        fs: float = 500.0,
        window_ms: float = 1000.0,
        min_window_ms: float = 600.0,
    ):
        self.fs = fs
        self.window_ms = window_ms
        self.min_window_ms = min_window_ms

        # half_len: amostras de cada lado do R-peak
        # window_len = 2 * half_len para alinhar com input shape (500, 1) do modelo.
        self.half_len = int(round((window_ms * fs) / 2000.0))
        self.min_half_len = int(round((min_window_ms * fs) / 2000.0))
        self.window_len = 2 * self.half_len
        self.min_window_len = 2 * self.min_half_len

        LOGGER.info(
            "ECGSegmenter fs=%.1f | window=%.0fms (%d samples) | " "min_window=%.0fms (%d samples)",
            fs,
            window_ms,
            self.window_len,
            min_window_ms,
            self.min_window_len,
        )

    def segment_with_labels(
        self,
        sig: np.ndarray,
        r_peaks: np.ndarray,
        labels: np.ndarray,
        rr_intervals_ms: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Segmenta batimentos em janelas centradas nos R-peaks.

        Parameters
        ----------
        sig : np.ndarray
            Sinal pré-processado 1-D (já em mV, filtrado, resampleado 500 Hz).
        r_peaks : np.ndarray
            Índices de amostra dos R-peaks.
        labels : np.ndarray
            Rótulos AAMI (N, V, S, F, Q) para cada batimento.
        rr_intervals_ms : np.ndarray, optional
            RR intervals in ms for each beat. If None, computed from r_peaks.

        Returns
        -------
        X : np.ndarray
            Array de segmentos shape (n_segments, window_len), float32.
        y : np.ndarray
            Rótulos shape (n_segments,).
        metadata : dict
            {
                "n_total": int,
                "n_descartados_bordas": int,
                "n_usados_600ms": int,
                "n_usados_1000ms": int,
                "window_len": int,
                "fs": float,
            }

        Raises
        ------
        ValueError
            Se sig não for 1-D, ou se r_peaks/labels tiverem tamanhos incompatíveis.
        """
        if sig.ndim != 1:
            raise ValueError(f"sig deve ser 1-D, recebido shape {sig.shape}")
        if len(r_peaks) != len(labels):
            raise ValueError(
                f"r_peaks ({len(r_peaks)}) e labels ({len(labels)}) devem ter mesmo tamanho"
            )

        n_total = len(r_peaks)
        if n_total == 0:
            LOGGER.warning("Nenhum R-peak fornecido — retornando arrays vazios")
            return (
                np.empty((0, self.window_len), dtype=np.float32),
                np.empty(0, dtype=object),
                {
                    "n_total": 0,
                    "n_descartados_bordas": 0,
                    "n_usados_600ms": 0,
                    "n_usados_1000ms": 0,
                    "kept_indices": [],
                    "window_len": self.window_len,
                    "fs": self.fs,
                },
            )

        # Calcular RR intervals se não fornecidos
        if rr_intervals_ms is None:
            if n_total >= 2:
                rr_samples = np.diff(r_peaks)
                rr_intervals_ms = np.concatenate(
                    [
                        np.array([rr_samples[0] / self.fs * 1000.0]),
                        rr_samples / self.fs * 1000.0,
                    ]
                )
            else:
                # Apenas 1 pico: usar janela padrão (sem RR para decidir)
                rr_intervals_ms = np.full(n_total, self.window_ms)
        else:
            rr_intervals_ms = np.asarray(rr_intervals_ms)
            if len(rr_intervals_ms) != n_total:
                raise ValueError(
                    f"rr_intervals_ms ({len(rr_intervals_ms)}) deve ter {n_total} elementos"
                )

        segments: List[np.ndarray] = []
        segment_labels: List[str] = []
        kept_indices: List[int] = []
        n_discarded = 0
        n_600ms = 0
        n_1000ms = 0

        for i in range(n_total):
            r_idx = int(r_peaks[i])
            rr_ms = float(rr_intervals_ms[i])

            # Escolher janela baseada no RR interval
            if rr_ms < self.min_window_ms:
                half = self.min_half_len
                n_600ms += 1
            else:
                half = self.half_len
                n_1000ms += 1

            # Verificar bordas — NUNCA padding com zeros
            start = r_idx - half
            end = r_idx + half  # exclusive; 2*half = window_len (par)

            if start < 0 or end > len(sig):
                LOGGER.debug(
                    "Descartando batimento %d (R@%d): bordas insuficientes [%d, %d) "
                    "para sig_len=%d",
                    i,
                    r_idx,
                    start,
                    end,
                    len(sig),
                )
                n_discarded += 1
                continue

            seg = sig[start:end].astype(np.float32)
            expected_len = 2 * half

            # Garantir shape correto
            if len(seg) != expected_len:
                LOGGER.debug(
                    "Descartando batimento %d: segmento tem %d amostras, esperado %d",
                    i,
                    len(seg),
                    expected_len,
                )
                n_discarded += 1
                continue

            # Padronizar para window_len (1000ms) usando edge padding
            # se o segmento for menor (fallback 600ms). Edge padding repete
            # o primeiro/último valor — não introduz descontinuidades como zeros.
            if len(seg) < self.window_len:
                pad_left = (self.window_len - len(seg)) // 2
                pad_right = self.window_len - len(seg) - pad_left
                seg = np.pad(
                    seg,
                    (pad_left, pad_right),
                    mode="edge",
                )

            segments.append(seg)
            segment_labels.append(str(labels[i]))
            kept_indices.append(i)

        n_kept = len(segments)
        LOGGER.info(
            "Segmentação: %d total | %d descartados (bordas) | "
            "%d usados 1000ms | %d usados 600ms",
            n_total,
            n_discarded,
            n_1000ms,
            n_600ms,
        )

        if n_kept == 0:
            return (
                np.empty((0, self.window_len), dtype=np.float32),
                np.empty(0, dtype=object),
                {
                    "n_total": n_total,
                    "n_descartados_bordas": n_discarded,
                    "n_usados_600ms": n_600ms,
                    "n_usados_1000ms": n_1000ms,
                    "kept_indices": [],
                    "window_len": self.window_len,
                    "fs": self.fs,
                },
            )

        X = np.stack(segments, axis=0)
        y = np.array(segment_labels)

        metadata = {
            "n_total": n_total,
            "n_descartados_bordas": n_discarded,
            "n_usados_600ms": n_600ms,
            "n_usados_1000ms": n_1000ms,
            "kept_indices": np.array(kept_indices, dtype=np.int64),
            "window_len": self.window_len,
            "fs": self.fs,
        }

        return X, y, metadata

    def segment_single_beat(
        self,
        sig: np.ndarray,
        r_idx: int,
        rr_ms: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Segmenta um único batimento.

        Parameters
        ----------
        sig : np.ndarray
            Sinal 1-D.
        r_idx : int
            Índice do R-peak.
        rr_ms : float, optional
            RR interval em ms. Se None, usa janela padrão (1000ms).

        Returns
        -------
        np.ndarray or None
            Segmento float32 shape (window_len,) ou None se bordas insuficientes.
        """
        if sig.ndim != 1:
            raise ValueError(f"sig deve ser 1-D, recebido shape {sig.shape}")

        half = (
            self.min_half_len
            if (rr_ms is not None and rr_ms < self.min_window_ms)
            else self.half_len
        )
        window_len = (
            self.min_window_len
            if (rr_ms is not None and rr_ms < self.min_window_ms)
            else self.window_len
        )

        start = r_idx - half
        end = r_idx + half

        if start < 0 or end > len(sig):
            return None

        seg = sig[start:end].astype(np.float32)
        if len(seg) != window_len:
            return None

        return seg
