"""Loader de batimentos F do PTB-XL a partir de registros com diagnóstico AFIB.

PTB-XL possui rótulos de diagnóstico global (SCP-ECG) por registro de 10 s.
Este módulo seleciona registros `AFIB=100` em `scp_codes`, detecta R-peaks no
sinal processado (lead II) e atribui label F a todos os batimentos.

O rótulo é fraco (diagnóstico global, não beat-level) e deve ser documentado
como tal no manifest.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def _parse_scp_codes(val) -> dict:
    """Parseia a coluna scp_codes do PTB-XL."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}
    return {}


def select_afib_records(
    ptbxl_database_csv: Path,
    confidence_threshold: float = 100.0,
) -> list[str]:
    """Retorna lista de ecg_id com AFIB acima do threshold de confiança.

    Parameters
    ----------
    ptbxl_database_csv : Path
        Caminho para `ptbxl_database.csv`.
    confidence_threshold : float
        Limiar de confiança do statement AFIB (default: 100.0).

    Returns
    -------
    list[str]
        Lista de ecg_id (ex: '00001').
    """
    import pandas as pd

    df = pd.read_csv(ptbxl_database_csv)
    scp_codes = [_parse_scp_codes(v) for v in df["scp_codes"]]
    mask = [d.get("AFIB", 0) >= confidence_threshold for d in scp_codes]
    selected = df[mask]
    return [f"{eid:05d}" for eid in selected["ecg_id"].tolist()]


def detect_r_peaks_ptbxl(
    signal: np.ndarray,
    fs: float = 500.0,
    min_hr: float = 40.0,
    max_hr: float = 200.0,
) -> np.ndarray:
    """Detecta R-peaks simples em sinal de ECG de 10 s.

    Usa scipy.signal.find_peaks com distancia minima derivada de max_hr.
    """
    min_distance = int(fs * 60.0 / max_hr)
    # Prominencia adaptativa baseada no desvio padrao local
    prominence = float(np.std(signal)) * 0.5
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    return np.asarray(peaks, dtype=np.int64)


def load_ptbxl_afib_beats(
    ecg_id: str,
    processed_dir: Path,
    fs: float = 500.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Carrega R-peaks e labels F para um registro PTB-XL com AFIB.

    Parameters
    ----------
    ecg_id : str
        Identificador do registro (ex: '00001').
    processed_dir : Path
        Diretorio com sinais processados (`{ecg_id}_lr_II.npy`).
    fs : float
        Frequencia de amostragem do sinal processado.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        (r_peaks, labels) ou None se o arquivo nao existir.
    """
    fname = f"{ecg_id}_lr_II.npy"
    npy_path = processed_dir / fname
    if not npy_path.exists():
        return None

    sig = np.load(npy_path).astype(np.float32)
    r_peaks = detect_r_peaks_ptbxl(sig, fs=fs)
    labels = np.full(len(r_peaks), "F", dtype=object)
    return r_peaks, labels
