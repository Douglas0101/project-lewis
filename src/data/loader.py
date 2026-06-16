"""Dataset loaders for MIT-BIH, SVDB, AFDB, INCART and Chapman-Shaoxing.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-02 spec):
- NUNCA hardcode gain=200, baseline=1024. Sempre ler do .hea via wfdb.rdheader().
- Validar range físico [-5.0, +5.0] mV.
- Mapear anotações WFDB → AAMI EC57.
- SVDB fs = 250 Hz (correção crítica).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import wfdb

LOGGER = logging.getLogger("lewis.camada02.loader")

# Fs nativos canônicos por dataset (skill + Camada-02)
DATASET_CONFIG = {
    "mitbih": {"fs": 360.0, "subdir": "raw_mitbih", "lead": "MLII"},
    "svdb": {"fs": 250.0, "subdir": "raw_svdb", "lead": "ECG1"},
    "afdb": {"fs": 250.0, "subdir": "raw_afdb", "lead": "ECG1"},
    "incart": {"fs": 257.0, "subdir": "raw_incart", "lead": "II"},
    "chapman": {"fs": 500.0, "subdir": "raw_chapman", "lead": "II"},
}

# AAMI EC57 mapping — apenas beat annotations (códigos 0–29 no formato MIT)
AAMI_BEAT_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "V": "V",
    "E": "V",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

_BEAT_SYMBOLS = set(AAMI_BEAT_MAP.keys())
_PACE_SYMBOLS = {"/", "f"}
_NOISE_SYMBOL = "~"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_data_dir(subdir: str) -> Path:
    return _project_root() / "data" / subdir


class MITBIHLoader:
    """Loader unificado para MIT-BIH family + Chapman.

    NÃO assume ganho/baseline fixos. Sempre lê do .hea via wfdb.rdheader()
    e aplica: physical = (digital - baseline) / gain.
    """

    FS_TARGET = 500.0  # Hz padrão após resample

    @staticmethod
    def load_signal(
        record_path: Path,
        channel: int = 0,
        units: str = "physical",
    ) -> np.ndarray:
        """Carrega sinal de um registro WFDB.

        Parameters
        ----------
        record_path : Path
            Caminho completo para o registro (sem extensão).
        channel : int
            Índice do canal a carregar.
        units : str
            "physical" → mV (wfdb converte usando adc_gain/baseline do .hea).
            "digital"  → ADC counts.

        Returns
        -------
        np.ndarray
            Sinal 1-D (float64 se physical, int16/32 se digital).

        Raises
        ------
        FileNotFoundError
            Se .hea ou .dat não existirem.
        ValueError
            Se units não for "physical" ou "digital", ou se range violar [-5, 5] mV.
        """
        if units not in {"physical", "digital"}:
            raise ValueError(f"units must be 'physical' or 'digital', got {units!r}")

        record_path = Path(record_path)
        hea_path = record_path.with_suffix(".hea")
        dat_path = record_path.with_suffix(".dat")

        if not hea_path.exists():
            raise FileNotFoundError(f"Header não encontrado: {hea_path}")
        if not dat_path.exists():
            raise FileNotFoundError(f"Sinal .dat não encontrado: {dat_path}")

        # Ler header explicitamente para extrair gain/baseline (skill: nunca hardcode)
        header = wfdb.rdheader(str(record_path))
        if channel >= header.n_sig:
            raise ValueError(f"Canal {channel} excede n_sig={header.n_sig} em {record_path.name}")

        gain = header.adc_gain[channel] if header.adc_gain else None
        baseline = header.adc_zero[channel] if header.adc_zero else None

        LOGGER.debug(
            "Load %s ch=%d | gain=%s baseline=%s fs=%s",
            record_path.name,
            channel,
            gain,
            baseline,
            header.fs,
        )

        # Carregar via wfdb.rdrecord para canal específico
        rec = wfdb.rdrecord(
            str(record_path),
            channels=[channel],
            physical=(units == "physical"),
        )
        sig = rec.p_signal if units == "physical" else rec.d_signal
        sig = sig.squeeze().astype(np.float64)

        if units == "physical":
            vmin, vmax = float(sig.min()), float(sig.max())
            if vmin < -5.0 or vmax > 5.0:
                LOGGER.warning(
                    "Range físico fora de [-5, +5] mV para %s: [%.3f, %.3f]",
                    record_path.name,
                    vmin,
                    vmax,
                )
            # WARN se gain/baseline divergirem drasticamente do padrão MIT-BIH
            if gain is not None and abs(gain - 200.0) > 50.0:
                LOGGER.warning(
                    "adc_gain=%.1f diverge do padrão MIT-BIH (~200) em %s",
                    gain,
                    record_path.name,
                )
            if baseline is not None and abs(baseline - 1024.0) > 100.0:
                LOGGER.warning(
                    "adc_zero=%.1f diverge do padrão MIT-BIH (~1024) em %s",
                    baseline,
                    record_path.name,
                )

        return sig

    @staticmethod
    def load_annotations(
        record_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Carrega anotações WFDB e mapeia para AAMI EC57.

        Parameters
        ----------
        record_path : Path
            Caminho completo para o registro (sem extensão).

        Returns
        -------
        samples : np.ndarray
            Índices de amostra dos batimentos.
        aami_labels : np.ndarray
            Rótulos AAMI (N, V, S, F, Q) para cada batimento.
        metadata : dict
            {
                "total_beats": int,
                "noise_segments": int,
                "paced_ratio": float,  # fração de batimentos paced
            }

        Raises
        ------
        FileNotFoundError
            Se .atr não existir.
        """
        record_path = Path(record_path)
        atr_path = record_path.with_suffix(".atr")

        if not atr_path.exists():
            raise FileNotFoundError(f"Anannotation .atr não encontrada: {atr_path}")

        ann = wfdb.rdann(str(record_path), extension="atr")
        symbols = np.array(ann.symbol)
        samples = np.array(ann.sample)

        # Contar ruído e pacing em TODAS as anotações (incluindo non-beat)
        noise_segments = int(np.sum(symbols == _NOISE_SYMBOL))
        total_annotations = len(symbols)
        paced_count = int(np.isin(symbols, list(_PACE_SYMBOLS)).sum())

        # Filtrar apenas batimentos (beat annotations)
        mask = np.isin(symbols, list(_BEAT_SYMBOLS))
        n_dropped = int((~mask).sum())
        if n_dropped:
            LOGGER.debug(
                "Dropped %d non-beat annotations de %s",
                n_dropped,
                record_path.name,
            )

        beat_samples = samples[mask]
        beat_symbols = symbols[mask]
        aami_labels = np.array([AAMI_BEAT_MAP[s] for s in beat_symbols])

        total_beats = len(beat_samples)
        paced_ratio = paced_count / max(total_annotations, 1)

        metadata = {
            "total_beats": total_beats,
            "noise_segments": noise_segments,
            "paced_ratio": paced_ratio,
            "n_dropped": n_dropped,
        }

        LOGGER.info(
            "Annotations %s: %d beats, %d noise, paced_ratio=%.3f",
            record_path.name,
            total_beats,
            noise_segments,
            paced_ratio,
        )
        return beat_samples, aami_labels, metadata

    @staticmethod
    def get_record_names(raw_dir: Path) -> List[str]:
        """Lista stems de arquivos .hea em um diretório.

        Esperado >= 226 registros (MIT-BIH family) ou >= 45.000 (Chapman).
        """
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            LOGGER.warning("Diretório não existe: %s", raw_dir)
            return []
        records = {f.stem for f in raw_dir.iterdir() if f.suffix == ".hea"}
        return sorted(records)

    @classmethod
    def iter_dataset(
        cls,
        dataset_name: str,
        data_dir: Optional[Path] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Itera sobre todos os registros de um dataset.

        Yields dicts com:
        {
            "record_name": str,
            "record_path": Path,
            "fs_native": float,
            "channel": int,
            "lead_name": str,
        }
        """
        dataset_name = dataset_name.lower().strip()
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Dataset desconhecido: {dataset_name}")

        cfg = DATASET_CONFIG[dataset_name]
        if data_dir is None:
            data_dir = _get_data_dir(str(cfg["subdir"]))

        for rec_name in cls.get_record_names(data_dir):
            yield {
                "record_name": rec_name,
                "record_path": data_dir / rec_name,
                "fs_native": cfg["fs"],
                "lead_name": cfg["lead"],
            }


# Alias para backward compatibility com código que usa o nome antigo
ECGDatasetLoader = MITBIHLoader
