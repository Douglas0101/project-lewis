"""High-level ECG preprocessing pipeline.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-02 spec):
- Filtro Butterworth bandpass 0.5–40 Hz, ordem 4, filtfilt (zero-phase)
- Detrend linear
- Normalização z-score global, eps=1e-12
- Idempotência via lineage JSON
- DLQ para falhas
- Lineage completo com raw_checksum e pipeline steps
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal

from . import resampler

LOGGER = logging.getLogger("lewis.camada02.preprocessor")

# Diretórios padrão do pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LINEAGE_DIR = PROJECT_ROOT / "data" / "lineage"
DLQ_PATH = PROJECT_ROOT / "data" / ".dlq" / "preprocess_failures.jsonl"

# Config default inline (fallback se YAML não existir)
DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "1.0.0",
    "resample": {
        "target_fs": 500.0,
        "method": "resample_poly",
        "padtype": "line",
    },
    "filter": {
        "type": "butterworth",
        "order": 4,
        "lowcut": 0.5,
        "highcut": 40.0,
        "implementation": "filtfilt",
    },
    "detrend": {
        "type": "linear",
    },
    "normalization": {
        "type": "zscore_global",
        "eps": 1.0e-12,
        "per_record": False,
    },
}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Calcula SHA256 de um arquivo em streaming."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _append_dlq(dlq_path: Path, event: dict) -> None:
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    with dlq_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return DEFAULT_CONFIG.copy()
    config_path = Path(config_path)
    if not config_path.exists():
        LOGGER.warning("Config %s não encontrado — usando defaults", config_path)
        return DEFAULT_CONFIG.copy()
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if cfg is None:
            return DEFAULT_CONFIG.copy()
        # Suporta tanto root com versão direta quanto nested
        if "version" in cfg and "filter" in cfg:
            return cfg
        # Caso esteja aninhado sob chave como "preprocess_v1.0"
        for key, val in cfg.items():
            if isinstance(val, dict) and "filter" in val:
                val["version"] = val.get("version", key)
                return val
        return DEFAULT_CONFIG.copy()
    except ImportError:
        LOGGER.warning("PyYAML não instalado — usando defaults")
        return DEFAULT_CONFIG.copy()
    except Exception as exc:
        LOGGER.warning("Erro ao carregar config: %s — usando defaults", exc)
        return DEFAULT_CONFIG.copy()


class ECGPreprocessor:
    """Pré-processamento determinístico e reprodutível de sinais ECG.

    Parameters
    ----------
    config_path : Path, optional
        Caminho para YAML de config. Se None, usa DEFAULT_CONFIG.
    dlq_path : Path, optional
        Caminho para o arquivo DLQ (dead-letter queue). Padrão:
        ``PROJECT_ROOT / "data" / ".dlq" / "preprocess_failures.jsonl"``.
        Pode ser sobrescrito em testes para isolar a DLQ de produção.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        dlq_path: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        lineage_dir: Optional[Path] = None,
    ):
        self.cfg = _load_config(config_path)
        self.config_version = self.cfg.get("version", "1.0.0")
        self._dlq_path = Path(dlq_path) if dlq_path is not None else DLQ_PATH
        self._processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
        self._lineage_dir = Path(lineage_dir) if lineage_dir is not None else LINEAGE_DIR

        # Parâmetros de filtro
        filter_cfg = self.cfg.get("filter", {})
        self.fs = float(filter_cfg.get("target_fs", 500.0))
        self.lowcut = float(filter_cfg.get("lowcut", 0.5))
        self.highcut = float(filter_cfg.get("highcut", 40.0))
        self.order = int(filter_cfg.get("order", 4))

        # Pré-computar coeficientes Butterworth (evita recálculo por registro)
        nyq = self.fs / 2.0
        self.b_band, self.a_band = signal.butter(
            self.order,
            [self.lowcut / nyq, self.highcut / nyq],
            btype="band",
        )

        # Parâmetros de normalização
        norm_cfg = self.cfg.get("normalization", {})
        self.eps = float(norm_cfg.get("eps", 1.0e-12))
        self.per_record = bool(norm_cfg.get("per_record", False))

        # Estatísticas globais (populadas externamente se per_record=False)
        self._global_mean: Optional[float] = None
        self._global_std: Optional[float] = None

        LOGGER.info(
            "ECGPreprocessor inicializado | config=%s | fs=%.1f | band=[%.2f, %.2f] | order=%d",
            self.config_version,
            self.fs,
            self.lowcut,
            self.highcut,
            self.order,
        )

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Aplica filtro passa-banda Butterworth zero-phase (filtfilt).

        filtfilt processa forward + backward, cancelando distorção de fase.
        """
        return signal.filtfilt(self.b_band, self.a_band, x)

    def detrend(self, x: np.ndarray) -> np.ndarray:
        """Remove drift DC linear."""
        return signal.detrend(x, type="linear")

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score global ou por registro.

        Raises
        ------
        ValueError
            Se ``per_record=False`` e as estatísticas globais não foram
            definidas via :meth:`set_global_stats`.
        """
        if self.per_record:
            mean = float(np.mean(x))
            std = float(np.std(x))
            return (x - mean) / (std + self.eps)

        if self._global_mean is None or self._global_std is None:
            raise ValueError(
                "Estatísticas globais não definidas. "
                "Chame set_global_stats(mean, std) antes da normalização "
                "quando normalization.per_record=False."
            )

        return (x - self._global_mean) / (self._global_std + self.eps)

    def _check_idempotency(
        self,
        dataset: str,
        record_id: str,
        raw_path: Path,
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """Checa se registro já foi processado com mesma config e raw_checksum.

        Retorna (signal, metadata) se idempotente, None caso contrário.
        """
        lineage_path = self._lineage_dir / dataset / f"{record_id}_lineage.json"
        if not lineage_path.exists():
            return None

        try:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if lineage.get("preprocess_config") != self.config_version:
            return None

        # Checksum do raw: preferimos .dat, fallback para .hea
        raw_dat = raw_path.with_suffix(".dat")
        raw_hea = raw_path.with_suffix(".hea")
        checksum_path = raw_dat if raw_dat.exists() else raw_hea
        if not checksum_path.exists():
            return None

        current_checksum = _sha256_file(checksum_path)
        if lineage.get("raw_checksum") != current_checksum:
            LOGGER.info(
                "Checksum mudou para %s/%s — reprocessando",
                dataset,
                record_id,
            )
            return None

        # Carregar sinal já processado
        out_path = Path(lineage["output"]["path"])
        if out_path.exists():
            LOGGER.info(
                "Idempotente: pulando %s/%s (config=%s)",
                dataset,
                record_id,
                self.config_version,
            )
            sig = np.load(out_path)
            return sig, lineage
        return None

    def _save_outputs(
        self,
        x: np.ndarray,
        lineage: dict,
        dataset: str,
        record_id: str,
        lead_name: str,
    ) -> Path:
        """Salva .npy e lineage.json."""
        out_dir = self._processed_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        lineage_dir = self._lineage_dir / dataset
        lineage_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / f"{record_id}_{lead_name}.npy"
        np.save(npy_path, x)

        lineage_path = lineage_dir / f"{record_id}_lineage.json"
        lineage["output"]["path"] = str(npy_path)
        lineage_path.write_text(
            json.dumps(lineage, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return npy_path

    def process(
        self,
        x: np.ndarray,
        *,
        record_id: str,
        dataset: str,
        fs_orig: float,
        raw_path: Path,
        lead_name: str,
        gain: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Pipeline completo: resample → filter → detrend → normalize.

        Parameters
        ----------
        x : np.ndarray
            Sinal bruto 1-D (já em mV ou ADC counts convertidos).
        record_id : str
            Identificador do registro.
        dataset : str
            Nome canônico do dataset.
        fs_orig : float
            Frequência nativa do sinal.
        raw_path : Path
            Caminho base do registro bruto (para checksum e lineage).
        lead_name : str
            Nome do lead selecionado.
        gain : float, optional
            ADC gain lido do .hea (para lineage).
        baseline : float, optional
            ADC zero lido do .hea (para lineage).

        Returns
        -------
        tuple
            (x_processed, metadata_dict)

        Raises
        ------
        ValueError, Exception
            Falhas são capturadas, logadas em DLQ e re-raised.
        """
        dataset = dataset.lower().strip()
        raw_path = Path(raw_path)

        # 1. Idempotência
        cached = self._check_idempotency(dataset, record_id, raw_path)
        if cached is not None:
            return cached

        # Checksum do raw para lineage
        raw_dat = raw_path.with_suffix(".dat")
        raw_hea = raw_path.with_suffix(".hea")
        checksum_path = raw_dat if raw_dat.exists() else raw_hea
        raw_checksum = _sha256_file(checksum_path) if checksum_path.exists() else ""

        lineage: Dict[str, Any] = {
            "record_id": record_id,
            "dataset": dataset,
            "version": "1.0.0",
            "raw_checksum": raw_checksum,
            "preprocess_config": self.config_version,
            "pipeline": [],
            "output": {},
            "quality_gate": {"QG1_pass": True, "QG2_pass": None},  # nosec B105
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        metadata: Dict[str, Any] = {
            "record_id": record_id,
            "dataset": dataset,
            "input_range_mV": [float(x.min()), float(x.max())],
        }

        step = "validate"
        try:
            # Validação prévia: estatísticas globais obrigatórias quando per_record=False
            if not self.per_record and (self._global_mean is None or self._global_std is None):
                raise ValueError(
                    "Estatísticas globais não definidas. "
                    "Chame set_global_stats(mean, std) antes de process() "
                    "quando normalization.per_record=False."
                )

            # Step 1: Resample
            step = "resample"
            if fs_orig != self.fs:
                x = resampler.resample_to_500hz(
                    x,
                    fs_orig,
                    padtype="line",
                    validate=True,
                )
            lineage["pipeline"].append(
                {
                    "step": "load",
                    "fs_orig": fs_orig,
                    "gain": gain,
                    "baseline": baseline,
                    "lead": lead_name,
                    "unit": "mV",
                }
            )
            from fractions import Fraction

            frac = Fraction(int(round(self.fs)), int(round(fs_orig))).limit_denominator(1000)
            lineage["pipeline"].append(
                {
                    "step": "resample",
                    "method": "resample_poly",
                    "up": frac.numerator,
                    "down": frac.denominator,
                    "fs_out": self.fs,
                }
            )

            # Step 2: Filter
            step = "filter"
            x = self.filter(x)
            lineage["pipeline"].append(
                {
                    "step": "filter",
                    "type": "butterworth",
                    "order": self.order,
                    "bandpass": [self.lowcut, self.highcut],
                    "implementation": "filtfilt",
                }
            )

            # Step 3: Detrend
            step = "detrend"
            x = self.detrend(x)
            lineage["pipeline"].append(
                {
                    "step": "detrend",
                    "type": "linear",
                }
            )

            # Step 4: Normalize
            step = "normalize"
            x = self.normalize(x)
            lineage["pipeline"].append(
                {
                    "step": "normalize",
                    "type": "zscore_global",
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                }
            )

            # Metadados de saída
            duration_sec = len(x) / self.fs
            metadata.update(
                {
                    "output_range_mV": [float(x.min()), float(x.max())],
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "duration_sec": duration_sec,
                }
            )

            lineage["output"].update(
                {
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "duration_sec": duration_sec,
                    "range_mV": [float(x.min()), float(x.max())],
                }
            )

            # Garantir float32 para consistência
            x = x.astype(np.float32, copy=False)

            # Salvar
            npy_path = self._save_outputs(x, lineage, dataset, record_id, lead_name)
            metadata["output_path"] = str(npy_path)

            LOGGER.info(
                "Processado %s/%s | len=%d | dur=%.1fs | range=[%.3f, %.3f] mV",
                dataset,
                record_id,
                len(x),
                duration_sec,
                x.min(),
                x.max(),
            )
            return x, metadata

        except Exception as exc:
            _append_dlq(
                self._dlq_path,
                {
                    "record_id": record_id,
                    "dataset": dataset,
                    "step": step,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "raw_path": str(raw_path),
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            LOGGER.error(
                "Falha no pré-processamento de %s/%s na etapa '%s': %s",
                dataset,
                record_id,
                step,
                exc,
            )
            raise

    def set_global_stats(self, mean: float, std: float) -> None:
        """Define estatísticas globais para z-score global (quando per_record=False)."""
        self._global_mean = mean
        self._global_std = std
