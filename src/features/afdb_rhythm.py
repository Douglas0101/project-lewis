"""Episódios de ritmo do AFDB (nível 3 da ontologia v3 — escopo: episódio).

Decisão D3 (``docs/rebuild_spec/01_clinical_ontology_decision.md``): AFDB entra
como **tarefa de ritmo**, nunca no classificador de batimentos. Este módulo
constrói episódios não sobrepostos de 10 s (5000 amostras a 500 Hz) a partir
dos sinais processados (C02) e dos intervalos de ritmo das anotações ``.atr``.

Regras:

- limites de intervalo reescalonados de 250 Hz (nativo) para 500 Hz
  (``round(b * 500 / 250)``, mesma regra de ``pipeline.py`` — DQ-01);
- um episódio só é rotulado se estiver **integralmente contido** em um único
  intervalo de ritmo; episódios de fronteira são descartados e contados;
- ritmos mapeados pela tabela única ``AFDB_RHYTHM_MAP_V3`` (ontology_v3);
- ``N`` → SINUS; ``J`` → JUNCTIONAL; ``AFIB`` → AFIB; ``AFL`` → AFL;
  demais → OTHER_RHYTHM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb

from src.features.ontology_v3 import AFDB_RHYTHM_MAP_V3, ONTOLOGY_VERSION
from src.training_integrity.integrity import (
    afdb_episode_sample_id,
    exclusive_publication,
    publish_staged_file_exclusive,
    temporary_staging_path,
    waveform_row_sha256,
)

LOGGER = logging.getLogger("lewis.camada03.afdb_rhythm")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw_afdb"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "afdb"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

FS_NATIVE = 250.0
TARGET_FS = 500.0
EPISODE_SECONDS = 10.0


def _safe_int(value: Any, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid integer for {context}: {value!r}") from error


EPISODE_LEN = _safe_int(EPISODE_SECONDS * TARGET_FS, context="AFDB episode length")  # 5000 amostras


def _load_rhythm_intervals_500(
    base: Path,
    sig_len_500: int | None = None,
) -> List[Tuple[int, int, int, int, str, str]]:
    """Carrega intervalos de ritmo do .atr e reescalona para o relógio de 500 Hz."""
    atr = wfdb.rdann(str(base), extension="atr")
    atr_samples = np.asarray(atr.sample, dtype=np.int64)
    aux_note = atr.aux_note if atr.aux_note is not None else []
    atr_aux = [str(a).strip().lstrip("(") for a in aux_note]  # '(AFIB' -> 'AFIB'

    # Fecha o último intervalo com o tamanho do sinal nativo (.hea).
    header = wfdb.rdheader(str(base))
    sig_len_native = _safe_int(header.sig_len, context="AFDB native signal length")
    if sig_len_500 is None:
        sig_len_500 = _safe_int(
            round(sig_len_native * TARGET_FS / FS_NATIVE),
            context="AFDB target signal length",
        )

    intervals: List[Tuple[int, int, int, int, str, str]] = []
    for i in range(len(atr_samples)):
        start_n = _safe_int(atr_samples[i], context="AFDB interval start")
        end_n = (
            _safe_int(atr_samples[i + 1], context="AFDB interval end")
            if i + 1 < len(atr_samples)
            else sig_len_native
        )
        raw_rhythm = atr_aux[i] if i < len(atr_aux) else "N"
        rhythm = AFDB_RHYTHM_MAP_V3.get(raw_rhythm, "OTHER_RHYTHM")
        start_500 = _safe_int(
            round(start_n * TARGET_FS / FS_NATIVE), context="AFDB target interval start"
        )
        end_500 = _safe_int(
            round(end_n * TARGET_FS / FS_NATIVE), context="AFDB target interval end"
        )
        end_500 = min(end_500, sig_len_500)
        if end_500 > start_500:
            intervals.append((start_n, end_n, start_500, end_500, raw_rhythm, rhythm))
    return intervals


def build_afdb_rhythm_episodes(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = FEATURES_DIR,
) -> Dict[str, Any]:
    """Constrói episódios de ritmo de todos os registros AFDB processados.

    Returns
    -------
    Dict[str, Any]
        Estatísticas da construção (por ritmo, por registro, descartes).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    X_all: List[np.ndarray] = []
    y_all: List[str] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "ontology_version": ONTOLOGY_VERSION,
        "episode_seconds": EPISODE_SECONDS,
        "fs_hz": TARGET_FS,
        "records_ok": 0,
        "records_skipped": [],
        "n_boundary_episodes_dropped": 0,
        "by_rhythm": {},
    }

    npy_files = sorted(processed_dir.glob("*_ECG1.npy"))
    if not npy_files:
        LOGGER.warning("Nenhum .npy processado em %s", processed_dir)
        return stats

    for npy_path in npy_files:
        record_id = npy_path.name.split("_")[0]
        base = raw_dir / record_id
        if not base.with_suffix(".hea").exists() or not base.with_suffix(".atr").exists():
            stats["records_skipped"].append(record_id)
            LOGGER.warning("AFDB %s sem .hea/.atr — pulado", record_id)
            continue

        sig = np.load(npy_path, allow_pickle=False).astype(np.float32)
        intervals = _load_rhythm_intervals_500(base, len(sig))
        if not intervals:
            stats["records_skipped"].append(record_id)
            continue

        n_episodes_record = 0
        for ep_start in range(0, len(sig) - EPISODE_LEN + 1, EPISODE_LEN):
            ep_end = ep_start + EPISODE_LEN
            interval = None
            for candidate in intervals:
                _, _, start, end, _, _ = candidate
                if start <= ep_start and ep_end <= end:
                    interval = candidate
                    break
            if interval is None:
                stats["n_boundary_episodes_dropped"] += 1
                continue
            (
                interval_start_native,
                interval_end_native,
                interval_start_target,
                interval_end_target,
                rhythm_original,
                rhythm_canonical,
            ) = interval
            episode_idx = ep_start // EPISODE_LEN
            ep_start_native = _safe_int(
                round(ep_start * FS_NATIVE / TARGET_FS),
                context="AFDB native episode start",
            )
            ep_end_native = _safe_int(
                round(ep_end * FS_NATIVE / TARGET_FS),
                context="AFDB native episode end",
            )
            sample_id = afdb_episode_sample_id(
                record_id,
                episode_idx,
                ep_start,
                ep_end,
            )
            waveform = sig[ep_start:ep_end].reshape(EPISODE_LEN, 1)
            waveform_hash = waveform_row_sha256(waveform)
            X_all.append(waveform)
            y_all.append(rhythm_canonical)
            rows.append(
                {
                    "dataset_id": "afdb",
                    "record_id": record_id,
                    "patient_id": "UNKNOWN_OR_UNVERIFIED",
                    "episode_idx": episode_idx,
                    "segment_id": sample_id,
                    "sample_id": sample_id,
                    "waveform_sha256": waveform_hash,
                    "source_sampling_rate": FS_NATIVE,
                    "target_sampling_rate": TARGET_FS,
                    "start_sample_native": ep_start_native,
                    "end_sample_native": ep_end_native,
                    "start_time_seconds": ep_start_native / FS_NATIVE,
                    "end_time_seconds": ep_end_native / FS_NATIVE,
                    "start_sample_target": ep_start,
                    "end_sample_target": ep_end,
                    "interval_start_native": interval_start_native,
                    "interval_end_native": interval_end_native,
                    "interval_start_target": interval_start_target,
                    "interval_end_target": interval_end_target,
                    "rhythm_original": rhythm_original,
                    "rhythm_canonical": rhythm_canonical,
                    "split": "rhythm_exploratory",
                    "fold": -1,
                }
            )
            n_episodes_record += 1
            stats["by_rhythm"][rhythm_canonical] = stats["by_rhythm"].get(rhythm_canonical, 0) + 1
        stats["records_ok"] += 1
        LOGGER.info(
            "AFDB %s: %d episódios (%d intervalos)", record_id, n_episodes_record, len(intervals)
        )

    if not X_all:
        LOGGER.warning("Nenhum episódio gerado")
        return stats

    X = np.stack(X_all).astype(np.float32)
    y = np.asarray(y_all, dtype=str)
    df = pd.DataFrame(rows)
    sample_ids = np.asarray(df["sample_id"].astype(str).tolist(), dtype=str)
    waveform_hashes = np.asarray(df["waveform_sha256"].astype(str).tolist(), dtype=str)

    npz_path = output_dir / "afdb_rhythm_episodes.npz"
    parquet_path = output_dir / "afdb_rhythm_episodes.parquet"
    lock_path = output_dir / ".afdb_rhythm_episodes.publish.lock"
    with exclusive_publication(lock_path, (npz_path, parquet_path)):
        with (
            temporary_staging_path(npz_path) as staged_npz,
            temporary_staging_path(parquet_path) as staged_parquet,
        ):
            np.savez(
                staged_npz,
                X=X,
                y=y,
                sample_id=sample_ids,
                waveform_sha256=waveform_hashes,
            )
            df.to_parquet(staged_parquet, index=False)
            publish_staged_file_exclusive(staged_npz, npz_path)
            publish_staged_file_exclusive(staged_parquet, parquet_path)

    stats.update(
        {
            "n_episodes": _safe_int(len(y), context="AFDB episode count"),
            "n_records": _safe_int(df["record_id"].nunique(), context="AFDB record count"),
            "npz": str(npz_path),
            "parquet": str(parquet_path),
        }
    )
    LOGGER.info(
        "AFDB ritmo: %d episódios de %d registros | por ritmo: %s | fronteira descartados: %d",
        stats["n_episodes"],
        stats["n_records"],
        stats["by_rhythm"],
        stats["n_boundary_episodes_dropped"],
    )
    return stats


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    stats = build_afdb_rhythm_episodes()
    print(f"AFDB ritmo: {stats.get('n_episodes', 0)} episódios | {stats['by_rhythm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
