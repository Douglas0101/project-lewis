"""Loader de batimentos do AFDB a partir de anotações de ritmo (.atr) e R-peaks (.qrs).

O MIT-BIH Atrial Fibrillation Database (AFDB) possui:
- `.qrs`: anotacoes de batimento (R-peaks), todas com simbolo 'N'.
- `.atr`: anotacoes de ritmo, com `aux_note` indicando o ritmo ativo:
  - '(N': ritmo normal
  - '(AFIB': fibrilacao atrial
  - '(AFL': flutter atrial
  - '(J': ritmo junctional

Este modulo atribui label AAMI por batimento com base no intervalo de ritmo
em que o R-peak cai.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb


def _find_rhythm_for_sample(
    rhythm_intervals: list[tuple[int, int, str]], sample: int
) -> str | None:
    """Retorna o ritmo ativo no sample fornecido.

    Os intervalos sao semi-abertos [start, end). Se o sample nao estiver em
    nenhum intervalo, retorna None.
    """
    for start, end, rhythm in rhythm_intervals:
        if start <= sample < end:
            return rhythm
    return None


def load_afdb_beats(
    record_id: str,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Carrega R-peaks e labels AAMI para um registro AFDB.

    Parameters
    ----------
    record_id : str
        Identificador do registro AFDB.
    raw_dir : Path
        Diretorio raiz onde os arquivos raw do AFDB estao localizados.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        (r_peaks, aami_labels) ou None se nao for possivel carregar.
    """
    base = raw_dir / record_id
    if not (base.with_suffix(".hea")).exists():
        matches = list(raw_dir.rglob(f"{record_id}.hea"))
        if not matches:
            return None
        base = matches[0].with_suffix("")

    try:
        qrs = wfdb.rdann(str(base), extension="qrs")
        atr = wfdb.rdann(str(base), extension="atr")
    except Exception as exc:
        raise ValueError(f"Falha ao carregar anotacoes AFDB para {record_id}: {exc}") from exc

    r_peaks = np.asarray(qrs.sample, dtype=np.int64)

    # Constroi intervalos de ritmo a partir das anotacoes .atr
    rhythm_intervals: list[tuple[int, int, str]] = []
    atr_samples = np.asarray(atr.sample, dtype=np.int64)
    aux_note = atr.aux_note if atr.aux_note is not None else []
    atr_aux = [str(a).strip("(") for a in aux_note]  # '(AFIB' -> 'AFIB'

    # Determina tamanho do sinal para fechar o ultimo intervalo
    try:
        record = wfdb.rdrecord(str(base))
        sig_len = record.sig_len
    except Exception:
        sig_len = int(r_peaks.max()) + 1 if len(r_peaks) > 0 else 1

    for i in range(len(atr_samples)):
        start = int(atr_samples[i])
        end = int(atr_samples[i + 1]) if i + 1 < len(atr_samples) else sig_len
        rhythm = atr_aux[i] if i < len(atr_aux) else "N"
        rhythm_intervals.append((start, end, rhythm))

    # Mapeia cada R-peak para o ritmo correspondente
    labels = []
    for sample in r_peaks:
        found_rhythm = _find_rhythm_for_sample(rhythm_intervals, int(sample))
        if found_rhythm is None:
            labels.append("N")
        else:
            labels.append(found_rhythm)

    labels_arr = np.array(labels, dtype=object)

    # Mapeamento AAMI para Stage 2:
    # AFIB/AFL -> F
    # J -> N (junctional, nao usado em Stage 2)
    # N -> N (nao entra em Stage 2)
    # Qualquer outro -> N
    aami_map = {
        "AFIB": "F",
        "AFL": "F",
        "N": "N",
        "J": "N",
    }
    aami_labels = np.array([aami_map.get(str(label), "N") for label in labels_arr], dtype=object)

    return r_peaks, aami_labels


def count_afdb_f_by_record(
    raw_dir: Path,
    record_ids: list[str],
) -> dict[str, int]:
    """Conta batimentos F por registro AFDB."""
    counts = {}
    for rid in record_ids:
        result = load_afdb_beats(rid, raw_dir)
        if result is None:
            counts[rid] = 0
            continue
        _, labels = result
        counts[rid] = int((labels == "F").sum())
    return counts
