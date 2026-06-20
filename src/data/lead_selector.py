"""Lead selection utilities — MLII-equivalent extraction per dataset.

Regras mandatórias (ecg-preprocessing-pipeline):
- NUNCA usar fuzzy matching ou fallbacks genéricos
- Mapeamento explícito por dataset:
  * Chapman  → lead "II"  (índice via lead_names.index("II"))
  * MIT-BIH  → lead "MLII" (índice 0)
  * SVDB     → lead "ECG1" (índice 0; caveat: equivalência não documentada)
  * AFDB     → lead "ECG1" (índice 0; mesmo caveat)
  * INCART   → lead "II"  (índice via lead_names.index("II"))
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger("lewis.camada02.lead_selector")

# Mapeamento canônico: dataset → lead_name esperado
DATASET_LEAD_MAP = {
    "chapman": "II",
    "mitbih": "MLII",
    "svdb": "ECG1",
    "afdb": "ECG1",
    "incart": "II",
    "ptbxl": "II",
}

# Para datasets onde o índice é fixo (não depende do nome no header)
DATASET_FIXED_INDEX = {
    "svdb": 0,
    "afdb": 0,
}


def select_lead(
    record,
    dataset_name: str,
) -> Tuple[np.ndarray, str]:
    """Extrair lead específico de registro multi-lead.

    Parameters
    ----------
    record : wfdb.Record ou tuple
        Objeto retornado por wfdb.rdrecord() ou tupla (p_signal, sig_name).
        Se for wfdb.Record, usa record.p_signal e record.sig_name.
    dataset_name : str
        Nome canônico do dataset: "mitbih", "svdb", "afdb", "incart", "chapman".

    Returns
    -------
    tuple
        (np.ndarray 1D em mV, lead_name)

    Raises
    ------
    ValueError
        Se dataset não for suportado ou lead não for encontrado.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in DATASET_LEAD_MAP:
        raise ValueError(
            f"Dataset '{dataset_name}' não suportado. " f"Esperado: {list(DATASET_LEAD_MAP.keys())}"
        )

    expected_lead = DATASET_LEAD_MAP[dataset_name]

    # Extrair sinal e nomes do record
    if hasattr(record, "p_signal") and hasattr(record, "sig_name"):
        p_signal = record.p_signal
        sig_name = record.sig_name
    elif isinstance(record, tuple) and len(record) >= 2:
        p_signal, sig_name = record[0], record[1]
    else:
        raise ValueError("record deve ser um objeto wfdb.Record ou tupla (p_signal, sig_name)")

    # Caso single-lead já
    if p_signal.ndim == 1:
        LOGGER.debug("Sinal já é 1-D para %s", dataset_name)
        return p_signal.astype(np.float64), expected_lead

    n_leads = p_signal.shape[1]

    # Para MIT-BIH: buscar "MLII" pelo nome exato. Alguns registros (102, 104)
    # não possuem MLII e usam V5/V2; nesses casos usamos índice 0 como fallback.
    if dataset_name == "mitbih":
        if sig_name is None:
            raise ValueError(f"Dataset {dataset_name}: sig_name ausente no header")
        if expected_lead in sig_name:
            idx = sig_name.index(expected_lead)
            LOGGER.debug(
                "Lead selecionado para %s: índice=%d nome=%s (busca por nome)",
                dataset_name,
                idx,
                expected_lead,
            )
            return p_signal[:, idx].astype(np.float64), expected_lead
        LOGGER.warning(
            "MIT-BIH: lead '%s' não encontrado em sig_name=%s — fallback para índice 0 (%s)",
            expected_lead,
            sig_name,
            sig_name[0] if sig_name else "?",
        )
        idx = 0
        if idx >= n_leads:
            raise ValueError(f"Dataset {dataset_name}: índice fixo {idx} excede n_leads={n_leads}")
        actual_name = sig_name[idx] if sig_name and idx < len(sig_name) else expected_lead
        return p_signal[:, idx].astype(np.float64), actual_name

    # Para SVDB, AFDB usamos índice fixo (0) conforme spec
    if dataset_name in DATASET_FIXED_INDEX:
        idx = DATASET_FIXED_INDEX[dataset_name]
        if idx >= n_leads:
            raise ValueError(f"Dataset {dataset_name}: índice fixo {idx} excede n_leads={n_leads}")
        actual_name = sig_name[idx] if sig_name and idx < len(sig_name) else expected_lead
        LOGGER.debug(
            "Lead selecionado para %s: índice=%d nome=%s (fixo)",
            dataset_name,
            idx,
            actual_name,
        )
        return p_signal[:, idx].astype(np.float64), actual_name

    # Para Chapman, INCART, PTB-XL: buscar pelo nome exato
    if sig_name is None:
        raise ValueError(f"Dataset {dataset_name}: sig_name ausente no header")

    try:
        idx = sig_name.index(expected_lead)
    except ValueError:
        raise ValueError(
            f"Dataset {dataset_name}: lead '{expected_lead}' não encontrado em sig_name={sig_name}"
        )

    LOGGER.debug(
        "Lead selecionado para %s: índice=%d nome=%s (busca por nome)",
        dataset_name,
        idx,
        expected_lead,
    )
    return p_signal[:, idx].astype(np.float64), expected_lead


def get_lead_index(
    sig_name: list,
    dataset_name: str,
) -> int:
    """Retorna o índice canônico do lead para um dataset.

    Útil para validação antes de chamar select_lead.
    """
    dataset_name = dataset_name.lower().strip()
    if sig_name is None:
        raise ValueError(f"Dataset {dataset_name}: sig_name é None")

    # MIT-BIH: preferir MLII pelo nome; fallback índice 0
    if dataset_name == "mitbih":
        expected = DATASET_LEAD_MAP[dataset_name]
        if expected in sig_name:
            return sig_name.index(expected)
        return 0

    if dataset_name in DATASET_FIXED_INDEX:
        return DATASET_FIXED_INDEX[dataset_name]

    expected = DATASET_LEAD_MAP[dataset_name]
    return sig_name.index(expected)
