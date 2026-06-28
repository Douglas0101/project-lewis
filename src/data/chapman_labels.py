"""Mapeamento de códigos SNOMED-CT do Chapman-Shaoxing para superclasses SCP-ECG.

As 5 superclasses SCP-ECG usadas no PTB-XL e Chapman-Shaoxing são:
    NORM — Normal
    CD   — Conduction Disturbance (bloqueios, BBB, WPW, etc.)
    MI   — Myocardial Infarction (infarto, ondas Q patológicas)
    HYP  — Hypertrophy (hipertrofia ventricular/atrial)
    STTC — ST/T Change (isquemia, alterações de repolarização)

Referências:
- Wagner et al., Sci Data 7, 154 (2020) — PTB-XL SCP-ECG superclasses
- Chapman-Shaoxing PhysioNet Challenge 2021 — ConditionNames_SNOMED-CT.csv
"""

from __future__ import annotations

from typing import Dict, List, Set

# Mapeamento código SNOMED-CT -> lista de superclasses SCP-ECG associadas.
# Um código pode pertencer a múltiplas superclasses (multi-label é permitido).
SNOMED_TO_SCP: Dict[str, List[str]] = {
    # NORM
    "426177001": ["NORM"],  # Normal sinus rhythm
    "426783006": ["NORM"],  # Sinus rhythm
    "427084000": ["NORM"],  # Sinus tachycardia
    "427393009": ["NORM"],  # Sinus bradycardia
    "106068003": ["NORM"],  # Sinus arrhythmia
    # Atrial bigeminy (ritmo, mas com ectopia) -> NORM/CD?
    # mantemos NORM por simplicidade
    "251173003": ["NORM"],
    # CD — Conduction Disturbance
    "270492004": ["CD"],  # 1st degree AV block
    "195042002": ["CD"],  # 2nd degree AV block
    "54016002": ["CD"],  # 2nd degree AV block type 1
    "28189009": ["CD"],  # 2nd degree AV block type 2
    "27885002": ["CD"],  # 3rd degree AV block
    "233917008": ["CD"],  # AV block
    "6374002": ["CD"],  # Bundle branch block
    "713426002": ["CD"],  # Complete right bundle branch block
    "713427006": ["CD"],  # Incomplete right bundle branch block
    "164909002": ["CD"],  # Left bundle branch block
    # Right bundle branch block; também mapeado para STTC (overlap no catálogo)
    "164947007": ["CD", "STTC"],
    "17338001": ["CD"],  # Right bundle branch block (outro código)
    "445118002": ["CD"],  # IVCD
    "698252002": ["CD"],  # Intraventricular conduction delay
    "111975006": ["CD"],  # Prolonged QT
    "361055000": ["CD"],  # Marked QT prolongation
    "164951009": ["CD"],  # Aberrant conduction
    "251223006": ["CD"],  # WPW pattern
    "251199005": ["CD"],  # Counterclockwise rotation
    "251198002": ["CD"],  # Clockwise rotation
    "39732003": ["CD"],  # Left axis deviation
    "47665007": ["CD"],  # Right axis deviation
    "251205003": ["CD"],  # Low voltage
    "425856008": ["CD"],  # S1,S2,S3 pattern
    "251164006": ["CD"],  # Junctional premature beat
    "426995002": ["CD"],  # Junctional escape beat
    "251180001": ["CD"],  # Ventricular escape trigeminy
    "11157007": ["CD"],  # Ventricular bigeminy
    # MI — Myocardial Infarction
    "164865005": ["MI"],  # Myocardial infarction
    "164861001": ["MI"],  # MI anterior
    "164873006": ["MI"],  # MI inferior
    "164869007": ["MI"],  # MI anterolateral
    "164884008": ["MI"],  # MI posterolateral
    "164872004": ["MI"],  # MI septal
    "425419005": ["MI"],  # MI lateral
    "164870008": ["MI"],  # MI apical
    # MI of anterior wall; também mapeado para STTC (overlap no catálogo)
    "164931005": ["MI", "STTC"],
    # MI of inferior wall; também mapeado para STTC (overlap no catálogo)
    "164930006": ["MI", "STTC"],
    "164937009": ["MI"],  # MI of anteroseptal region
    "164912004": ["MI"],  # MI of lateral wall
    "164942001": ["MI"],  # fQRS wave
    "164917005": ["MI"],  # Abnormal Q wave
    "10370003": ["MI"],  # Anterior MI
    "59931005": ["MI"],  # Inferior MI
    "29320008": ["MI"],  # Posterior MI
    "55930002": ["MI"],  # Lateral MI
    "55827005": ["MI"],  # Anteroseptal MI
    "13640000": ["MI"],  # Subendocardial MI
    "426761007": ["MI"],  # STEMI
    # HYP — Hypertrophy
    "164873001": ["HYP"],  # Left ventricular hypertrophy
    "89792004": ["HYP"],  # Right ventricular hypertrophy
    "446358003": ["HYP"],  # Right atrial hypertrophy
    "164934002": ["HYP"],  # Atrial hypertrophy
    "59118001": ["HYP"],  # Left atrial enlargement
    "365413008": ["HYP"],  # Right atrial enlargement
    "61721007": ["HYP"],  # Left ventricular hypertrophy (outro)
    "427172004": ["HYP"],  # Left ventricular strain
    # STTC — ST/T Change
    "164889003": ["STTC"],  # Ischemia
    "164890007": ["STTC"],  # ST-T change
    "428750005": ["STTC"],  # ST-T abnormality
    "429622005": ["STTC"],  # ST elevation
    "428417006": ["STTC"],  # Early repolarization
    "713422000": ["STTC"],  # Acute T wave abnormality
    "713423005": ["STTC"],  # T wave abnormality
    "365414002": ["STTC"],  # T wave inversion
    "164933006": ["STTC"],  # Nonspecific ST-T changes
}

SCP_SUPERCLASSES: List[str] = ["NORM", "CD", "MI", "HYP", "STTC"]


def snomed_to_superclass_codes(snomed_codes: List[str]) -> Set[str]:
    """Converte lista de códigos SNOMED-CT em conjunto de superclasses SCP-ECG.

    Parameters
    ----------
    snomed_codes : list[str]
        Códigos SNOMED-CT (ex: ["426177001", "164889003"]).

    Returns
    -------
    set[str]
        Superclasses SCP-ECG encontradas.
    """
    supers: Set[str] = set()
    for code in snomed_codes:
        supers.update(SNOMED_TO_SCP.get(code.strip(), []))
    return supers


def superclass_to_multihot(superclasses: Set[str]) -> List[int]:
    """Converte conjunto de superclasses em vetor one-hot multi-label.

    Parameters
    ----------
    superclasses : set[str]
        Superclasses presentes no registro.

    Returns
    -------
    list[int]
        Vetor binário [NORM, CD, MI, HYP, STTC].
    """
    return [1 if cls in superclasses else 0 for cls in SCP_SUPERCLASSES]


def diagnosis_string_to_multihot(diagnosis: str) -> List[int]:
    """Converte string de diagnósticos do catalog em multi-hot.

    Os códigos SNOMED-CT devem estar separados por vírgula.

    Parameters
    ----------
    diagnosis : str
        String com códigos SNOMED-CT separados por vírgula.

    Returns
    -------
    list[int]
        Vetor binário [NORM, CD, MI, HYP, STTC].
    """
    codes = [c.strip() for c in diagnosis.split(",") if c.strip()]
    supers = snomed_to_superclass_codes(codes)
    # Se nenhuma superclass foi mapeada e há códigos, retornar vetor zero
    # (não forçar NORM para evitar labels incorretos).
    return superclass_to_multihot(supers)
