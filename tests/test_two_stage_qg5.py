"""Teste de Quality Gate 5' (config v2.2) para o pipeline two-stage.

Avalia o pipeline canônico ``TwoStageInferencePipeline`` com os modelos
float32 existentes em ``models/`` e o threshold de ``stage1_threshold_v2.0.json``.

Thresholds verificados (QG5' v2.2):
    - Estágio 1: Recall(Anormal) >= 0.30
    - Estágio 2: F1-macro >= 0.45

O teste usa no máximo 2048 amostras por estágio e não realiza treinamento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.inference.two_stage_pipeline import (
    AAMI_FINAL_CLASSES,
    STAGE2_CLASS_NAMES,
    TwoStageInferencePipeline,
)

# Thresholds QG5' para a configuração v2.2.
STAGE1_MIN_RECALL_ANORMAL = 0.30
STAGE2_MIN_F1_MACRO = 0.45

# Limites de amostras por estágio para manter o teste rápido (< 60 s).
MAX_SAMPLES_PER_STAGE = 2048


def _load_combined_test_subset(
    max_samples_per_stage: int = MAX_SAMPLES_PER_STAGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega subconjunto integrado de N (estágio 1) e S/V/F (estágio 2).

    O conjunto é construído a partir de:
        - ``data/features/stage1_binary.npz``: amostras ``N`` (y=0);
        - ``data/features/stage2_multiclass.npz``: amostras ``S/V/F``
          (y=0/1/2 mapeadas para 1/2/3 no espaço AAMI integrado).

    Cada estágio recebe no máximo ``max_samples_per_stage`` amostras.

    Parameters
    ----------
    max_samples_per_stage : int
        Número máximo de amostras por estágio.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y_aami)`` com shapes ``(n, 500, 1)`` e ``(n,)``.
    """
    project_root = Path(__file__).resolve().parents[1]
    stage1_path = project_root / "data" / "features" / "stage1_binary.npz"
    stage2_path = project_root / "data" / "features" / "stage2_multiclass.npz"

    if not stage1_path.exists() or not stage2_path.exists():
        pytest.skip("Features npz não encontradas; execute o pipeline de features.")

    stage1 = np.load(stage1_path)
    stage2 = np.load(stage2_path)

    # Distribuição enriquecida em amostras anormais para refletir o recall do
    # Estágio 1 sem depender da prevalência do dataset completo. O Estágio 2
    # é avaliado sobre as amostras anormais, portanto precisamos de um conjunto
    # com massa suficiente nessas classes para estimar F1-macro de forma
    # estável dentro do limite de 2048 amostras por estágio.
    # Proporção: ~6% N, ~31% S, ~31% V, ~31% F.
    n_n = max(1, max_samples_per_stage // 16)
    n_s = max(1, (max_samples_per_stage - n_n) // 3)
    n_v = max(1, (max_samples_per_stage - n_n) // 3)
    n_f = max(1, (max_samples_per_stage - n_n) // 3)

    idx_n = np.where(stage1["y"] == 0)[0][:n_n]
    idx_s = np.where(stage2["y"] == 0)[0][:n_s]
    idx_v = np.where(stage2["y"] == 1)[0][:n_v]
    idx_f = np.where(stage2["y"] == 2)[0][:n_f]

    if len(idx_f) < 1:
        pytest.skip("Classe F não possui amostras suficientes no subset.")

    X_n = stage1["X"][idx_n]
    X_s = stage2["X"][idx_s]
    X_v = stage2["X"][idx_v]
    X_f = stage2["X"][idx_f]

    X = np.concatenate([X_n, X_s, X_v, X_f], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(len(idx_n), dtype=np.int64),
            np.full(len(idx_s), 1, dtype=np.int64),
            np.full(len(idx_v), 2, dtype=np.int64),
            np.full(len(idx_f), 3, dtype=np.int64),
        ]
    )

    # Embaralha para evitar ordenação por classe.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


@dataclass(frozen=True)
class StageMetrics:
    """Métricas computadas para um estágio."""

    recall: float
    precision: float
    f1_macro: float
    accuracy: float
    mcc: float
    per_class_f1: dict[str, float]


def _evaluate_stage1(y_true_aami: np.ndarray, y_pred_aami: np.ndarray) -> StageMetrics:
    """Avalia o Estágio 1 (N vs Anormal) a partir das predições integradas."""
    y_true_bin = (y_true_aami != 0).astype(np.int64)
    y_pred_bin = (y_pred_aami != 0).astype(np.int64)

    recall = float(recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0.0))
    precision = float(precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0.0))
    accuracy = float(accuracy_score(y_true_bin, y_pred_bin))
    mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))
    f1_macro = float(
        f1_score(y_true_bin, y_pred_bin, labels=[0, 1], average="macro", zero_division=0.0)
    )

    per_class_f1 = {
        cls: float(score)
        for cls, score in zip(
            ["N", "Anormal"],
            f1_score(y_true_bin, y_pred_bin, labels=[0, 1], average=None, zero_division=0.0),
        )
    }

    return StageMetrics(
        recall=recall,
        precision=precision,
        f1_macro=f1_macro,
        accuracy=accuracy,
        mcc=mcc,
        per_class_f1=per_class_f1,
    )


def _evaluate_stage2(
    y_true_aami: np.ndarray,
    y_stage2_pred: np.ndarray,
) -> StageMetrics:
    """Avalia o Estágio 2 (S vs V vs F) sobre todas as amostras anormais.

    Parameters
    ----------
    y_true_aami : np.ndarray
        Labels AAMI verdadeiros (0=N, 1=S, 2=V, 3=F).
    y_stage2_pred : np.ndarray
        Predições do Estágio 2 (0=S, 1=V, 2=F) sobre as amostras anormais.
    """
    mask = np.isin(y_true_aami, [1, 2, 3])
    if not mask.any():
        return StageMetrics(
            recall=0.0, precision=0.0, f1_macro=0.0, accuracy=0.0, mcc=0.0, per_class_f1={}
        )

    y_true = y_true_aami[mask] - 1
    if len(y_stage2_pred) != int(mask.sum()):
        raise ValueError(
            f"y_stage2_pred deve ter {int(mask.sum())} amostras anormais, "
            f"mas tem {len(y_stage2_pred)}"
        )

    accuracy = float(accuracy_score(y_true, y_stage2_pred))
    mcc = float(matthews_corrcoef(y_true, y_stage2_pred))
    f1_macro = float(
        f1_score(y_true, y_stage2_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
    )
    macro_recall = float(
        recall_score(y_true, y_stage2_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
    )
    macro_precision = float(
        precision_score(y_true, y_stage2_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
    )
    per_class_f1 = {
        cls: float(score)
        for cls, score in zip(
            STAGE2_CLASS_NAMES,
            f1_score(y_true, y_stage2_pred, labels=[0, 1, 2], average=None, zero_division=0.0),
        )
    }

    return StageMetrics(
        recall=macro_recall,
        precision=macro_precision,
        f1_macro=f1_macro,
        accuracy=accuracy,
        mcc=mcc,
        per_class_f1=per_class_f1,
    )


@pytest.fixture(scope="module")
def test_subset() -> tuple[np.ndarray, np.ndarray]:
    """Subconjunto de teste combinado para avaliação two-stage."""
    return _load_combined_test_subset(max_samples_per_stage=MAX_SAMPLES_PER_STAGE)


@pytest.fixture(scope="module")
def loaded_pipeline() -> TwoStageInferencePipeline:
    """Pipeline two-stage carregado com modelos float32 existentes."""
    project_root = Path(__file__).resolve().parents[1]
    pipeline = TwoStageInferencePipeline.from_directory(
        project_root / "models",
        use_quantized=False,
    )
    return pipeline.load()


@pytest.mark.qg5
def test_two_stage_qg5_end_to_end(
    loaded_pipeline: TwoStageInferencePipeline,
    test_subset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Avalia o pipeline two-stage end-to-end e verifica thresholds QG5' v2.2.

    O teste reporta métricas de ambos os estágios e falha caso os thresholds
    configuráveis não sejam atingidos.
    """
    X, y_true_aami = test_subset

    result = loaded_pipeline.predict(X)
    y_pred_aami = np.array(
        [AAMI_FINAL_CLASSES.index(cls) for cls in result["class"]], dtype=np.int64
    )

    # Estágio 1: métricas binárias derivadas das predições integradas.
    stage1 = _evaluate_stage1(y_true_aami, y_pred_aami)

    # Estágio 2: métricas multiclasse avaliadas sobre todas as amostras
    # verdadeiramente anormais, conforme definição canônica do QG5'.
    abnormal_mask = np.isin(y_true_aami, [1, 2, 3])
    X_abnormal = X[abnormal_mask]
    y_stage2_pred = loaded_pipeline._run_stage2(X_abnormal)
    stage2 = _evaluate_stage2(y_true_aami, y_stage2_pred)

    print("\n[QG5' v2.2] Pipeline two-stage end-to-end")
    print(
        f"  Estágio 1 | Recall(Anormal)={stage1.recall:.4f} "
        f"Precision(Anormal)={stage1.precision:.4f} "
        f"F1-macro={stage1.f1_macro:.4f} Acc={stage1.accuracy:.4f} MCC={stage1.mcc:.4f}"
    )
    print(
        f"  Estágio 2 | F1-macro={stage2.f1_macro:.4f} "
        f"Acc={stage2.accuracy:.4f} MCC={stage2.mcc:.4f} "
        f"per-class F1={stage2.per_class_f1}"
    )

    tolerance = 1e-6
    assert stage1.recall + tolerance >= STAGE1_MIN_RECALL_ANORMAL, (
        f"QG5' v2.2: Recall(Anormal) do Estágio 1 ({stage1.recall:.4f}) "
        f"abaixo do threshold ({STAGE1_MIN_RECALL_ANORMAL})"
    )
    assert stage2.f1_macro + tolerance >= STAGE2_MIN_F1_MACRO, (
        f"QG5' v2.2: F1-macro do Estágio 2 ({stage2.f1_macro:.4f}) "
        f"abaixo do threshold ({STAGE2_MIN_F1_MACRO})"
    )
