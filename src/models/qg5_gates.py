"""Quality Gates QG5 redesenhados para a research branch v2.4.

Divide o antigo QG5 unico em:
- QG5_SMOKE_BALANCED: teste diagnostico rapido em subset balanceado;
- QG5_PATIENTWISE: avaliacao inter-paciente real via OOF/GroupKFold;
- QG5_STABILITY: variabilidade entre folds;
- QG5_CALIBRATION: metricas probabilisticas;
- QG5_REPRODUCIBILITY: validacao de manifests e seeds;
- QG5_PUBLICATION: agregado que so autoriza publicacao quando todos os gates formais passam.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import tensorflow as tf
from src.models.keras_loader import load_keras_model
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline

LOGGER = logging.getLogger("lewis.qg5_gates")


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"Falha ao carregar {path}: {exc}") from exc


def _balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    max_samples_per_class: int = 683,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna indices balanceados por classe para teste diagnostico."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in range(int(y.max()) + 1):
        idx = np.where(y == cls)[0]
        n = min(len(idx), max_samples_per_class)
        selected.extend(rng.choice(idx, size=n, replace=False).tolist())
    selected_arr = np.array(selected)
    rng.shuffle(selected_arr)
    return X[selected_arr], y[selected_arr], selected_arr


def _stage2_metrics(
    model: tf.keras.Model,
    scaler: Any,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calcula metricas do Estagio 2."""
    X_scaled = scaler.transform(X)
    proba = model.predict(X_scaled, batch_size=4096, verbose=0)
    if thresholds:
        y_pred = TwoStageMLPPipeline._apply_stage2_thresholds(proba, thresholds)
    else:
        y_pred = np.argmax(proba, axis=1).astype(np.int64)

    per_class_f1 = {
        cls: float(score)
        for cls, score in zip(
            ["S", "V", "F"],
            np.asarray(
                f1_score(
                    y,
                    y_pred,
                    labels=[0, 1, 2],
                    average=None,  # type: ignore
                    zero_division=0,  # type: ignore
                )
            ).tolist(),
        )
    }
    per_class_precision = {
        cls: float(score)
        for cls, score in zip(
            ["S", "V", "F"],
            np.asarray(
                precision_score(
                    y,
                    y_pred,
                    labels=[0, 1, 2],
                    average=None,  # type: ignore
                    zero_division=0,  # type: ignore
                )
            ).tolist(),
        )
    }
    per_class_recall = {
        cls: float(score)
        for cls, score in zip(
            ["S", "V", "F"],
            np.asarray(
                recall_score(
                    y,
                    y_pred,
                    labels=[0, 1, 2],
                    average=None,  # type: ignore
                    zero_division=0,  # type: ignore
                )
            ).tolist(),
        )
    }

    # Brier multiclasse: media das Brier one-vs-rest
    brier = 0.0
    for k in range(3):
        y_true_bin = (y == k).astype(np.int64)
        brier += brier_score_loss(y_true_bin, proba[:, k])
    brier /= 3.0

    try:
        logloss = log_loss(y, proba, labels=[0, 1, 2])
    except ValueError:
        logloss = float("inf")

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])

    return {
        "f1_macro": float(
            f1_score(
                y,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,  # type: ignore
            )
        ),
        "f1_weighted": float(
            f1_score(
                y,
                y_pred,
                labels=[0, 1, 2],
                average="weighted",
                zero_division=0,  # type: ignore
            )
        ),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "log_loss": float(logloss),
        "multiclass_brier": float(brier),
        "confusion_matrix": cm.tolist(),
    }


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    diagnostic_only: bool
    metrics: dict[str, Any]
    failures: list[str]
    notes: list[str]


class QG5SmokeBalancedGate:
    """Gate diagnostico rapido em subset balanceado."""

    def __init__(
        self,
        min_f1_s: float = 0.55,
        min_f1_v: float = 0.70,
        min_f1_f: float = 0.50,
        min_f1_macro: float = 0.45,
        max_samples_per_class: int = 683,
    ) -> None:
        self.min_f1_s = min_f1_s
        self.min_f1_v = min_f1_v
        self.min_f1_f = min_f1_f
        self.min_f1_macro = min_f1_macro
        self.max_samples_per_class = max_samples_per_class

    def evaluate(
        self,
        model: tf.keras.Model,
        scaler: Any,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: dict[str, float] | None = None,
    ) -> GateResult:
        X_sub, y_sub, _ = _balanced_subset(X, y, self.max_samples_per_class)
        metrics = _stage2_metrics(model, scaler, X_sub, y_sub, thresholds)
        failures = []
        if metrics["per_class_f1"]["S"] < self.min_f1_s:
            failures.append(f"F1(S)={metrics['per_class_f1']['S']:.4f} < {self.min_f1_s}")
        if metrics["per_class_f1"]["V"] < self.min_f1_v:
            failures.append(f"F1(V)={metrics['per_class_f1']['V']:.4f} < {self.min_f1_v}")
        if metrics["per_class_f1"]["F"] < self.min_f1_f:
            failures.append(f"F1(F)={metrics['per_class_f1']['F']:.4f} < {self.min_f1_f}")
        if metrics["f1_macro"] < self.min_f1_macro:
            failures.append(f"F1-macro={metrics['f1_macro']:.4f} < {self.min_f1_macro}")
        return GateResult(
            gate_name="QG5_SMOKE_BALANCED",
            passed=len(failures) == 0,
            diagnostic_only=True,
            metrics=metrics,
            failures=failures,
            notes=["Avaliacao em subset balanceado; nao substitui generalizacao inter-paciente."],
        )


class QG5PatientwiseGate:
    """Gate de generalizacao inter-paciente via OOF/GroupKFold."""

    def __init__(
        self,
        min_f1_f_mean: float = 0.50,
        min_f1_f_min: float = 0.0,
        min_f1_macro_mean: float = 0.45,
        min_f1_s_mean: float = 0.55,
        min_f1_v_mean: float = 0.70,
    ) -> None:
        self.min_f1_f_mean = min_f1_f_mean
        self.min_f1_f_min = min_f1_f_min
        self.min_f1_macro_mean = min_f1_macro_mean
        self.min_f1_s_mean = min_f1_s_mean
        self.min_f1_v_mean = min_f1_v_mean

    def evaluate(
        self,
        experiment_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        thresholds: dict[str, float] | None = None,
        n_splits: int = 5,
    ) -> GateResult:
        gkf = GroupKFold(n_splits=n_splits)
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            fold_dir = experiment_dir / f"fold_{fold_idx}"
            if not fold_dir.exists():
                raise FileNotFoundError(f"Diretorio do fold nao encontrado: {fold_dir}")
            model = load_keras_model(str(fold_dir / "model.keras"), compile=False)
            scaler = joblib.load(fold_dir / "input_scaler.pkl")
            metrics = _stage2_metrics(model, scaler, X[val_idx], y[val_idx], thresholds)
            fold_results.append(metrics)
            LOGGER.info(
                "Fold %d | F1_macro=%.4f | F1(S)=%.4f | F1(V)=%.4f | F1(F)=%.4f",
                fold_idx,
                metrics["f1_macro"],
                metrics["per_class_f1"]["S"],
                metrics["per_class_f1"]["V"],
                metrics["per_class_f1"]["F"],
            )

        f1_f_values = [f["per_class_f1"]["F"] for f in fold_results]
        f1_s_values = [f["per_class_f1"]["S"] for f in fold_results]
        f1_v_values = [f["per_class_f1"]["V"] for f in fold_results]
        f1_macro_values = [f["f1_macro"] for f in fold_results]

        aggregated = {
            "F1_macro_mean": float(np.mean(f1_macro_values)),
            "F1_macro_std": float(np.std(f1_macro_values)),
            "F1_S_mean": float(np.mean(f1_s_values)),
            "F1_V_mean": float(np.mean(f1_v_values)),
            "F1_F_mean": float(np.mean(f1_f_values)),
            "F1_F_std": float(np.std(f1_f_values)),
            "F1_F_min": float(np.min(f1_f_values)),
            "F1_F_max": float(np.max(f1_f_values)),
            "per_fold": fold_results,
        }

        try:
            f1_f_mean = cast(float, aggregated["F1_F_mean"])
            f1_f_min = cast(float, aggregated["F1_F_min"])
            f1_macro_mean = cast(float, aggregated["F1_macro_mean"])
            f1_s_mean = cast(float, aggregated["F1_S_mean"])
            f1_v_mean = cast(float, aggregated["F1_V_mean"])
        except Exception as exc:
            raise ValueError(f"Falha ao extrair metricas agregadas: {exc}") from exc
        failures = []
        if f1_f_mean < self.min_f1_f_mean:
            failures.append(f"mean F1(F)={f1_f_mean:.4f} < {self.min_f1_f_mean}")
        if f1_f_min < self.min_f1_f_min:
            failures.append(f"min F1(F)={f1_f_min:.4f} < {self.min_f1_f_min}")
        if f1_macro_mean < self.min_f1_macro_mean:
            failures.append(f"mean F1-macro={f1_macro_mean:.4f} < {self.min_f1_macro_mean}")
        if f1_s_mean < self.min_f1_s_mean:
            failures.append(f"mean F1(S)={f1_s_mean:.4f} < {self.min_f1_s_mean}")
        if f1_v_mean < self.min_f1_v_mean:
            failures.append(f"mean F1(V)={f1_v_mean:.4f} < {self.min_f1_v_mean}")

        return GateResult(
            gate_name="QG5_PATIENTWISE",
            passed=len(failures) == 0,
            diagnostic_only=False,
            metrics=aggregated,
            failures=failures,
            notes=["Avaliacao inter-paciente via GroupKFold OOF."],
        )


class QG5StabilityGate:
    """Gate de estabilidade entre folds e seeds."""

    def __init__(self, max_f1_f_std: float | None = None) -> None:
        self.max_f1_f_std = max_f1_f_std

    def evaluate(self, patientwise_metrics: dict[str, Any]) -> GateResult:
        f1_f_values = [f["per_class_f1"]["F"] for f in patientwise_metrics["per_fold"]]
        f1_macro_values = [f["f1_macro"] for f in patientwise_metrics["per_fold"]]
        try:
            metrics = {
                "F1_F_std": float(np.std(f1_f_values)),
                "F1_macro_std": float(np.std(f1_macro_values)),
                "worst_fold_F1_F": float(np.min(f1_f_values)),
            }
        except Exception as exc:
            raise ValueError(f"Falha ao calcular metricas de estabilidade: {exc}") from exc
        failures = []
        if self.max_f1_f_std is not None and metrics["F1_F_std"] > self.max_f1_f_std:
            failures.append(f"F1_F_std={metrics['F1_F_std']:.4f} > {self.max_f1_f_std}")
        return GateResult(
            gate_name="QG5_STABILITY",
            passed=len(failures) == 0,
            diagnostic_only=False,
            metrics=metrics,
            failures=failures,
            notes=["Limites de estabilidade ainda nao definidos na research branch (E10)."],
        )


class QG5CalibrationGate:
    """Gate de calibracao probabilistica."""

    def __init__(self, max_log_loss: float | None = None, max_brier: float | None = None) -> None:
        self.max_log_loss = max_log_loss
        self.max_brier = max_brier

    def evaluate(self, patientwise_metrics: dict[str, Any]) -> GateResult:
        log_losses = [f["log_loss"] for f in patientwise_metrics["per_fold"]]
        briers = [f["multiclass_brier"] for f in patientwise_metrics["per_fold"]]
        try:
            metrics = {
                "mean_log_loss": float(np.mean(log_losses)),
                "std_log_loss": float(np.std(log_losses)),
                "mean_multiclass_brier": float(np.mean(briers)),
                "std_multiclass_brier": float(np.std(briers)),
            }
        except Exception as exc:
            raise ValueError(f"Falha ao calcular metricas de calibracao: {exc}") from exc
        failures = []
        if self.max_log_loss is not None and metrics["mean_log_loss"] > self.max_log_loss:
            failures.append(f"mean log_loss={metrics['mean_log_loss']:.4f} > {self.max_log_loss}")
        if self.max_brier is not None and metrics["mean_multiclass_brier"] > self.max_brier:
            failures.append(f"mean Brier={metrics['mean_multiclass_brier']:.4f} > {self.max_brier}")
        return GateResult(
            gate_name="QG5_CALIBRATION",
            passed=len(failures) == 0,
            diagnostic_only=False,
            metrics=metrics,
            failures=failures,
            notes=["Brier e log_loss one-vs-rest; nao confundir com calibracao pura."],
        )


class QG5ReproducibilityGate:
    """Gate de reproducibilidade: verifica manifests e seeds."""

    def __init__(
        self,
        expected_dataset_hash: str | None = None,
        expected_feature_schema_hash: str | None = None,
    ) -> None:
        self.expected_dataset_hash = expected_dataset_hash
        self.expected_feature_schema_hash = expected_feature_schema_hash

    def evaluate(
        self,
        split_manifest_path: Path | None = None,
        dataset_manifest_path: Path | None = None,
        feature_manifest_path: Path | None = None,
        seed: int | None = None,
    ) -> GateResult:
        metrics: dict[str, Any] = {}
        metrics["seed"] = seed
        failures = []
        notes = []
        for name, path in [
            ("split_manifest", split_manifest_path),
            ("dataset_manifest", dataset_manifest_path),
            ("feature_manifest", feature_manifest_path),
        ]:
            if path is None:
                failures.append(f"{name} nao fornecido")
                continue
            if not path.exists():
                failures.append(f"{name} nao encontrado: {path}")
                continue
            try:
                data = _load_json(path)
                metrics[name] = {"hash": self._hash_dict(data), "path": str(path)}
            except Exception as exc:
                failures.append(f"{name} invalido: {exc}")
        if seed is None:
            notes.append("seed nao fornecida")
        return GateResult(
            gate_name="QG5_REPRODUCIBILITY",
            passed=len(failures) == 0,
            diagnostic_only=False,
            metrics=metrics,
            failures=failures,
            notes=notes,
        )

    @staticmethod
    def _hash_dict(data: dict) -> str:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()


class QG5PublicationGate:
    """Agrega os gates formais para decidir publicacao."""

    def __init__(self) -> None:
        self.gates: list[GateResult] = []

    def add(self, result: GateResult) -> None:
        self.gates.append(result)

    def can_publish(self) -> bool:
        for gate in self.gates:
            if gate.diagnostic_only:
                continue
            if not gate.passed:
                return False
        return True

    def status(self) -> str:
        if self.can_publish():
            return "PUBLICATION_READY"
        if any(g.passed for g in self.gates):
            return "RESEARCH_CANDIDATE_NOT_PUBLICATION_READY"
        return "FAIL"

    def report(self) -> dict[str, Any]:
        return {
            "status": self.status(),
            "can_publish": self.can_publish(),
            "gates": [
                {
                    "name": g.gate_name,
                    "passed": g.passed,
                    "diagnostic_only": g.diagnostic_only,
                    "metrics": g.metrics,
                    "failures": g.failures,
                    "notes": g.notes,
                }
                for g in self.gates
            ],
        }
