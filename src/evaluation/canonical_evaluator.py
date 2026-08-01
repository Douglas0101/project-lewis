"""Avaliador canônico do ML Protocol v2 (evaluator_version v2.0).

Orquestra métricas equalizadas, calibração (temperature scaling), thresholds e
contrato de comparabilidade, gerando o conjunto ``evaluation_v2/`` de artefatos
(docs/ml_protocol_v2.md §9). Determinístico; não depende de logs de treino
Keras; não modifica artefatos originais da run.

Uso (CLI)::

    uv run python -m src.evaluation.canonical_evaluator \
        --run-dir experiments/<run> \
        --task-profile pretrain_scp_ecg_multilabel \
        --split-name chapman-record-disjoint-val0.1-seed13 \
        --output-dir experiments/<run>/evaluation_v2 \
        --temperature-source calibration.json \
        --n-bins 15 --seed 13

Predições: ``--predictions x.npz`` > ``<output_dir>/predictions/predictions.npz``
> regeneração read-only a partir do checkpoint (imports de TF lazy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.evaluation.calibration_metrics import (
    apply_temperature,
    calibration_report,
    fit_temperature_multilabel,
)
from src.evaluation.metric_definitions import (
    f1_at_thresholds,
    macro_auroc,
    macro_pr_auc,
    nll_softmax,
    per_class_metrics,
)
from src.evaluation.metrics_ci import bootstrap_ci
from src.evaluation.schema import (
    EVALUATOR_VERSION,
    MetricsBlock,
    MetricsJson,
)
from src.evaluation.thresholding import ThresholdPolicy, fit_thresholds

LOGGER = logging.getLogger("lewis.evaluation.canonical")

DEFAULT_CLASSES: dict[str, tuple[str, ...]] = {
    "pretrain_scp_ecg_multilabel": ("NORM", "CD", "MI", "HYP", "STTC"),
    "beat_classification_aami": ("N", "S", "V", "F", "Q"),
}


@dataclass
class EvaluationResult:
    """Pacote completo de artefatos de uma avaliação canônica."""

    metrics: dict
    metrics_per_class: dict
    calibration: dict
    thresholds: dict
    reliability: dict
    confidence_intervals: dict
    reconciliation: dict = field(default_factory=dict)


def _validate_multilabel(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.ndim != 2 or y_score.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError(
            "task_profile multi-label espera y_true e y_score 2D com o mesmo "
            f"shape (n, k); recebido y_true{y_true.shape} vs y_score{y_score.shape}. "
            "Para rótulos inteiros 1D use beat_classification_aami (softmax)."
        )
    uniques = np.unique(y_true)
    if not np.all(np.isin(uniques, [0, 1])):
        raise ValueError(
            "task_profile multi-label espera y_true binário {0,1}; "
            f"valores encontrados: {uniques.tolist()[:10]}"
        )


def _validate_softmax(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.ndim != 1 or y_score.ndim != 2 or y_score.shape[0] != y_true.shape[0]:
        raise ValueError(
            "task_profile beat_classification_aami espera y_true 1D de índices "
            f"inteiros e y_score 2D (n, k); recebido y_true{y_true.shape} vs "
            f"y_score{y_score.shape}. Para multi-label binário use "
            "pretrain_scp_ecg_multilabel."
        )


def _macro_auc_pair(y_true: np.ndarray, y_score: np.ndarray, class_names: Sequence[str]):
    """(macro_pr_auc, macro_auroc) via per_class_metrics @0.5."""
    mpc = per_class_metrics(y_true, y_score, class_names, threshold=0.5)
    return macro_pr_auc(mpc), macro_auroc(mpc)


def _confidence_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """IC95 bootstrap (reamostragem de amostras) das métricas primárias."""
    if n_bootstrap <= 0:
        return {}

    def fn_pr(yt: np.ndarray, ys: np.ndarray) -> float:
        mpc = per_class_metrics(yt, ys, class_names, threshold=0.5)
        value = macro_pr_auc(mpc)
        return value if value is not None else float("nan")

    def fn_roc(yt: np.ndarray, ys: np.ndarray) -> float:
        mpc = per_class_metrics(yt, ys, class_names, threshold=0.5)
        value = macro_auroc(mpc)
        return value if value is not None else float("nan")

    out: dict[str, dict] = {}
    for name, fn in (("macro_pr_auc", fn_pr), ("macro_auroc", fn_roc)):
        mean, lower, upper = bootstrap_ci(
            y_true, y_score, fn, n_bootstrap=n_bootstrap, random_state=seed
        )
        out[name] = {"mean": mean, "ci_95": [lower, upper], "n_bootstrap": n_bootstrap}
    return out


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    task_profile: str,
    split_id: str,
    ontology_version: str,
    n_bins: int = 15,
    temperature: Optional[float] = None,
    fit_temperature: bool = False,
    threshold_policy: Optional[ThresholdPolicy] = None,
    calibration_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    protocol_status: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    run_id: str = "adhoc",
    preprocessing_version: str = "v1.0",
    n_bootstrap: int = 200,
    seed: int = 13,
) -> EvaluationResult:
    """Avaliação canônica completa sobre (y_true, y_score) de UM split de avaliação.

    ``calibration_data`` (y_cal, p_cal) separa o ajuste de T/thresholds do split
    avaliado (modo PROSPECTIVE); sem ela, o ajuste ocorre nos próprios dados
    avaliados e o resultado é carimbado RETROSPECTIVE.
    """
    if task_profile not in DEFAULT_CLASSES:
        raise ValueError(
            f"task_profile desconhecido '{task_profile}'; "
            f"opções: {sorted(DEFAULT_CLASSES)}"
        )
    names = list(class_names or DEFAULT_CLASSES[task_profile])
    policy = threshold_policy or ThresholdPolicy(name="max_f1_per_class")
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    multilabel = task_profile == "pretrain_scp_ecg_multilabel"
    if multilabel:
        _validate_multilabel(y_true, y_score)
    else:
        _validate_softmax(y_true, y_score)

    # --- protocolo: onde T/thresholds são ajustados -------------------------
    if calibration_data is not None:
        y_fit, p_fit = calibration_data
        status = "PROSPECTIVE"
    else:
        y_fit, p_fit = y_true, y_score
        fitted_here = fit_temperature or policy.name != "fixed_0.5"
        status = "RETROSPECTIVE" if fitted_here else "FROZEN_PARAMS"
    if protocol_status is not None:
        status = protocol_status

    # --- métricas de discriminação (sempre no split avaliado) ---------------
    y_labels: Optional[np.ndarray] = None  # rótulos inteiros (perfil softmax)
    if multilabel:
        prob_eval = y_score
    else:
        y_labels = y_true.astype(np.int64)
        onehot = np.zeros_like(y_score, dtype=int)
        onehot[np.arange(len(y_labels)), y_labels] = 1
        y_true = onehot
        prob_eval = y_score
        # thresholds/ECE operam one-vs-rest: o fit também precisa de one-hot
        if calibration_data is None:
            y_fit = y_true
        else:
            y_cal, p_cal = calibration_data
            cal_onehot = np.zeros_like(p_cal, dtype=int)
            cal_onehot[np.arange(len(y_cal)), y_cal.astype(np.int64)] = 1
            y_fit, p_fit = cal_onehot, p_cal
    mpc = per_class_metrics(y_true, prob_eval, names, threshold=0.5)

    # --- temperatura ---------------------------------------------------------
    temp: Optional[float] = temperature
    if fit_temperature:
        if not multilabel:
            raise ValueError("fit de temperatura multilabel não se aplica ao perfil softmax")
        temp = fit_temperature_multilabel(y_fit, p_fit)
        LOGGER.info("temperatura ajustada (%s): T=%.6f", status, temp)

    def _apply_t(probs: np.ndarray) -> np.ndarray:
        """Aplica T: sigmoid(logit/T) multi-label; softmax(log p/T) multiclasse."""
        if temp is None:
            return probs
        if multilabel:
            return apply_temperature(probs, temp)
        z = np.log(np.clip(probs, 1e-12, 1.0)) / temp
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # --- calibração ----------------------------------------------------------
    pre = calibration_report(y_true, prob_eval, n_bins, names)
    post: Optional[dict] = None
    prob_cal = prob_eval
    if temp is not None:
        prob_cal = _apply_t(prob_eval)
        post = calibration_report(y_true, prob_cal, n_bins, names)

    # --- thresholds (fit em y_fit/p_fit calibrados; apply no split avaliado) -
    thresholds = fit_thresholds(y_fit, _apply_t(p_fit), names, policy)
    tuned = f1_at_thresholds(y_true, prob_cal, thresholds, names)
    fixed = f1_at_thresholds(y_true, prob_cal, {n: 0.5 for n in names}, names)

    # --- bloco de métricas schema 2.0 ----------------------------------------
    metrics_block = MetricsBlock(
        macro_pr_auc=macro_pr_auc(mpc),
        macro_auroc=macro_auroc(mpc),
        macro_f1_at_0_5=fixed["macro_f1"],
        macro_f1_tuned=tuned["macro_f1"],
        bce=pre["macro"]["bce"] if multilabel else None,
        bce_post_temperature=post["macro"]["bce"] if (post and multilabel) else None,
        nll=(
            pre["macro"]["bce"]
            if multilabel
            else nll_softmax(y_labels if y_labels is not None else y_true, prob_eval)
        ),
        nll_post_temperature=(post["macro"]["nll"] if (post and multilabel) else (
            nll_softmax(y_labels, prob_cal) if (post and y_labels is not None) else None
        )),
        brier_mean=pre["macro"]["brier"],
        ece_pre_calibration=pre["macro"]["ece"],
        ece_post_calibration=post["macro"]["ece"] if post else None,
        mce_post_calibration=post["macro"]["mce"] if post else None,
        temperature=temp,
    )
    metrics_json = MetricsJson(
        run_id=run_id,
        task_profile=task_profile,
        split_id=split_id,
        ontology_version=ontology_version,
        n_samples=int(len(y_score)),
        protocol_status=status,  # type: ignore[arg-type]
        metrics=metrics_block,
        per_class={
            "at_0.5": mpc["per_class"],
            "tuned": tuned["per_class"],
            "calibration_pre": pre["per_class"],
            "calibration_post": post["per_class"] if post else None,
        },
        thresholds={
            "policy": policy.name,
            "fit_split": "calibration" if calibration_data is not None else "evaluation",
            "values": thresholds,
        },
        provenance={
            "evaluator_version": EVALUATOR_VERSION,
            "n_bins": n_bins,
            "preprocessing_version": preprocessing_version,
            "seed": seed,
        },
    )
    LOGGER.info(
        "avaliação %s | macro_pr_auc=%.4f | macro_auroc=%.4f | ece %.4f → %s | T=%s",
        run_id,
        metrics_block.macro_pr_auc or float("nan"),
        metrics_block.macro_auroc or float("nan"),
        metrics_block.ece_pre_calibration or float("nan"),
        f"{metrics_block.ece_post_calibration:.4f}"
        if metrics_block.ece_post_calibration is not None
        else "n/a",
        f"{temp:.4f}" if temp is not None else "n/a",
    )
    return EvaluationResult(
        metrics=metrics_json.model_dump(),
        metrics_per_class=metrics_json.per_class,
        calibration={
            "temperature": temp,
            "method": "temperature_scaling" if temp is not None else "none",
            "pre": pre,
            "post": post,
            "protocol_status": status,
        },
        thresholds=metrics_json.thresholds,
        reliability={"pre": pre["reliability"], "post": post["reliability"] if post else None},
        confidence_intervals=_confidence_intervals(
            y_true, prob_eval, names, n_bootstrap, seed
        ),
    )


# ---------------------------------------------------------------------------
# predições: carga, regeneração read-only, reconciliação com legado
# ---------------------------------------------------------------------------


def load_predictions_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega npz com chaves ``y_true`` e ``y_score`` (ou ``y_prob``)."""
    data = np.load(path)
    y_true = data["y_true"]
    y_score = data["y_score"] if "y_score" in data else data["y_prob"]
    return y_true, y_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate_predictions(run_dir: Path, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Regenera (y_true, y_prob) do split de validação — read-only na run.

    Reconstrói o split determinístico via provenance.json (seed, batch,
    validation_steps), prediz com o checkpoint salvo e grava
    ``<output_dir>/predictions/predictions.npz`` + ``predictions_meta.json``.
    Imports de TensorFlow são lazy (somente neste caminho).
    """
    from src.models.keras_loader import load_keras_model
    from src.models.pretrain_chapman import build_datasets
    from src.models.pretrain_evaluation import predict_validation

    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    training = provenance["training"]
    seed = provenance["seed"]
    model_path = run_dir / "backbone_pretrained.keras"
    sha = _sha256(model_path)
    expected = provenance.get("hashes", {}).get("model_sha256")
    if expected and sha != expected:
        raise RuntimeError(f"checkpoint hash mismatch: {sha} != provenance {expected}")
    _, val_ds, _, val_steps = build_datasets(
        val_ratio=0.1,
        batch_size=training["batch_size"],
        segment_len=provenance["model"]["input_shape"][1],
        seed=seed,
        steps_per_epoch=1,
        validation_steps=training["validation_steps"],
    )
    model = load_keras_model(str(model_path), compile=False)
    y_true, y_prob = predict_validation(model, val_ds, val_steps)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / "predictions.npz", y_true=y_true, y_score=y_prob)
    meta = {
        "sha256_model": sha,
        "seed": seed,
        "validation_steps": val_steps,
        "n_samples": int(len(y_prob)),
        "source": "regenerated_from_checkpoint",
    }
    (pred_dir / "predictions_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    LOGGER.info("predições regeneradas e salvas em %s (%d amostras)", pred_dir, len(y_prob))
    return y_true, y_prob


def reconcile_with_legacy(run_dir: Path, result: EvaluationResult) -> dict:
    """Compara o resultado v2 com artefatos legados da run (se existirem)."""
    reconciliation: dict = {"run_dir": str(run_dir), "legacy_files": [], "deltas": {}}
    legacy_mpc_path = run_dir / "metrics_per_class.json"
    if legacy_mpc_path.exists():
        legacy = json.loads(legacy_mpc_path.read_text(encoding="utf-8"))
        reconciliation["legacy_files"].append("metrics_per_class.json")
        new_pc = result.metrics_per_class["at_0.5"]
        deltas: dict = {}
        for cls, m in legacy.get("per_class", {}).items():
            if cls in new_pc:
                deltas[cls] = {
                    "auc_roc": {
                        "legacy": m["auc_roc"],
                        "v2": new_pc[cls]["auc_roc"],
                        "delta": abs((new_pc[cls]["auc_roc"] or 0) - (m["auc_roc"] or 0)),
                    },
                    "auc_pr": {
                        "legacy": m["auc_pr"],
                        "v2": new_pc[cls]["auc_pr"],
                        "delta": abs((new_pc[cls]["auc_pr"] or 0) - (m["auc_pr"] or 0)),
                    },
                    "f1_at_0.5": {
                        "legacy": m["f1"],
                        "v2": new_pc[cls]["f1"],
                        "delta": abs(new_pc[cls]["f1"] - m["f1"]),
                    },
                }
        reconciliation["deltas"]["per_class"] = deltas
        legacy_aucs = [m["auc_roc"] for m in legacy["per_class"].values() if m["auc_roc"]]
        if legacy_aucs:
            reconciliation["deltas"]["macro_auroc"] = {
                "legacy": float(np.mean(legacy_aucs)),
                "v2": result.metrics["metrics"]["macro_auroc"],
                "delta": abs(
                    result.metrics["metrics"]["macro_auroc"] - float(np.mean(legacy_aucs))
                ),
            }
    legacy_cal_path = run_dir / "calibration.json"
    if legacy_cal_path.exists():
        legacy = json.loads(legacy_cal_path.read_text(encoding="utf-8"))
        reconciliation["legacy_files"].append("calibration.json")
        reconciliation["deltas"]["temperature"] = {
            "legacy": legacy.get("temperature"),
            "v2": result.calibration["temperature"],
        }
        reconciliation["deltas"]["ece"] = {
            "legacy_pre": legacy.get("ece_before"),
            "legacy_post": legacy.get("ece_after"),
            "v2_pre": result.metrics["metrics"]["ece_pre_calibration"],
            "v2_post": result.metrics["metrics"]["ece_post_calibration"],
        }
    return reconciliation


def write_artifacts(result: EvaluationResult, output_dir: Path) -> None:
    """Grava o conjunto evaluation_v2 (nunca sobrescreve artefatos da run)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "metrics.json": result.metrics,
        "metrics_per_class.json": result.metrics_per_class,
        "calibration.json": result.calibration,
        "thresholds.json": result.thresholds,
        "reliability.json": result.reliability,
        "confidence_intervals.json": result.confidence_intervals,
    }
    if result.reconciliation:
        payloads["reconciliation.json"] = result.reconciliation
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    LOGGER.info("artefatos evaluation_v2 escritos em %s (%d arquivos)", output_dir, len(payloads))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--task-profile", required=True, choices=sorted(DEFAULT_CLASSES))
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--calibration-predictions", type=Path, default=None)
    parser.add_argument(
        "--temperature-source",
        default="none",
        help="'none' | 'fit' | caminho p/ calibration.json com 'temperature'",
    )
    parser.add_argument("--threshold-policy", default="max_f1_per_class")
    parser.add_argument("--ontology-version", default="v3")
    parser.add_argument("--preprocessing-version", default="v1.0")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if args.predictions is not None:
        y_true, y_score = load_predictions_npz(args.predictions)
    elif (args.output_dir / "predictions" / "predictions.npz").exists():
        y_true, y_score = load_predictions_npz(
            args.output_dir / "predictions" / "predictions.npz"
        )
    elif args.run_dir is not None:
        y_true, y_score = regenerate_predictions(args.run_dir, args.output_dir)
    else:
        LOGGER.error("sem --predictions nem --run-dir para regenerar predições")
        return 2

    calibration_data = None
    if args.calibration_predictions is not None:
        calibration_data = load_predictions_npz(args.calibration_predictions)

    fit_temp = args.temperature_source == "fit"
    temperature: Optional[float] = None
    if args.temperature_source not in ("none", "fit"):
        source = Path(args.temperature_source)
        if not source.is_absolute() and args.run_dir is not None:
            source = args.run_dir / source
        payload = json.loads(source.read_text(encoding="utf-8"))
        if "temperature" not in payload or payload["temperature"] is None:
            LOGGER.error("%s não contém 'temperature'", source)
            return 3
        temperature = float(payload["temperature"])

    run_id = args.run_dir.name if args.run_dir is not None else args.output_dir.name
    result = evaluate(
        y_true,
        y_score,
        task_profile=args.task_profile,
        split_id=args.split_name,
        ontology_version=args.ontology_version,
        n_bins=args.n_bins,
        temperature=temperature,
        fit_temperature=fit_temp,
        threshold_policy=ThresholdPolicy(name=args.threshold_policy),
        calibration_data=calibration_data,
        run_id=run_id,
        preprocessing_version=args.preprocessing_version,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    if args.run_dir is not None:
        result.reconciliation = reconcile_with_legacy(args.run_dir, result)
    write_artifacts(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
