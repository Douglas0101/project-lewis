"""Advanced evaluation for pretrain runs (FASE 8).

Per-class metrics, calibration (ECE/MCE/Brier), temperature scaling and
confusion counts — computed on the validation split for evaluation/reporting
only (never for training decisions). Threshold tuning results are reported
as analysis, not applied to gates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.chapman_labels import SCP_SUPERCLASSES
from src.models.pretrain_provenance import compute_per_class_metrics

LOGGER = logging.getLogger("lewis.camada04.evaluation")


def ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Expected/Maximum Calibration Error for one label column."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        gap = abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
        ece += mask.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def sigmoid_to_logits(y_prob: np.ndarray) -> np.ndarray:
    p = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def nll_multilabel(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary NLL over samples×labels."""
    z = np.clip(logits, -30, 30)
    return float(np.mean(np.maximum(z, 0) - z * y_true + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Scalar temperature minimizing validation NLL (grid + refine)."""
    from scipy.optimize import minimize_scalar

    logits = sigmoid_to_logits(y_prob)

    def objective(t: float) -> float:
        return nll_multilabel(y_true, logits / t)

    result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def apply_temperature(y_prob: np.ndarray, temperature: float) -> np.ndarray:
    logits = sigmoid_to_logits(y_prob) / temperature
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def confusion_per_class(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """TP/FP/TN/FN counts per class at ``threshold``."""
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, dict] = {}
    for idx, cls in enumerate(SCP_SUPERCLASSES[: y_true.shape[1]]):
        yt, yp = y_true[:, idx], y_pred[:, idx]
        out[cls] = {
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "tn": int(((yt == 0) & (yp == 0)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
        }
    return out


def best_f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Per-class best-F1 threshold (analysis only — NOT used by any gate)."""
    from sklearn.metrics import f1_score

    out: dict[str, dict] = {}
    for idx, cls in enumerate(SCP_SUPERCLASSES[: y_true.shape[1]]):
        yt, ys = y_true[:, idx], y_prob[:, idx]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.96, 0.05):
            f1 = f1_score(yt, (ys >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
        out[cls] = {"threshold": best_t, "f1": best_f1}
    return out


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Per-class reliability diagram data (mean predicted vs mean observed per bin)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    per_class: dict[str, list] = {}
    for idx, cls in enumerate(SCP_SUPERCLASSES[: y_true.shape[1]]):
        yt, yp = y_true[:, idx], y_prob[:, idx]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (yp >= lo) & (yp < hi if hi < 1.0 else yp <= hi)
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "count": int(mask.sum()),
                    "mean_pred": float(yp[mask].mean()) if mask.any() else None,
                    "mean_obs": float(yt[mask].mean()) if mask.any() else None,
                }
            )
        per_class[cls] = rows
    return {"n_bins": n_bins, "per_class": per_class}


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Macro + per-class ECE/MCE/Brier + reliability diagram data."""
    per_class: dict[str, dict] = {}
    eces, briers = [], []
    for idx, cls in enumerate(SCP_SUPERCLASSES[: y_true.shape[1]]):
        ece, mce = ece_mce(y_true[:, idx], y_prob[:, idx], n_bins)
        brier = brier_score(y_true[:, idx], y_prob[:, idx])
        per_class[cls] = {"ece": ece, "mce": mce, "brier": brier}
        eces.append(ece)
        briers.append(brier)
    return {
        "n_bins": n_bins,
        "macro": {"ece": float(np.mean(eces)), "brier": float(np.mean(briers))},
        "per_class": per_class,
        "reliability": reliability_bins(y_true, y_prob, n_bins),
    }


def _macro_auc_roc(metrics_per_class: dict) -> float | None:
    """Mean of non-None per-class AUC-ROC; None if unavailable for all classes."""
    aucs = [
        m["auc_roc"]
        for m in metrics_per_class["per_class"].values()
        if m["auc_roc"] is not None
    ]
    return float(np.mean(aucs)) if aucs else None


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Full advanced evaluation bundle for a (y_true, y_prob) validation pair."""
    report: dict = {
        "metrics_per_class": compute_per_class_metrics(y_true, y_prob),
        "confusion_threshold_0_5": confusion_per_class(y_true, y_prob, 0.5),
        "calibration_before": calibration_summary(y_true, y_prob, n_bins=n_bins),
    }
    temperature = fit_temperature(y_true, y_prob)
    y_cal = apply_temperature(y_prob, temperature)
    metrics_cal = compute_per_class_metrics(y_true, y_cal)
    report["temperature_scaling"] = {
        "temperature": temperature,
        "nll_before": nll_multilabel(y_true, sigmoid_to_logits(y_prob)),
        "nll_after": nll_multilabel(y_true, sigmoid_to_logits(y_cal)),
        "auc_roc_macro_before": _macro_auc_roc(report["metrics_per_class"]),
        "auc_roc_macro_after": _macro_auc_roc(metrics_cal),
        "calibration_after": calibration_summary(y_true, y_cal, n_bins=n_bins),
    }
    report["best_f1_thresholds_analysis_only"] = best_f1_thresholds(y_true, y_prob)
    pr_auc = {
        cls: m["auc_pr"]
        for cls, m in report["metrics_per_class"]["per_class"].items()
        if m["auc_pr"] is not None
    }
    if pr_auc:
        worst = sorted(pr_auc, key=lambda c: pr_auc[c])[:2]
        report["error_analysis"] = {
            "worst_pr_auc_classes": [{"class": c, "auc_pr": pr_auc[c]} for c in worst]
        }
    return report


def predict_validation(
    model: tf.keras.Model, val_dataset: tf.data.Dataset, validation_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Predict on the validation split; returns (y_true, y_prob)."""
    y_prob = model.predict(val_dataset, steps=validation_steps, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in val_dataset.take(validation_steps)], axis=0)
    return y_true[: len(y_prob)], y_prob


def write_reliability_diagram(run_dir: Path, reliability: dict, ece: float) -> bool:
    """Render a reliability diagram PNG (matplotlib Agg); False if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib indisponível; reliability diagram não renderizado")
        return False
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfeita")
    for cls, rows in reliability["per_class"].items():
        xs = [r["mean_pred"] for r in rows if r["count"] > 0]
        ys = [r["mean_obs"] for r in rows if r["count"] > 0]
        ax.plot(xs, ys, marker="o", label=cls)
    ax.set_xlabel("confiança média predita")
    ax.set_ylabel("frequência observada")
    ax.set_title(f"Reliability diagram (ECE={ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "reliability_diagram.png", dpi=120)
    plt.close(fig)
    return True


def write_evaluation_reports(
    run_dir: Path, report: dict, contract: dict | None = None
) -> None:
    """Persist evaluation_report.json/.md and calibration.json in the run dir.

    When ``contract`` is given, calibration.json also carries the flat T1
    contract metadata (run/model identity, split, seed, sha256) plus flat
    ``temperature``/``ece_before``/``ece_after``/``n_bins`` summary fields.
    """
    run_dir = Path(run_dir)
    (run_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ts = report["temperature_scaling"]
    calibration: dict = {}
    if contract:
        calibration.update(contract)
        calibration.update(
            {
                "temperature": ts["temperature"],
                "ece_before": report["calibration_before"]["macro"]["ece"],
                "ece_after": ts["calibration_after"]["macro"]["ece"],
                "n_bins": report["calibration_before"]["n_bins"],
            }
        )
    calibration.update(
        {
            "before": report["calibration_before"],
            "temperature_scaling": ts,
        }
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if write_reliability_diagram(
        run_dir,
        report["calibration_before"]["reliability"],
        report["calibration_before"]["macro"]["ece"],
    ):
        LOGGER.info("reliability_diagram.png salvo em %s", run_dir)
    lines = [
        f"# Avaliação avançada — {run_dir.name}",
        "",
        "## Calibração (macro)",
        f"- ECE antes: {report['calibration_before']['macro']['ece']:.4f}",
        f"- Brier antes: {report['calibration_before']['macro']['brier']:.4f}",
        f"- Temperature: {ts['temperature']:.3f} "
        f"(NLL {ts['nll_before']:.4f} → {ts['nll_after']:.4f})",
        f"- ECE depois: {ts['calibration_after']['macro']['ece']:.4f}",
        "",
        "## Métricas por classe",
        "| classe | support | auc_roc | auc_pr | P | R | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, m in report["metrics_per_class"]["per_class"].items():
        auc_roc = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "n/a"
        auc_pr = f"{m['auc_pr']:.4f}" if m["auc_pr"] is not None else "n/a"
        lines.append(
            f"| {cls} | {m['support']} | {auc_roc} | {auc_pr} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines.append("")
    lines.append("> Thresholds alternativos: análise apenas — não aplicados a gates.")
    (run_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("evaluation_report salvo em %s", run_dir)
