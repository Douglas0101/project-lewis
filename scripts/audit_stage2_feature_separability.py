"""Auditoria de separabilidade das 16 features atuais para a classe F (E05).

Avalia se o espaco de features contem sinal generalizavel suficiente para
separar a classe F dos demais, especialmente fora dos registros 208/213.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("audit_stage2_feature_separability")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_dict(d: dict) -> dict:
    """Converte valores numpy para tipos nativos."""
    return {
        k: (
            float(v)
            if isinstance(v, (np.floating, float))
            else int(v) if isinstance(v, (np.integer, int)) else v
        )
        for k, v in d.items()
    }


def _feature_statistics(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    """Estatisticas descritivas por feature."""
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        series = pd.Series(df[col])
        try:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            stats[col] = _safe_dict(
                {
                    "count": int(series.count()),
                    "missing": int(series.isna().sum()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "median": float(series.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        except Exception as exc:
            raise ValueError(f"Falha ao calcular estatisticas de {col}: {exc}") from exc
    return stats


def _class_statistics(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Estatisticas por feature estratificadas por classe."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for cls in sorted(df["y"].unique()):
        subset = pd.DataFrame(df[df["y"] == cls])
        result[f"class_{cls}"] = _feature_statistics(subset, feature_cols)
    return result


def _mutual_information_discrete(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimativa simples de mutual information via discretizacao."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    try:
        hist_2d, _, _ = np.histogram2d(x_arr, y_arr, bins=bins)
        pxy = hist_2d / float(hist_2d.sum())
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mi = 0.0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
        return float(mi)
    except Exception as exc:
        raise ValueError(f"Falha ao calcular MI: {exc}") from exc


def _compute_mutual_information(
    df: pd.DataFrame, feature_cols: list[str], task: str
) -> dict[str, float]:
    """MI de cada feature para a tarefa indicada."""
    df_work = pd.DataFrame(df)
    if task == "F_vs_rest":
        y_task = (df_work["y"] == 2).astype(np.int64).to_numpy()
    elif task == "F_vs_S":
        df_work = pd.DataFrame(df_work[df_work["y"].isin([0, 2])])
        y_task = (df_work["y"] == 2).astype(np.int64).to_numpy()
    elif task == "F_vs_V":
        df_work = pd.DataFrame(df_work[df_work["y"].isin([1, 2])])
        y_task = (df_work["y"] == 2).astype(np.int64).to_numpy()
    elif task == "S_vs_V":
        df_work = pd.DataFrame(df_work[df_work["y"].isin([0, 1])])
        y_task = df_work["y"].astype(np.int64).to_numpy()
    else:
        raise ValueError(f"Tarefa desconhecida: {task}")

    mi: dict[str, float] = {}
    for col in feature_cols:
        try:
            mi[col] = _mutual_information_discrete(df_work[col].to_numpy(), y_task)
        except Exception as exc:
            raise ValueError(f"Falha ao calcular MI para {col}: {exc}") from exc
    return mi


def _leave_one_feature_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Treina RandomForest sem cada feature e mede perda de F1-macro."""
    gkf = GroupKFold(n_splits=5)
    baseline_f1 = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_val)
            baseline_f1.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))
        except Exception as exc:
            raise ValueError(f"Falha no LOFO baseline: {exc}") from exc

    baseline_mean = float(np.mean(baseline_f1))
    results: dict[str, Any] = {"baseline_f1_macro": baseline_mean}
    for i, col in enumerate(feature_cols):
        try:
            X_lofo = np.delete(X, i, axis=1)
            fold_f1: list[float] = []
            for train_idx, val_idx in gkf.split(X_lofo, y, groups):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_lofo[train_idx])
                X_val = scaler.transform(X_lofo[val_idx])
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                clf.fit(X_train, y[train_idx])
                y_pred = clf.predict(X_val)
                fold_f1.append(
                    f1_score(y[val_idx], y_pred, average="macro", zero_division=0)
                )  # type: ignore
            results[col] = {
                "f1_macro_without": float(np.mean(fold_f1)),
                "delta": float(np.mean(fold_f1)) - baseline_mean,
            }
        except Exception as exc:
            raise ValueError(f"Falha no LOFO para {col}: {exc}") from exc
    return results


def _permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
) -> dict[str, float]:
    """Permutation importance de F1-macro usando GroupKFold."""
    gkf = GroupKFold(n_splits=5)
    importances = np.zeros(len(feature_cols))
    for train_idx, val_idx in gkf.split(X, y, groups):
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y[train_idx])
            result = permutation_importance(
                clf,
                X_val,
                y[val_idx],
                n_repeats=5,
                random_state=42,
                scoring="f1_macro",
                n_jobs=-1,
            )
            importances += result.importances_mean
        except Exception as exc:
            raise ValueError(f"Falha na permutation importance: {exc}") from exc
    importances /= 5.0
    return {col: float(importances[i]) for i, col in enumerate(feature_cols)}


def _leave_group_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    group_targets: list[int],
) -> dict[str, Any]:
    """Treina sem grupos alvo e avalia apenas nos grupos alvo."""
    results: dict[str, Any] = {}
    for target in group_targets:
        try:
            train_mask = groups != target
            val_mask = groups == target
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_val = scaler.transform(X[val_mask])
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y[train_mask])
            y_pred = clf.predict(X_val)
            f1_per_class = f1_score(
                y[val_mask], y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )  # type: ignore
            f1_macro = f1_score(
                y[val_mask], y_pred, labels=[0, 1, 2], average="macro", zero_division=0
            )  # type: ignore
            results[f"without_group_{target}"] = {
                "train_groups": int(np.unique(groups[train_mask]).shape[0]),
                "test_n": int(val_mask.sum()),
                "f1_macro": float(f1_macro),
                "f1_S": float(f1_per_class[0]),
                "f1_V": float(f1_per_class[1]),
                "f1_F": float(f1_per_class[2]),
            }
        except Exception as exc:
            raise ValueError(f"Falha no leave-group-out para {target}: {exc}") from exc
    return results


def _run_separability_audit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Executa a auditoria completa de separabilidade."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=feature_cols)
    df["y"] = y
    df["group"] = groups

    regimes = {
        "global": df,
        "record_208": df[df["group"] == 27],
        "record_213": df[df["group"] == 30],
        "excluding_208_213": df[~df["group"].isin([27, 30])],
    }

    regime_reports = {}
    for regime_name, regime_df in regimes.items():
        LOGGER.info("Analisando regime: %s", regime_name)
        stats = _feature_statistics(regime_df, feature_cols)
        class_stats = _class_statistics(regime_df, feature_cols)

        mi_tasks = {}
        for task in ["F_vs_rest", "F_vs_S", "F_vs_V", "S_vs_V"]:
            try:
                mi_tasks[task] = _compute_mutual_information(regime_df, feature_cols, task)
            except Exception as exc:
                LOGGER.warning("MI %s/%s falhou: %s", regime_name, task, exc)
                mi_tasks[task] = {}

        regime_reports[regime_name] = {
            "n_samples": int(len(regime_df)),
            "class_counts": {
                int(c): int((regime_df["y"] == c).sum()) for c in sorted(regime_df["y"].unique())
            },
            "feature_statistics": stats,
            "class_statistics": class_stats,
            "mutual_information": mi_tasks,
        }

    # LOFO e permutation importance no regime global
    LOGGER.info("Calculando LOFO e permutation importance")
    lofo = _leave_one_feature_out(X, y, groups, feature_cols)
    perm = _permutation_importance(X, y, groups, feature_cols)

    # Leave-group-out
    LOGGER.info("Calculando leave-group-out")
    lgo = _leave_group_out(X, y, groups, feature_cols, [27, 30])

    # Classificacao diagnostico inverso: treinar com 208+213, testar resto
    inverse_mask = df["group"].isin([27, 30])
    if inverse_mask.any() and (~inverse_mask).any():
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[inverse_mask])
            X_val = scaler.transform(X[~inverse_mask])
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y[inverse_mask])
            y_pred = clf.predict(X_val)
            f1_per_class = f1_score(
                y[~inverse_mask], y_pred, labels=[0, 1, 2], average=None, zero_division=0
            )  # type: ignore
            f1_macro = f1_score(
                y[~inverse_mask], y_pred, labels=[0, 1, 2], average="macro", zero_division=0
            )  # type: ignore
            lgo["dominated_by_208_213"] = {
                "train_groups": 2,
                "test_n": int((~inverse_mask).sum()),
                "f1_macro": float(f1_macro),
                "f1_S": float(f1_per_class[0]),
                "f1_V": float(f1_per_class[1]),
                "f1_F": float(f1_per_class[2]),
            }
        except Exception as exc:
            LOGGER.warning("Leave-group-out inverso falhou: %s", exc)

    report = {
        "regimes": regime_reports,
        "leave_one_feature_out": lofo,
        "permutation_importance": perm,
        "leave_group_out": lgo,
    }

    with open(output_dir / "feature_separability_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # CSV resumido
    rows = []

    def _nested_get(d: dict[str, Any], key: str, sub: str, default: float = 0.0) -> float:
        try:
            return float(d.get(key, {}).get(sub, default))  # type: ignore
        except Exception:
            return default

    for regime_name, regime in regime_reports.items():
        for col in feature_cols:
            rows.append(
                {
                    "regime": regime_name,
                    "feature": col,
                    "mi_F_vs_rest": _nested_get(
                        cast(dict[str, Any], regime.get("mutual_information", {})),
                        "F_vs_rest",
                        col,
                    ),
                    "lofo_delta": (
                        _nested_get(lofo, col, "delta") if regime_name == "global" else None
                    ),
                    "perm_importance": perm.get(col, 0.0) if regime_name == "global" else None,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "feature_separability_summary.csv", index=False)

    md_lines = [
        "# Auditoria de Separabilidade das 16 Features (Stage 2)",
        "",
        "## Regimes analisados",
        "",
    ]
    for regime_name, regime in regime_reports.items():
        md_lines.append(f"### {regime_name}")
        md_lines.append(f"- n_samples: {regime['n_samples']}")
        md_lines.append(f"- class_counts: {regime['class_counts']}")
        md_lines.append("")

    md_lines.append("## Leave-one-feature-out (global)")
    md_lines.append("")
    lofo_df = pd.DataFrame.from_dict(lofo, orient="index")
    md_lines.append(str(lofo_df.to_markdown()))
    md_lines.append("")

    md_lines.append("## Permutation importance (global)")
    md_lines.append("")
    perm_df = pd.DataFrame(list(perm.items()), columns=["feature", "importance"])
    perm_df = perm_df.sort_values("importance", ascending=False)
    md_lines.append(str(perm_df.to_markdown(index=False)))
    md_lines.append("")

    md_lines.append("## Leave-group-out")
    md_lines.append("")
    lgo_df = pd.DataFrame.from_dict(lgo, orient="index")
    md_lines.append(str(lgo_df.to_markdown()))
    md_lines.append("")

    md_lines.append("## NOTA")
    md_lines.append(
        "PCA/UMAP e outras analises exploratorias nao sao evidencia de performance. "
        "As metricas acima sao diagnosticas para guiar a research branch."
    )

    (output_dir / "feature_separability_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    LOGGER.info("Separability audit saved to %s", output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditoria de separabilidade das 16 features para a classe F."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E05_feature_separability",
    )
    args = parser.parse_args()

    try:
        npz = np.load(args.input_npz)
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64)
        groups = np.asarray(npz["groups"])
        with open(args.input_json) as f:
            feature_cols = json.load(f)["feature_names"]
        _run_separability_audit(X, y, groups, feature_cols, args.output_dir)
    except Exception as exc:
        LOGGER.error("Falha na auditoria de separabilidade: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
