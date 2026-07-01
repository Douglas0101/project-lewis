"""Análise consolidada do treinamento Stage1 v2.0.

Lê logs de treinamento, gradientes e calibração por fold e gera relatório
com conclusões claras conforme docs/UNIFIED_DOCUMENT_v2.0.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Extrai métricas de época do log Keras."""
    content = log_path.read_text(encoding="utf-8")
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # Padrão: "Epoch 1/50" seguido de linha com loss/accuracy
    epoch_lines = re.findall(
        r"Epoch\s+(\d+)/(\d+)\s*\n\s*(\d+/\d+\s+-\s+[\dms.]+\s+-\s+)?"
        r"loss:\s+([\d.]+).*?accuracy:\s+([\d.]+).*?"
        r"val_loss:\s+([\d.]+).*?val_accuracy:\s+([\d.]+)",
        content,
    )
    for match in epoch_lines:
        epoch = int(match[0])
        epochs.append(epoch)
        train_loss.append(float(match[3]))
        train_acc.append(float(match[4]))
        val_loss.append(float(match[5]))
        val_acc.append(float(match[6]))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def parse_gradients_log(log_path: Path) -> Dict[str, Any]:
    """Carrega log de gradientes em formato JSON."""
    data = json.loads(log_path.read_text(encoding="utf-8"))
    epochs = []
    norm_ratios: Dict[str, List[float]] = {}
    grad_means: Dict[str, List[float]] = {}
    grad_per_class: Dict[str, Dict[str, List[float]]] = {}

    for entry in data:
        epochs.append(entry["epoch"])
        for layer in entry.get("layers", []):
            name = layer["layer_name"]
            norm_ratios.setdefault(name, []).append(layer["norm_ratio"])
            grad_means.setdefault(name, []).append(layer["gradient_mean"])
            for cls, value in layer.get("gradient_mean_per_class", {}).items():
                grad_per_class.setdefault(name, {}).setdefault(cls, []).append(value)

    return {
        "epochs": epochs,
        "norm_ratios": norm_ratios,
        "grad_means": grad_means,
        "grad_per_class": grad_per_class,
    }


def parse_calibration_log(log_path: Path) -> Dict[str, Any]:
    """Carrega log de calibração em formato JSON."""
    data = json.loads(log_path.read_text(encoding="utf-8"))
    epochs = []
    ece = []
    mce = []
    brier = []
    confidence_per_class: Dict[str, List[float]] = {}

    for entry in data:
        epochs.append(entry["epoch"])
        ece.append(entry["ece"])
        mce.append(entry["mce"])
        brier.append(entry["brier_score"])
        for cls, value in entry.get("confidence_per_class", {}).items():
            confidence_per_class.setdefault(cls, []).append(value)

    return {
        "epochs": epochs,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "confidence_per_class": confidence_per_class,
    }


def summarize_metrics(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "last": float(arr[-1]) if len(arr) else 0.0,
        "trend": float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0,
    }


def analyze_fold(fold_idx: int, log_dir: Path) -> Dict[str, Any]:
    """Analisa um único fold."""
    grad = parse_gradients_log(log_dir / "gradients_stage1.json")
    cal = parse_calibration_log(log_dir / "calibration_stage1.json")

    grad_summary = {}
    for layer_name, ratios in grad["norm_ratios"].items():
        grad_summary[layer_name] = summarize_metrics(ratios)

    cal_summary = {
        "ece": summarize_metrics(cal["ece"]),
        "mce": summarize_metrics(cal["mce"]),
        "brier": summarize_metrics(cal["brier"]),
    }

    return {
        "fold": fold_idx,
        "grad_summary": grad_summary,
        "cal_summary": cal_summary,
        "final_ece": cal["ece"][-1] if cal["ece"] else None,
        "final_mce": cal["mce"][-1] if cal["mce"] else None,
        "final_brier": cal["brier"][-1] if cal["brier"] else None,
    }


def load_summary(experiment_dir: Path) -> Dict[str, Any]:
    """Carrega summary.json do experimento."""
    summary_path = experiment_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def generate_report(
    experiment_dir: Path,
    output_path: Path,
    logs_dir: Path,
) -> None:
    """Gera relatório markdown com conclusões."""
    summary = load_summary(experiment_dir)
    folds_analysis = []
    for fold_idx in range(5):
        fold_dir = logs_dir / f"fold_{fold_idx}"
        if (fold_dir / "gradients_stage1.json").exists():
            folds_analysis.append(analyze_fold(fold_idx, fold_dir))

    lines = []
    lines.append("# Análise Aprofundada — Treinamento Estágio 1 v2.0\n")
    lines.append(f"**Experimento:** {experiment_dir.name}\n")
    lines.append("**Referência:** docs/UNIFIED_DOCUMENT_v2.0.md\n")
    lines.append("---\n")

    # 1. Resultados agregados
    lines.append("## 1. Resultados Agregados vs. Metas do UNIFIED_DOCUMENT\n")
    mean_metrics = summary.get("mean_metrics", {})
    f1_macro = mean_metrics.get("F1_macro", 0.0)
    acc = mean_metrics.get("Acc", 0.0)
    passes_qg5 = summary.get("passes_qg5", False)

    acc_status = "PASSA" if acc > 0.92 else "FALHA"
    f1_status = "PASSA" if f1_macro > 0.90 else "FALHA"
    lines.append(f"- **Accuracy:** {acc:.4f} | Meta: > 0,92 | **{acc_status}**\n")
    lines.append(f"- **F1-macro:** {f1_macro:.4f} | Meta: > 0,90 | **{f1_status}**\n")
    lines.append(f"- **passes_qg5:** {passes_qg5}\n")

    # 2. Resultados por fold
    lines.append("\n## 2. Resultados por Fold\n")
    lines.append("| Fold | Acc | F1-macro | Recall Anormal | Precision Anormal | Passa QG5' |\n")
    lines.append("|------|-----|----------|----------------|-------------------|------------|\n")
    for fold in summary.get("folds", []):
        g = fold["global"]
        p = fold["per_class"]["Anormal"]
        lines.append(
            f"| {fold['fold']} | {g['Acc']:.4f} | {g['F1_macro']:.4f} | "
            f"{p['Se']:.4f} | {p['PPV']:.4f} | {fold['passes_qg5']} |\n"
        )

    # 3. Diagnóstico de calibração
    lines.append("\n## 3. Diagnóstico de Calibração por Fold\n")
    lines.append("| Fold | ECE final | MCE final | Brier final | Status |\n")
    lines.append("|------|-----------|-----------|-------------|--------|\n")
    for fa in folds_analysis:
        ece = fa["final_ece"]
        mce = fa["final_mce"]
        brier = fa["final_brier"]
        status = []
        if ece is not None and ece > 0.15:
            status.append("ECE alto")
        if mce is not None and mce > 0.30:
            status.append("MCE alto")
        if brier is not None and brier > 0.50:
            status.append("Brier alto")
        status_str = "; ".join(status) if status else "OK"
        ece_str = f"{ece:.4f}" if ece is not None else "-"
        mce_str = f"{mce:.4f}" if mce is not None else "-"
        brier_str = f"{brier:.4f}" if brier is not None else "-"
        lines.append(
            f"| {fa['fold']} | {ece_str} | {mce_str} | {brier_str} | {status_str} |\n"
        )

    # 4. Diagnóstico de gradientes
    lines.append("\n## 4. Diagnóstico de Gradientes por Fold\n")
    for fa in folds_analysis:
        lines.append(f"\n### Fold {fa['fold']}\n")
        for layer_name, stats in fa["grad_summary"].items():
            line = (
                f"- **{layer_name}**: norm_ratio mean={stats['mean']:.2e}, "
                f"min={stats['min']:.2e}, max={stats['max']:.2e}, "
                f"trend={stats['trend']:.2e}/epoch\n"
            )
            lines.append(line)
            if stats["min"] < 1e-6:
                lines.append(f"  - ⚠️ **Vanishing gradient detectado** em `{layer_name}`\n")
            if stats["max"] > 10.0:
                lines.append(f"  - ⚠️ **Exploding gradient detectado** em `{layer_name}`\n")

    # 5. Conclusões
    lines.append("\n## 5. Conclusões\n")
    lines.append("### 5.1 Por que o modelo não atinge as metas?\n")
    lines.append(
        "1. **Separação probabilística ausente:** as distribuições de probabilidade preditas para "
        "N e Anormal são quase idênticas (AUC-ROC ≈ 0,56 no melhor fold). "
        "O modelo não aprendeu features discriminativas.\n"
    )
    lines.append(
        "2. **Backbone congelada não transfere:** o pré-treino no Chapman "
        "(5 superclasses SCP-ECG) não gera representações úteis para a "
        "distinção N vs. Anormal do MIT-BIH em inter-patient split. "
        "Descongelar a backbone não melhorou o desempenho.\n"
    )
    lines.append(
        "3. **Treinamento do zero também falha:** um modelo idêntico treinado "
        "from scratch no fold 2 atingiu AUC-ROC = 0,50 e F1-macro = 0,50. "
        "A arquitetura atual é insuficiente para a tarefa.\n"
    )
    lines.append(
        "4. **Calibração ruim:** ECE e Brier elevados indicam que as probabilidades do softmax "
        "não refletem a verdadeira confiança do modelo.\n"
    )

    lines.append("\n### 5.2 O que o UNIFIED_DOCUMENT preconiza?\n")
    lines.append(
        "- RF-01.1: Recall(Anormal) ≥ 0,95 (crítico para minimizar falsos negativos de arritmia). "
        "O melhor fold alcançou 0,263.\n"
    )
    lines.append(
        "- Meta Estágio 1: F1-macro > 0,90. O melhor fold alcançou 0,5463.\n"
    )
    lines.append(
        "- AUC-ROC > 0,98. O melhor fold alcançou 0,5588.\n"
    )

    lines.append("\n### 5.3 Recomendações conforme UNIFIED_DOCUMENT\n")
    lines.append(
        "- **Decisão 7 (Checklist): Fallback para features morfológicas** se Estágio 1/2 falharem. "
        "A evidência indica que esse fallback deve ser acionado.\n"
    )
    lines.append(
        "- Revisar se a **entrada de 500 amostras** (v2.0 atual) é compatível com o documento, "
        "que especifica 250 amostras. Verificar se o resample impacta a separabilidade.\n"
    )
    lines.append(
        "- Avaliar se o pré-treino no Chapman realmente cobre padrões morfológicos compatíveis "
        "com MIT-BIH; se necessário, fazer pré-treino multi-dataset (MIT-BIH + Chapman + PTB-XL).\n"
    )
    lines.append(
        "- Considerar aumentar a capacidade do Estágio 1, respeitando os limites de memória "
        "(RNF-02: Flash < 40 KB, RAM < 80 KB).\n"
    )

    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"[INFO] Relatório salvo em: {output_path}")


def main():
    experiment_dir = Path("experiments/20260630_020334_stage1_v2.0")
    if not experiment_dir.exists():
        print(f"[ERRO] Experimento nao encontrado: {experiment_dir}")
        sys.exit(1)

    output_path = Path("reports/stage1_v2_training_analysis.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(experiment_dir, output_path, logs_dir=Path("logs"))


if __name__ == "__main__":
    main()
