#!/usr/bin/env python3
"""
scripts/analyze_training_dynamics.py

Análise correlacional de dinâmica de treinamento.
Lê logs de treinamento, gradientes e calibração para gerar
insights acionáveis sobre a relação entre gradientes, calibração e métricas F1.

Uso:
    python scripts/analyze_training_dynamics.py \
        --training_log logs/finetune_v1.1_full_v3.log \
        --gradients_log logs/gradients_v1.1_full_v3.json \
        --calibration_log logs/calibration_v1.1_full_v3.json \
        --output_dir logs/figures/ \
        --report_path logs/training_dynamics_analysis.md

Restrições:
    - Apenas numpy, scipy, matplotlib (sem dependências novas)
    - Código em português (docstrings), comentários técnicos em inglês

Autor: Douglas Souza
Data: 2026-06-21
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path: str) -> Dict:
    """Extrai métricas de época do log de treinamento Keras.

    Suporta formato padrão de logs do Keras:
        Epoch 1/50
        loss: 0.4567 - accuracy: 0.7890 - val_loss: 0.3456 - val_accuracy: 0.8123

    Args:
        log_path: Caminho para o arquivo .log

    Returns:
        Dicionário com listas de métricas por época.
    """
    epochs: List[int] = []
    train_loss: List[float] = []
    train_acc: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []
    val_f1_macro: List[float] = []

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex flexível para extrair métricas de cada época.
    # Matches "Epoch N/M" seguido de uma linha contendo loss/accuracy.
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s*\n"
        r".*?"
        r"loss:\s+([\d.]+).*?"
        r"accuracy:\s+([\d.]+).*?"
        r"val_loss:\s+([\d.]+).*?"
        r"val_accuracy:\s+([\d.]+)"
    )

    for match in epoch_pattern.finditer(content):
        epoch = int(match.group(1))
        epochs.append(epoch)
        train_loss.append(float(match.group(3)))
        train_acc.append(float(match.group(4)))
        val_loss.append(float(match.group(5)))
        val_acc.append(float(match.group(6)))

    # Tentar extrair F1-macro se presente no log.
    f1_pattern = re.compile(r"val_f1_macro:\s+([\d.]+)")
    for match in f1_pattern.finditer(content):
        val_f1_macro.append(float(match.group(1)))

    # Ajustar tamanhos se F1-macro não estiver presente.
    if len(val_f1_macro) < len(epochs):
        val_f1_macro.extend([0.0] * (len(epochs) - len(val_f1_macro)))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1_macro": val_f1_macro[: len(epochs)],
    }


def parse_gradients_log(log_path: str) -> Dict:
    """Carrega log de gradientes em formato JSON.

    Args:
        log_path: Caminho para gradients_*.json

    Returns:
        Dicionário com métricas de gradiente por época e camada.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs: List[int] = []
    norm_ratios: Dict[str, List[float]] = {}
    p95_gradients: Dict[str, List[float]] = {}

    for entry in data:
        epochs.append(entry["epoch"])
        for layer in entry.get("layers", []):
            name = layer["layer_name"]
            if name not in norm_ratios:
                norm_ratios[name] = []
                p95_gradients[name] = []
            norm_ratios[name].append(float(layer["norm_ratio"]))
            p95_gradients[name].append(float(layer["p95_gradient"]))

    return {
        "epochs": epochs,
        "norm_ratios": norm_ratios,
        "p95_gradients": p95_gradients,
    }


def parse_calibration_log(log_path: str) -> Dict:
    """Carrega log de calibração em formato JSON.

    Args:
        log_path: Caminho para calibration_*.json

    Returns:
        Dicionário com métricas de calibração por época.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs: List[int] = []
    ece: List[float] = []
    mce: List[float] = []
    brier: List[float] = []
    brier_s: List[float] = []
    brier_v: List[float] = []
    brier_f: List[float] = []
    reliability_bins: List[List[Dict]] = []

    for entry in data:
        epochs.append(entry["epoch"])
        ece.append(float(entry["ece"]))
        mce.append(float(entry["mce"]))
        brier.append(float(entry["brier_score"]))

        bpc = entry.get("brier_per_class", {})
        brier_s.append(float(bpc.get("S", 0.0)))
        brier_v.append(float(bpc.get("V", 0.0)))
        brier_f.append(float(bpc.get("F", 0.0)))

        reliability_bins.append(entry.get("reliability_bins", []))

    return {
        "epochs": epochs,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "brier_s": brier_s,
        "brier_v": brier_v,
        "brier_f": brier_f,
        "reliability_bins": reliability_bins,
    }


def compute_correlations(
    training: Dict, gradients: Dict, calibration: Dict
) -> Dict[str, float]:
    """Computa correlações entre gradientes, calibração e F1-macro.

    Returns:
        Dicionário com correlações nomeadas.
    """
    n_epochs = min(
        len(training["epochs"]),
        len(gradients["epochs"]),
        len(calibration["epochs"]),
    )

    # Alinhar dados ao mesmo número de épocas.
    f1_macro = np.array(training["val_f1_macro"][:n_epochs], dtype=float)
    ece = np.array(calibration["ece"][:n_epochs], dtype=float)
    brier_s = np.array(calibration["brier_s"][:n_epochs], dtype=float)
    brier_v = np.array(calibration["brier_v"][:n_epochs], dtype=float)
    brier_f = np.array(calibration["brier_f"][:n_epochs], dtype=float)

    correlations: Dict[str, float] = {}

    def _has_variance(arr: np.ndarray) -> bool:
        # np.isclose avoids false positives with near-zero floating-point std.
        return not np.isclose(np.std(arr), 0.0)

    # Correlação: norm_ratio dos Dense vs F1-macro.
    for layer_name, ratios in gradients["norm_ratios"].items():
        ratios_arr = np.array(ratios[:n_epochs], dtype=float)
        if (
            len(ratios_arr) == n_epochs
            and _has_variance(ratios_arr)
            and _has_variance(f1_macro)
        ):
            corr = float(np.corrcoef(ratios_arr, f1_macro)[0, 1])
            correlations[f"norm_ratio_{layer_name}_vs_f1_macro"] = corr

    # Correlação: ECE vs F1-macro.
    if _has_variance(ece) and _has_variance(f1_macro):
        correlations["ece_vs_f1_macro"] = float(np.corrcoef(ece, f1_macro)[0, 1])

    # Correlação: Brier por classe vs F1-macro (proxy para recall).
    for cls_name, brier_arr in [("S", brier_s), ("V", brier_v), ("F", brier_f)]:
        if _has_variance(brier_arr) and _has_variance(f1_macro):
            correlations[f"brier_{cls_name}_vs_f1_macro"] = float(
                np.corrcoef(brier_arr, f1_macro)[0, 1]
            )

    return correlations


def generate_figures(
    training: Dict,
    gradients: Dict,
    calibration: Dict,
    correlations: Dict[str, float],
    output_dir: str,
) -> None:
    """Gera visualizações e salva em output_dir.

    Gera:
        1. Heatmap de correlação (barplot horizontal)
        2. Curva dual-axis: ECE e F1-macro
        3. Reliability diagram para a última época
    """
    os.makedirs(output_dir, exist_ok=True)

    n_epochs = min(
        len(training["epochs"]),
        len(gradients["epochs"]),
        len(calibration["epochs"]),
    )
    epochs = training["epochs"][:n_epochs]

    # Figura 1: Heatmap de correlação.
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_items = list(correlations.items())
    corr_names = [k for k, _ in corr_items]
    corr_values = [v for _, v in corr_items]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in corr_values]
    ax.barh(range(len(corr_names)), corr_values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(corr_names)))
    ax.set_yticklabels(corr_names, fontsize=8)
    ax.set_xlabel("Coeficiente de Correlação de Pearson", fontsize=11)
    ax.set_title(
        "Correlações: Gradientes × Calibração × F1-macro", fontsize=12, fontweight="bold"
    )
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=0.5, color="green", linestyle="--", alpha=0.5, label="Forte positiva")
    ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.5, label="Forte negativa")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Figura 2: Dual-axis — ECE e F1-macro.
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ece = calibration["ece"][:n_epochs]
    f1 = training["val_f1_macro"][:n_epochs]

    color1 = "#e74c3c"
    ax1.set_xlabel("Época", fontsize=11)
    ax1.set_ylabel("ECE (Expected Calibration Error)", color=color1, fontsize=11)
    ax1.plot(epochs, ece, color=color1, linewidth=2, label="ECE")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(y=0.15, color=color1, linestyle="--", alpha=0.7, label="Limite ECE (0.15)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#2ecc71"
    ax2.set_ylabel("F1-macro", color=color2, fontsize=11)
    ax2.plot(epochs, f1, color=color2, linewidth=2, label="F1-macro")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(y=0.55, color=color2, linestyle="--", alpha=0.7, label="Meta QG5' (0.55)")
    ax2.legend(loc="upper right", fontsize=9)

    plt.title("ECE vs F1-macro ao Longo do Treinamento", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ece_vs_f1_dual_axis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Figura 3: Reliability diagram (última época).
    if calibration.get("reliability_bins"):
        last_bins = calibration["reliability_bins"][-1]
        if last_bins:
            fig, ax = plt.subplots(figsize=(8, 6))
            bin_centers = [(b["lower_edge"] + b["upper_edge"]) / 2 for b in last_bins]
            accuracies = [b["accuracy"] for b in last_bins]
            confidences = [b["confidence"] for b in last_bins]

            ax.plot([0, 1], [0, 1], "k--", label="Perfeitamente calibrado", linewidth=1)
            ax.bar(
                bin_centers,
                accuracies,
                width=0.06,
                alpha=0.6,
                label="Acurácia",
                color="#3498db",
                edgecolor="black",
            )
            ax.plot(bin_centers, confidences, "ro-", label="Confiança", linewidth=2, markersize=6)

            ax.set_xlabel("Confiança Predita", fontsize=11)
            ax.set_ylabel("Acurácia / Confiança", fontsize=11)
            ax.set_title("Reliability Diagram — Última Época", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "reliability_diagram.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    print(f"[INFO] Figuras salvas em: {output_dir}")


def generate_report(
    training: Dict,
    gradients: Dict,
    calibration: Dict,
    correlations: Dict[str, float],
    output_path: str,
) -> None:
    """Gera relatório markdown com insights acionáveis."""
    lines: List[str] = []
    lines.append("# Análise de Dinâmica de Treinamento")
    lines.append(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    # Resumo de correlações.
    lines.append("## 1. Correlações Principais\n")
    for name, value in sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        direction = "positiva" if value > 0 else "negativa"
        if abs(value) > 0.5:
            strength = "forte"
        elif abs(value) > 0.3:
            strength = "moderada"
        else:
            strength = "fraca"
        lines.append(f"- **{name}**: {value:.3f} ({strength} {direction})\n")

    # Insights.
    lines.append("\n## 2. Insights Acionáveis\n")

    # Verificar vanishing/exploding.
    for layer_name, ratios in gradients["norm_ratios"].items():
        min_ratio = min(ratios)
        max_p95 = max(gradients["p95_gradients"].get(layer_name, [0.0]))
        if min_ratio < 1e-6:
            lines.append(
                f"🔴 **GRADIENTE VANISHING** detectado em `{layer_name}`: "
                f"norm_ratio mínimo = {min_ratio:.2e}\n"
            )
        if max_p95 > 10.0:
            lines.append(
                f"🔴 **GRADIENTE EXPLODING** detectado em `{layer_name}`: "
                f"p95 = {max_p95:.2f}\n"
            )

    # Verificar calibração.
    latest_ece = calibration["ece"][-1] if calibration["ece"] else 0.0
    latest_mce = calibration["mce"][-1] if calibration["mce"] else 0.0
    if latest_ece > 0.15:
        lines.append(f"🔴 **CALIBRAÇÃO RUIM**: ECE = {latest_ece:.3f} > 0.15\n")
    if latest_mce > 0.30:
        lines.append(f"🔴 **MÁXIMA CALIBRAÇÃO RUIM**: MCE = {latest_mce:.3f} > 0.30\n")

    # Recomendações.
    lines.append("\n## 3. Recomendações Automáticas\n")
    if any(min(r) < 1e-6 for r in gradients["norm_ratios"].values()):
        lines.append("1. **Aumentar learning rate** para camadas com vanishing gradient.\n")
        lines.append("2. **Substituir inicialização** para HeNormal em camadas afetadas.\n")
    if latest_ece > 0.15:
        lines.append("3. **Aplicar temperature scaling** pós-treinamento para calibrar softmax.\n")
        lines.append("4. **Aumentar peso da loss** para classes minoritárias (S, V, F).\n")

    lines.append("\n---\n")
    lines.append("*Relatório gerado automaticamente por analyze_training_dynamics.py*\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[INFO] Relatório salvo em: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Análise de dinâmica de treinamento MIT-BIH/AAMI"
    )
    parser.add_argument(
        "--training_log", required=True, help="Caminho do log de treinamento"
    )
    parser.add_argument(
        "--gradients_log", required=True, help="Caminho do log de gradientes"
    )
    parser.add_argument(
        "--calibration_log", required=True, help="Caminho do log de calibração"
    )
    parser.add_argument(
        "--output_dir", default="logs/figures", help="Diretório de saída"
    )
    parser.add_argument(
        "--report_path",
        default="logs/training_dynamics_analysis.md",
        help="Caminho do relatório",
    )
    args = parser.parse_args()

    print("[INFO] Analisando dinâmica de treinamento...")

    # Parse logs.
    training = parse_training_log(args.training_log)
    gradients = parse_gradients_log(args.gradients_log)
    calibration = parse_calibration_log(args.calibration_log)

    print(f"  - Épocas de treinamento: {len(training['epochs'])}")
    print(f"  - Camadas monitoradas: {list(gradients['norm_ratios'].keys())}")
    print("  - Métricas de calibração: ECE, MCE, Brier")

    # Correlações.
    correlations = compute_correlations(training, gradients, calibration)
    print(f"  - Correlações computadas: {len(correlations)}")

    # Figuras.
    generate_figures(training, gradients, calibration, correlations, args.output_dir)

    # Relatório.
    generate_report(training, gradients, calibration, correlations, args.report_path)

    print("[INFO] Análise concluída.")


if __name__ == "__main__":
    main()
