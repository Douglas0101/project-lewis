"""Auditoria da distribuição da classe F por registro/paciente no Stage 2.

Recalcula diretamente a partir do parquet local. Não codifica os valores
documentais de 208/213; apenas os reporta quando presentes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("audit_stage2_patient_distribution")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def herfindahl(values: pd.Series | np.ndarray) -> float:
    """Índice de concentração tipo Herfindahl (diagnóstico interno)."""
    try:
        arr = np.asarray(values, dtype=np.float64)
        total = float(arr.sum())
        if total == 0:
            return 0.0
        shares = arr / total
        return float((shares**2).sum())
    except Exception as exc:
        raise ValueError(f"Falha ao calcular Herfindahl: {exc}") from exc


def run_audit(df: pd.DataFrame, output_dir: Path) -> dict:
    """Executa a auditoria e persiste os artefatos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {"record_id", "label_aami", "y", "dataset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    # Tabela de rastreabilidade
    aami_to_stage2 = {
        "S": 0,
        "V": 1,
        "F": 2,
    }
    try:
        trace_rows = []
        for label_aami, y in aami_to_stage2.items():
            count = int((df["label_aami"] == label_aami).sum())
            trace_rows.append(
                {
                    "original_annotation": label_aami,
                    "mapped_aami_class": label_aami,
                    "stage1_target": "Anormal" if label_aami != "N" else "Normal",
                    "stage2_target": y,
                    "count": count,
                }
            )
        traceability = pd.DataFrame(trace_rows)
    except Exception as exc:
        raise ValueError(f"Falha ao construir tabela de rastreabilidade: {exc}") from exc
    traceability.to_csv(output_dir / "aami_traceability.csv", index=False)
    traceability.to_json(output_dir / "aami_traceability.json", orient="records", indent=2)

    # Distribuição por grupo
    try:
        grouped = (
            df.groupby("record_id")
            .agg(
                dataset=("dataset", "first"),
                total_stage2=("y", "size"),
                S_count=("label_aami", lambda s: int((s == "S").sum())),
                V_count=("label_aami", lambda s: int((s == "V").sum())),
                F_count=("label_aami", lambda s: int((s == "F").sum())),
            )
            .reset_index()
        )
        grouped["F_percentage_within_group"] = np.round(
            100.0 * grouped["F_count"] / grouped["total_stage2"], 4
        )

        total_f = int(grouped["F_count"].sum())
        grouped["percentage_of_all_F"] = np.round(
            100.0 * grouped["F_count"] / total_f if total_f > 0 else 0.0, 4
        )
        grouped = grouped.sort_values("F_count", ascending=False).reset_index(drop=True)
        grouped["cumulative_F_percentage"] = np.round(grouped["percentage_of_all_F"].cumsum(), 4)
    except Exception as exc:
        raise ValueError(f"Falha ao agrupar por paciente: {exc}") from exc

    grouped.to_csv(output_dir / "patient_class_distribution.csv", index=False)
    grouped.to_json(output_dir / "patient_class_distribution.json", orient="records", indent=2)

    try:
        f_groups = grouped[grouped["F_count"] > 0]
        non_f_groups = grouped[grouped["F_count"] == 0]
        f_counts = np.asarray(f_groups["F_count"], dtype=np.float64)

        concentration = {
            "total_F": total_f,
            "number_of_groups_with_F": int(len(f_groups)),
            "number_of_groups_without_F": int(len(non_f_groups)),
            "top1_F_concentration": grouped.iloc[0].to_dict() if len(grouped) else None,
            "top2_F_concentration": grouped.iloc[1].to_dict() if len(grouped) > 1 else None,
            "top3_F_concentration": grouped.iloc[2].to_dict() if len(grouped) > 2 else None,
            "Herfindahl_like_F_concentration": herfindahl(f_counts),
            "median_F_per_F_group": float(np.median(f_counts)) if len(f_counts) else 0.0,
            "mean_F_per_F_group": float(np.mean(f_counts)) if len(f_counts) else 0.0,
            "std_F_per_F_group": float(np.std(f_counts)) if len(f_counts) else 0.0,
        }
    except Exception as exc:
        raise ValueError(f"Falha ao calcular métricas de concentração: {exc}") from exc

    try:
        with open(output_dir / "f_concentration_report.json", "w") as f:
            json.dump(concentration, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar f_concentration_report.json: {exc}") from exc

    md_lines = [
        "# Distribuição da Classe F por Registro/Paciente (Stage 2)",
        "",
        f"- Total de exemplos Stage 2: {len(df)}",
        f"- Total de grupos/registros: {df['record_id'].nunique()}",
        f"- Total de batimentos classe F: {total_f}",
        f"- Grupos com F: {concentration['number_of_groups_with_F']}",
        f"- Grupos sem F: {concentration['number_of_groups_without_F']}",
        "",
        "## Top 5 grupos com mais F",
        "",
        grouped.head().to_markdown(index=False),
        "",
        "## Métricas de concentração",
        "",
        f"- Herfindahl-like F concentration: "
        f"{concentration['Herfindahl_like_F_concentration']:.4f}",
        f"- Median F per F-group: {concentration['median_F_per_F_group']:.1f}",
        f"- Mean F per F-group: {concentration['mean_F_per_F_group']:.1f}",
        f"- Std F per F-group: {concentration['std_F_per_F_group']:.1f}",
        "",
        "## Tabela de rastreabilidade AAMI",
        "",
        traceability.to_markdown(index=False),
        "",
        "## Notas",
        "",
        "Os registros 208 e 213, quando presentes no dataset local, "
        "são reportados explicitamente na tabela acima. "
        "Não foram codificados como referência; os valores são "
        "recalculados a partir da fonte local.",
    ]
    (output_dir / "f_concentration_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    LOGGER.info("Audit reports saved to %s", output_dir)
    LOGGER.info(
        "TOP F groups: %s",
        list(
            grouped.head(3)[
                ["record_id", "F_count", "percentage_of_all_F", "cumulative_F_percentage"]
            ].to_dict(
                orient="records"
            )  # type: ignore[call-arg]
        ),
    )
    return concentration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditoria da distribuição da classe F por registro/paciente no Stage 2."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E01_patient_distribution",
    )
    args = parser.parse_args()

    try:
        df = pd.read_parquet(args.input_parquet)
        run_audit(df, args.output_dir)
    except Exception as exc:
        LOGGER.error("Falha na auditoria: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
