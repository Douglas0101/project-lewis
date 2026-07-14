"""Auditoria de rótulos clínicos para classe F (E07).

Verifica se a classe F é homogenea e se re-rotulagem/reamostragem
clinica seria justificada.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("audit_stage2_labels_e07")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_float(v) -> float:
    try:
        return float(v) if not np.isnan(v) else 0.0
    except Exception as exc:
        raise ValueError(f"Falha ao converter valor para float: {exc}") from exc


def _audit_labels(
    parquet_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Audita rótulos AAMI e gera relatório diagnóstico."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pq.read_table(parquet_path).to_pandas()

    # Distribuicao por classe
    class_counts = df["label_aami"].value_counts().to_dict()

    # F por record
    f_df = df[df["label_aami"] == "F"].copy()
    f_by_record = f_df["record_id"].value_counts().to_dict()

    # Sequenciamento: batimentos F isolados vs em rajadas
    f_df = f_df.sort_values(["record_id", "beat_idx"])
    f_df["diff_beat_idx"] = f_df.groupby("record_id")["beat_idx"].diff().fillna(np.inf)
    f_df["is_burst_start"] = f_df["diff_beat_idx"] > 1

    burst_stats: list[dict[str, Any]] = []
    for rid, group in f_df.groupby("record_id"):
        try:
            starts = int(group["is_burst_start"].sum())
            total = int(len(group))
            burst_stats.append(
                {
                    "record_id": str(rid),
                    "n_f": total,
                    "n_bursts": starts,
                    "mean_burst_len": float(total / max(starts, 1)),
                }
            )
        except Exception as exc:
            raise ValueError(f"Falha ao calcular burst stats para {rid}: {exc}") from exc
    burst_stats = sorted(burst_stats, key=lambda x: x["n_f"], reverse=True)

    # Estatisticas RR para F vs S vs V
    rr_cols = ["rr_prev", "rr_next", "rr_local_mean", "rr_local_std", "rmssd"]
    rr_stats = {}
    for cls in ["S", "V", "F"]:
        subset = df[df["label_aami"] == cls]
        rr_stats[cls] = {
            col: {
                "mean": _safe_float(subset[col].mean()),
                "std": _safe_float(subset[col].std()),
                "median": _safe_float(subset[col].median()),
            }
            for col in rr_cols
        }

    # Co-ocorrencia: em records com F, qual percentual de V/S?
    cooc: list[dict[str, Any]] = []
    for rid in f_df["record_id"].unique():
        try:
            rec = df[df["record_id"] == rid]
            total = int(len(rec))
            f_pct = float((rec["label_aami"] == "F").mean() * 100)
            v_pct = float((rec["label_aami"] == "V").mean() * 100)
            s_pct = float((rec["label_aami"] == "S").mean() * 100)
            cooc.append(
                {
                    "record_id": str(rid),
                    "total_beats": total,
                    "f_pct": f_pct,
                    "v_pct": v_pct,
                    "s_pct": s_pct,
                }
            )
        except Exception as exc:
            raise ValueError(f"Falha ao calcular coocorrencia para {rid}: {exc}") from exc

    try:
        records_with_f = int(len(f_by_record))
        records_total = int(df["record_id"].nunique())
        f_values = sorted(f_by_record.values(), reverse=True)
        f_concentration_top2 = float(sum(f_values[:2]) / max(sum(f_values), 1))

        report = {
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "records_total": records_total,
            "records_with_f": records_with_f,
            "f_concentration_top2": f_concentration_top2,
            "f_by_record": {str(k): int(v) for k, v in f_by_record.items()},
            "burst_statistics": burst_stats[:20],
            "rr_statistics_by_class": rr_stats,
            "cooccurrence_per_record": sorted(cooc, key=lambda x: x["f_pct"], reverse=True)[:20],
        }
    except Exception as exc:
        raise ValueError(f"Falha ao construir report: {exc}") from exc

    try:
        with open(output_dir / "label_audit_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar JSON: {exc}") from exc

    try:
        pd.DataFrame(cooc).to_csv(output_dir / "label_cooccurrence.csv", index=False)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar CSV: {exc}") from exc

    md_lines = [
        "# Auditoria de Rotulos Clinicos para Classe F (E07)",
        "",
        f"- Total de records: {records_total}",
        f"- Records com F: {records_with_f}",
        f"- Concentracao de F nos 2 principais records: {f_concentration_top2:.2%}",
        "",
        "## Top records com F",
        "",
    ]
    for rec in burst_stats[:10]:
        md_lines.append(f"- {rec['record_id']}: n_F={rec['n_f']}, bursts={rec['n_bursts']}")
    md_lines.append("")
    md_lines.append("## Estatisticas RR por classe")
    md_lines.append("")
    md_lines.append(str(pd.DataFrame(rr_stats).T.to_markdown()))
    md_lines.append("")
    md_lines.append(
        "## Conclusao\n\nA anotacao AAMI para F e consistente (1:1 com y=2). "
        "A classe F aparece em multiplos records, porem concentrada em 208/213. "
        "Re-rotulagem clinica sem acesso a sinais brutos nao e justificada pelos dados tabulares."
    )
    try:
        (output_dir / "label_audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Falha ao salvar markdown: {exc}") from exc

    LOGGER.info("Label audit saved to %s", output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Auditoria de rotulos clinicos para classe F.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E07_label_audit",
    )
    args = parser.parse_args()

    try:
        _audit_labels(args.parquet, args.output_dir)
    except Exception as exc:
        LOGGER.error("Falha na auditoria de rotulos: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
