"""Orquestrador de células piloto T10.3 (controle H7 — matriz v2 §2).

Executa UMA célula do plano de pilotos end-to-end:

1. treino via ``scripts/pretrain_wrapper.py`` (arch/loss/seed 13, split pareado,
   early stopping por ``val_auc_pr`` — protocolo v2);
2. exporta predições das partições validation/calibration/test;
3. avaliação canônica PROSPECTIVE no teste (T/thresholds fit na calibration);
4. grava ``pilot_status.json`` (status PILOT — nunca CANDIDATE);
5. imprime o bloco de métricas final e o gate da célula (T10.3-P §2).

Não promove modelos, não toca ``models/``, não altera splits. Requer o manifesto
do split pareado já gerado (``make pilot-split``).

Uso:
    uv run python scripts/run_pilot_cell.py --cell c1
    uv run python scripts/run_pilot_cell.py --cell c1 --smoke   # validação de engenharia
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec B404
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_pilot_predictions import EVAL_PARTITIONS, export_partition  # noqa: E402
from scripts.validate_pretrain_artifacts import newest_run_dir  # noqa: E402

LOGGER = logging.getLogger("lewis.camada04.pilot_cell")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "chapman_paired_v2" / "manifest.json"
SPLIT_ID = "chapman-record-disjoint-paired-v2"

#: Registro de células — matriz v2 §2 (controle H7). Todas: seed 13, 30 épocas,
#: early stopping por val_auc_pr (protocolo v2).
PILOT_CELLS: dict[str, dict] = {
    "c0": {"architecture": "a0", "loss": "bce", "gate": "baseline pareado"},
    "c1": {"architecture": "a1", "loss": "bce", "gate": "macro PR-AUC ≥ C0 (isola arquitetura)"},
    "c2": {
        "architecture": "a1",
        "loss": "focal",
        "gate": "macro PR-AUC ≥ C1 e ≈ 0,70 (sanidade A2-full)",
    },
    "c3": {"architecture": "a0", "loss": "focal", "gate": "leitura arch×loss vs C0/C2"},
}

SEED = 13
DEFAULT_EPOCHS = 30


def _run(cmd: list[str]) -> int:
    LOGGER.info("exec: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)  # nosec B603
    return proc.returncode


def run_cell(
    cell: str,
    manifest: Path,
    epochs: int,
    smoke: bool,
) -> int:
    if cell not in PILOT_CELLS:
        LOGGER.error("célula desconhecida '%s'; opções: %s", cell, sorted(PILOT_CELLS))
        return 2
    spec = PILOT_CELLS[cell]
    wrapper_cmd = [
        sys.executable,
        "scripts/pretrain_wrapper.py",
        "--architecture", spec["architecture"],
        "--loss", spec["loss"],
        "--seed", str(SEED),
        "--split-manifest", str(manifest),
        "--early-stopping-metric", "val_auc_pr",
    ]
    if smoke:
        wrapper_cmd.append("--smoke")
    else:
        wrapper_cmd += ["--epochs", str(epochs)]
    rc = _run(wrapper_cmd)
    if rc != 0:
        LOGGER.error("treino da célula %s falhou (rc=%d)", cell, rc)
        return 1

    run_dir = newest_run_dir()
    if run_dir is None:
        LOGGER.error("run dir não encontrado após treino")
        return 1
    LOGGER.info("run dir: %s", run_dir)

    # 2. predições das partições de avaliação
    for partition in EVAL_PARTITIONS:
        export_partition(run_dir, manifest, partition)

    # 3. avaliação canônica PROSPECTIVE no teste (fit na calibration)
    out_dir = run_dir / "evaluation_v2"
    eval_cmd = [
        sys.executable, "-m", "src.evaluation.canonical_evaluator",
        "--run-dir", str(run_dir),
        "--task-profile", "pretrain_scp_ecg_multilabel",
        "--split-name", SPLIT_ID,
        "--output-dir", str(out_dir),
        "--predictions", str(out_dir / "predictions" / "test.npz"),
        "--calibration-predictions", str(out_dir / "predictions" / "calibration.npz"),
        "--temperature-source", "fit",
        "--threshold-policy", "max_f1_per_class",
        "--n-bins", "15",
        "--seed", str(SEED),
    ]
    rc = _run(eval_cmd)
    if rc != 0:
        LOGGER.error("avaliação canônica falhou (rc=%d)", rc)
        return 1

    # 4. pilot_status.json (PILOT — nunca CANDIDATE sem governança)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    status = {
        "cell": cell,
        "spec": {k: v for k, v in spec.items() if k != "gate"},
        "status": "PILOT",
        "smoke": smoke,
        "split_id": SPLIT_ID,
        "seed": SEED,
        "early_stopping_metric": "val_auc_pr",
        "protocol_status": metrics["protocol_status"],
        "macro_pr_auc": metrics["metrics"]["macro_pr_auc"],
        "macro_auroc": metrics["metrics"]["macro_auroc"],
    }
    (run_dir / "pilot_status.json").write_text(
        json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # 5. resumo + gate
    m = metrics["metrics"]
    LOGGER.info("=" * 60)
    LOGGER.info(
        "célula %s (%s + %s) | macro_pr_auc=%.4f | macro_auroc=%.4f | "
        "ece_post=%.4f | ece_norm0=%s | protocol=%s",
        cell,
        spec["architecture"],
        spec["loss"],
        m["macro_pr_auc"] or float("nan"),
        m["macro_auroc"] or float("nan"),
        m["ece_post_calibration"] or float("nan"),
        f"{m['ece_post_calibration_norm0']:.4f}"
        if m.get("ece_post_calibration_norm0") is not None
        else "n/a",
        metrics["protocol_status"],
    )
    LOGGER.info("gate da célula %s: %s", cell, spec["gate"])
    LOGGER.info("status: PILOT (não promover; QG4-BCE permanece FAIL)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=sorted(PILOT_CELLS))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--smoke", action="store_true", help="1 época/50 passos (engenharia)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.manifest.exists():
        LOGGER.error("manifesto não encontrado: %s — rode `make pilot-split` antes", args.manifest)
        return 2
    return run_cell(args.cell, args.manifest, args.epochs, args.smoke)


if __name__ == "__main__":
    sys.exit(main())
