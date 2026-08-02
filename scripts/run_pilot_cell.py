"""Orquestrador de células piloto T10.3 (controle H7 — matriz v2 §2).

Executa UMA célula do plano de pilotos end-to-end (PRD CPU-first, pós-adequação):

1. treino via ``scripts/pretrain_wrapper.py`` (arch/loss/seed 13, split pareado,
   early stopping por ``val_auc_pr``, perfil runtime uniforme);
2. exporta predições de validation + calibration (teste BLOQUEADO até freeze —
   RF-DATA-005);
3. avaliação canônica PROSPECTIVE na **validation** (T/thresholds fit na
   calibration) — a comparação entre células acontece em desenvolvimento, nunca
   no teste (RF-SEL-001);
4. gate da célula com códigos de saída (RF-QG-003): exit 3 se o gate reprovar;
5. grava ``pilot_status.json`` (status PILOT — nunca CANDIDATE).

Falhas de esteira que abortam: "ran out of data" no treino (RF-TFDATA-002),
desalinhamento IDs×predições, hash mismatch de checkpoint.

Não promove modelos, não toca ``models/``, não altera splits. Requer o manifesto
do split pareado já gerado (``make pilot-split``).

Uso:
    uv run python scripts/run_pilot_cell.py --cell c1
    uv run python scripts/run_pilot_cell.py --cell c1 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_pilot_predictions import export_partition  # noqa: E402
from scripts.validate_pretrain_artifacts import newest_run_dir  # noqa: E402

LOGGER = logging.getLogger("lewis.camada04.pilot_cell")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "chapman_paired_v2" / "manifest.json"
SPLIT_ID = "chapman-record-disjoint-paired-v2"
RUNTIME_PROFILE = "fast"  # pilotos = exploração (PRD RF-CPU-003); qualificação = strict

#: Códigos de saída (PRD RF-QG-003).
EXIT_OK = 0
EXIT_EXECUTION_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_GATE_FAILED = 3
EXIT_DATA_LEAKAGE = 4

#: Marker de cardinalidade — RF-TFDATA-002: esgotamento é falha, não aviso.
CARDINALITY_FATAL = "ran out of data"

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

#: Predecessora comparativa de cada célula (para o gate).
PREDECESSOR: dict[str, Optional[str]] = {"c0": None, "c1": "c0", "c2": "c1", "c3": None}

SEED = 13
DEFAULT_EPOCHS = 30
GATE_NOISE_BAND = 0.005  # 0,5 p.p. — banda de ruído para gates comparativos
C2_SANITY_FLOOR = 0.60  # sanidade mínima de macro PR-AUC para C2 (A2-full ≈ 0,70)


def evaluate_cell_gate(
    cell: str,
    metrics: dict,
    baseline_metrics: Optional[dict],
) -> tuple[bool, str]:
    """Gate comparativo da célula (T10.3-P §2). Retorna (passed, reason).

    C0/C3 não têm gate comparativo (sempre passam — são leituras).
    C1 deve ser ≥ C0 (banda de ruído 0,5 p.p.); C2 deve ser ≥ C1 e ≥ 0,60.
    """
    pr_auc = metrics["metrics"]["macro_pr_auc"]
    if PREDECESSOR.get(cell) is None:
        return True, "sem gate comparativo (baseline/leitura)"
    if baseline_metrics is None:
        return True, f"predecessora {PREDECESSOR[cell]} não encontrada — gate não aplicado (aviso)"
    base_pr_auc = baseline_metrics["metrics"]["macro_pr_auc"]
    if cell == "c2" and pr_auc < C2_SANITY_FLOOR:
        return False, f"C2 macro PR-AUC {pr_auc:.4f} < piso de sanidade {C2_SANITY_FLOOR}"
    if pr_auc < base_pr_auc - GATE_NOISE_BAND:
        return False, (
            f"{cell.upper()} macro PR-AUC {pr_auc:.4f} < {PREDECESSOR[cell].upper()} "
            f"{base_pr_auc:.4f} − {GATE_NOISE_BAND} (banda de ruído)"
        )
    return True, f"{cell.upper()} ≥ {PREDECESSOR[cell].upper()} dentro da banda de ruído"


def gate_for_cell(
    cell: str,
    metrics: dict,
    baseline_metrics: Optional[dict],
    smoke: bool,
) -> tuple[bool, str]:
    """Gate da célula; em smoke (validação de engenharia) o gate é desativado."""
    if smoke:
        return True, "smoke — validação de engenharia; gate desativado"
    return evaluate_cell_gate(cell, metrics, baseline_metrics)


def find_predecessor_metrics(cell: str) -> Optional[dict]:
    """Métricas da run mais recente da célula predecessora (pilot_status.json).

    Runs de smoke (``pilot_status.smoke == true``) são ignoradas: não são
    evidência científica e não podem servir de baseline de gate.
    """
    predecessor = PREDECESSOR.get(cell)
    if predecessor is None:
        return None
    candidates = sorted(PROJECT_ROOT.glob("experiments/*_pretrain_chapman"), reverse=True)
    for run_dir in candidates:
        status_path = run_dir / "pilot_status.json"
        metrics_path = run_dir / "evaluation_v2" / "metrics.json"
        if not status_path.exists() or not metrics_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if status.get("smoke"):
            continue
        if status.get("cell") == predecessor and status.get("status") == "PILOT":
            LOGGER.info("predecessora de %s: %s (%s)", cell, predecessor, run_dir.name)
            return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


def _run_stream(cmd: list[str]) -> tuple[int, str]:
    """Executa comando streamando saída; retorna (rc, log_text)."""
    LOGGER.info("exec: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(  # nosec B603
        cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    return proc.wait(), "".join(lines)


def run_cell(cell: str, manifest: Path, epochs: int, smoke: bool) -> int:
    if cell not in PILOT_CELLS:
        LOGGER.error("célula desconhecida '%s'; opções: %s", cell, sorted(PILOT_CELLS))
        return EXIT_CONFIG_ERROR
    spec = PILOT_CELLS[cell]
    wrapper_cmd = [
        sys.executable,
        "scripts/pretrain_wrapper.py",
        "--architecture", spec["architecture"],
        "--loss", spec["loss"],
        "--seed", str(SEED),
        "--split-manifest", str(manifest),
        "--early-stopping-metric", "val_auc_pr",
        "--runtime-profile", RUNTIME_PROFILE,
    ]
    if smoke:
        wrapper_cmd.append("--smoke")
    else:
        wrapper_cmd += ["--epochs", str(epochs)]
    rc, train_log = _run_stream(wrapper_cmd)
    if CARDINALITY_FATAL in train_log:
        LOGGER.error("RF-TFDATA-002: esgotamento detectado no treino da célula %s", cell)
        return EXIT_EXECUTION_ERROR
    if rc != 0:
        LOGGER.error("treino da célula %s falhou (rc=%d)", cell, rc)
        return EXIT_EXECUTION_ERROR

    run_dir = newest_run_dir()
    if run_dir is None:
        LOGGER.error("run dir não encontrado após treino")
        return EXIT_EXECUTION_ERROR
    LOGGER.info("run dir: %s", run_dir)

    # 2. predições de desenvolvimento (teste BLOQUEADO — RF-DATA-005)
    for partition in ("validation", "calibration"):
        try:
            export_partition(run_dir, manifest, partition)
        except RuntimeError as exc:
            LOGGER.error("export %s falhou: %s", partition, exc)
            return EXIT_EXECUTION_ERROR

    # 3. avaliação canônica PROSPECTIVE na validation (fit na calibration)
    out_dir = run_dir / "evaluation_v2"
    eval_cmd = [
        sys.executable, "-m", "src.evaluation.canonical_evaluator",
        "--run-dir", str(run_dir),
        "--task-profile", "pretrain_scp_ecg_multilabel",
        "--split-name", SPLIT_ID,
        "--output-dir", str(out_dir),
        "--predictions", str(out_dir / "predictions" / "validation.npz"),
        "--calibration-predictions", str(out_dir / "predictions" / "calibration.npz"),
        "--temperature-source", "fit",
        "--threshold-policy", "max_f1_per_class",
        "--n-bins", "15",
        "--seed", str(SEED),
        "--runtime-profile", RUNTIME_PROFILE,
    ]
    rc, _ = _run_stream(eval_cmd)
    if rc != 0:
        LOGGER.error("avaliação canônica falhou (rc=%d)", rc)
        return EXIT_EXECUTION_ERROR

    # 4. gate da célula (RF-QG-003) — lê o evaluation_v2 do checkpoint implantável
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline_metrics = None if smoke else find_predecessor_metrics(cell)
    gate_pass, gate_reason = gate_for_cell(cell, metrics, baseline_metrics, smoke)
    LOGGER.info("gate %s: %s", cell, gate_reason)

    # 5. pilot_status.json (PILOT — nunca CANDIDATE sem governança)
    status = {
        "cell": cell,
        "spec": {k: v for k, v in spec.items() if k != "gate"},
        "status": "PILOT",
        "smoke": smoke,
        "split_id": SPLIT_ID,
        "seed": SEED,
        "early_stopping_metric": "val_auc_pr",
        "runtime_profile": RUNTIME_PROFILE,
        "evaluation_split": "validation",
        "test_status": "locked_until_model_freeze",
        "protocol_status": metrics["protocol_status"],
        "macro_pr_auc": metrics["metrics"]["macro_pr_auc"],
        "macro_auroc": metrics["metrics"]["macro_auroc"],
        "gate": {"pass": gate_pass, "reason": gate_reason},
    }
    (run_dir / "pilot_status.json").write_text(
        json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    m = metrics["metrics"]
    LOGGER.info("=" * 60)
    LOGGER.info(
        "célula %s (%s + %s) | macro_pr_auc=%.4f | macro_auroc=%.4f | "
        "ece_post=%.4f | ece_norm0=%s | protocol=%s | gate=%s",
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
        gate_pass,
    )
    LOGGER.info("status: PILOT (não promover; QG4-BCE permanece FAIL)")
    return EXIT_OK if gate_pass else EXIT_GATE_FAILED


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
        return EXIT_CONFIG_ERROR
    return run_cell(args.cell, args.manifest, args.epochs, args.smoke)


if __name__ == "__main__":
    sys.exit(main())
