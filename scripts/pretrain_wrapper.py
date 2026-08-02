"""Safe wrapper for Chapman pre-training (FASE 3/4 — SDD exit policy).

Runs ``python -m src.models.pretrain_chapman`` as a subprocess, streams its
output, validates the produced artifacts (strict), and maps the exit code:

Default mode (execution semantics):
- 0  execution succeeded (training concluded + artifacts valid) — QG4 pass/fail
     is a *scientific result*, not a process failure;
- 1  real failure (crash, Traceback, missing artifacts);
- 2  configuration/usage error.

Gate enforcement (``--enforce-qg4``):
- 0  execution succeeded AND QG4 pass;
- 10 execution succeeded AND QG4 fail;
- 1/2 as above.

The known GeneratorDataset interpreter-teardown error never masks failures:
any ``Traceback`` in the log is treated as a real failure.

Usage:
    python scripts/pretrain_wrapper.py [--smoke] [--enforce-qg4] [--epochs N]
        [--steps-per-epoch N] [--validation-steps N] [--config PATH]
        [--architecture a0|a1|a2] [--loss bce|bce_weighted|focal] [--seed N]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess  # nosec B404
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_pretrain_artifacts import (  # noqa: E402
    newest_run_dir,
    validate_run_dir,
)

LOGGER = logging.getLogger("lewis.camada04.wrapper")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("config/pretrain_v1.0.yaml")

SUCCESS_MARKER = "Pré-treino concluído"
TEARDOWN_ERROR = "Python interpreter state is not initialized"
FATAL_MARKERS = ("Traceback (most recent call last)",)
QG4_FAIL_EXIT = 10
QG4_RE = re.compile(r"QG4 \|.*pass=(True|False)")
EXPERIMENT_RE = re.compile(r"experiment_dir=(\S+)")


def _parse_qg4(log_text: str) -> bool | None:
    """Return QG4 outcome from the log, or None if not evaluated."""
    match = QG4_RE.search(log_text)
    if not match:
        return None
    return match.group(1) == "True"


def _has_fatal(log_text: str) -> bool:
    return any(marker in log_text for marker in FATAL_MARKERS)


def decide_exit_code(
    *,
    returncode: int,
    log_text: str,
    artifacts_ok: bool,
    smoke: bool,
    enforce_qg4: bool = False,
) -> int:
    """Map the subprocess exit code per the SDD exit policy (section 11)."""
    concluded = SUCCESS_MARKER in log_text and artifacts_ok and not _has_fatal(log_text)
    if not concluded:
        return returncode if returncode != 0 else 1
    if smoke:
        return 0
    if enforce_qg4:
        qg4 = _parse_qg4(log_text)
        if qg4 is True:
            return 0
        LOGGER.warning("QG4 fail — resultado científico registrado; exit %d", QG4_FAIL_EXIT)
        return QG4_FAIL_EXIT
    if returncode != 0:
        LOGGER.warning(
            "exit code do subprocesso (%d) ignorado: execução concluída e artefatos "
            "válidos (teardown GeneratorDataset tolerado)",
            returncode,
        )
    return 0


def _find_experiment_dir(log_text: str) -> Path | None:
    match = EXPERIMENT_RE.search(log_text)
    if match:
        candidate = Path(match.group(1))
        if candidate.is_dir():
            return candidate
    return newest_run_dir()


def _load_training_cfg(config: Path | None) -> dict:
    """Read deterministic mode + seed from the training config (best effort)."""
    import yaml

    path = config or DEFAULT_CONFIG
    try:
        cfg = yaml.safe_load((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return cfg if isinstance(cfg, dict) else {}


def build_subprocess_env(cfg: dict, seed: int | None) -> dict:
    """Environment for the training subprocess (RF-DET-001/002).

    When ``deterministic.mode == strict``, oneDNN custom ops and nondeterministic
    TF ops are disabled **before** TensorFlow is imported in the subprocess;
    PYTHONHASHSEED is pinned to the resolved seed.
    """
    env = os.environ.copy()
    resolved_seed = seed if seed is not None else int(cfg.get("training", {}).get("seed", 42))
    env["PYTHONHASHSEED"] = str(resolved_seed)
    mode = str(cfg.get("deterministic", {}).get("mode", "fast"))
    if mode == "strict":
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["TF_DETERMINISTIC_OPS"] = "1"
        LOGGER.info("deterministic strict: oneDNN custom ops OFF, deterministic ops ON")
    return env


def run_training(
    *,
    epochs: int | None,
    steps_per_epoch: int | None,
    validation_steps: int | None,
    config: Path | None,
    architecture: str | None = None,
    loss: str | None = None,
    seed: int | None = None,
    split_manifest: Path | None = None,
    early_stopping_metric: str | None = None,
    env: dict | None = None,
) -> tuple[int, str]:
    """Run the pretrain module, streaming output. Returns (returncode, log)."""
    cmd = [sys.executable, "-m", "src.models.pretrain_chapman"]
    if config is not None:
        cmd += ["--config", str(config)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if steps_per_epoch is not None:
        cmd += ["--steps-per-epoch", str(steps_per_epoch)]
    if validation_steps is not None:
        cmd += ["--validation-steps", str(validation_steps)]
    if architecture is not None:
        cmd += ["--architecture", architecture]
    if loss is not None:
        cmd += ["--loss", loss]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if split_manifest is not None:
        cmd += ["--split-manifest", str(split_manifest)]
    if early_stopping_metric is not None:
        cmd += ["--early-stopping-metric", early_stopping_metric]
    LOGGER.info("wrapper: running %s", " ".join(cmd))
    proc = subprocess.Popen(  # nosec B603
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    return proc.wait(), "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="engineering check only")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--architecture", choices=["a0", "a1", "a2"], default=None)
    parser.add_argument("--loss", choices=["bce", "bce_weighted", "focal"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["val_loss", "val_auc_pr"],
        default=None,
        help="Métrica de EarlyStopping (protocolo v2: val_auc_pr)",
    )
    parser.add_argument(
        "--enforce-qg4",
        action="store_true",
        help="gate enforcement: exit 10 quando execução OK mas QG4 falha",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    epochs = args.epochs
    steps = args.steps_per_epoch
    val_steps = args.validation_steps
    if args.smoke:
        epochs = epochs or 1
        steps = steps or 50
        val_steps = val_steps or 50

    returncode, log_text = run_training(
        epochs=epochs,
        steps_per_epoch=steps,
        validation_steps=val_steps,
        config=args.config,
        architecture=args.architecture,
        loss=args.loss,
        seed=args.seed,
        split_manifest=args.split_manifest,
        early_stopping_metric=args.early_stopping_metric,
        env=build_subprocess_env(_load_training_cfg(args.config), args.seed),
    )

    run_dir = _find_experiment_dir(log_text)
    problems = validate_run_dir(run_dir, strict=True) if run_dir else ["run directory not found"]
    artifacts_ok = not problems
    for problem in problems:
        LOGGER.error("artifact check: %s", problem)

    decision = decide_exit_code(
        returncode=returncode,
        log_text=log_text,
        artifacts_ok=artifacts_ok,
        smoke=args.smoke,
        enforce_qg4=args.enforce_qg4,
    )
    LOGGER.info(
        "wrapper: subprocess rc=%d | artifacts_ok=%s | smoke=%s | enforce_qg4=%s | exit=%d",
        returncode,
        artifacts_ok,
        args.smoke,
        args.enforce_qg4,
        decision,
    )
    return decision


if __name__ == "__main__":
    sys.exit(main())
