"""Inspetor somente-leitura de artefatos de treino (T10.3 — auditoria forense).

Construído a partir dos schemas REAIS observados nos runs de ``experiments/``
(auditoria ``artifacts/artifacts_audit/artifacts_audit_before_changes.md``).
Nunca escreve em diretórios de run, nunca carrega modelos por padrão e abre
``.npz`` sempre com ``allow_pickle=False``.

Verificações (graves → exit 3):
  * JSON corrompido em artefatos centrais;
  * hash declarado × hash calculado (``--verify-hashes``);
  * ``test.npz`` presente sem ``model_freeze.json`` (RF-DATA-005);
  * QG4/run_status sem checkpoint ``backbone_pretrained.keras``;
  * predições com scores/labels desalinhados, não-finitos ou fora de [0, 1].

Avisos (não mudam o exit code):
  * ``deterministic_mode`` × ``onednn_enabled`` × ``runtime_profile`` divergentes;
  * época do QG4 ≠ época do checkpoint (monitor ES/ModelCheckpoint);
  * braço de perda do QG4 rotulado ``val_loss`` em run focal (valor = ``val_bce_monitor``);
  * ``.npz`` sem IDs (record/segment/patient) — overlap não verificável;
  * ``split_policy`` estático ou manifesto sem hash na proveniência;
  * artefatos de schema legado (``predictions.npz`` solto, ausência de qg4/run_status).

Uso:
    uv run python scripts/inspect_training_artifacts.py --run-dir experiments/<RUN>
    uv run python scripts/inspect_training_artifacts.py \
        --experiments-dir experiments --runs latest-c0 latest-c1 latest-c2 latest-c3 \
        --verify-hashes --inspect-predictions --compare-runs \
        --output artifacts_audit.json --format json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("lewis.audit.inspect")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GRAVE = 3

CORE_JSON = (
    "config.json",
    "provenance.json",
    "qg4_result.json",
    "run_status.json",
    "pilot_status.json",
    "history.json",
    "metrics_per_class.json",
)
CHECKPOINT = "backbone_pretrained.keras"
GATE_LOSS_FOR_FOCAL = "val_bce_monitor"
SEGMENTS_PER_RECORD = 10

GRAVE = "grave"
AVISO = "aviso"
INFO = "info"


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest em streaming (string vazia se ausente)."""
    path = Path(path)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    """Lê JSON; retorna (dados, erro). erro é 'ausente' ou 'corrompido: …'."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "ausente"
    except json.JSONDecodeError as exc:
        return None, f"corrompido: {exc}"
    except OSError as exc:
        return None, f"erro: {exc}"


def _argbest(values: list, mode: str) -> int | None:
    idx = [(i, float(v)) for i, v in enumerate(values or []) if v is not None]
    if not idx:
        return None
    return (max if mode == "max" else min)(idx, key=lambda t: t[1])[0]


def analyze_history(history: dict) -> dict:
    """Extrai épocas candidatas do history.json (schema real: chave 'history')."""
    hh = history.get("history", history) if isinstance(history, dict) else {}
    series = {k: v for k, v in hh.items() if isinstance(v, list)}
    out: dict = {"metrics": sorted(series), "epochs": len(series.get("loss", []))}
    for metric, mode in (
        ("val_auc_pr", "max"),
        ("val_auc_roc", "max"),
        ("val_loss", "min"),
        (GATE_LOSS_FOR_FOCAL, "min"),
    ):
        best = _argbest(series.get(metric) or [], mode)
        if best is not None:
            out[f"best_{metric}"] = {"epoch": best + 1, "value": float(series[metric][best])}
    lr = series.get("learning_rate") or []
    if lr:
        out["lr_first_last"] = [float(lr[0]), float(lr[-1])]
        out["lr_reductions"] = sum(1 for i in range(1, len(lr)) if lr[i] < lr[i - 1])
    return out


def inspect_keras_zip(path: Path) -> dict:
    """Inspeção não destrutiva do .keras (ZIP v3): entradas + metadados."""
    out: dict = {"path": str(path), "size": path.stat().st_size}
    try:
        with zipfile.ZipFile(path) as archive:
            out["entries"] = {i.filename: i.file_size for i in archive.infolist()}
            if "metadata.json" in out["entries"]:
                out["keras_metadata"] = json.loads(archive.read("metadata.json").decode("utf-8"))
            if "config.json" in out["entries"]:
                cfg = json.loads(archive.read("config.json").decode("utf-8"))
                out["keras_class"] = cfg.get("class_name")
                layers = cfg.get("config", {}).get("layers", [])
                out["n_layers"] = len(layers)
    except zipfile.BadZipFile:
        out["erro"] = "não é ZIP Keras v3 (ou corrompido)"
    return out


def load_model_info(path: Path) -> dict:
    """Carregamento EXPLÍCITO do modelo (safe_mode=True, compile=False).

    Importa TensorFlow lazily. Nunca regrava o modelo.
    """
    import tensorflow as tf  # lazy — custo alto, só com flag

    model = tf.keras.saving.load_model(path, compile=False, safe_mode=True)
    return {
        "name": model.name,
        "input_shape": [None if s is None else int(s) for s in model.input_shape],
        "output_shape": [None if s is None else int(s) for s in model.output_shape],
        "params": int(model.count_params()),
        "layers": [layer.__class__.__name__ for layer in model.layers],
        "dtypes": sorted({w.dtype for w in model.weights}),
    }


def inspect_npz(path: Path) -> dict:
    """Resumo de um .npz (allow_pickle=False): chaves, shapes, dtypes, sanidade."""
    out: dict = {"path": str(path), "size": path.stat().st_size, "arrays": {}}
    with np.load(path, allow_pickle=False) as data:
        out["keys"] = sorted(data.files)
        for key in out["keys"]:
            arr = data[key]
            info: dict = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            if arr.size and np.issubdtype(arr.dtype, np.number):
                finite = np.isfinite(arr)
                info["finite_frac"] = float(finite.mean())
                if finite.any():
                    info["min"] = float(arr[finite].min())
                    info["max"] = float(arr[finite].max())
            out["arrays"][key] = info
    return out


def check_predictions(run_dir: Path, issues: list) -> dict:
    """Valida os .npz de evaluation_v2/predictions (isolamento, IDs, cardinalidade)."""
    preds_dir = run_dir / "evaluation_v2" / "predictions"
    summary: dict = {}
    record_ids: dict[str, np.ndarray] = {}
    for part in ("validation", "calibration", "test"):
        path = preds_dir / f"{part}.npz"
        if not path.exists():
            summary[part] = "ausente"
            continue
        info = inspect_npz(path)
        summary[part] = info
        arrays = info["arrays"]
        if "y_score" in arrays and "y_true" in arrays:
            if arrays["y_score"]["shape"][0] != arrays["y_true"]["shape"][0]:
                issues.append((GRAVE, f"{part}.npz: scores × labels desalinhados"))
            for name in ("y_score", "y_true"):
                if arrays[name].get("finite_frac", 1.0) < 1.0:
                    issues.append((GRAVE, f"{part}.npz: {name} com NaN/Inf"))
            smin, smax = arrays["y_score"].get("min", 0.0), arrays["y_score"].get("max", 1.0)
            if smin < 0.0 or smax > 1.0:
                issues.append((GRAVE, f"{part}.npz: y_score fora de [0, 1]"))
        if "record_ids" not in arrays:
            issues.append(
                (
                    AVISO,
                    f"{part}.npz: sem IDs (record/segment/patient) — "
                    "overlap e segmentação não verificáveis",
                )
            )
        else:
            with np.load(path, allow_pickle=False) as data:
                rid = data["record_ids"]
            record_ids[part] = rid
            uniq, counts = np.unique(rid, return_counts=True)
            info["n_records"] = int(len(uniq))
            info["segments_per_record"] = sorted({int(c) for c in counts})
            if info["segments_per_record"] != [SEGMENTS_PER_RECORD]:
                issues.append(
                    (
                        AVISO,
                        f"{part}.npz: segmentos/registro "
                        f"{info['segments_per_record']} ≠ [{SEGMENTS_PER_RECORD}]",
                    )
                )
    parts = sorted(record_ids)
    for i, a in enumerate(parts):
        for b in parts[i + 1 :]:
            inter = np.intersect1d(record_ids[a], record_ids[b])
            summary[f"overlap_{a}_{b}"] = int(len(inter))
            if len(inter):
                issues.append((GRAVE, f"overlap {a} × {b}: {len(inter)} records"))
    return summary


def inspect_run(
    run_dir: Path,
    *,
    verify_hashes: bool = False,
    inspect_model: bool = False,
    load_model: bool = False,
    inspect_predictions: bool = False,
) -> dict:
    """Ficha completa de um run (somente leitura)."""
    run_dir = Path(run_dir)
    issues: list[tuple[str, str]] = []
    ficha: dict = {"run_dir": str(run_dir), "run_id": run_dir.name, "issues": issues}
    if not run_dir.is_dir():
        issues.append((GRAVE, "run dir inexistente"))
        return ficha

    docs: dict[str, dict | None] = {}
    for name in CORE_JSON:
        data, err = _read_json(run_dir / name)
        docs[name] = data
        if err == "ausente" and name in ("qg4_result.json", "run_status.json"):
            issues.append((INFO, f"{name} ausente (pré-contrato 10.4/10.6?)"))
        elif err and err != "ausente":
            issues.append((GRAVE, f"{name}: {err}"))
        elif err == "ausente" and name in ("config.json", "provenance.json"):
            issues.append((GRAVE, f"{name} ausente"))
    cfg, prov = docs.get("config.json") or {}, docs.get("provenance.json") or {}
    qg4, status = docs.get("qg4_result.json") or {}, docs.get("run_status.json") or {}
    pilot = docs.get("pilot_status.json") or {}
    hist_doc = docs.get("history.json")

    ficha["config"] = {k: cfg.get(k) for k in ("architecture", "loss", "seed", "epochs", "name")}
    ficha["pilot"] = (
        {
            k: pilot.get(k)
            for k in (
                "cell",
                "smoke",
                "runtime_profile",
                "split_id",
                "seed",
                "early_stopping_metric",
                "protocol_status",
                "status",
            )
        }
        if pilot
        else None
    )
    ficha["provenance"] = (
        {
            "git_commit": prov.get("git_commit"),
            "seed": prov.get("seed"),
            "deterministic_mode": prov.get("deterministic_mode"),
            "onednn_enabled": prov.get("onednn_enabled"),
            "runtime": prov.get("runtime"),
            "training": prov.get("training"),
            "metrics": prov.get("metrics"),
            "split_policy": (prov.get("dataset") or {}).get("split_policy"),
        }
        if prov
        else None
    )

    # --- cruzamentos config × provenance -----------------------------------
    training = prov.get("training") or {}
    if cfg and prov:
        if cfg.get("seed") != prov.get("seed"):
            issues.append(
                (GRAVE, f"seed config({cfg.get('seed')}) ≠ provenance({prov.get('seed')})")
            )
        if cfg.get("architecture") != training.get("architecture"):
            issues.append((GRAVE, "architecture config ≠ provenance.training"))
        if cfg.get("loss") != training.get("loss"):
            issues.append((GRAVE, "loss config ≠ provenance.training"))

    # --- runtime solicitado × efetivo (D2) ----------------------------------
    profile_req = pilot.get("runtime_profile")
    det_mode = prov.get("deterministic_mode")
    onednn = prov.get("onednn_enabled")
    runtime_block = prov.get("runtime") or {}
    profile_eff = runtime_block.get("profile")
    if profile_req and profile_eff and profile_req != profile_eff:
        issues.append((GRAVE, f"runtime solicitado ({profile_req}) ≠ efetivo ({profile_eff})"))
    if profile_req == "fast" and det_mode == "strict" and not runtime_block:
        issues.append(
            (
                AVISO,
                "runtime_profile=fast (pilot_status) mas provenance registra "
                "deterministic_mode=strict — perfil não propagado ao treino",
            )
        )
    if det_mode == "strict" and onednn is True and not runtime_block:
        issues.append(
            (
                AVISO,
                "deterministic_mode=strict com onednn_enabled=true (rótulo "
                "não reflete o ambiente efetivo)",
            )
        )

    # --- history: época do checkpoint × época do QG4 (D3) -------------------
    hist = analyze_history(hist_doc) if hist_doc else None
    ficha["history"] = hist
    es_metric = training.get("early_stopping_metric", "val_loss")
    ckpt_epoch = None
    if hist:
        if es_metric == "val_auc_pr" and hist.get("best_val_auc_pr"):
            ckpt_epoch = hist["best_val_auc_pr"]["epoch"]
        elif hist.get("best_val_loss"):
            ckpt_epoch = hist["best_val_loss"]["epoch"]
    ficha["checkpoint_epoch"] = ckpt_epoch
    ficha["checkpoint_monitor"] = es_metric
    qg4_epoch = (status.get("qg4") or {}).get("best_epoch")
    ficha["qg4_epoch"] = qg4_epoch
    if ckpt_epoch is not None and qg4_epoch is not None and ckpt_epoch != qg4_epoch:
        issues.append(
            (
                AVISO,
                f"época do QG4 ({qg4_epoch}) ≠ época do checkpoint "
                f"({ckpt_epoch}, monitor {es_metric})",
            )
        )

    # --- rótulo do braço de perda em runs focais (D4) -----------------------
    loss_name = cfg.get("loss") or training.get("loss")
    if loss_name and loss_name != "bce" and qg4:
        arms = qg4.get("arms") or {}
        if "val_loss" in arms and GATE_LOSS_FOR_FOCAL not in arms:
            issues.append(
                (
                    AVISO,
                    f"run {loss_name}: braço do QG4 rotulado 'val_loss' "
                    f"(valor efetivo: {GATE_LOSS_FOR_FOCAL})",
                )
            )

    # --- QG4 × run_status × provenance --------------------------------------
    if qg4 and status:
        if qg4.get("pass") != (status.get("qg4") or {}).get("pass"):
            issues.append((GRAVE, "qg4_result.pass ≠ run_status.qg4.pass"))
    if qg4 and prov and (prov.get("qg4") or {}).get("pass") is not None:
        if qg4.get("pass") != (prov.get("qg4") or {}).get("pass"):
            issues.append((GRAVE, "qg4_result.pass ≠ provenance.qg4.pass"))
    if (qg4 or status) and not (run_dir / CHECKPOINT).exists():
        issues.append((GRAVE, "QG4/run_status sem checkpoint backbone_pretrained.keras"))

    # --- split (D5) ----------------------------------------------------------
    hashes = prov.get("hashes") or {}
    if training.get("split_id") and "paired" in str(training.get("split_id")):
        if "split_manifest_sha256" not in hashes:
            issues.append(
                (
                    AVISO,
                    "split pareado sem split_manifest_sha256 na proveniência "
                    "(linhagem só por split_id)",
                )
            )
        if (prov.get("dataset") or {}).get(
            "split_policy"
        ) == "record_disjoint (val_ratio=0.1, seeded shuffle)":
            issues.append(
                (
                    AVISO,
                    "provenance.dataset.split_policy descreve split legado, "
                    "não o manifesto pareado",
                )
            )
    ficha["split_id"] = training.get("split_id")

    # --- isolamento do teste (D1) -------------------------------------------
    test_npz = run_dir / "evaluation_v2" / "predictions" / "test.npz"
    if test_npz.exists() and not (run_dir / "model_freeze.json").exists():
        issues.append((GRAVE, "test.npz presente sem model_freeze.json (RF-DATA-005)"))
    ficha["test_npz"] = test_npz.exists()

    # --- hashes declarados × calculados --------------------------------------
    ficha["hash_checks"] = {}
    if verify_hashes and prov:
        keras = run_dir / CHECKPOINT
        if keras.exists() and hashes.get("model_sha256"):
            ok = sha256_file(keras) == hashes["model_sha256"]
            ficha["hash_checks"]["model"] = ok
            if not ok:
                issues.append((GRAVE, "model_sha256 declarado ≠ calculado"))
        for name, key in (
            ("config.json", "config_sha256"),
            ("history.json", "history_sha256"),
            ("metrics_per_class.json", "metrics_per_class_sha256"),
        ):
            path = run_dir / name
            if path.exists() and hashes.get(key):
                ok = sha256_file(path) == hashes[key]
                ficha["hash_checks"][name] = ok
                if not ok:
                    issues.append((GRAVE, f"{key} declarado ≠ calculado"))
        vmeta, _ = _read_json(run_dir / "evaluation_v2" / "predictions" / "validation_meta.json")
        if vmeta and vmeta.get("sha256_model") and keras.exists():
            ok = sha256_file(keras) == vmeta["sha256_model"]
            ficha["hash_checks"]["validation_meta_model"] = ok
            if not ok:
                issues.append((GRAVE, "validation_meta.sha256_model ≠ checkpoint"))

    # --- modelo ---------------------------------------------------------------
    keras = run_dir / CHECKPOINT
    if inspect_model and keras.exists():
        ficha["keras_zip"] = inspect_keras_zip(keras)
        if "erro" in ficha["keras_zip"]:
            issues.append((GRAVE, f"{CHECKPOINT}: {ficha['keras_zip']['erro']}"))
        elif load_model:
            ficha["keras_model"] = load_model_info(keras)

    # --- predições -------------------------------------------------------------
    if inspect_predictions and (run_dir / "evaluation_v2" / "predictions").is_dir():
        ficha["predictions"] = check_predictions(run_dir, issues)
    if (run_dir / "evaluation_v2" / "predictions" / "predictions.npz").exists():
        issues.append((INFO, "predictions.npz de schema legado (sem partição no nome)"))

    # --- métricas de avaliação (para --compare-runs) ---------------------------
    ev2, _ = _read_json(run_dir / "evaluation_v2" / "metrics.json")
    if ev2:
        m = ev2.get("metrics", {})
        ficha["evaluation"] = {
            "protocol_status": ev2.get("protocol_status"),
            "split_id": ev2.get("split_id"),
            "macro_pr_auc": m.get("macro_pr_auc"),
            "macro_auroc": m.get("macro_auroc"),
            "ece_post_calibration": m.get("ece_post_calibration"),
            "temperature": m.get("temperature"),
        }
    return ficha


def select_runs(experiments_dir: Path, tokens: list[str]) -> list[Path]:
    """Resolve tokens 'latest-cX'/'latest-smoke'/'latest' ou nomes literais de run."""
    all_runs = sorted((p for p in experiments_dir.iterdir() if p.is_dir()), reverse=True)
    selected: list[Path] = []
    for token in tokens:
        if token == "latest":
            if all_runs:
                selected.append(all_runs[0])
            continue
        if token.startswith("latest-"):
            want = token[len("latest-") :]
            for run in all_runs:
                pilot, _ = _read_json(run / "pilot_status.json")
                if not pilot:
                    continue
                if want == "smoke" and pilot.get("smoke"):
                    selected.append(run)
                    break
                if pilot.get("cell") == want and not pilot.get("smoke"):
                    selected.append(run)
                    break
            else:
                LOGGER.warning("nenhum run encontrado para '%s'", token)
            continue
        candidate = experiments_dir / token
        if candidate.is_dir():
            selected.append(candidate)
        else:
            LOGGER.warning("run não encontrado: %s", token)
    return selected


def compare_table(fichas: list[dict]) -> list[dict]:
    """Matriz de comparação entre runs (matriz dos runs do relatório)."""
    rows = []
    for f in fichas:
        ev = f.get("evaluation") or {}
        pilot = f.get("pilot") or {}
        cfg = f.get("config") or {}
        prov = f.get("provenance") or {}
        rows.append(
            {
                "run": f["run_id"],
                "cell": pilot.get("cell"),
                "architecture": cfg.get("architecture"),
                "loss": cfg.get("loss"),
                "seed": cfg.get("seed"),
                "runtime_profile": pilot.get("runtime_profile"),
                "deterministic_mode": prov.get("deterministic_mode"),
                "onednn_enabled": prov.get("onednn_enabled"),
                "checkpoint_epoch": f.get("checkpoint_epoch"),
                "qg4_epoch": f.get("qg4_epoch"),
                "macro_pr_auc": ev.get("macro_pr_auc"),
                "macro_auroc": ev.get("macro_auroc"),
                "ece_post": ev.get("ece_post_calibration"),
                "temperature": ev.get("temperature"),
                "protocol": ev.get("protocol_status"),
                "test_npz": f.get("test_npz"),
                "n_grave": sum(1 for s, _ in f["issues"] if s == GRAVE),
                "n_aviso": sum(1 for s, _ in f["issues"] if s == AVISO),
            }
        )
    return rows


def render_markdown(fichas: list[dict], compare: bool) -> str:
    lines = ["# Inspeção de artefatos de treino", ""]
    if compare:
        rows = compare_table(fichas)
        cols = [
            "run",
            "cell",
            "architecture",
            "loss",
            "seed",
            "runtime_profile",
            "deterministic_mode",
            "checkpoint_epoch",
            "qg4_epoch",
            "macro_pr_auc",
            "macro_auroc",
            "ece_post",
            "temperature",
            "test_npz",
            "n_grave",
            "n_aviso",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c)) for c in cols) + " |")
        lines.append("")
    for f in fichas:
        lines.append(f"## {f['run_id']}")
        for sev, msg in f["issues"]:
            lines.append(f"- **{sev}**: {msg}")
        if not f["issues"]:
            lines.append("- sem inconsistências detectadas")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, action="append", default=[], help="Run específico (repetível)"
    )
    parser.add_argument("--experiments-dir", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[],
        help="Tokens: latest-c0..c3, latest-smoke, latest, ou nome do run",
    )
    parser.add_argument(
        "--cell",
        choices=["c0", "c1", "c2", "c3"],
        default=None,
        help="Todos os runs (não-smoke) de uma célula",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Run não-smoke mais recente de cada célula (equivale a latest-c*)",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--inspect-model", action="store_true", help="Inspeção ZIP do .keras")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Carrega o modelo (safe_mode=True, compile=False; importa TF)",
    )
    parser.add_argument("--inspect-predictions", action="store_true")
    parser.add_argument("--compare-runs", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dirs: list[Path] = list(args.run_dir)
    tokens = list(args.runs)
    if args.latest:
        tokens += ["latest-c0", "latest-c1", "latest-c2", "latest-c3"]
    if args.cell:
        tokens.append(f"cell-all-{args.cell}")
    expanded: list[str] = []
    for token in tokens:
        if token.startswith("cell-all-"):
            cell = token[len("cell-all-") :]
            for run in sorted(args.experiments_dir.iterdir(), reverse=True):
                if not run.is_dir():
                    continue
                pilot, _ = _read_json(run / "pilot_status.json")
                if pilot and pilot.get("cell") == cell and not pilot.get("smoke"):
                    expanded.append(run.name)
        else:
            expanded.append(token)
    run_dirs += select_runs(args.experiments_dir, expanded)
    if not run_dirs:
        LOGGER.error("nenhum run selecionado (--run-dir, --runs, --cell ou --latest)")
        return EXIT_USAGE

    fichas = [
        inspect_run(
            d,
            verify_hashes=args.verify_hashes,
            inspect_model=args.inspect_model,
            load_model=args.load_model,
            inspect_predictions=args.inspect_predictions,
        )
        for d in run_dirs
    ]
    result = {"runs": fichas}
    if args.compare_runs:
        result["comparison"] = compare_table(fichas)

    if args.format == "json":
        rendered = json.dumps(result, ensure_ascii=False, indent=1, default=str)
    else:
        rendered = render_markdown(fichas, args.compare_runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        LOGGER.info("relatório escrito em %s", args.output)
    else:
        print(rendered)

    graves = [msg for f in fichas for sev, msg in f["issues"] if sev == GRAVE]
    for msg in graves:
        LOGGER.error("grave: %s", msg)
    return EXIT_GRAVE if graves else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
