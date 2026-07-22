"""Prepara datasets para o pipeline de duas etapas (Project-Lewis v3.0.0).

A partir do dataset de fine-tuning original (5 classes AAMI: N, S, V, F, Q),
gera:

- ``stage1_binary``: N vs Anormal(S/V/FUSION). **v3 (decisão D4): a classe Q
  (Q_OR_UNKNOWN, classe de rejeição) é excluída dos alvos por padrão** — ela
  permanece no family dataset para a cabeça de abstenção, mas não é mais
  rotulada como "Anormal" (encerra DQ-05).
- ``stage2_multiclass``: apenas os batimentos S, V, F, remapeados para
  0 = S, 1 = V, 2 = F.

Adicionalmente (v3):
- deduplica janelas byte-idênticas com rótulos conflitantes (DQ-04), removendo
  todas as instâncias do grupo e registrando em ``v3_exclusions.jsonl``;
- anexa flags de qualidade por janela (``qf_flatline``, ``qf_clip``,
  ``qf_off_center``) — flags informativas, não exclusão silenciosa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training_integrity.contracts import (  # noqa: E402
    DatasetRole,
    PatientIdentityManifest,
    PatientSplitManifest,
    SplitBundlePublication,
)
from src.training_integrity.integrity import (  # noqa: E402
    beat_sample_id,
    exclusive_publication,
    hash_canonical,
    publish_staged_file_exclusive,
    resolve_project_path,
    temporary_staging_path,
    verify_detached_sha256,
    waveform_row_sha256,
)
from src.training_integrity.splits import validate_split_identity_consistency  # noqa: E402

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("lewis.camada04.prepare_two_stage")

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

SOURCE_NPZ = FEATURE_DIR / "finetuning_mitbih_family.npz"
SOURCE_PARQUET = FEATURE_DIR / "finetuning_mitbih_family.parquet"

STAGE1_NPZ = FEATURE_DIR / "stage1_binary.npz"
STAGE1_PARQUET = FEATURE_DIR / "stage1_binary.parquet"
STAGE2_NPZ = FEATURE_DIR / "stage2_multiclass.npz"
STAGE2_PARQUET = FEATURE_DIR / "stage2_multiclass.parquet"
EXCLUSIONS_JSONL = FEATURE_DIR / "v3_exclusions.jsonl"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "groupkfold_5_stratified" / "v3.1.0"

# Mapeamento canônico AAMI: int -> string
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# Limites das flags de qualidade (v3 — ver docs/rebuild_spec/01 §2)
FLATLINE_STD = 0.01
CLIP_ABS = 9.99
OFF_CENTER_LO, OFF_CENTER_HI = 225, 275


def _safe_int(value: Any, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid integer for {context}: {value!r}") from error


def _aami_int_to_str(y: np.ndarray) -> np.ndarray:
    labels: list[str] = []
    for value in y:
        index = _safe_int(value, context="AAMI class")
        if index < 0 or index >= len(AAMI_CLASSES):
            raise ValueError(f"AAMI class index out of range: {index}")
        labels.append(AAMI_CLASSES[index])
    return np.array(labels, dtype=object)


def _compute_quality_flags(X: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Anexa flags de qualidade por janela (v3 — informativas, não exclusivas)."""
    std = X.std(axis=(1, 2))
    clip = (np.abs(X) >= CLIP_ABS).any(axis=(1, 2))
    r_in_seg = df["r_peak_in_segment"].to_numpy(dtype=float)
    off_center = (r_in_seg < OFF_CENTER_LO) | (r_in_seg > OFF_CENTER_HI)
    df = df.copy()
    df["qf_flatline"] = std < FLATLINE_STD
    df["qf_clip"] = clip
    df["qf_off_center"] = off_center
    return df


def _drop_conflicting_duplicates(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    *,
    exclusions_path: Path = EXCLUSIONS_JSONL,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Remove grupos de janelas byte-idênticas com rótulos conflitantes (DQ-04).

    Todas as instâncias do grupo são removidas e registradas em
    ``v3_exclusions.jsonl`` (id, motivo, regra, etapa, classe, paciente).
    """
    hashes = np.empty(len(X), dtype=object)
    for i in range(len(X)):
        hashes[i] = hashlib.sha256(np.ascontiguousarray(X[i]).tobytes()).hexdigest()[:20]

    df = df.reset_index(drop=True).copy()
    df["_xh"] = hashes
    conflict_mask = np.zeros(len(X), dtype=bool)
    n_groups = 0
    exclusions: list[dict] = []
    grouped = df.groupby("_xh")
    for xh, grp in grouped:
        if len(grp) > 1 and grp["label_aami"].nunique() > 1:
            n_groups += 1
            idx = grp.index.to_numpy()
            conflict_mask[idx] = True
            for _, row in grp.iterrows():
                exclusions.append(
                    {
                        "record_id": row["record_id"],
                        "beat_idx": _safe_int(row["beat_idx"], context="beat_idx"),
                        "dataset": row["dataset"],
                        "label_aami": row["label_aami"],
                        "reason": "duplicate_window_conflicting_label",
                        "rule": "DQ-04: remover todas as instâncias do grupo (v3)",
                        "stage": "prepare_two_stage_datasets",
                        "patient_id": None,
                        "patient_identity_status": "NOT_AVAILABLE_AT_DATASET_PREPARATION",
                    }
                )
    if exclusions:
        exclusions_path.parent.mkdir(parents=True, exist_ok=True)
        with exclusions_path.open("a", encoding="utf-8") as fh:
            for rec in exclusions:
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **rec,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        LOGGER.warning(
            "DQ-04: %d grupos de duplicatas conflitantes removidos (%d janelas) — ver %s",
            n_groups,
            _safe_int(conflict_mask.sum(), context="conflicting duplicate count"),
            exclusions_path,
        )
    keep = ~conflict_mask
    return X[keep], y[keep], df.loc[keep].drop(columns=["_xh"]), n_groups


def _prepare_stage1(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    npz_path: Path,
    parquet_path: Path,
    *,
    identity: PatientIdentityManifest,
    split: PatientSplitManifest,
    exclude_q: bool = False,
    feature_columns: tuple[str, ...] = ("rr_prev", "qrs_width_ms"),
) -> None:
    """Cria dataset binário N vs Anormal.

    Parameters
    ----------
    exclude_q : bool
        Se True, remove amostras da classe Q do treino do Estágio 1.
        A classe Anormal passa a ser formada apenas por S, V e F.
    feature_columns : tuple[str, ...]
        Colunas do DataFrame fonte a serem incluídas como features
        morfológicas/contextuais no Estágio 1.
    """
    validate_split_identity_consistency(identity, split)
    if exclude_q:
        keep_mask = y != 4
        X = X[keep_mask]
        y = y[keep_mask]
        df = df.iloc[np.nonzero(keep_mask)[0]].copy()
        # v3: exclusão de Q_OR_UNKNOWN é o padrão canônico (D4); o arquivo é
        # stage1_binary e a coluna stage reflete o artefato, não a política.
        stage_name = "stage1_binary"
    else:
        df = df.copy()
        stage_name = "stage1_binary"

    y_bin = np.where(y == 0, 0, 1).astype(np.int64)
    identity_columns = {"dataset", "record_id", "beat_idx", "r_peak_sample"}
    missing_identity = identity_columns - set(df.columns)
    if missing_identity:
        raise ValueError(
            f"Sample identity columns missing in source DataFrame: {sorted(missing_identity)}"
        )
    sample_ids = np.asarray(
        [
            beat_sample_id(
                str(row.dataset),
                str(row.record_id),
                _safe_int(row.beat_idx, context="beat_idx"),
                _safe_int(row.r_peak_sample, context="r_peak_sample"),
            )
            for row in df.loc[:, ["dataset", "record_id", "beat_idx", "r_peak_sample"]].itertuples(
                index=False
            )
        ],
        dtype=str,
    )
    waveform_hashes = np.asarray([waveform_row_sha256(row) for row in X], dtype=str)

    # Features auxiliares já computadas na Camada 3
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns missing in source DataFrame: {missing}")
    features = df[list(feature_columns)].to_numpy(dtype=np.float32)

    df_out = df.copy()
    df_out["sample_id"] = sample_ids
    df_out["segment_id"] = sample_ids
    df_out["waveform_sha256"] = waveform_hashes
    df_out["dataset_id"] = df_out["dataset"].astype(str)
    df_out["beat_index"] = df_out["beat_idx"]
    df_out["annotation_index_target"] = df_out["r_peak_sample"]
    identity_by_key = {record.record_key: record for record in identity.records}
    patient_fold = {
        patient_id: fold.fold for fold in split.folds for patient_id in fold.outer_test_patient_ids
    }
    record_fold = {
        record_key: fold.fold for fold in split.folds for record_key in fold.outer_test_record_keys
    }
    patient_ids: list[str] = []
    split_names: list[str] = []
    fold_ids: list[int] = []
    for dataset_id, record_id in df_out.loc[:, ["dataset_id", "record_id"]].itertuples(
        index=False, name=None
    ):
        record_key = f"{dataset_id}/{record_id}"
        record = identity_by_key.get(record_key)
        if record is None:
            raise ValueError(f"missing identity for Stage 1 record: {record_key}")
        if record.role is DatasetRole.CONFIRMATORY_CORE:
            if record.patient_id is None:
                raise ValueError(f"confirmatory patient identity is absent: {record_key}")
            expected_patient_fold = patient_fold.get(record.patient_id)
            expected_record_fold = record_fold.get(record_key)
            if expected_patient_fold is None or expected_patient_fold != expected_record_fold:
                raise ValueError(f"split assignment mismatch for Stage 1 record: {record_key}")
            patient_ids.append(record.patient_id)
            split_names.append("outer_test")
            fold_ids.append(expected_patient_fold)
        elif record.role is DatasetRole.DOMAIN_SENSITIVITY:
            patient_ids.append("UNKNOWN_OR_UNVERIFIED")
            split_names.append("domain_sensitivity")
            fold_ids.append(-1)
        else:
            raise ValueError(f"rhythm-only record entered Stage 1: {record_key}")
    df_out["patient_id"] = patient_ids
    df_out["split"] = split_names
    df_out["fold"] = fold_ids
    df_out["class_canonical"] = (
        df_out["label_aami"].astype(str).replace({"F": "FUSION", "Q": "Q_OR_UNKNOWN"})
    )
    quality_flag_count = (
        df_out[["qf_flatline", "qf_clip", "qf_off_center"]].astype(bool).sum(axis=1)
    )
    df_out["quality_label"] = np.select(
        [
            quality_flag_count > 1,
            df_out["qf_flatline"].astype(bool),
            df_out["qf_clip"].astype(bool),
            df_out["qf_off_center"].astype(bool),
        ],
        ["MULTIPLE_FLAGS", "FLATLINE", "CLIP", "OFF_CENTER"],
        default="VALID",
    )
    df_out["y"] = y_bin
    df_out["stage"] = stage_name
    required_source_lineage = {
        "source_sampling_rate",
        "target_sampling_rate",
        "annotation_index_native",
        "annotation_time_seconds",
        "annotation_index_target",
        "class_original",
        "class_canonical",
    }
    missing_source_lineage = required_source_lineage - set(df_out.columns)
    if missing_source_lineage:
        raise ValueError(
            "Stage 1 source lacks canonical annotation custody: "
            f"{sorted(missing_source_lineage)}"
        )
    if df_out[list(required_source_lineage)].isna().any(axis=None):
        raise ValueError("Stage 1 source contains null annotation custody")

    lock_path = npz_path.parent / f".{npz_path.stem}.publish.lock"
    with exclusive_publication(lock_path, (npz_path, parquet_path)):
        with (
            temporary_staging_path(npz_path) as staged_npz,
            temporary_staging_path(parquet_path) as staged_parquet,
        ):
            np.savez(
                staged_npz,
                X=X.astype(np.float32),
                y=y_bin,
                sample_id=sample_ids,
                waveform_sha256=waveform_hashes,
                features=features,
                feature_columns=np.array(feature_columns),
            )
            df_out.to_parquet(staged_parquet, index=False)
            publish_staged_file_exclusive(staged_npz, npz_path)
            publish_staged_file_exclusive(staged_parquet, parquet_path)

    n_normal = _safe_int((y_bin == 0).sum(), context="normal count")
    n_abnormal = _safe_int((y_bin == 1).sum(), context="abnormal count")
    LOGGER.info(
        "Stage1 binary saved: n=%d | Normal=%d | Anormal=%d | "
        "features=%s | exclude_q=%s | path=%s",
        len(y_bin),
        n_normal,
        n_abnormal,
        list(feature_columns),
        exclude_q,
        npz_path,
    )


def _prepare_stage2(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    npz_path: Path,
    parquet_path: Path,
) -> None:
    """Cria dataset S/V/F (exclui N e Q)."""
    mask = np.isin(y, [1, 2, 3])
    X_sub = X[mask]
    y_sub = y[mask]
    df_sub = df.iloc[np.nonzero(mask)[0]].copy()

    # Remapear: S=1->0, V=2->1, F=3->2
    remap = {1: 0, 2: 1, 3: 2}
    y_remapped = np.vectorize(remap.get)(y_sub).astype(np.int64)

    df_sub["y"] = y_remapped
    df_sub["stage"] = "stage2_multiclass"
    lock_path = npz_path.parent / f".{npz_path.stem}.publish.lock"
    with exclusive_publication(lock_path, (npz_path, parquet_path)):
        with (
            temporary_staging_path(npz_path) as staged_npz,
            temporary_staging_path(parquet_path) as staged_parquet,
        ):
            np.savez(
                staged_npz,
                X=X_sub.astype(np.float32),
                y=y_remapped,
            )
            df_sub.to_parquet(staged_parquet, index=False)
            publish_staged_file_exclusive(staged_npz, npz_path)
            publish_staged_file_exclusive(staged_parquet, parquet_path)

    counts = {
        cls: _safe_int((y_remapped == index).sum(), context=f"{cls} count")
        for index, cls in enumerate(["S", "V", "F"])
    }
    LOGGER.info(
        "Stage2 multiclass saved: n=%d | S=%d | V=%d | F=%d | path=%s",
        len(y_remapped),
        counts["S"],
        counts["V"],
        counts["F"],
        npz_path,
    )


def _load_split_contracts(
    split_dir: Path,
) -> tuple[PatientIdentityManifest, PatientSplitManifest]:
    identity_path = split_dir / "patient_identity_manifest.json"
    split_path = split_dir / "index.json"
    completion_path = split_dir / "SPLIT_BUNDLE_COMPLETE.json"
    verify_detached_sha256(completion_path)
    verify_detached_sha256(identity_path)
    verify_detached_sha256(split_path)
    identity = PatientIdentityManifest.model_validate_json(
        identity_path.read_text(encoding="utf-8")
    )
    split = PatientSplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
    completion = SplitBundlePublication.model_validate_json(
        completion_path.read_text(encoding="utf-8")
    )
    expected_completion = SplitBundlePublication(
        schema_version="patient-split-publication-v3.1.0",
        patient_identity_hash=hash_canonical("patient-identity", identity),
        patient_split_hash=hash_canonical("patient-split", split),
        status="SPLIT_BUNDLE_COMPLETE",
    )
    if completion != expected_completion:
        raise ValueError("split completion marker does not bind the published bundle")
    if split.patient_identity_hash != hash_canonical("patient-identity", identity):
        raise ValueError("published split does not bind its patient identity manifest")
    return identity, split


def _resolve_cli_path(path: Path) -> Path:
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"path escapes project root: {path}")
        return resolved
    return resolve_project_path(PROJECT_ROOT, path.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare two-stage datasets (v3).")
    parser.add_argument(
        "--exclude-q-from-stage1",
        dest="exclude_q_from_stage1",
        action="store_true",
        default=True,
        help="[PADRÃO v3] Remove Q_OR_UNKNOWN dos alvos do Estágio 1 (decisão D4).",
    )
    parser.add_argument(
        "--include-q-from-stage1",
        dest="exclude_q_from_stage1",
        action="store_false",
        help="Restaura comportamento v2.x (Q como 'Anormal' no Estágio 1) — legado.",
    )
    parser.add_argument("--source-npz", type=Path, default=SOURCE_NPZ)
    parser.add_argument("--source-parquet", type=Path, default=SOURCE_PARQUET)
    parser.add_argument("--stage1-npz", type=Path, default=STAGE1_NPZ)
    parser.add_argument("--stage1-parquet", type=Path, default=STAGE1_PARQUET)
    parser.add_argument("--stage2-npz", type=Path, default=STAGE2_NPZ)
    parser.add_argument("--stage2-parquet", type=Path, default=STAGE2_PARQUET)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--exclusions-jsonl", type=Path, default=EXCLUSIONS_JSONL)
    args = parser.parse_args()

    source_npz = _resolve_cli_path(args.source_npz)
    source_parquet = _resolve_cli_path(args.source_parquet)
    stage1_npz = _resolve_cli_path(args.stage1_npz)
    stage1_parquet = _resolve_cli_path(args.stage1_parquet)
    stage2_npz = _resolve_cli_path(args.stage2_npz)
    stage2_parquet = _resolve_cli_path(args.stage2_parquet)
    split_dir = _resolve_cli_path(args.split_dir)
    exclusions_path = _resolve_cli_path(args.exclusions_jsonl)
    outputs = (stage1_npz, stage1_parquet, stage2_npz, stage2_parquet)
    existing_outputs = [str(path) for path in outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            f"two-stage outputs are write-once; existing paths: {existing_outputs}"
        )
    identity, split = _load_split_contracts(split_dir)

    LOGGER.info("Loading source dataset from %s", source_npz)
    data = np.load(source_npz, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    LOGGER.info("Loading metadata from %s", source_parquet)
    df = pd.read_parquet(source_parquet)

    if len(X) != len(df) or len(y) != len(df):
        raise ValueError(f"Mismatch: X={len(X)}, y={len(y)}, df={len(df)}")
    source_custody = {
        "source_sampling_rate",
        "target_sampling_rate",
        "annotation_index_native",
        "annotation_time_seconds",
        "annotation_index_target",
        "class_original",
        "class_canonical",
    }
    missing_custody = source_custody - set(df.columns)
    if missing_custody or df[list(source_custody & set(df.columns))].isna().any(axis=None):
        raise ValueError(
            "source family dataset lacks complete canonical annotation custody: "
            f"{sorted(missing_custody)}"
        )

    # v3: deduplicação de janelas idênticas com rótulos conflitantes (DQ-04)
    X, y, df, n_conflicts = _drop_conflicting_duplicates(X, y, df, exclusions_path=exclusions_path)
    LOGGER.info("Após dedup DQ-04: n=%d (grupos removidos=%d)", len(y), n_conflicts)

    # v3: flags de qualidade por janela (informativas)
    df = _compute_quality_flags(X, df)

    _prepare_stage1(
        X,
        y,
        df,
        npz_path=stage1_npz,
        parquet_path=stage1_parquet,
        identity=identity,
        split=split,
        exclude_q=args.exclude_q_from_stage1,
    )
    _prepare_stage2(
        X,
        y,
        df,
        npz_path=stage2_npz,
        parquet_path=stage2_parquet,
    )

    LOGGER.info("Two-stage datasets prepared successfully (v3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
