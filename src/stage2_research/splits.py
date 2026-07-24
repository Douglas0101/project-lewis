"""One-time frozen nested patient/group split manifests."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np

from src.models.e06_protocol import build_outer_splits, select_inner_split
from src.stage2_research.contracts import (
    ExitCode,
    InnerFoldManifest,
    InnerSplitManifest,
    OuterFoldManifest,
    ResearchConfig,
    ResearchError,
    SplitManifest,
    SplitPartition,
)
from src.stage2_research.data import INDEX_TO_LABEL, Stage2Dataset
from src.stage2_research.integrity import (
    atomic_write_json,
    atomic_write_text,
    hash_canonical,
    load_json,
)


def _safe_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResearchError(
            f"{name} is not an integer",
            ExitCode.DATA_INTEGRITY,
        ) from error


def _partition(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> SplitPartition:
    index_values = np.asarray(indices, dtype=np.int64)
    group_values = tuple(sorted(set(groups[index_values].astype(str).tolist())))
    unique_labels, counts = np.unique(labels[index_values], return_counts=True)
    class_counts = {"S": 0, "V": 0, "F": 0}
    for label, count in zip(unique_labels, counts, strict=True):
        class_counts[INDEX_TO_LABEL[_safe_int(label, "split class label")]] = _safe_int(
            count,
            "split class count",
        )
    record_ids = groups[index_values].astype(str)
    f_mask = labels[index_values] == 2
    return SplitPartition(
        indices=tuple(_safe_int(value, "split index") for value in index_values),
        groups=group_values,
        class_counts=class_counts,
        f_208=_safe_int(np.sum(f_mask & (record_ids == "208")), "F support in 208"),
        f_213=_safe_int(np.sum(f_mask & (record_ids == "213")), "F support in 213"),
        f_outside_208_213=_safe_int(
            np.sum(f_mask & ~np.isin(record_ids, ["208", "213"])),
            "F support outside 208/213",
        ),
        indices_hash=hash_canonical(index_values.tolist()),
        groups_hash=hash_canonical(group_values),
    )


def generate_split_manifests(
    config: ResearchConfig,
    dataset: Stage2Dataset,
) -> tuple[SplitManifest, InnerSplitManifest]:
    """Generate deterministic outer and nested inner manifests in memory."""
    outer_entries: list[OuterFoldManifest] = []
    inner_entries: list[InnerFoldManifest] = []
    outer_splits = build_outer_splits(
        dataset.labels,
        dataset.groups,
        config.split_contract,
    )
    for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
        fold = fold_index + 1
        inner_train, inner_val = select_inner_split(
            outer_train,
            dataset.labels,
            dataset.groups,
            config.split_contract,
            fold_index=fold_index,
        )
        train_groups = set(dataset.groups[outer_train].astype(str).tolist())
        test_groups = set(dataset.groups[outer_test].astype(str).tolist())
        inner_train_groups = set(dataset.groups[inner_train].astype(str).tolist())
        inner_val_groups = set(dataset.groups[inner_val].astype(str).tolist())
        outer_overlap = tuple(sorted(train_groups & test_groups))
        inner_overlap = tuple(sorted(inner_train_groups & inner_val_groups))
        val_test_overlap = tuple(sorted(inner_val_groups & test_groups))
        if outer_overlap or inner_overlap or val_test_overlap:
            raise ResearchError(
                f"group leakage in fold {fold}",
                ExitCode.LEAKAGE,
                details={
                    "outer_overlap": outer_overlap,
                    "inner_overlap": inner_overlap,
                    "validation_test_overlap": val_test_overlap,
                },
            )
        if not set(inner_train).issubset(set(outer_train)) or not set(inner_val).issubset(
            set(outer_train)
        ):
            raise ResearchError(
                f"inner split escapes outer train in fold {fold}",
                ExitCode.LEAKAGE,
            )
        outer_entries.append(
            OuterFoldManifest(
                fold=fold,
                train=_partition(outer_train, dataset.labels, dataset.groups),
                test=_partition(outer_test, dataset.labels, dataset.groups),
                overlap_groups=outer_overlap,
            )
        )
        inner_entries.append(
            InnerFoldManifest(
                fold=fold,
                train=_partition(inner_train, dataset.labels, dataset.groups),
                validation=_partition(inner_val, dataset.labels, dataset.groups),
                outer_test_groups=tuple(sorted(test_groups)),
                train_validation_overlap=inner_overlap,
                validation_outer_test_overlap=val_test_overlap,
            )
        )

    outer_payload = {
        "schema_version": "stage2-splits-v2.4",
        "dataset_manifest_hash": dataset.manifest_hash,
        "splitter": "StratifiedGroupKFold",
        "split_random_state": config.split_contract.random_seed,
        "outer_folds": [entry.model_dump(mode="json") for entry in outer_entries],
    }
    outer_manifest = SplitManifest(
        dataset_manifest_hash=dataset.manifest_hash,
        split_random_state=config.split_contract.random_seed,
        outer_folds=tuple(outer_entries),
        manifest_hash=hash_canonical(outer_payload),
    )
    inner_payload = {
        "schema_version": "stage2-inner-splits-v2.4",
        "dataset_manifest_hash": dataset.manifest_hash,
        "outer_split_manifest_hash": outer_manifest.manifest_hash,
        "split_random_state": config.split_contract.random_seed,
        "inner_folds": [entry.model_dump(mode="json") for entry in inner_entries],
    }
    inner_manifest = InnerSplitManifest(
        dataset_manifest_hash=dataset.manifest_hash,
        outer_split_manifest_hash=outer_manifest.manifest_hash,
        split_random_state=config.split_contract.random_seed,
        inner_folds=tuple(inner_entries),
        manifest_hash=hash_canonical(inner_payload),
    )
    return outer_manifest, inner_manifest


def _validate_outer_hash(manifest: SplitManifest) -> None:
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    if hash_canonical(payload) != manifest.manifest_hash:
        raise ResearchError(
            "outer split manifest self-hash mismatch",
            ExitCode.DATA_INTEGRITY,
        )


def _validate_inner_hash(manifest: InnerSplitManifest) -> None:
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    if hash_canonical(payload) != manifest.manifest_hash:
        raise ResearchError(
            "inner split manifest self-hash mismatch",
            ExitCode.DATA_INTEGRITY,
        )


def _diagnostics_csv(
    outer: SplitManifest,
    inner: InnerSplitManifest,
) -> str:
    buffer = io.StringIO()
    fields = [
        "fold",
        "partition",
        "n_samples",
        "n_groups",
        "S",
        "V",
        "F",
        "F_208",
        "F_213",
        "F_outside_208_213",
        "indices_hash",
        "groups_hash",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    inner_by_fold = {entry.fold: entry for entry in inner.inner_folds}
    for outer_fold in outer.outer_folds:
        inner_fold = inner_by_fold[outer_fold.fold]
        partitions = (
            ("outer_train", outer_fold.train),
            ("outer_test", outer_fold.test),
            ("inner_train", inner_fold.train),
            ("inner_validation", inner_fold.validation),
        )
        for name, partition in partitions:
            writer.writerow(
                {
                    "fold": outer_fold.fold,
                    "partition": name,
                    "n_samples": len(partition.indices),
                    "n_groups": len(partition.groups),
                    "S": partition.class_counts["S"],
                    "V": partition.class_counts["V"],
                    "F": partition.class_counts["F"],
                    "F_208": partition.f_208,
                    "F_213": partition.f_213,
                    "F_outside_208_213": partition.f_outside_208_213,
                    "indices_hash": partition.indices_hash,
                    "groups_hash": partition.groups_hash,
                }
            )
    return buffer.getvalue()


def freeze_or_validate_splits(
    config: ResearchConfig,
    dataset: Stage2Dataset,
) -> tuple[SplitManifest, InnerSplitManifest, bool]:
    """Create split manifests once, then reject any future drift."""
    split_dir = config.output_root / "splits"
    outer_path = split_dir / "outer_splits_v2.4.json"
    inner_path = split_dir / "inner_splits_v2.4.json"
    diagnostics_path = split_dir / "split_diagnostics.csv"
    generated_outer, generated_inner = generate_split_manifests(config, dataset)
    created = False
    if outer_path.exists() != inner_path.exists():
        raise ResearchError(
            "only one split manifest exists",
            ExitCode.DATA_INTEGRITY,
        )
    if outer_path.exists():
        try:
            stored_outer = SplitManifest.model_validate(load_json(outer_path))
            stored_inner = InnerSplitManifest.model_validate(load_json(inner_path))
        except ValueError as error:
            raise ResearchError(
                "stored split manifest schema is invalid",
                ExitCode.DATA_INTEGRITY,
            ) from error
        _validate_outer_hash(stored_outer)
        _validate_inner_hash(stored_inner)
        if stored_outer != generated_outer or stored_inner != generated_inner:
            raise ResearchError(
                "frozen split manifests drift from the canonical splitter",
                ExitCode.REGRESSION,
            )
        outer_manifest, inner_manifest = stored_outer, stored_inner
    else:
        split_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(outer_path, generated_outer.model_dump(mode="json"))
        atomic_write_json(inner_path, generated_inner.model_dump(mode="json"))
        outer_manifest, inner_manifest = generated_outer, generated_inner
        created = True
    diagnostics = _diagnostics_csv(outer_manifest, inner_manifest)
    if diagnostics_path.exists():
        try:
            previous = diagnostics_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ResearchError(
                "cannot read split diagnostics",
                ExitCode.DATA_INTEGRITY,
            ) from error
        if previous != diagnostics:
            raise ResearchError("split diagnostics drift", ExitCode.REGRESSION)
    else:
        atomic_write_text(diagnostics_path, diagnostics)
    return outer_manifest, inner_manifest, created


def split_indices(
    outer: SplitManifest,
    inner: InnerSplitManifest,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one-based display fold to frozen index vectors."""
    try:
        outer_fold = next(item for item in outer.outer_folds if item.fold == fold)
        inner_fold = next(item for item in inner.inner_folds if item.fold == fold)
    except StopIteration as error:
        raise ResearchError(f"unknown fold: {fold}", ExitCode.ARGUMENT_ERROR) from error
    return (
        np.asarray(outer_fold.train.indices, dtype=np.int64),
        np.asarray(outer_fold.test.indices, dtype=np.int64),
        np.asarray(inner_fold.train.indices, dtype=np.int64),
        np.asarray(inner_fold.validation.indices, dtype=np.int64),
    )
