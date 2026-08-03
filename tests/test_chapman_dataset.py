"""Tests for the Chapman pre-training dataset split (Camada 4).

Covers the stability fixes for QG4:
* train split reshuffles record order every epoch (deterministic given seed);
* val split is deterministic;
* ``estimate_n_segments`` counts segments without iterating the generator.

Fixtures are fully synthetic: a tiny JSONL catalog + constant-value ``.npy``
signals whose value encodes the record index, making yield order observable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.models.chapman_dataset as cd
from src.models.pretrain_chapman import _checkpoint_epoch_metrics

SEGMENT_LEN = 500
SIG_LEN = 1000  # 2 segments per record
VAL_RATIO = 0.25
SEED = 42


def _write_npy(processed_dir: Path, record_name: str, value: float) -> None:
    np.save(processed_dir / f"{record_name}_II.npy", np.full(SIG_LEN, value, dtype=np.float32))


@pytest.fixture()
def synthetic_data(tmp_path):
    """Catalog with 12 mapped records + noise entries, and .npy signals."""
    catalog = tmp_path / "dataset_catalog.jsonl"
    processed = tmp_path / "processed" / "chapman"
    processed.mkdir(parents=True)

    entries = []
    for i in range(12):
        name = f"JS{i + 1:05d}"
        entries.append(
            {
                "record_name": name,
                "dataset": "chapman",
                "diagnosis": "426177001" if i % 2 == 0 else "164865005",
            }
        )
        _write_npy(processed, name, float(i + 1))
    # noise: non-chapman record, unmapped diagnosis, missing npy
    entries.append({"record_name": "100", "dataset": "mitdb", "diagnosis": "426177001"})
    entries.append({"record_name": "JS99998", "dataset": "chapman", "diagnosis": "999999999"})
    entries.append({"record_name": "JS99999", "dataset": "chapman", "diagnosis": "426177001"})

    catalog.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return {"catalog": catalog, "processed": processed}


@pytest.fixture()
def split_sets(synthetic_data):
    """The record sets produced by the split for the synthetic catalog."""
    train_set, val_set = cd.chapman_split_record_sets(
        val_ratio=VAL_RATIO,
        seed=SEED,
        catalog_path=synthetic_data["catalog"],
    )
    usable = {f"JS{i + 1:05d}" for i in range(12)}
    return {"train": train_set & usable, "val": val_set & usable}


def _split_kwargs(synthetic_data):
    return dict(
        val_ratio=VAL_RATIO,
        batch_size=4,
        segment_len=SEGMENT_LEN,
        seed=SEED,
        catalog_path=synthetic_data["catalog"],
        processed_dir=synthetic_data["processed"],
    )


def _pass_record_ids(ds) -> list[int]:
    """Record ids (constant signal values) seen in one full pass over ``ds``."""
    ids: list[int] = []
    for X_batch, _y in ds:
        ids.extend(int(v) for v in X_batch.numpy()[:, 0, 0])
    return ids


def test_train_split_reshuffles_each_iteration(synthetic_data, split_sets):
    train_ds, _ = cd.chapman_train_val_split(**_split_kwargs(synthetic_data))
    pass1 = _pass_record_ids(train_ds)
    pass2 = _pass_record_ids(train_ds)

    assert sorted(pass1) == sorted(pass2), "same segments in every pass"
    assert len(pass1) == len(split_sets["train"]) * 2
    assert pass1 != pass2, "train order must reshuffle between epochs"


def test_train_split_deterministic_given_seed(synthetic_data):
    ds_a, _ = cd.chapman_train_val_split(**_split_kwargs(synthetic_data))
    ds_b, _ = cd.chapman_train_val_split(**_split_kwargs(synthetic_data))
    assert _pass_record_ids(ds_a) == _pass_record_ids(ds_b)


def test_val_split_is_deterministic(synthetic_data, split_sets):
    _, val_ds = cd.chapman_train_val_split(**_split_kwargs(synthetic_data))
    pass1 = _pass_record_ids(val_ds)
    pass2 = _pass_record_ids(val_ds)
    assert pass1 == pass2
    assert len(pass1) == len(split_sets["val"]) * 2


def test_estimate_n_segments_matches_usable_records(synthetic_data, split_sets):
    common = dict(
        catalog_path=synthetic_data["catalog"],
        processed_dir=synthetic_data["processed"],
        segment_len=SEGMENT_LEN,
    )
    assert cd.estimate_n_segments(split_sets["train"], **common) == len(split_sets["train"]) * 2
    assert cd.estimate_n_segments(split_sets["val"], **common) == len(split_sets["val"]) * 2


def test_checkpoint_epoch_metrics_legacy_val_loss_monitor():
    """Monitor legado val_loss: checkpoint = argmin val_loss (compatível)."""
    history = {
        "val_loss": [0.50, 0.30, 0.44],
        "val_auc_roc": [0.60, 0.71, 0.65],
    }
    best = _checkpoint_epoch_metrics(history)
    assert best["best_epoch"] == 2
    assert best["val_loss"] == pytest.approx(0.30)
    assert best["val_auc_roc"] == pytest.approx(0.71)


def test_checkpoint_epoch_metrics_val_auc_pr_monitor():
    """Protocolo v2: QG4 julga a época do checkpoint (argmax val_auc_pr),
    lendo a perda do gate NESSA época — não no argmin da perda."""
    history = {
        "val_loss": [0.40, 0.30, 0.35],  # argmin na época 2
        "val_auc_pr": [0.60, 0.65, 0.70],  # checkpoint (argmax) na época 3
        "val_auc_roc": [0.80, 0.85, 0.83],
    }
    best = _checkpoint_epoch_metrics(history, checkpoint_monitor="val_auc_pr")
    assert best["best_epoch"] == 3
    assert best["val_loss"] == pytest.approx(0.35)  # perda na época do checkpoint
    assert best["val_auc_roc"] == pytest.approx(0.83)


def test_checkpoint_epoch_metrics_focal_reads_bce_monitor_at_checkpoint():
    """Focal/weighted runs: gate lê o BCE monitor na época do checkpoint."""
    history = {
        "val_loss": [0.10, 0.05, 0.03],  # focal: sempre decrescente
        "val_bce_monitor": [0.40, 0.35, 0.42],
        "val_auc_pr": [0.60, 0.70, 0.65],  # checkpoint na época 2
        "val_auc_roc": [0.80, 0.83, 0.85],
    }
    best = _checkpoint_epoch_metrics(
        history, checkpoint_monitor="val_auc_pr", gate_loss_key="val_bce_monitor"
    )
    assert best["best_epoch"] == 2
    assert best["val_loss"] == pytest.approx(0.35)
    assert best["val_auc_roc"] == pytest.approx(0.83)


def test_checkpoint_epoch_metrics_handles_missing_val():
    best = _checkpoint_epoch_metrics({"loss": [0.1]})
    assert np.isnan(best["val_loss"])
    assert np.isnan(best["val_auc_roc"])


# ---------------------------------------------------------------------------
# FASE 2 — pipeline de dados: repeat, shapes, labels, NaN, sem warning
# ---------------------------------------------------------------------------


def test_batches_have_consistent_shapes_and_binary_labels(synthetic_data):
    from src.models.pretrain_chapman import build_datasets

    train_ds, val_ds, steps, val_steps = build_datasets(
        val_ratio=VAL_RATIO,
        batch_size=4,
        segment_len=SEGMENT_LEN,
        seed=SEED,
        steps_per_epoch=3,
        validation_steps=2,
        catalog_path=synthetic_data["catalog"],
        processed_dir=synthetic_data["processed"],
    )
    for X, y in train_ds.take(3):
        assert X.shape[1:] == (SEGMENT_LEN, 1)
        assert y.shape[1:] == (5,)
        assert not bool(np.isnan(X.numpy()).any())
        assert not bool(np.isinf(X.numpy()).any())
        assert set(np.unique(y.numpy())).issubset({0.0, 1.0})
    assert steps == 3
    assert val_steps == 2


def test_datasets_are_repeated_when_steps_defined(synthetic_data):
    import tensorflow as tf

    from src.models.pretrain_chapman import build_datasets

    train_ds, val_ds, _, _ = build_datasets(
        val_ratio=VAL_RATIO,
        batch_size=4,
        segment_len=SEGMENT_LEN,
        seed=SEED,
        steps_per_epoch=3,
        validation_steps=2,
        catalog_path=synthetic_data["catalog"],
        processed_dir=synthetic_data["processed"],
    )
    infinite = tf.data.INFINITE_CARDINALITY
    assert train_ds.cardinality() == infinite
    assert val_ds.cardinality() == infinite


def test_fit_one_epoch_has_no_ran_out_of_data_warning(synthetic_data):
    """Regression: validation-to-exhaustion must not warn (Keras 3 false positive)."""
    import warnings

    from src.models.backbone_1d import build_backbone_1d_multilabel
    from src.models.pretrain_chapman import build_datasets

    train_ds, val_ds, steps, val_steps = build_datasets(
        val_ratio=VAL_RATIO,
        batch_size=4,
        segment_len=SEGMENT_LEN,
        seed=SEED,
        steps_per_epoch=3,
        validation_steps=2,
        catalog_path=synthetic_data["catalog"],
        processed_dir=synthetic_data["processed"],
    )
    model = build_backbone_1d_multilabel(input_len=SEGMENT_LEN, num_classes=5)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            epochs=1,
            verbose=0,
        )
    messages = [str(w.message) for w in caught]
    assert not any("ran out of data" in m for m in messages), messages
