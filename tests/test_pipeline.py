"""End-to-end pipeline tests for Camada 1 → Camada 2."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.data.aggregator import ECGAggregator


@pytest.fixture
def isolated_dirs(monkeypatch):
    """Provide temporary processed/lineage/DLQ directories."""
    tmp = Path(tempfile.mkdtemp(prefix="lewis_pipeline_test_"))
    processed = tmp / "processed"
    lineage = tmp / "lineage"
    dlq = tmp / ".dlq" / "preprocess_failures.jsonl"

    processed.mkdir(parents=True, exist_ok=True)
    lineage.mkdir(parents=True, exist_ok=True)
    dlq.parent.mkdir(parents=True, exist_ok=True)

    # Monkey-patch aggregator module defaults
    import src.data.aggregator as agg_mod
    import src.data.preprocessor as prep_mod

    monkeypatch.setattr(agg_mod, "CATALOG_PATH", tmp / "dataset_catalog.jsonl")
    monkeypatch.setattr(prep_mod, "PROCESSED_DIR", processed)
    monkeypatch.setattr(prep_mod, "LINEAGE_DIR", lineage)

    # Build a tiny catalog with one record per dataset
    catalog = tmp / "dataset_catalog.jsonl"
    entries = [
        {"record_name": "100", "dataset": "mitdb", "fs": 360.0, "n_sig": 2, "sig_len": 650000},
        {"record_name": "800", "dataset": "svdb", "fs": 250.0, "n_sig": 2, "sig_len": 450000},
        {"record_name": "I01", "dataset": "incart", "fs": 257.0, "n_sig": 12, "sig_len": 462600},
    ]
    with catalog.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    yield {"processed": processed, "lineage": lineage, "dlq": dlq}

    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.qg1
def test_pipeline_processes_mitbih_family(isolated_dirs):
    """Smoke test: process one record from each MIT-BIH-family dataset."""
    aggregator = ECGAggregator(dlq_path=isolated_dirs["dlq"])
    stats = aggregator.run(datasets=["mitdb", "svdb", "incart"])

    assert stats["mitdb"]["processed"] == 1
    assert stats["svdb"]["processed"] == 1
    assert stats["incart"]["processed"] == 1
    assert all(s["failed"] == 0 for s in stats.values())

    # Check processed outputs and lineage exist
    processed = isolated_dirs["processed"]
    lineage = isolated_dirs["lineage"]

    for dataset in ("mitdb", "svdb", "incart"):
        npy_files = list((processed / dataset).glob("*.npy"))
        lineage_files = list((lineage / dataset).glob("*_lineage.json"))
        assert len(npy_files) == 1, f"{dataset}: expected 1 .npy, got {len(npy_files)}"
        assert len(lineage_files) == 1, f"{dataset}: expected 1 lineage, got {len(lineage_files)}"

        # Verify lineage schema
        lin = json.loads(lineage_files[0].read_text(encoding="utf-8"))
        assert lin["dataset"] == dataset
        assert lin["pipeline"][0]["step"] == "load"
        assert lin["pipeline"][1]["step"] == "resample"
        assert lin["pipeline"][2]["step"] == "filter"
        assert lin["pipeline"][3]["step"] == "detrend"
        assert lin["pipeline"][4]["step"] == "normalize"


@pytest.mark.qg1
def test_pipeline_idempotency(isolated_dirs):
    """Second run must skip already processed records."""
    aggregator = ECGAggregator(dlq_path=isolated_dirs["dlq"])
    stats1 = aggregator.run(datasets=["mitdb"])
    assert stats1["mitdb"]["processed"] == 1

    stats2 = aggregator.run(datasets=["mitdb"])
    assert stats2["mitdb"]["processed"] == 0
    assert stats2["mitdb"]["failed"] == 0
    assert stats2["mitdb"]["skipped"] == 0


@pytest.mark.qg1
def test_pipeline_dlq_empty_on_success(isolated_dirs):
    """DLQ must remain empty when processing succeeds."""
    aggregator = ECGAggregator(dlq_path=isolated_dirs["dlq"])
    aggregator.run(datasets=["mitdb"])
    assert not isolated_dirs["dlq"].exists() or isolated_dirs["dlq"].read_text().strip() == ""
