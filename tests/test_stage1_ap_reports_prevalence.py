"""Guard against interpreting Average Precision without its prevalence baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_stage1_qg5 import ap_reference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "docs" / "stage1_inference_contract.md"


def test_stage1_ap_lift_uses_positive_prevalence() -> None:
    """The observed AP is below the abnormal-prevalence reference."""
    y_true = np.concatenate([np.ones(1920, dtype=np.int64), np.zeros(128, dtype=np.int64)])

    prevalence, lift, interpretation = ap_reference(0.9136421076851974, y_true)

    assert prevalence == pytest.approx(0.9375, abs=0.0)
    assert lift == pytest.approx(-0.02385789231480262, abs=1e-15)
    assert interpretation == "BELOW_PREVALENCE_REFERENCE"


def test_stage1_report_never_presents_ap_without_reference() -> None:
    """The forensic report must publish AP, prevalence, and AP lift together."""
    report = CONTRACT_PATH.read_text(encoding="utf-8")

    assert "average precision: `0.9136421077`" in report
    assert "positive prevalence: `0.9375`" in report
    assert "AP lift: `-0.0238578923`" in report
    assert "não há evidência de ganho de ranking" in report
    assert "A AP alta" not in report
