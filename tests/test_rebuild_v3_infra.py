"""Testes da infraestrutura v3: calibração, bundle, attestation, auditoria de domínio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.bundle.bundle_v3 import (
    BUNDLE_COMPONENTS,
    BundleState,
    build_bundle_manifest,
    canonical_bundle_digest,
    verify_bundle,
)
from src.evaluation.calibration_v3 import (
    evaluate_binary,
    fit_binary_calibrator,
    nll_binary,
    select_binary_calibrator,
    select_multiclass_calibrator,
)
from src.evaluation.domain_audit import (
    conditional_metrics,
    counterfactual_suite,
    dataset_id_probe,
)
from src.security.artifact_bundle_contracts import (
    ARTIFACT_BUNDLE_PREDICATE_V2,
    BundleOutcome,
    BundlePredicate,
    BundleStatement,
    BundleSubject,
)


def _sha(n: int = 1) -> str:
    return f"{n:064x}"[:64]


# ---------------------------------------------------------------------------
# calibração
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_temperature_improves_nll_on_overconfident_binary(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal(4000)
        p_true = 1 / (1 + np.exp(-1.0 * z))
        y = (rng.random(4000) < p_true).astype(float)
        p_over = 1 / (1 + np.exp(-3 * z))  # superconfiante (slope 3 vs 1 real)
        cal = fit_binary_calibrator(p_over, y, "temperature")
        assert cal.params["T"] > 1.0
        assert nll_binary(y, cal.transform(p_over)) < nll_binary(y, p_over)

    def test_select_binary_never_worse_than_identity_on_partition(self):
        rng = np.random.default_rng(1)
        p = rng.random(2000)
        y = (rng.random(2000) < p).astype(float)
        best, scores = select_binary_calibrator(p, y)
        assert scores[best.method] <= scores["identity"] + 1e-9

    def test_multiclass_temperature_shapes(self):
        rng = np.random.default_rng(2)
        proba = rng.dirichlet([2, 1, 1], size=500)
        y = rng.choice(3, size=500)
        best, _ = select_multiclass_calibrator(proba, y, methods=("identity", "temperature"))
        out = best.transform(proba)
        assert out.shape == proba.shape
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_binary_keys(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        res = evaluate_binary(y, p)
        assert set(res) == {"nll", "brier", "ece", "reliability_bins"}
        assert res["brier"] == pytest.approx(0.085, abs=1e-9)


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def _make_components(tmp: Path) -> dict[str, Path]:
    files = {}
    for name in BUNDLE_COMPONENTS:
        p = tmp / f"{name}.bin"
        p.write_bytes(f"conteúdo-{name}".encode("utf-8"))
        files[name] = p
    return files


class TestBundle:
    def test_build_and_verify_valid(self, tmp_path: Path):
        files = _make_components(tmp_path)
        manifest = build_bundle_manifest("run-1", "2026-07-18T00:00:00Z", files)
        resolver = lambda name: files[name]  # noqa: E731
        state, problems = verify_bundle(manifest, resolver)
        assert state == BundleState.BUNDLE_VALID
        assert problems == []

    def test_incomplete_rejected(self, tmp_path: Path):
        files = _make_components(tmp_path)
        del files["model"]
        with pytest.raises(ValueError, match="componentes ausentes"):
            build_bundle_manifest("run-1", "2026-07-18T00:00:00Z", files)

    def test_hash_mismatch_on_tamper(self, tmp_path: Path):
        files = _make_components(tmp_path)
        manifest = build_bundle_manifest("run-1", "2026-07-18T00:00:00Z", files)
        files["model"].write_bytes(b"adulterado")
        state, problems = verify_bundle(manifest, lambda name: files[name])
        assert state == BundleState.HASH_MISMATCH
        assert any("model" in p for p in problems)

    def test_generation_mismatch(self, tmp_path: Path):
        files = _make_components(tmp_path)
        manifest = build_bundle_manifest(
            "run-1", "2026-07-18T00:00:00Z", files,
            extra={"generation_ids": {"model": "run-1", "scaler": "run-1",
                                      "calibrator": "run-1", "threshold_policy": "run-OLD"}},
        )
        state, _ = verify_bundle(manifest, lambda name: files[name])
        assert state == BundleState.GENERATION_MISMATCH

    def test_gate_failed(self, tmp_path: Path):
        files = _make_components(tmp_path)
        manifest = build_bundle_manifest("run-1", "2026-07-18T00:00:00Z", files)
        state, _ = verify_bundle(manifest, lambda name: files[name], metrics_gate=lambda: False)
        assert state == BundleState.GATE_FAILED

    def test_digest_canonical_domain_separated(self):
        c1 = {"a": _sha(1), "b": _sha(2)}
        c2 = {"ab": _sha(1), "": _sha(2)}
        assert canonical_bundle_digest(c1) != canonical_bundle_digest(c2)
        with pytest.raises(ValueError):
            canonical_bundle_digest({"x": "not-a-hash"})


# ---------------------------------------------------------------------------
# attestation
# ---------------------------------------------------------------------------

class TestAttestationContracts:
    def _predicate(self) -> BundlePredicate:
        return BundlePredicate(
            bundleDigest=_sha(1),
            components={"model": _sha(2), "scaler": _sha(3)},
            trainingRunId="run-1",
            sourceRevision="abc123",
            environmentHash=_sha(4),
            decisionId="dec-1",
            nonce="nonce-1",
            sequence=0,
            validFromUtc="2026-07-18T00:00:00Z",
            validUntilUtc="2026-07-19T00:00:00Z",
            outcome=BundleOutcome.APPROVED_FOR_AUDIT,
        )

    def test_statement_roundtrip(self):
        stmt = BundleStatement(
            subject=[BundleSubject(name="bundle-v3", digest={"sha256": _sha(1)})],
            predicate=self._predicate(),
        )
        data = json.loads(stmt.model_dump_json(by_alias=True))
        assert data["_type"] == "https://in-toto.io/Statement/v1"
        assert data["predicateType"] == ARTIFACT_BUNDLE_PREDICATE_V2
        assert data["predicate"]["shadow"] is True
        assert data["predicate"]["operational"] is False

    def test_strict_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            BundlePredicate(**{**self._predicate().model_dump(), "campo_extra": 1})

    def test_strict_rejects_bad_hash(self):
        with pytest.raises(ValidationError):
            BundlePredicate(**{**self._predicate().model_dump(), "bundleDigest": "xyz"})


# ---------------------------------------------------------------------------
# auditoria de domínio
# ---------------------------------------------------------------------------

class TestDomainAudit:
    def test_probe_detects_shortcut(self):
        rng = np.random.default_rng(0)
        n = 600
        ds = np.repeat(["a", "b", "c"], n // 3)
        emb = np.column_stack(
            [(ds == "a") * 3.0, (ds == "b") * 3.0, (ds == "c") * 3.0]
        ) + rng.standard_normal((n, 3)) * 0.1
        res = dataset_id_probe(emb, ds, groups=np.arange(n) % 20)
        assert res.shortcut
        assert res.balanced_acc > 0.9

    def test_probe_clean_embeddings(self):
        rng = np.random.default_rng(1)
        n = 600
        ds = np.repeat(["a", "b", "c"], n // 3)
        emb = rng.standard_normal((n, 8))
        res = dataset_id_probe(emb, ds, groups=np.arange(n) % 20)
        assert not res.shortcut
        assert res.balanced_acc < 0.6

    def test_conditional_metrics(self):
        import pandas as pd

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        df = pd.DataFrame({"dataset": ["d1", "d1", "d1", "d1", "d2", "d2"]})
        out = conditional_metrics(y_true, y_pred, df, ["N", "Anormal"], by="dataset")
        assert out["d1"]["N"] == 0.5
        assert out["d1"]["Anormal"] == 0.5
        assert out["d2"]["N"] == 1.0
        assert out["d2"]["Anormal"] == 1.0

    def test_counterfactual_invariant_model(self):
        X = np.random.default_rng(0).standard_normal((8, 500, 1)).astype(np.float32)
        predict_fn = lambda x: np.tile([0.7, 0.3], (len(x), 1))  # noqa: E731
        out = counterfactual_suite(predict_fn, X)
        for name, res in out.items():
            assert res["delta_p_mean"] == 0.0, name

    def test_counterfactual_sensitive_model(self):
        X = np.ones((8, 500, 1), dtype=np.float32)

        def predict_fn(x):
            mean = x.mean(axis=(1, 2))
            return np.column_stack([mean > 0, mean <= 0]).astype(float)

        out = counterfactual_suite(predict_fn, X)
        assert out["polarity_invert"]["argmax_flip_rate"] == 1.0
        assert out["amplitude_x2.0"]["argmax_flip_rate"] == 0.0
