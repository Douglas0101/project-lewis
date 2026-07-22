"""Gates de alinhamento temporal v3 (docs/rebuild_spec/02, DQ-01/DQ-02).

Prova que os índices de anotação são reescalonados do relógio nativo para o
relógio canônico de 500 Hz antes da segmentação e das features temporais:

- G-T2/G-T3: janelas produzidas pelo caminho produtivo correlacionam ≥0,99 com
  o sinal na posição reescalonada (e não na posição nativa);
- G-T5: RR dual-clock coincide dentro da tolerância de arredondamento;
- G-T7: ida e volta amostra↔tempo dentro da tolerância;
- contrato: índices fora do alcance do sinal são rejeitados.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wfdb

from src.features.pipeline import TARGET_FS, _load_raw_annotations, build_beat_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("100", "mitdb", "data/raw_mitbih/100", "data/processed/mitdb/100_MLII.npy", 360.0),
    ("800", "svdb", "data/raw_svdb/800", "data/processed/svdb/800_ECG1.npy", 128.0),
    ("I01", "incart", "data/raw_incart/I01", "data/processed/incart/I01_II.npy", 257.0),
]

DROP = {"~", "+", "x", "|"}


def _filtered_native_samples(raw_base: str) -> np.ndarray:
    ann = wfdb.rdann(str(PROJECT_ROOT / raw_base), "atr")
    keep = np.array([s not in DROP for s in ann.symbol])
    return np.asarray(ann.sample, dtype=np.int64)[keep]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("rec,ds,raw_path,npy_path,fs_nat", CASES)
def test_annotation_indices_rescaled_to_target_fs(rec, ds, raw_path, npy_path, fs_nat):
    if not (PROJECT_ROOT / npy_path).exists():
        pytest.skip("processed .npy ausente")
    r_peaks, labels = _load_raw_annotations(rec, ds)
    native = _filtered_native_samples(raw_path)
    assert len(r_peaks) == len(native) == len(labels)
    expected = np.rint(native.astype(np.float64) * (TARGET_FS / fs_nat)).astype(np.int64)
    assert np.array_equal(
        r_peaks, expected
    ), f"{ds}/{rec}: índices não correspondem a round(s*500/{fs_nat}) (DQ-01)"


@pytest.mark.parametrize("rec,ds,raw_path,npy_path,fs_nat", CASES)
def test_windows_align_with_rescaled_positions(rec, ds, raw_path, npy_path, fs_nat):
    """G-T3: janela do caminho produtivo ≈ sinal na posição reescalonada (≥0,99)."""
    if not (PROJECT_ROOT / npy_path).exists():
        pytest.skip("processed .npy ausente")
    sig_full = np.load(PROJECT_ROOT / npy_path).astype(np.float32)
    sig = sig_full[:60_000]  # 2 min bastam para o gate e mantêm o teste rápido
    r_peaks_all, labels_all = _load_raw_annotations(rec, ds)
    in_range = (r_peaks_all >= 250) & (r_peaks_all < len(sig) - 250)
    r_peaks = r_peaks_all[in_range]
    labels = labels_all[in_range]
    beats, X, y = build_beat_records(
        sig=sig,
        r_peaks=r_peaks,
        aami_labels=labels,
        record_id=rec,
        dataset=ds,
        lineage_path="test",
    )
    assert len(beats) > 0
    # usa apenas janelas 1000 ms (sem edge-pad do fallback 600 ms)
    rr_ms = np.diff(r_peaks) / TARGET_FS * 1000.0
    long_rr = np.concatenate([[rr_ms[0] if len(rr_ms) else 1e9], rr_ms]) >= 600.0
    checked = 0
    for seg_i, beat in enumerate(beats):
        bi = beat.beat_idx
        if bi >= len(r_peaks) or not long_rr[bi]:
            continue
        r = int(r_peaks[bi])
        xw = X[seg_i]
        c_rescaled = _corr(xw, sig[r - 250 : r + 250])
        assert c_rescaled >= 0.99, f"{ds}/{rec} beat {bi}: corr={c_rescaled:.4f} < 0.99"
        checked += 1
    assert checked >= 20, f"{ds}/{rec}: amostra insuficiente para o gate ({checked})"


@pytest.mark.parametrize("rec,ds,raw_path,npy_path,fs_nat", CASES)
def test_rr_dual_clock_matches(rec, ds, raw_path, npy_path, fs_nat):
    """G-T5: RR no relógio nativo (ms) == RR no relógio 500 Hz (ms), ±tol."""
    r_peaks, _ = _load_raw_annotations(rec, ds)
    native = _filtered_native_samples(raw_path)
    rr_native_ms = np.diff(native[:200]) / fs_nat * 1000.0
    rr_500_ms = np.diff(r_peaks[:200]) / TARGET_FS * 1000.0
    tol_ms = 1000.0 / TARGET_FS + 1e-6  # 2 ms (≤1 amostra @500 Hz de cada lado /2)
    delta = np.abs(rr_native_ms - rr_500_ms)
    assert (
        float(delta.max()) <= tol_ms + 1.0
    ), f"{ds}/{rec}: RR dual-clock diverge {delta.max():.3f} ms (tol {tol_ms:.3f})"


@pytest.mark.parametrize("rec,ds,raw_path,npy_path,fs_nat", CASES)
def test_roundtrip_sample_time_sample(rec, ds, raw_path, npy_path, fs_nat):
    """G-T7: s → t → s' dentro da tolerância declarada por dataset."""
    native = _filtered_native_samples(raw_path)[:1000]
    t = np.rint(native.astype(np.float64) * (TARGET_FS / fs_nat))
    s_back = np.rint(t * (fs_nat / TARGET_FS))
    tol = np.ceil(fs_nat / TARGET_FS) + 1
    assert int(np.abs(s_back - native).max()) <= tol


def test_temporal_error_bound_per_index():
    """G-T2: |t_i/f_t − τ_i| ≤ 0,5/f_t + ε para todo índice."""
    for rec, ds, raw_path, npy_path, fs_nat in CASES:
        r_peaks, _ = _load_raw_annotations(rec, ds)
        native = _filtered_native_samples(raw_path)
        err = np.abs(r_peaks / TARGET_FS - native / fs_nat)
        assert float(err.max()) <= 0.5 / TARGET_FS + 1e-9, f"{ds}/{rec}: {err.max():.6f}s"


def test_single_boundary_annotation_preserves_original_beat_indices():
    """A tolerated boundary drop must not renumber source annotation identities."""
    signal = np.random.default_rng(0).standard_normal(61_000).astype(np.float32)
    peaks = 300 + np.arange(101, dtype=np.int64) * 600
    peaks[50] = len(signal)
    symbols = np.asarray(["N"] * len(peaks), dtype=str)
    beats, _, _ = build_beat_records(
        sig=signal,
        r_peaks=peaks,
        aami_labels=symbols,
        record_id="boundary",
        dataset="mitdb",
        lineage_path="test",
        native_annotation_indices=peaks,
        original_symbols=symbols,
        source_sampling_rate=500.0,
    )

    assert [beat.beat_idx for beat in beats] == [index for index in range(101) if index != 50]


def test_native_clock_indices_rejected():
    """Contrato: índices fora do alcance do sinal (relógio errado) são rejeitados."""
    sig = np.random.default_rng(0).standard_normal(10_000).astype(np.float32)
    bad_peaks = np.array([500, 50_000])  # 50k excede len(sig) — típico de relógio nativo
    with pytest.raises(ValueError, match="relógio nativo|fora do alcance"):
        build_beat_records(
            sig=sig,
            r_peaks=bad_peaks,
            aami_labels=np.array(["N", "N"], dtype=object),
            record_id="test",
            dataset="mitdb",
            lineage_path="test",
        )
