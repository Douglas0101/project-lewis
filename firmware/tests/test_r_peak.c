#include "harness.h"
#include "dsp/r_peak_detector.h"
#include "fixtures/generated/fixture_rpeak.h"
#include <math.h>
#include <stdint.h>

static void test_r_peak_synthetic(harness_result_t* r) {
    /* Sinal sintetico: 2 segundos @ 500 Hz = 1000 amostras, com 2 picos nitidos */
    const size_t len = 1000;
    const float fs = 500.0f;
    float sig[1000];

    for (size_t i = 0; i < len; i++) {
        sig[i] = 0.0f;
    }
    /* Pico em 250 e 750 */
    sig[250] = 1.0f;
    sig[750] = 1.0f;

    size_t peaks[LEWIS_RPEAK_MAX_PEAKS];
    size_t n_peaks = 0;
    int rc = lewis_detect_r_peaks(sig, len, fs, peaks, &n_peaks);

    harness_assert_int_eq(r, 0, rc, "detect_r_peaks rc");
    harness_assert_int_eq(r, 2, (int64_t)n_peaks, "n_peaks");
    /* Detector leve permite pequeno deslocamento (±5 amostras @ 500 Hz = 10 ms) */
    if (n_peaks >= 1) harness_assert_int_close(r, 250, (int64_t)peaks[0], 5, "peak[0]");
    if (n_peaks >= 2) harness_assert_int_close(r, 750, (int64_t)peaks[1], 5, "peak[1]");
}

static void test_r_peak_empty(harness_result_t* r) {
    float sig[10] = {0.0f};
    size_t peaks[LEWIS_RPEAK_MAX_PEAKS];
    size_t n_peaks = 0;
    int rc = lewis_detect_r_peaks(sig, 10, 500.0f, peaks, &n_peaks);
    harness_assert_int_eq(r, 0, rc, "empty rc");
    harness_assert_int_eq(r, 0, (int64_t)n_peaks, "empty n_peaks");
}

static void test_r_peak_vs_ampt(harness_result_t* r) {
    size_t peaks[LEWIS_RPEAK_MAX_PEAKS];
    size_t n_peaks = 0;
    int rc = lewis_detect_r_peaks(
        fixture_rpeak_signal, LEWIS_FIXTURE_RPEAK_LEN, 500.0f, peaks, &n_peaks);
    harness_assert_int_eq(r, 0, rc, "detect_r_peaks rc");

    int tp = 0, fn = 0, fp = 0;
    for (size_t i = 0; i < LEWIS_FIXTURE_RPEAK_EXPECTED_COUNT; i++) {
        bool matched = false;
        for (size_t j = 0; j < n_peaks; j++) {
            uint32_t diff = (peaks[j] > fixture_rpeak_expected[i])
                ? (uint32_t)(peaks[j] - fixture_rpeak_expected[i])
                : (uint32_t)(fixture_rpeak_expected[i] - peaks[j]);
            if (diff <= LEWIS_FIXTURE_RPEAK_TOL_SAMPLES) {
                matched = true;
                tp++;
                break;
            }
        }
        if (!matched) {
            fn++;
        }
    }
    fp = (int)n_peaks - tp;
    if (fp < 0) fp = 0;

    float sens = (tp + fn) > 0 ? (float)tp / (float)(tp + fn) : 0.0f;
    float ppv = (tp + fp) > 0 ? (float)tp / (float)(tp + fp) : 0.0f;

    harness_assert_true(r, sens >= 0.90f, "Sens >= 0.90");
    harness_assert_true(r, ppv >= 0.90f, "PPV >= 0.90");
}

void suite_r_peak_register(void) {
    harness_register("RPEAK", "synthetic_two_peaks", test_r_peak_synthetic);
    harness_register("RPEAK", "empty_signal", test_r_peak_empty);
    harness_register("RPEAK", "vs_ampt_python", test_r_peak_vs_ampt);
}
