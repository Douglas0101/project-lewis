#include "harness.h"
#include "dsp/filter.h"
#include "dsp/normalizer.h"
#include "ml/inference.h"
#include "ml/quantization_params.h"
#include "fixtures/generated/fixture_pipeline.h"
#include <string.h>
#include <stdint.h>

#define PIPELINE_LEN 500

static void quantize_float_to_int8(const float* in, int8_t* out, size_t len) {
    const float scale = LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE;
    const int32_t zp = LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT;
    for (size_t i = 0; i < len; i++) {
        float normalized = in[i] / scale;
        int32_t q = (int32_t)(normalized + (normalized >= 0.0f ? 0.5f : -0.5f)) + zp;
        if (q > 127) q = 127;
        else if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

static void test_pipeline_beat(harness_result_t* r, int idx) {
    float frame[PIPELINE_LEN];
    int8_t quantized[PIPELINE_LEN];
    int8_t output[LEWIS_OUTPUT_LEN];
    lewis_filter_chain_t chain;

    /* Seleciona a fixture correta por indice. */
    const float* input = NULL;
    const int8_t* expected = NULL;
    switch (idx) {
        case 0: input = fixture_pipeline_input_0; expected = fixture_pipeline_expected_0; break;
        case 1: input = fixture_pipeline_input_1; expected = fixture_pipeline_expected_1; break;
        case 2: input = fixture_pipeline_input_2; expected = fixture_pipeline_expected_2; break;
        case 3: input = fixture_pipeline_input_3; expected = fixture_pipeline_expected_3; break;
        case 4: input = fixture_pipeline_input_4; expected = fixture_pipeline_expected_4; break;
        default:
            harness_assert_true(r, false, "invalid fixture index");
            return;
    }

    memcpy(frame, input, sizeof(frame));
    lewis_filter_chain_init(&chain);
    lewis_filter_chain_reset(&chain);
    lewis_filter_chain_process(&chain, frame, frame, PIPELINE_LEN);
    lewis_zscore_normalize(frame, PIPELINE_LEN);
    quantize_float_to_int8(frame, quantized, PIPELINE_LEN);

    bool ok = lewis_inference_init();
    harness_assert_true(r, ok, "inference_init");

    ok = lewis_inference_run(quantized, output);
    harness_assert_true(r, ok, "inference_run");

    for (int i = 0; i < LEWIS_OUTPUT_LEN; i++) {
        harness_assert_int8_eq(r, expected[i], output[i], "pipeline output");
    }
}

static void test_pipeline_beat_0(harness_result_t* r) { test_pipeline_beat(r, 0); }
static void test_pipeline_beat_1(harness_result_t* r) { test_pipeline_beat(r, 1); }
static void test_pipeline_beat_2(harness_result_t* r) { test_pipeline_beat(r, 2); }
static void test_pipeline_beat_3(harness_result_t* r) { test_pipeline_beat(r, 3); }
static void test_pipeline_beat_4(harness_result_t* r) { test_pipeline_beat(r, 4); }

void suite_pipeline_register(void) {
    harness_register("PIPELINE", "beat_0", test_pipeline_beat_0);
    harness_register("PIPELINE", "beat_1", test_pipeline_beat_1);
    harness_register("PIPELINE", "beat_2", test_pipeline_beat_2);
    harness_register("PIPELINE", "beat_3", test_pipeline_beat_3);
    harness_register("PIPELINE", "beat_4", test_pipeline_beat_4);
}
